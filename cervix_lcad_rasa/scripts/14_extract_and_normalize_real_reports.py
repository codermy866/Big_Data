#!/usr/bin/env python3
"""Prompt B: Extract and normalize real reports for Enshi/Jingzhou."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.report_discovery import discover_report_path
from src.reports.report_extraction import composite_from_row, extract_raw_text
from src.reports.report_normalization import normalize_sections, SECTION_KEYS
from src.utils.io import write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="outputs/manifests/full_manifest_with_pseudo_reports.csv")
    p.add_argument("--output_manifest", default="outputs/publishable/manifests/full_manifest_with_normalized_real_reports.csv")
    p.add_argument("--output_dir", default="outputs/publishable/reports")
    args = p.parse_args()
    df = pd.read_csv(ROOT / args.manifest)
    out_dir = ROOT / args.output_dir / "real_report_texts"
    out_dir.mkdir(parents=True, exist_ok=True)
    status_rows = []
    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        center_id = str(row["center_id"])
        has_real = int(row.get("has_real_report", 0)) == 1
        rep_path = str(row.get("real_report_path", "") or "").strip()
        if not rep_path or not Path(rep_path).is_file():
            rep_path = discover_report_path(row.get("colposcopy_paths", ""))
        raw, src_type, flags = ("", "none", [])
        if has_real and rep_path:
            raw, src_type, flags = extract_raw_text(Path(rep_path))
        if has_real and len(raw) < 30:
            raw, src_type = composite_from_row(row.to_dict())
            flags.append("used_composite_fallback")
        norm = normalize_sections(raw, row.to_dict()) if has_real else {"normalized_report": {k: "" for k in SECTION_KEYS}, "reference_report_text": "", "extraction_confidence": 0.0, "normalization_flags": []}
        payload = {
            "case_id": case_id,
            "center_id": center_id,
            "report_source_path": rep_path,
            "report_source_type": src_type,
            "raw_extracted_text": raw[:8000],
            "normalized_report": norm["normalized_report"],
            "normalization_flags": norm.get("normalization_flags", []) + flags,
            "needs_manual_review": len(raw) < 30 and has_real,
            "extraction_confidence": norm.get("extraction_confidence", 0.0),
        }
        (out_dir / center_id).mkdir(parents=True, exist_ok=True)
        (out_dir / center_id / f"{case_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        ref_text = norm.get("reference_report_text", "")
        for k in SECTION_KEYS:
            df.loc[row.name, f"reference_{k}"] = norm["normalized_report"].get(k, "")
        df.loc[row.name, "reference_report_text"] = ref_text
        df.loc[row.name, "real_report_source_type"] = src_type
        status_rows.append(
            {
                "case_id": case_id,
                "center_id": center_id,
                "has_real_report": has_real,
                "success": int(has_real and len(ref_text) >= 20),
                "source_type": src_type,
                "text_len": len(ref_text),
            }
        )
    out_m = ROOT / args.output_manifest
    out_m.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_m, index=False)
    write_csv(pd.DataFrame(status_rows), ROOT / "outputs/publishable/tables/real_report_extraction_status.csv")
    print(f"Normalized manifest: {out_m} | real success rate: {pd.DataFrame(status_rows)['success'].mean():.1%}")


if __name__ == "__main__":
    main()
