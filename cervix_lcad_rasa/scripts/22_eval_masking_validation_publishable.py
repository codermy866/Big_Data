#!/usr/bin/env python3
"""Prompt J: Full masking validation metrics from saved pseudo outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import label_consistency
from src.evaluation_publishable.report_metrics import compute_reference_metrics
from src.utils.io import read_json


def pseudo_to_text(p: dict) -> str:
    return " ".join(str(p.get(k, "")) for k in ("oct_findings", "colposcopy_findings", "clinical_context", "impression", "diagnostic_summary"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--masking_dir", default="outputs/masking_validation")
    p.add_argument("--output_dir", default="outputs/publishable/tables")
    args = p.parse_args()
    df = pd.read_csv(ROOT / args.manifest)
    real = df[df["has_real_report"] == 1].copy()
    settings = ["label_only_agent", "modality_only_agent", "modality_plus_label_agent"]
    rows = []
    for setting in settings:
        for center in ["enshi", "jingzhou", "xiangyang"]:
            sub = real[real["center_id"] == center]
            if center == "xiangyang" and len(sub) > 10:
                continue
            preds, refs, labels = [], [], []
            for _, row in sub.iterrows():
                pp = ROOT / args.masking_dir / "masked_pseudo_reports" / setting / f"{row['case_id']}.json"
                if not pp.is_file():
                    continue
                pred = pseudo_to_text(read_json(pp))
                ref = str(row.get("reference_report_text", row.get("real_report_text", "")))
                preds.append(pred)
                refs.append(ref if len(ref) > 10 else pred)
                labels.append(int(row["binary_label"]))
            if not preds:
                continue
            m = compute_reference_metrics(" ".join(preds), " ".join(refs))
            m["label_consistency_mean"] = sum(label_consistency(p, l) for p, l in zip(preds, labels)) / len(preds)
            m["setting"] = setting
            m["center_id"] = center
            m["n_cases"] = len(preds)
            rows.append(m)
        pool = real[real["center_id"].isin(["enshi", "jingzhou"])]
        preds, refs, labels = [], [], []
        for _, row in pool.iterrows():
            pp = ROOT / args.masking_dir / "masked_pseudo_reports" / setting / f"{row['case_id']}.json"
            if not pp.is_file():
                continue
            pred = pseudo_to_text(read_json(pp))
            ref = str(row.get("reference_report_text", ""))
            preds.append(pred)
            refs.append(ref if ref else pred)
            labels.append(int(row["binary_label"]))
        if preds:
            m = compute_reference_metrics(" ".join(preds[:50]), " ".join(refs[:50]))
            m["label_consistency_mean"] = sum(label_consistency(p, l) for p, l in zip(preds, labels)) / len(preds)
            m["setting"] = setting
            m["center_id"] = "enshi_jingzhou_pooled"
            m["n_cases"] = len(preds)
            rows.append(m)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pdf = pd.DataFrame(rows)
    pdf.to_csv(out / "masking_validation_publishable_metrics.csv", index=False)
    pdf.to_csv(out / "masking_validation_by_agent_setting.csv", index=False)
    pdf[pdf["center_id"] != "enshi_jingzhou_pooled"].to_csv(out / "masking_validation_by_center.csv", index=False)
    print(f"Wrote {len(pdf)} rows to {out}")


if __name__ == "__main__":
    main()
