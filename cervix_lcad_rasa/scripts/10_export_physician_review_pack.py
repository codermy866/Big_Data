#!/usr/bin/env python3
"""Step 11: Physician blind review package export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_project_root
from src.utils.io import write_csv
from src.utils.logger import get_logger
from src.utils.privacy import sanitize_text

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/physician_review.yaml")
    p.add_argument("--manifest", default=None)
    p.add_argument("--generated_reports", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--cases_per_center", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, resolve_project_root())
    manifest = Path(args.manifest or cfg["manifest"])
    gen_dir = Path(args.generated_reports or cfg["generated_reports"])
    out_dir = Path(args.output_dir or cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_per = args.cases_per_center or cfg.get("cases_per_center", 50)
    weak_centers = cfg.get("report_missing_centers", [])

    df = pd.read_csv(manifest)
    df = df[df["center_id"].isin(weak_centers) & (df.get("pseudo_report_pass_qc", 0) == 1)]
    rows = []
    materials = out_dir / "case_materials"
    materials.mkdir(parents=True, exist_ok=True)
    for cid in weak_centers:
        sub = df[df["center_id"] == cid].head(n_per)
        for _, r in sub.iterrows():
            case_id = r["case_id"]
            text = sanitize_text(str(r.get("pseudo_report_text", ""))[:1500])
            mat_path = materials / f"{cid}_{case_id}.txt"
            mat_path.write_text(text, encoding="utf-8")
            rows.append(
                {
                    "case_id": case_id,
                    "center_id": cid,
                    "binary_label": r.get("binary_label"),
                    "material_path": str(mat_path),
                    "oct_consistency_score": "",
                    "colposcopy_consistency_score": "",
                    "clinical_context_score": "",
                    "impression_score": "",
                    "recommendation_score": "",
                    "overdiagnosis_risk": "",
                    "overall_quality": "",
                    "comments": "",
                }
            )
    write_csv(pd.DataFrame(rows), out_dir / "review_cases.csv")
    guide = out_dir / "review_guide.md"
    guide.write_text(
        "# Physician review guide\n\n"
        "Pseudo reports are **weak supervision**, not real clinical reports.\n"
        "Score 1–5 per dimension. Do not enter patient identifiers.\n",
        encoding="utf-8",
    )
    try:
        pd.DataFrame(rows).to_excel(out_dir / "review_form_template.xlsx", index=False)
    except Exception:
        write_csv(pd.DataFrame(rows), out_dir / "review_form_template.csv")
    logger.info("Exported %d review cases to %s", len(rows), out_dir)


if __name__ == "__main__":
    main()
