#!/usr/bin/env python3
"""Prompt K: Physician review pack from publishable generated reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.privacy import sanitize_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--generated_reports", required=True)
    p.add_argument("--output_dir", default="outputs/publishable/physician_review")
    p.add_argument("--cases_per_group", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    df = pd.read_csv(ROOT / args.manifest)
    weak_centers = ["wuda", "shiyan", "xiangyang", "jingzhou"]
    sub = df[(df["center_id"].isin(weak_centers)) & (df["needs_pseudo_report"] == 1)]
    gen = ROOT / args.generated_reports
    out = ROOT / args.output_dir
    mats = out / "case_materials"
    mats.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid in weak_centers:
        csub = sub[sub["center_id"] == cid].sample(min(args.cases_per_group, len(sub[sub["center_id"] == cid])), random_state=args.seed)
        for _, r in csub.iterrows():
            case = str(r["case_id"])
            gp = gen / f"{case}.json"
            text = ""
            if gp.is_file():
                text = json.loads(gp.read_text()).get("generated_report", "")
            text = sanitize_text(text)[:2000]
            (mats / f"{cid}_{case}.txt").write_text(text, encoding="utf-8")
            rows.append({"case_id": case, "center_id": cid, "binary_label": r["binary_label"], "review_group": "report_missing"})
    pd.DataFrame(rows).to_csv(out / "review_cases.csv", index=False)
    (out / "review_guide.md").write_text(
        "# Physician blind review (publishable)\n\nRate 1-5: OCT consistency, colposcopy consistency, clinical context, impression, recommendation, overdiagnosis risk, overall quality.\n",
        encoding="utf-8",
    )
    print(f"Exported {len(rows)} cases to {out}")


if __name__ == "__main__":
    main()
