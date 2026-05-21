#!/usr/bin/env python3
"""Step 6: Pseudo-report QC and manifest update."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distillation.quality_control import qc_pseudo_report
from src.utils.config import load_config, resolve_project_root
from src.utils.io import read_json, write_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=None)
    p.add_argument("--pseudo_report_dir", default=None)
    p.add_argument("--output_manifest", default=None)
    p.add_argument("--config", default="configs/data.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, resolve_project_root())
    manifest_path = Path(args.manifest or cfg["manifest"]["full"])
    pseudo_dir = Path(args.pseudo_report_dir or cfg["outputs"]["pseudo_reports"])
    out_manifest = Path(args.output_manifest or cfg["manifest"]["with_pseudo"])

    df = pd.read_csv(manifest_path)
    for col in ("pseudo_report_path", "pseudo_report_text"):
        if col in df.columns:
            df[col] = df[col].astype(object)
    qc_rows = []
    for i, row in df.iterrows():
        if int(row.get("has_real_report", 0)) == 1:
            continue
        pr_path = pseudo_dir / str(row["center_id"]) / f"{row['case_id']}.json"
        if not pr_path.is_file():
            continue
        report = read_json(pr_path)
        qc = qc_pseudo_report(report, row.to_dict())
        df.at[i, "has_pseudo_report"] = 1
        df.at[i, "pseudo_report_path"] = str(pr_path)
        df.at[i, "pseudo_report_text"] = json.dumps(report, ensure_ascii=False)[:2000]
        for k, v in qc.items():
            df.at[i, k] = v
        qc_rows.append({"case_id": row["case_id"], **qc})

    df.to_csv(out_manifest, index=False, encoding="utf-8-sig")
    qc_dir = Path(cfg["outputs"]["qc"])
    write_csv(pd.DataFrame(qc_rows), qc_dir / "pseudo_report_qc_cases.csv")
    summary = pd.DataFrame(
        [
            {
                "n_qc": len(qc_rows),
                "pass_rate": pd.DataFrame(qc_rows)["pseudo_report_pass_qc"].mean() if qc_rows else 0,
                "mean_qc_score": pd.DataFrame(qc_rows)["qc_score"].mean() if qc_rows else 0,
                "mean_weight": pd.DataFrame(qc_rows)["pseudo_training_weight"].mean() if qc_rows else 0,
            }
        ]
    )
    write_csv(summary, Path(cfg["outputs"]["tables"]) / "pseudo_report_quality_summary.csv")
    logger.info("QC done: %s -> %s", len(qc_rows), out_manifest)


if __name__ == "__main__":
    main()
