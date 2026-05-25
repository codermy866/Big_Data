#!/usr/bin/env python3
"""
Compute unified metrics from predictions CSV.

Input CSV required columns: y_true, y_prob
Optional: patient_id, exam_id, split

Output matches results/metrics_template.csv schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

JBD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JBD_ROOT))

from src.experiment_log import build_run_record, append_jsonl, git_commit
from src.metrics import METRIC_COLUMNS, bootstrap_ci


def parse_args():
    p = argparse.ArgumentParser(description="JBD unified metrics")
    p.add_argument("--predictions", required=True, help="CSV with y_true, y_prob")
    p.add_argument("--output", required=True, help="Output metrics CSV path")
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--model-variant", required=True)
    p.add_argument("--config-path", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold-id", default="main")
    p.add_argument("--train-centers", default="", help="comma-separated")
    p.add_argument("--test-center", default="")
    p.add_argument("--enshi-reports-used", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--log-jsonl", default=str(JBD_ROOT / "logs" / "experiment_runs.jsonl"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred = pd.read_csv(args.predictions)
    for col in ("y_true", "y_prob"):
        if col not in pred.columns:
            raise SystemExit(f"Missing column: {col}")

    m = bootstrap_ci(
        pred["y_true"],
        pred["y_prob"],
        n_boot=args.n_bootstrap,
        seed=args.seed,
    )
    row = {
        "experiment_id": args.experiment_id,
        "model_variant": args.model_variant,
        "config_path": args.config_path,
        "git_commit": git_commit(JBD_ROOT.parents[1]),
        "seed": args.seed,
        "fold_id": args.fold_id,
        "train_centers": args.train_centers,
        "test_center": args.test_center,
        "enshi_reports_used_training": int(args.enshi_reports_used),
        "report_input_at_inference": 0,
        "split_evaluated": pred["split"].iloc[0] if "split" in pred.columns else "test",
        "n_patients": pred["patient_id"].nunique() if "patient_id" in pred.columns else "",
        "n_examinations": len(pred),
    }
    row.update(m.to_dict())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)

    record = build_run_record(
        experiment_id=args.experiment_id,
        model_variant=args.model_variant,
        config_path=args.config_path,
        seed=args.seed,
        fold_id=args.fold_id,
        train_centers=[c.strip() for c in args.train_centers.split(",") if c.strip()],
        test_center=args.test_center or None,
        enshi_reports_used=args.enshi_reports_used,
        report_input_at_inference=False,
        repo_root=JBD_ROOT.parents[1],
        extra={"metrics_path": str(out), "metrics": {k: row[k] for k in METRIC_COLUMNS if k in row}},
    )
    append_jsonl(Path(args.log_jsonl), record)
    print(json.dumps({k: row[k] for k in METRIC_COLUMNS if k in row}, indent=2))


if __name__ == "__main__":
    main()
