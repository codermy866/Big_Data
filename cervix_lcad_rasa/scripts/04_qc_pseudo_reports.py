#!/usr/bin/env python3
"""Quality control for pseudo-reports (length, label keywords)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.distillation.qc import run_pseudo_report_qc
from src.utils.config import load_config, resolve_project_root
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QC pseudo-reports before training.")
    p.add_argument("--config", default="configs/distill.yaml")
    p.add_argument(
        "--reports",
        default=None,
        help="Path to pseudo_reports.jsonl (default from config).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root()
    cfg = load_config(args.config, root)
    reports_path = Path(args.reports) if args.reports else Path(cfg["outputs"]["pseudo_reports"]) / "pseudo_reports.jsonl"
    out = run_pseudo_report_qc(reports_path, cfg)
    logger.info("QC results: %s", out)


if __name__ == "__main__":
    main()
