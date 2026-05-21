#!/usr/bin/env python3
"""Generate label-constrained pseudo-reports via agent distillation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.modality import load_modality_summaries
from src.distillation.agent import generate_pseudo_reports
from src.utils.config import load_config, resolve_project_root
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label-Constrained Agent pseudo-report generation.")
    p.add_argument("--config", default="configs/distill.yaml")
    p.add_argument(
        "--summaries",
        default=None,
        help="Path to modality_summaries.jsonl (default from config).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root()
    cfg = load_config(args.config, root)
    summaries_path = Path(args.summaries) if args.summaries else Path(cfg["outputs"]["modality_summaries"]) / "modality_summaries.jsonl"
    df = load_modality_summaries(summaries_path)
    out = generate_pseudo_reports(df, cfg)
    logger.info("Pseudo reports: %s", out)


if __name__ == "__main__":
    main()
