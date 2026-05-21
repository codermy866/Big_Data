#!/usr/bin/env python3
"""Evaluate LCAD-RASA on held-out split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.runner import evaluate_lcad_rasa
from src.utils.config import load_config, resolve_project_root
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate report generation metrics.")
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--mock", action="store_true", default=None)
    p.add_argument("--no-mock", dest="mock", action="store_false")
    p.set_defaults(mock=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root()
    cfg = load_config(args.config, root)
    train_cfg = load_config(cfg.get("train_config", "configs/train.yaml"), root)
    cfg["_train"] = train_cfg
    data_cfg = load_config(cfg.get("data_config", "configs/data.yaml"), root)
    cfg["_data"] = data_cfg
    mock = cfg.get("mock", {}).get("enabled", True) if args.mock is None else args.mock
    out = evaluate_lcad_rasa(cfg, mock=mock)
    logger.info("Evaluation written: %s", out)


if __name__ == "__main__":
    main()
