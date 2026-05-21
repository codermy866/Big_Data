#!/usr/bin/env python3
"""Train LCAD-RASA report generation model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.trainer import train_lcad_rasa
from src.utils.config import load_config, resolve_project_root
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LCAD-RASA with LCAD + RASA losses.")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--mock", action="store_true", default=None)
    p.add_argument("--no-mock", dest="mock", action="store_false")
    p.set_defaults(mock=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root()
    cfg = load_config(args.config, root)
    mock = cfg.get("mock", {}).get("enabled", True) if args.mock is None else args.mock
    ckpt = train_lcad_rasa(cfg, mock=mock)
    logger.info("Training finished: %s", ckpt)


if __name__ == "__main__":
    main()
