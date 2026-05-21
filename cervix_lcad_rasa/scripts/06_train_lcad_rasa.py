#!/usr/bin/env python3
"""Step 8: Train LCAD-RASA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.train_eval import run_train
from src.utils.config import load_config, resolve_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--experiment", default="full_lcad_rasa")
    return p.parse_args()


def main():
    args = parse_args()
    root = resolve_project_root()
    cfg = load_config(args.config, root)
    dcfg = load_config("configs/data.yaml", root)
    cfg["outputs"] = cfg.get("outputs", dcfg.get("outputs", {}))
    cfg.setdefault("mock", {})["enabled"] = False
    ckpt = run_train(args.experiment, cfg)
    logger.info("Checkpoint: %s", ckpt)


if __name__ == "__main__":
    main()
