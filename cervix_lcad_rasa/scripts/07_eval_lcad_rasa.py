#!/usr/bin/env python3
"""Step 9: Evaluate LCAD-RASA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.runner import evaluate_lcad_rasa
from src.utils.config import load_config, resolve_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--experiment", default="full_lcad_rasa")
    return p.parse_args()


def main():
    args = parse_args()
    root = resolve_project_root()
    ecfg = load_config(args.config, root)
    train_cfg = load_config(ecfg.get("train_config", "configs/train.yaml"), root)
    dcfg = load_config("configs/data.yaml", root)
    ecfg["_data"] = dcfg
    ecfg["_train"] = train_cfg
    ecfg.setdefault("outputs", {})["generated_reports"] = dcfg["outputs"]["generated_reports"]
    ecfg.setdefault("outputs", {})["tables"] = dcfg["outputs"]["tables"]
    ecfg["manifest"] = {"path": str(args.manifest or dcfg["manifest"]["with_pseudo"])}
    ckpt = args.checkpoint or str(root / f"outputs/checkpoints/{args.experiment}/best.ckpt")
    ecfg["evaluation"] = dict(ecfg.get("evaluation", {}))
    ecfg["evaluation"]["checkpoint"] = ckpt
    ecfg["evaluation"]["experiment"] = args.experiment
    out = evaluate_lcad_rasa(ecfg, mock=False)
    logger.info("Metrics written: %s", out)


if __name__ == "__main__":
    main()
