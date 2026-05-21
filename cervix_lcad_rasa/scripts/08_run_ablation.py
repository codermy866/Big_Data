#!/usr/bin/env python3
"""Step 10: Ablation experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.modality_ablation import run_modality_ablations
from src.pipeline.train_eval import run_ablation_grid
from src.utils.config import load_config, resolve_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ablation.yaml")
    p.add_argument("--manifest", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = resolve_project_root()
    ab = load_config(args.config, root)
    train = load_config(ab["train_config"], root)
    ev = load_config(ab["eval_config"], root)
    dcfg = load_config("configs/data.yaml", root)
    ev["_data"] = dcfg
    ev.setdefault("outputs", {})["generated_reports"] = dcfg["outputs"]["generated_reports"]
    ev.setdefault("outputs", {})["tables"] = dcfg["outputs"]["tables"]
    train.setdefault("mock", {})["enabled"] = False
    train["training"]["max_steps_per_epoch"] = 50
    manifest = Path(args.manifest or ab["manifest"])
    run_ablation_grid(ab, train, ev, manifest)
    run_modality_ablations(
        manifest,
        Path(dcfg["outputs"]["modality_evidence"]),
        Path(dcfg["outputs"]["tables"]),
    )
    logger.info("Ablation tables written under outputs/tables/")


if __name__ == "__main__":
    main()
