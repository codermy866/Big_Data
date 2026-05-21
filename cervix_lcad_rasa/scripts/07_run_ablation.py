#!/usr/bin/env python3
"""Run ablation grid over LCAD / RASA loss weights."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.runner import evaluate_lcad_rasa
from src.training.trainer import train_lcad_rasa
from src.utils.config import ensure_dir, load_config, resolve_project_root
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation study for LCAD-RASA components.")
    p.add_argument("--config", default="configs/experiments.yaml")
    p.add_argument("--mock", action="store_true", default=None)
    p.add_argument("--no-mock", dest="mock", action="store_false")
    p.set_defaults(mock=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root()
    exp_cfg = load_config(args.config, root)
    mock = exp_cfg.get("mock", {}).get("enabled", True) if args.mock is None else args.mock
    train_base = load_config(exp_cfg["train_config"], root)
    eval_base = load_config(exp_cfg["eval_config"], root)

    results = []
    for ab in exp_cfg.get("ablations", []):
        name = ab["name"]
        logger.info("=== Ablation: %s ===", name)
        tcfg = copy.deepcopy(train_base)
        tcfg["loss"]["lcad_weight"] = float(ab.get("lcad_weight", 0.3))
        tcfg["loss"]["rasa_weight"] = float(ab.get("rasa_weight", 0.5))
        tcfg["training"]["experiment_name"] = f"ablation_{name}"
        if mock:
            tcfg["training"]["num_epochs"] = int(exp_cfg["mock"].get("epochs_per_run", 1))
            tcfg["mock"]["num_steps_per_epoch"] = int(exp_cfg["mock"].get("num_steps_per_epoch", 3))
        ckpt = train_lcad_rasa(tcfg, mock=mock)

        ecfg = copy.deepcopy(eval_base)
        ecfg["evaluation"]["checkpoint"] = str(ckpt)
        ecfg["_train"] = tcfg
        ecfg["_data"] = load_config(ecfg.get("data_config", "configs/data.yaml"), root)
        metrics_path = evaluate_lcad_rasa(ecfg, mock=mock)
        with metrics_path.open() as f:
            metrics = json.load(f)
        results.append({"ablation": name, **metrics})

    out_dir = ensure_dir(exp_cfg["outputs"]["tables"])
    out_path = out_dir / "ablation_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Ablation summary: %s", out_path)


if __name__ == "__main__":
    main()
