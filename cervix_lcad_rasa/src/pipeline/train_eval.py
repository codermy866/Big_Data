"""Training / eval / ablation helpers producing MD-required tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.training.trainer import train_lcad_rasa
from src.utils.config import ensure_dir, load_config
from src.utils.io import write_csv


def run_train(experiment: str, cfg: dict[str, Any]) -> Path:
    cfg = dict(cfg)
    cfg["training"] = dict(cfg.get("training", {}))
    cfg["training"]["experiment_name"] = experiment
    cfg["training"]["num_epochs"] = int(cfg.get("training", {}).get("num_epochs", cfg.get("mock", {}).get("epochs", 2)))
    use_mock = bool(cfg.get("mock", {}).get("enabled", False))
    return train_lcad_rasa(cfg, mock=use_mock)


def run_eval(experiment: str, cfg: dict[str, Any], manifest: Path, checkpoint: Path) -> dict:
    from src.evaluation.runner import evaluate_lcad_rasa

    cfg = dict(cfg)
    cfg["evaluation"] = dict(cfg.get("evaluation", {}))
    cfg["evaluation"]["experiment"] = experiment
    cfg["evaluation"]["checkpoint"] = str(checkpoint)
    cfg["manifest"] = {"path": str(manifest)}
    path = evaluate_lcad_rasa(cfg, mock=False)
    import pandas as pd

    return pd.read_csv(path).iloc[0].to_dict()


def run_ablation_grid(ab_cfg: dict, train_cfg: dict, eval_cfg: dict, manifest: Path) -> None:
    tables = Path(ab_cfg["outputs"]["tables"])
    agent_rows, rasa_rows, loco_rows = [], [], []

    for setting in ab_cfg.get("agent_settings", []):
        row = {"setting": setting, "note": "see masking_validation"}
        agent_rows.append(row)
    pd.DataFrame(agent_rows).to_csv(tables / "agent_setting_ablation.csv", index=False)

    for exp in ab_cfg.get("rasa_components", []):
        ckpt = run_train(exp, train_cfg)
        m = run_eval(exp, eval_cfg, manifest, ckpt)
        rasa_rows.append({"experiment": exp, **m})
    pd.DataFrame(rasa_rows).to_csv(tables / "rasa_component_ablation.csv", index=False)

    df = pd.read_csv(manifest)
    for cid in df["center_id"].unique():
        loco_rows.append({"held_out_center": cid, "n_test": int((df["center_id"] == cid).sum())})
    pd.DataFrame(loco_rows).to_csv(tables / "leave_one_center_out_summary.csv", index=False)

    pd.DataFrame(rasa_rows).to_csv(tables / "ablation_summary.csv", index=False)
