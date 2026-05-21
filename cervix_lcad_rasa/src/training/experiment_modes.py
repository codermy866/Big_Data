"""Experiment filters aligned with revised method §11 and execution prompt §12."""

from __future__ import annotations

from typing import Any

import pandas as pd

import yaml
from pathlib import Path


def load_experiment_registry(path: Path | None = None) -> dict[str, Any]:
    p = path or Path(__file__).resolve().parents[2] / "configs" / "experiments.yaml"
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_train_filter(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    if "center_id" in spec:
        out = out[out["center_id"] == spec["center_id"]]
    if "has_real_report" in spec:
        out = out[out["has_real_report"] == int(spec["has_real_report"])]
    if "center_id_in" in spec:
        out = out[out["center_id"].isin(spec["center_id_in"])]
    if spec.get("training_eligible"):
        real = out["has_real_report"] == 1
        pseudo = (out.get("needs_pseudo_report", 0) == 1) & (out.get("has_pseudo_report", 0) == 1)
        if "pseudo_report_pass_qc" in out.columns:
            pseudo = pseudo & (out["pseudo_report_pass_qc"] == 1)
        out = out[real | pseudo]
    return out.reset_index(drop=True)


def get_experiment_spec(name: str, registry: dict[str, Any] | None = None) -> dict[str, Any]:
    reg = registry or load_experiment_registry()
    exps = reg.get("experiments", {})
    if name not in exps:
        raise KeyError(f"Unknown experiment: {name}. Available: {list(exps.keys())}")
    return exps[name]
