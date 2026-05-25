#!/usr/bin/env python3
"""Load YAML configs from JBD_2026/configs/."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

JBD_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = JBD_ROOT / "configs"


def load_yaml(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_configs() -> Dict[str, Dict[str, Any]]:
    return {
        "data": load_yaml("config_data.yaml"),
        "training": load_yaml("config_training.yaml"),
        "hydra_core": load_yaml("config_model_hydra_core.yaml"),
        "ra_hydra_llm": load_yaml("config_model_ra_hydra_llm.yaml"),
        "experiments": load_yaml("config_experiments.yaml"),
    }
