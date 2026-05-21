"""YAML configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def resolve_project_root(start: Path | None = None) -> Path:
    """Walk up from *start* until a directory containing configs/data.yaml is found."""
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "configs" / "data.yaml").is_file():
            return parent
    return cur


def load_config(path: str | Path, project_root: Path | None = None) -> dict[str, Any]:
    """Load a YAML config and resolve relative paths against *project_root*."""
    root = project_root or resolve_project_root()
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    nested = cfg.get("data_config")
    if nested and isinstance(nested, str):
        data_cfg = load_config(nested, root)
        cfg["_data"] = data_cfg

    return _resolve_paths(cfg, root)


def _resolve_paths(obj: Any, root: Path) -> Any:
    if isinstance(obj, dict):
        return {k: _resolve_paths(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_paths(v, root) for v in obj]
    if isinstance(obj, str):
        if obj in (".", "./"):
            return str(root)
        if obj.startswith("outputs/") or obj.startswith("configs/"):
            return str((root / obj).resolve())
    return obj


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
