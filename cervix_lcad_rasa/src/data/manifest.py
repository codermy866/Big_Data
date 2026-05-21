"""Manifest construction and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import ensure_dir
from src.utils.logging_utils import get_logger
from src.utils.mock import generate_mock_manifest

logger = get_logger(__name__)


def resolve_manifest_path(cfg: dict[str, Any], mock: bool) -> Path:
    """Single source of truth: JBD Exp0 modeling CSV; mock writes a separate file."""
    mcfg = cfg.get("manifest", {})
    if mock:
        out_dir = ensure_dir(cfg["outputs"]["manifests"])
        name = mcfg.get("mock_filename", mcfg.get("filename", "patient_manifest_mock.csv"))
        return out_dir / name
    modeling = mcfg.get("modeling_csv")
    if modeling:
        path = Path(modeling)
        if path.is_file():
            return path
    # Fallback: JBD manifests next to repo
    fallback = Path(mcfg.get("jbd_manifest_dir", "../manifests")) / "patient_manifest_modeling.csv"
    return fallback.resolve()


def build_manifest(cfg: dict[str, Any], mock: bool = True) -> Path:
    out_path = resolve_manifest_path(cfg, mock)

    if mock:
        df = generate_mock_manifest(cfg)
        ensure_dir(out_path.parent)
        df.to_csv(out_path, index=False)
        logger.info("Built mock manifest (%d exams) -> %s", len(df), out_path)
        return out_path

    if out_path.is_file():
        df = pd.read_csv(out_path)
        logger.info("Using existing JBD manifest (%d rows): %s", len(df), out_path)
        return out_path

    registry_path = Path(cfg["raw"]["registry_csv"])
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"JBD manifest not found at {out_path} and registry missing: {registry_path}. "
            "Run JBD scripts/exp0_data_ledger_leakage_audit.py first."
        )
    df = pd.read_csv(registry_path)
    if "modeling_eligible" in df.columns:
        df = df[df["modeling_eligible"] == 1]
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    logger.info("Wrote manifest to %s", out_path)
    return out_path


def load_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
