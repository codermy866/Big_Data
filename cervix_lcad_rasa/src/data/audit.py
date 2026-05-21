"""Data ledger audit: paths, splits, and basic integrity checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import ensure_dir
from src.utils.logging_utils import get_logger
from src.utils.mock import generate_mock_manifest

logger = get_logger(__name__)


def run_data_audit(cfg: dict[str, Any], mock: bool = True) -> dict[str, Any]:
    data_root = Path(cfg.get("data_root", "/data"))
    out_dir = ensure_dir(cfg["outputs"]["audit"])

    report: dict[str, Any] = {
        "data_root": str(data_root),
        "data_root_exists": data_root.is_dir(),
        "mock": mock,
    }

    if mock:
        df = generate_mock_manifest(cfg)
        report["n_exams_mock"] = len(df)
        centre_col = "centre" if "centre" in df.columns else "center"
        report["centres"] = df[centre_col].value_counts().to_dict()
        report["split_counts"] = df["split"].value_counts().to_dict()
        dup_patients = df.groupby("patient_id")["split"].nunique()
        report["patients_spanning_splits"] = int((dup_patients > 1).sum())
    else:
        from src.data.manifest import resolve_manifest_path

        registry = Path(cfg["raw"]["registry_csv"])
        report["registry_exists"] = registry.is_file()
        if registry.is_file():
            reg_df = pd.read_csv(registry)
            report["registry_rows"] = len(reg_df)
        for key in ("imaging_root", "reports_root", "labels_root"):
            p = Path(cfg["raw"][key])
            report[f"{key}_exists"] = p.is_dir()
        mpath = resolve_manifest_path(cfg, mock=False)
        report["jbd_modeling_manifest"] = str(mpath)
        if mpath.is_file():
            mdf = pd.read_csv(mpath)
            report["modeling_n"] = len(mdf)
            sc = "center" if "center" in mdf.columns else "centre"
            report["split_counts"] = mdf["split"].value_counts().to_dict()
            report["center_counts"] = mdf[sc].value_counts().to_dict()
            if "patient_id" in mdf.columns:
                dup = mdf.groupby("patient_id")["split"].nunique()
                report["patients_spanning_splits"] = int((dup > 1).sum())

    out_path = out_dir / "audit_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote audit report to %s", out_path)
    return report
