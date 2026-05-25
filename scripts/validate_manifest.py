#!/usr/bin/env python3
"""Validate patient_manifest_v1.csv against JBD protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

JBD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JBD_ROOT))

from src.split_policy import validate_patient_splits

REQUIRED = [
    "patient_id",
    "center",
    "exam_id",
    "colpo_paths",
    "oct_paths",
    "age",
    "hpv",
    "tct",
    "report_available",
    "report_source",
    "raw_report_path",
    "standardized_report_path",
    "pathology_raw",
    "cin_grade",
    "cin2plus",
    "cin3plus",
    "split",
    "fold_id",
]


def main() -> None:
    path = JBD_ROOT / "manifests" / "patient_manifest_v1.csv"
    if not path.exists():
        raise SystemExit(f"Missing manifest: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    ok, errs = validate_patient_splits(df)
    if not ok:
        raise SystemExit("FAIL patient splits:\n" + "\n".join(errs))

    empty_oct = (df.oct_paths.fillna("") == "").sum()
    empty_col = (df.colpo_paths.fillna("") == "").sum()
    if empty_oct or empty_col:
        print(f"WARNING: empty oct_paths={empty_oct}, colpo_paths={empty_col}")

    print("OK manifest validation passed")
    print(f"  examinations: {len(df)}")
    print(f"  patients: {df.patient_id.nunique()}")
    print(df.groupby('split').patient_id.nunique().to_string())


if __name__ == "__main__":
    main()
