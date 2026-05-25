#!/usr/bin/env python3
"""Patient-level split assignment — no image/B-scan leakage."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def assign_patient_stratified_splits(
    patients: pd.DataFrame,
    patient_id_col: str = "patient_id",
    center_col: str = "center",
    label_col: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 2026,
) -> Dict[str, str]:
    """Return patient_id -> split (train|val|test)."""
    rng = np.random.default_rng(seed)
    out: Dict[str, str] = {}
    grouped = patients.drop_duplicates(patient_id_col)
    for (_, group) in grouped.groupby([center_col, label_col], dropna=False):
        ids = group[patient_id_col].astype(str).tolist()
        rng.shuffle(ids)
        n = len(ids)
        if n == 1:
            out[ids[0]] = "train"
            continue
        n_test = max(1, int(round(n * (1 - train_ratio - val_ratio))))
        n_val = max(1, int(round(n * val_ratio)))
        if n_test + n_val >= n:
            n_test = 1 if n > 2 else 0
            n_val = 1 if n - n_test > 1 else 0
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train = 1
            n_val = max(0, n - n_train - 1)
            n_test = n - n_train - n_val
        for i, pid in enumerate(ids):
            if i < n_train:
                out[pid] = "train"
            elif i < n_train + n_val:
                out[pid] = "val"
            else:
                out[pid] = "test"
    return out


def validate_patient_splits(manifest: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Ensure one split per patient_id."""
    errors: List[str] = []
    g = manifest.groupby("patient_id")["split"].nunique()
    bad = g[g > 1]
    if len(bad):
        errors.append(f"{len(bad)} patients appear in multiple splits")
        for pid in bad.index[:10]:
            splits = manifest.loc[manifest.patient_id == pid, "split"].unique().tolist()
            errors.append(f"  patient_id={pid} splits={splits}")
    return len(errors) == 0, errors


def assign_lco_fold(
    manifest: pd.DataFrame,
    test_center: str,
    center_col: str = "center",
) -> pd.Series:
    """Leave-one-centre-out: test_center held out; others train."""
    return manifest[center_col].map(
        lambda c: "test" if str(c) == test_center else "train"
    )
