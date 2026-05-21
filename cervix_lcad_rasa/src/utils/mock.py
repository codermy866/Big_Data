"""Synthetic cohort generation for offline / mock pipeline runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CENTRES = ["Centre_A", "Centre_B", "Centre_C", "Centre_D", "Centre_E"]
SPLITS = ["train", "val", "test"]
MODALITIES = ["colposcopy", "cytology", "hpv", "clinical"]


def generate_mock_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    mock = cfg.get("mock", {})
    n = int(mock.get("n_exams", 32))
    n_centres = int(mock.get("n_centres", 3))
    seed = int(mock.get("seed", 42))
    rng = np.random.default_rng(seed)

    centres = CENTRES[:n_centres]
    rows = []
    for i in range(n):
        exam_id = f"EXAM_{i:05d}"
        patient_id = f"PAT_{i // 2:04d}"
        split = SPLITS[int(rng.integers(0, len(SPLITS)))]
        label = int(rng.integers(0, 2))
        centre = centres[int(rng.integers(0, len(centres)))]
        rows.append(
            {
                "exam_id": exam_id,
                "patient_id": patient_id,
                "split": split,
                "cin2plus": label,
                "centre": centre,
                "has_colposcopy": 1,
                "has_cytology": int(rng.integers(0, 2)),
                "has_hpv": 1,
                "has_clinical": 1,
            }
        )
    return pd.DataFrame(rows)


def mock_modality_summary(modality: str, label: int, rng: np.random.Generator) -> str:
    templates = {
        "colposcopy": [
            "Acetowhite area at 3 o'clock, punctation present.",
            "Normal transformation zone, no acetowhite lesion.",
        ],
        "cytology": ["LSIL.", "NILM.", "ASC-US."],
        "hpv": ["HPV16 positive.", "HPV negative.", "Other HR-HPV positive."],
        "clinical": [
            "Parity 1; last screening 2 years ago.",
            "Post-menopausal; prior treatment noted.",
        ],
    }
    opts = templates.get(modality, ["No summary available."])
    base = opts[int(rng.integers(0, len(opts)))]
    if label == 1:
        return f"{base} Suspicious for CIN2+."
    return base
