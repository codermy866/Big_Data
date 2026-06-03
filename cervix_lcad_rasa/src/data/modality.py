"""Per-modality summary extraction (mock or file-based)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import ensure_dir
from src.utils.logging_utils import get_logger
from src.utils.mock import MODALITIES, mock_modality_summary

logger = get_logger(__name__)


def extract_modality_summaries(
    manifest_df: pd.DataFrame,
    cfg: dict[str, Any],
    mock: bool = True,
) -> Path:
    out_dir = ensure_dir(cfg["outputs"]["modality_summaries"])
    seed = int(cfg.get("mock", {}).get("seed", 42))
    rng = np.random.default_rng(seed)

    records = []
    for _, row in manifest_df.iterrows():
        exam_id = row["exam_id"]
        label = int(row.get("cin2plus", row.get("cin2_plus", 0)))
        for mod in MODALITIES:
            if f"has_{mod}" in row and row[f"has_{mod}"] == 0:
                continue
            if mock:
                summary = mock_modality_summary(mod, label, rng)
            else:
                summary = _summary_from_manifest_row(row, mod)
                if summary.startswith("[missing"):
                    summary = _load_from_disk(cfg, exam_id, mod)
            records.append(
                {
                    "exam_id": exam_id,
                    "modality": mod,
                    "summary": summary,
                    "label": label,
                }
            )

    out_path = out_dir / "modality_summaries.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d modality summaries to %s", len(records), out_path)
    return out_path


def _summary_from_manifest_row(row: pd.Series, modality: str) -> str:
    """Build text summaries from locked cohort manifest columns (report-free features)."""
    def _s(val: Any) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        return str(val).strip()

    if modality == "colposcopy":
        n = row.get("colpo_image_count", "")
        center = _s(row.get("center_name", row.get("center", "")))
        path = _s(row.get("pathology_raw", ""))
        return f"Centre {center}; colposcopy images n={n}. Pathology note: {path or 'not available'}."
    if modality == "cytology":
        return f"TCT: {_s(row.get('tct')) or 'missing'}; class: {_s(row.get('tct_class')) or 'unclassified'}."
    if modality == "hpv":
        return f"HPV: {_s(row.get('hpv')) or 'missing'}; class: {_s(row.get('hpv_class')) or 'unclassified'}."
    if modality == "clinical":
        age = row.get("age", "")
        treat = _s(row.get("treatment_text", ""))
        oct_abn = _s(row.get("oct_abnormal", ""))
        return f"Age {age}; OCT abnormal flag: {oct_abn or 'NA'}; treatment: {treat or 'none recorded'}."
    return "[missing modality]"


def _load_from_disk(cfg: dict[str, Any], exam_id: str, modality: str) -> str:
    root = Path(cfg["data_root"]) / "summaries" / modality
    path = root / f"{exam_id}.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return f"[missing summary for {exam_id}/{modality}]"


def load_modality_summaries(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)
