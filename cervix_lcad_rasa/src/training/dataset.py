"""Manifest + real/pseudo report dataset for LCAD-RASA training."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.models_publishable.lcad_rasa_model import stable_token_id


def _pseudo_to_text(report: dict) -> str:
    return " ".join(
        str(report.get(k, ""))
        for k in (
            "diagnostic_summary",
            "oct_findings",
            "colposcopy_findings",
            "clinical_context",
            "impression",
            "recommendation",
        )
    )


def _row_report_text(row: pd.Series, pseudo_root: Path | None) -> tuple[str, float]:
    """Return (text, sample_weight) for one manifest row."""
    case_id = str(row["case_id"])
    if int(row.get("has_real_report", 0)) == 1:
        text = str(row.get("real_report_text", "") or "").strip()
        if not text or len(text) < 20:
            text = str(row.get("other_clinical_attributes", ""))
        return text, 1.0

    if pseudo_root:
        p = pseudo_root / str(row["center_id"]) / f"{case_id}.json"
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                rep = json.load(f)
            w = float(row.get("pseudo_training_weight", row.get("qc_score", 0.5)))
            return _pseudo_to_text(rep), w
    return "", 0.0


class CervixReportDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        pseudo_root: Path | None = None,
        report_source: str = "mixed",
        vocab_size: int = 4096,
        max_len: int = 64,
        min_weight: float = 0.0,
        require_qc_pass: bool = False,
    ):
        self.manifest = manifest_df.reset_index(drop=True)
        self.reports: dict[str, str] = {}
        self.weights: dict[str, float] = {}
        pseudo_root = Path(pseudo_root) if pseudo_root else None

        for _, row in self.manifest.iterrows():
            case_id = str(row["case_id"])
            use_real = int(row.get("has_real_report", 0)) == 1
            if report_source == "real" and not use_real:
                continue
            if report_source == "pseudo" and use_real:
                continue
            if require_qc_pass and not use_real:
                if int(row.get("pseudo_report_pass_qc", 0)) != 1:
                    continue
            text, w = _row_report_text(row, pseudo_root)
            if not text.strip():
                continue
            self.reports[case_id] = text
            self.weights[case_id] = 1.0 if use_real else w

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.min_weight = min_weight
        self.indices = [
            i
            for i in range(len(self.manifest))
            if self._row_id(i) in self.reports
            and self.weights.get(self._row_id(i), 0.0) >= min_weight
        ]

    def _row_id(self, idx: int) -> str:
        return str(self.manifest.iloc[idx]["case_id"])

    def __len__(self) -> int:
        return len(self.indices)

    def _text_to_ids(self, text: str) -> torch.Tensor:
        ids = [stable_token_id(w, self.vocab_size) for w in text.split()[: self.max_len]]
        ids += [0] * max(0, self.max_len - len(ids))
        return torch.tensor(ids[: self.max_len], dtype=torch.long)

    def __getitem__(self, idx: int) -> dict:
        ri = self.indices[idx]
        row = self.manifest.iloc[ri]
        case_id = self._row_id(ri)
        text = self.reports[case_id]
        input_ids = self._text_to_ids(text)
        label = int(row.get("binary_label", 0))
        weight = float(self.weights.get(case_id, 1.0))
        is_real = int(row.get("has_real_report", 0))
        return {
            "input_ids": input_ids,
            "labels": torch.tensor(label, dtype=torch.long),
            "target_ids": input_ids.clone(),
            "weight": torch.tensor(weight, dtype=torch.float),
            "is_real_report": torch.tensor(is_real, dtype=torch.long),
            "case_id": case_id,
        }
