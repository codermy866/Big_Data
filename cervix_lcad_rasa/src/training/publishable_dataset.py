"""Dataset loading visual .npy embeddings + training report text."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.models_publishable.lcad_rasa_model import instr_vector, load_visual_emb


def _resolve_weight(row: pd.Series, weight_mode: str) -> float:
    if str(row.get("training_report_type", "")) == "real":
        return 1.0
    if weight_mode == "pseudo_all_no_qc":
        return 1.0
    if weight_mode == "pseudo_qc_pass_only":
        return 1.0 if int(row.get("pseudo_report_pass_qc", 0)) == 1 else 0.0
    if weight_mode == "pseudo_confidence_only":
        return float(row.get("pseudo_report_confidence", 0.5))
    if weight_mode == "pseudo_qc_score_only":
        return float(row.get("qc_score", 0.5))
    return float(row.get("pseudo_training_weight", row.get("pseudo_report_confidence", 0.5) * row.get("qc_score", 0.5)))


class PublishableDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab_size: int = 8192,
        max_len: int = 128,
        min_weight: float = 0.0,
        use_pseudo_report: bool = True,
        use_real_report: bool = True,
        require_qc_pass: bool = True,
        weight_mode: str = "default",
        use_report_loss: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.use_report_loss = use_report_loss
        self.weight_mode = weight_mode
        self.indices = []
        for i, row in self.df.iterrows():
            if not use_report_loss:
                if int(row.get("has_visual_embedding", 1)) == 0:
                    continue
                self.indices.append(i)
                continue
            rtype = str(row.get("training_report_type", "none"))
            if rtype == "none" or not str(row.get("training_report_text", "")).strip():
                continue
            if rtype == "real" and not use_real_report:
                continue
            if rtype == "pseudo":
                if not use_pseudo_report:
                    continue
                if require_qc_pass and weight_mode != "pseudo_all_no_qc" and int(row.get("pseudo_report_pass_qc", 0)) != 1:
                    continue
            w = _resolve_weight(row, weight_mode)
            if w >= min_weight:
                self.indices.append(i)

    def __len__(self) -> int:
        return len(self.indices)

    def _text_ids(self, text: str) -> torch.Tensor:
        ids = [hash(w) % self.vocab_size for w in text.split()[: self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids[: self.max_len], dtype=torch.long)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[self.indices[idx]]
        text = str(row.get("training_report_text", "")) if self.use_report_loss else "classification"
        oct_p = str(row.get("oct_embedding_path", ""))
        col_p = str(row.get("colposcopy_embedding_path", ""))
        fus_p = str(row.get("fused_visual_embedding_path", ""))
        return {
            "oct_emb": torch.tensor(load_visual_emb(oct_p), dtype=torch.float32),
            "col_emb": torch.tensor(load_visual_emb(col_p), dtype=torch.float32),
            "fused_emb": torch.tensor(load_visual_emb(fus_p), dtype=torch.float32),
            "instr": torch.tensor(instr_vector(row.to_dict()), dtype=torch.float32),
            "input_ids": self._text_ids(text),
            "target_ids": self._text_ids(text),
            "labels": torch.tensor(int(row["binary_label"]), dtype=torch.long),
            "weight": torch.tensor(
                _resolve_weight(row, self.weight_mode) if self.use_report_loss else 1.0,
                dtype=torch.float32,
            ),
            "case_id": str(row["case_id"]),
        }
