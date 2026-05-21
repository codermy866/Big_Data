"""Dataset loading visual .npy embeddings + training report text."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.models_publishable.lcad_rasa_model import instr_vector, load_visual_emb


class PublishableDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab_size: int = 8192, max_len: int = 128, min_weight: float = 0.0):
        self.df = df.reset_index(drop=True)
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.indices = []
        for i, row in self.df.iterrows():
            if str(row.get("training_report_type", "none")) == "none":
                continue
            if not str(row.get("training_report_text", "")).strip():
                continue
            w = 1.0 if row.get("training_report_type") == "real" else float(row.get("pseudo_training_weight", 0.5))
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
        text = str(row["training_report_text"])
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
                1.0 if row["training_report_type"] == "real" else float(row.get("pseudo_training_weight", 0.5)),
                dtype=torch.float32,
            ),
            "case_id": str(row["case_id"]),
        }
