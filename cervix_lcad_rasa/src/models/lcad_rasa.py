"""LCAD-RASA multimodal report generator (mock-friendly)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LCADRASAModel(nn.Module):
    """
    Lightweight encoder-decoder for cervical report generation.
    In mock mode, operates on bag-of-words style features without downloading transformers.
    """

    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_size: int = 256,
        num_modalities: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_length: int = 256,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.modality_proj = nn.Linear(hidden_size, hidden_size)
        self.label_embed = nn.Embedding(2, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.risk_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.token_embed(input_ids.clamp(0, self.token_embed.num_embeddings - 1))
        if modality_mask is not None:
            x = x + self.modality_proj(modality_mask)
        if labels is not None:
            x = x + self.label_embed(labels).unsqueeze(1)
        h = self.encoder(x)
        logits = self.decoder(h)
        risk_logit = self.risk_head(h.mean(dim=1))
        return {"logits": logits, "hidden": h, "risk_logit": risk_logit}

    def rasa_alignment_loss(self, hidden: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Report-Anchored Semantic Alignment (cosine)."""
        h = F.normalize(hidden.mean(dim=1), dim=-1)
        a = F.normalize(anchor, dim=-1)
        return 1.0 - (h * a).sum(dim=-1).mean()

    def lcad_distill_loss(self, logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Label-Constrained Agent Distillation (KL)."""
        return F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )


def build_model(cfg: dict[str, Any]) -> LCADRASAModel:
    mcfg = cfg.get("model", {})
    return LCADRASAModel(
        hidden_size=int(mcfg.get("hidden_size", 256)),
        num_layers=int(mcfg.get("num_layers", 2)),
        dropout=float(mcfg.get("dropout", 0.1)),
        max_seq_length=int(cfg.get("training", {}).get("max_seq_length", 256)),
    )
