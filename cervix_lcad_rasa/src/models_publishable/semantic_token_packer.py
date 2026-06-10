"""STREAM-inspired semantic token packer for LCAD-RASA."""

from __future__ import annotations

import torch
import torch.nn as nn


class SemanticTokenPacker(nn.Module):
    """Pack variable multimodal/entity tokens into fixed semantic prompts.

    STREAM packs multi-view and temporal image features into visual prompts.
    Here the same idea is adapted to cervical analytics: OCT, colposcopy,
    fused visual, clinical instruction, and retrieved entity vectors are
    attended by a small set of learnable semantic queries.
    """

    def __init__(self, hidden_size: int, num_queries: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_size) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return one packed semantic vector per case.

        Args:
            tokens: Tensor shaped ``[batch, n_tokens, hidden]``.
        """
        batch = tokens.size(0)
        query = self.query_tokens.unsqueeze(0).expand(batch, -1, -1)
        packed, _ = self.attn(query, tokens, tokens, need_weights=False)
        packed = self.norm(packed + query)
        packed = self.out_norm(packed + self.ffn(packed))
        return packed.mean(dim=1)

