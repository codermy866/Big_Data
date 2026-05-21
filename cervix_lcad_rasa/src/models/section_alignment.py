"""Section-level alignment between modality hidden states and report sections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SectionAlignmentModule(nn.Module):
  def __init__(self, hidden_size: int = 256):
    super().__init__()
    self.proj_oct = nn.Linear(hidden_size, hidden_size)
    self.proj_colpo = nn.Linear(hidden_size, hidden_size)
    self.proj_instr = nn.Linear(hidden_size, hidden_size)
    self.proj_fused = nn.Linear(hidden_size, hidden_size)

  def forward(
    self,
    hidden: torch.Tensor,
    section_embeds: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    """Cosine alignment loss across OCT/colposcopy/instruction/impression sections."""
    pooled = hidden.mean(dim=1)
    losses = []
    pairs = (
      ("oct", self.proj_oct(pooled)),
      ("colposcopy", self.proj_colpo(pooled)),
      ("instruction", self.proj_instr(pooled)),
      ("impression", self.proj_fused(pooled)),
    )
    for name, proj in pairs:
      if name in section_embeds:
        t = F.normalize(section_embeds[name], dim=-1)
        s = F.normalize(proj, dim=-1)
        losses.append(1.0 - (s * t).sum(dim=-1))
    if not losses:
      return torch.tensor(0.0, device=hidden.device)
    return torch.stack(losses, dim=0).mean()
