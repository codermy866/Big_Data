"""LCAD-RASA composite loss with section alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def section_embeds_from_hidden(hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split sequence hidden states into section proxies."""
    n = hidden.size(1)
    q = max(1, n // 4)
    return {
        "oct": hidden[:, :q].mean(dim=1),
        "colposcopy": hidden[:, q : 2 * q].mean(dim=1),
        "instruction": hidden[:, 2 * q : 3 * q].mean(dim=1),
        "impression": hidden[:, 3 * q :].mean(dim=1),
    }


def label_consistency_loss(impression_proxy: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Penalise risk head vs binary label (proxy for impression consistency)."""
    risk_logit = impression_proxy
    return torch.nn.functional.binary_cross_entropy_with_logits(
        risk_logit.squeeze(-1), labels.float()
    )


def compute_total_loss(
    model,
    out: dict,
    target_ids: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    loss_cfg: dict,
    section_align=None,
) -> tuple[torch.Tensor, dict]:
    logits = out["logits"]
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        reduction="none",
    )
    ce = ce.view(target_ids.size(0), -1).mean(dim=1)
    ce = (ce * weights).mean()

    hidden = out["hidden"]
    if section_align is not None and float(loss_cfg.get("rasa_weight", 0.5)) > 0:
        anchors = section_embeds_from_hidden(hidden)
        align = section_align(hidden, anchors)
    else:
        align = torch.tensor(0.0, device=hidden.device)

    teacher = logits.detach()
    lcad = (
        model.lcad_distill_loss(logits, teacher)
        if float(loss_cfg.get("lcad_weight", 0.3)) > 0
        else torch.tensor(0.0, device=hidden.device)
    )
    risk_logit = out.get("risk_logit")
    if risk_logit is None or float(loss_cfg.get("cls_weight", 0.2)) <= 0:
        risk = torch.tensor(0.0, device=hidden.device)
        cons = torch.tensor(0.0, device=hidden.device)
    else:
        risk = F.binary_cross_entropy_with_logits(risk_logit.squeeze(-1), labels.float())
        cons = label_consistency_loss(risk_logit, labels)

    total = (
        float(loss_cfg.get("ce_weight", 1.0)) * ce
        + float(loss_cfg.get("rasa_weight", 0.5)) * align
        + float(loss_cfg.get("lcad_weight", 0.3)) * lcad
        + float(loss_cfg.get("cls_weight", 0.2)) * risk
        + float(loss_cfg.get("cons_weight", 0.1)) * cons
    )
    return total, {
        "ce": float(ce.item()),
        "align": float(align.item()),
        "lcad": float(lcad.item()),
        "risk": float(risk.item()) if torch.is_tensor(risk) else 0.0,
        "cons": float(cons.item()) if torch.is_tensor(cons) else 0.0,
    }
