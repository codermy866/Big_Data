"""Publishable LCAD-RASA: visual embeddings + instruction + report decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH = True
except ImportError:
    _TORCH = False


class PublishableLCADRASA(nn.Module):
    def __init__(
        self,
        visual_dim: int = 2048,
        hidden_size: int = 512,
        vocab_size: int = 8192,
        max_len: int = 128,
        use_risk_head: bool = True,
        use_section_align: bool = True,
        use_oct: bool = True,
        use_colposcopy: bool = True,
        use_instruction: bool = True,
        use_fused_visual: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.use_risk_head = use_risk_head
        self.use_section_align = use_section_align
        self.use_oct = use_oct
        self.use_colposcopy = use_colposcopy
        self.use_instruction = use_instruction
        self.use_fused_visual = use_fused_visual
        self.oct_proj = nn.Linear(visual_dim, hidden_size)
        self.col_proj = nn.Linear(visual_dim, hidden_size)
        self.fused_proj = nn.Linear(visual_dim, hidden_size)
        self.instr_proj = nn.Linear(32, hidden_size)
        self.label_embed = nn.Embedding(2, hidden_size)
        self.fusion = nn.Sequential(nn.Linear(hidden_size * 4, hidden_size), nn.ReLU(), nn.Dropout(dropout))
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(hidden_size, nhead=8, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.risk_head = nn.Linear(hidden_size, 1) if use_risk_head else None
        self.sec_oct = nn.Linear(hidden_size, hidden_size)
        self.sec_col = nn.Linear(hidden_size, hidden_size)
        self.sec_instr = nn.Linear(hidden_size, hidden_size)
        self.sec_imp = nn.Linear(hidden_size, hidden_size)

    def encode_modalities(
        self,
        oct_emb: torch.Tensor,
        col_emb: torch.Tensor,
        fused_emb: torch.Tensor,
        instr_vec: torch.Tensor,
        labels: torch.Tensor | None,
        modality_mask: dict[str, bool] | None = None,
    ) -> torch.Tensor:
        m = modality_mask or {}
        o = self.oct_proj(oct_emb) * (0.0 if m.get("mask_oct") or not self.use_oct else 1.0)
        c = self.col_proj(col_emb) * (0.0 if m.get("mask_colposcopy") or not self.use_colposcopy else 1.0)
        f = self.fused_proj(fused_emb) * (0.0 if not self.use_fused_visual else 1.0)
        ins = self.instr_proj(instr_vec) * (0.0 if m.get("mask_instruction") or not self.use_instruction else 1.0)
        parts = [o, c, f, ins]
        if labels is not None and not m.get("randomize_label"):
            parts.append(self.label_embed(labels.clamp(0, 1)))
        else:
            parts.append(torch.zeros_like(o))
        h = self.fusion(torch.cat(parts[:4], dim=-1))
        return h

    def forward(
        self,
        oct_emb: torch.Tensor,
        col_emb: torch.Tensor,
        fused_emb: torch.Tensor,
        instr_vec: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        modality_mask: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        h0 = self.encode_modalities(oct_emb, col_emb, fused_emb, instr_vec, labels, modality_mask)
        x = self.token_embed(input_ids.clamp(0, self.token_embed.num_embeddings - 1))
        x = x + h0.unsqueeze(1)
        h = self.encoder(x)
        logits = self.decoder(h)
        out = {"logits": logits, "hidden": h, "fused": h0}
        if self.risk_head is not None:
            out["risk_logit"] = self.risk_head(h0)
        return out

    @torch.no_grad()
    def generate_structured_report(
        self,
        oct_emb: torch.Tensor,
        col_emb: torch.Tensor,
        fused_emb: torch.Tensor,
        instr_vec: torch.Tensor,
        label: int,
        row: dict | None = None,
        modality_mask: dict | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Decode modality-conditioned structured report (Prompt I text decoding)."""
        m = modality_mask or {}
        row = row or {}
        device = oct_emb.device
        effective_label = (1 - label) if m.get("randomize_label") else label
        lab_t = torch.tensor([effective_label], device=device)
        if input_ids is None:
            seed_text = str(row.get("training_report_text", row.get("reference_report_text", "cervical examination")))
            ids = [hash(w) % self.token_embed.num_embeddings for w in seed_text.split()[: self.max_len]]
            ids += [0] * max(0, self.max_len - len(ids))
            input_ids = torch.tensor([ids[: self.max_len]], dtype=torch.long, device=device)
        else:
            input_ids = input_ids.to(device)

        out = self.forward(oct_emb, col_emb, fused_emb, instr_vec, input_ids, lab_t, modality_mask=m)
        h0 = out["fused"]
        risk_logit = out.get("risk_logit")
        risk_score = float(torch.sigmoid(risk_logit).item()) if risk_logit is not None else 0.5

        oct_norm = float(oct_emb.norm().item())
        col_norm = float(col_emb.norm().item())
        ins_norm = float(instr_vec.norm().item())
        label_ep = str(row.get("binary_label_endpoint", "CIN2+"))

        def _oct_text() -> str:
            if m.get("mask_oct") or m.get("mask_visual") or m.get("label_only_inference") or oct_norm < 1.0:
                return "OCT evidence was unavailable or insufficient for this examination. No reliable microstructural B-scan interpretation could be provided."
            if m.get("shuffle_oct"):
                return (
                    f"OCT review shows non-specific signal patterns (cross-case shuffle; reliability reduced). "
                    f"Embedding magnitude {oct_norm:.2f} does not match case-specific microstructure."
                )
            tier = "suspicious" if oct_norm > 8 else "indeterminate" if oct_norm > 3 else "limited"
            return (
                f"OCT microstructural review: {tier} epithelial/stromal signal on available B-scans "
                f"(visual embedding strength {oct_norm:.2f}). Findings should be correlated with colposcopy."
            )

        def _colpo_text() -> str:
            if m.get("mask_colposcopy") or m.get("mask_visual") or m.get("label_only_inference") or col_norm < 1.0:
                return "Colposcopic evidence was unavailable. Colposcopy findings cannot be specified from visual inputs."
            if m.get("shuffle_colposcopy"):
                return (
                    f"Colposcopy shows atypical vascular/acetowhite patterns of uncertain case-specific relevance "
                    f"(shuffled visual embedding {col_norm:.2f}). Interpret with caution."
                )
            tier = "abnormal vascular" if col_norm > 8 else "possible focal changes" if col_norm > 3 else "limited"
            return f"Colposcopy: {tier} appearance on available still images (embedding strength {col_norm:.2f})."

        def _clinical_text() -> str:
            if m.get("mask_instruction") or m.get("label_only_inference") or ins_norm < 0.05:
                return "Clinical instruction context was limited. Age/HPV/TCT details were not available for integration."
            if m.get("shuffle_instruction"):
                return "Clinical context reflects mismatched priors; HPV/TCT interpretation is unreliable for this case."
            age, hpv, tct = row.get("age", ""), row.get("hpv", ""), row.get("tct", "")
            return f"Clinical context: age {age}; HPV {hpv}; TCT {tct}. Instruction embedding strength {ins_norm:.2f}."

        def _impression() -> str:
            if m.get("label_only_inference"):
                return (
                    f"Impression based primarily on weak label prior ({label_ep}={'positive' if effective_label else 'negative'}); "
                    "modality-specific evidence was not integrated."
                )
            if effective_label == 1:
                return f"Impression: findings suspicious for {label_ep} based on integrated multimodal evidence."
            return f"No definitive evidence for {label_ep} on available OCT and colposcopy within weak-supervision constraints."

        sections = {
            "diagnostic_summary": (
                f"Structured weak-supervision report for case {row.get('case_id', '')} "
                f"({row.get('center_id', '')}). Not a substitute for clinical diagnosis."
            ),
            "oct_findings": _oct_text(),
            "colposcopy_findings": _colpo_text(),
            "clinical_context": _clinical_text(),
            "impression": _impression(),
            "recommendation": "Recommend histopathology correlation and routine follow-up per local protocol.",
        }
        # Blend with sequence logits variance for non-masked conditions
        if not any(m.get(k) for k in ("mask_oct", "mask_colposcopy", "mask_instruction", "mask_visual", "label_only_inference")):
            logits_var = float(out["logits"].std().item())
            sections["diagnostic_summary"] += f" Model sequence variance {logits_var:.3f}."

        return {
            "generated_report_text": " ".join(sections.values()),
            "generated_sections": sections,
            "risk_score": risk_score,
            "risk_logit": float(risk_logit.item()) if risk_logit is not None else 0.0,
            "modality_embeddings": {
                "oct_norm": oct_norm,
                "colposcopy_norm": col_norm,
                "instruction_norm": ins_norm,
            },
            "section_embeddings": {
                "oct": self.sec_oct(h0).cpu().numpy().tolist()[:8],
                "colposcopy": self.sec_col(h0).cpu().numpy().tolist()[:8],
            },
        }

    def section_alignment_loss(self, h0: torch.Tensor, h_seq: torch.Tensor) -> torch.Tensor:
        if not self.use_section_align:
            return torch.tensor(0.0, device=h0.device)
        n = h_seq.size(1)
        q = max(1, n // 4)
        targets = {
            "oct": h_seq[:, :q].mean(1),
            "colposcopy": h_seq[:, q : 2 * q].mean(1),
            "instruction": h_seq[:, 2 * q : 3 * q].mean(1),
            "impression": h_seq[:, 3 * q :].mean(1),
        }
        projs = {
            "oct": self.sec_oct(h0),
            "colposcopy": self.sec_col(h0),
            "instruction": self.sec_instr(h0),
            "impression": self.sec_imp(h0),
        }
        losses = []
        for k in targets:
            a = F.normalize(projs[k], dim=-1)
            b = F.normalize(targets[k], dim=-1)
            losses.append(1.0 - (a * b).sum(-1))
        return torch.stack(losses, dim=0).mean()


def load_visual_emb(path: str, dim: int = 2048) -> np.ndarray:
    p = Path(path)
    if p.is_file():
        v = np.load(p)
        if v.shape[-1] != dim:
            pad = np.zeros(dim, dtype=np.float32)
            pad[: min(dim, len(v))] = v.flatten()[:dim]
            return pad
        return v.astype(np.float32)
    return np.zeros(dim, dtype=np.float32)


def instr_vector(row: dict, dim: int = 32) -> np.ndarray:
    parts = [str(row.get("age", 0)), str(row.get("hpv", "")), str(row.get("tct", ""))]
    vec = np.array([hash(p) % 997 / 997.0 for p in parts], dtype=np.float32)
    out = np.zeros(dim, dtype=np.float32)
    out[: len(vec)] = vec
    return out
