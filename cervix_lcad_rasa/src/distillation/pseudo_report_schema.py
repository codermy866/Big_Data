"""Structured pseudo-report JSON schema."""

from __future__ import annotations

from typing import Any


def build_pseudo_report(
    case_id: str,
    center_id: str,
    label: int,
    endpoint: str,
    oct_sum: str,
    colpo_sum: str,
    clinical: str,
    setting: str,
    confidence: float = 0.75,
) -> dict[str, Any]:
    label_phrase = (
        f"weak supervision: {endpoint} positive (label-constrained inference)"
        if label == 1
        else f"weak supervision: no {endpoint} (label-constrained inference)"
    )
    impression = (
        f"Impression suggests findings suspicious for {endpoint}."
        if label == 1
        else f"No definitive evidence for {endpoint} on available modalities."
    )
    return {
        "case_id": case_id,
        "center_id": center_id,
        "agent_setting": setting,
        "diagnostic_summary": f"Multimodal weak report-level supervision for case {case_id}.",
        "oct_findings": oct_sum or "OCT evidence unavailable.",
        "colposcopy_findings": colpo_sum or "Colposcopy evidence unavailable.",
        "clinical_context": clinical or "Clinical instruction partially available.",
        "impression": impression,
        "recommendation": "Correlate with histopathology; not a substitute for clinical diagnosis.",
        "evidence_support": {
            "oct_supported": bool(oct_sum and "unavailable" not in oct_sum.lower()),
            "colposcopy_supported": bool(colpo_sum and "unavailable" not in colpo_sum.lower()),
            "instruction_supported": bool(clinical),
            "label_supported": setting != "modality_only_agent",
        },
        "sentence_level_evidence": [
            {
                "statement": impression,
                "source": ["label"] if setting == "label_only_agent" else ["OCT", "colposcopy", "instruction", "label"],
                "evidence_type": "constrained" if label == 1 else "observed",
                "confidence": confidence,
            }
        ],
        "label_consistency": "consistent",
        "confidence": confidence,
        "quality_flags": [],
        "weak_supervision_disclaimer": "Pseudo report — not a real clinical report.",
        "label_phrase": label_phrase,
    }
