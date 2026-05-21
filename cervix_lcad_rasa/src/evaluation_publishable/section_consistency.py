"""Section completeness and evidence support scores."""

from __future__ import annotations

SECTION_KEYS = (
    "diagnostic_summary",
    "oct_findings",
    "colposcopy_findings",
    "clinical_context",
    "impression",
    "recommendation",
)


def section_completeness(sections: dict[str, str]) -> dict[str, float]:
    present = {f"{k}_present": float(len(str(sections.get(k, "")).strip()) >= 15) for k in SECTION_KEYS}
    present["overall_section_completeness"] = sum(present.values()) / len(SECTION_KEYS)
    return present


def section_supported_scores(sections: dict[str, str], condition: str) -> dict[str, float]:
    def score(text: str, unavailable_ok: bool) -> float:
        t = (text or "").lower()
        if unavailable_ok and any(x in t for x in ("unavailable", "insufficient", "not provided", "limited")):
            return 0.2
        return min(1.0, len(t) / 120.0)

    mask_oct = "mask_oct" in condition or condition in ("mask_visual", "label_only_inference")
    mask_col = "mask_colposcopy" in condition or condition in ("mask_visual", "label_only_inference")
    mask_ins = "mask_instruction" in condition or condition == "label_only_inference"
    return {
        "oct_section_supported_score": score(sections.get("oct_findings", ""), mask_oct),
        "colposcopy_section_supported_score": score(sections.get("colposcopy_findings", ""), mask_col),
        "clinical_context_supported_score": score(sections.get("clinical_context", ""), mask_ins),
        "impression_supported_score": score(sections.get("impression", ""), False),
    }
