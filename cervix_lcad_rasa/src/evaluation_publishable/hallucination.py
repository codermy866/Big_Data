"""Hallucination detection when modality is masked."""

from __future__ import annotations

import re

SPECIFIC_OCT = re.compile(r"(epithelium|stroma|b-scan|microstructural|layer|signal|oct image)", re.I)
SPECIFIC_COLPO = re.compile(r"(acetowhite|mosaic|punctation|vascular|colposcop|lesion|zone)", re.I)
SPECIFIC_INSTR = re.compile(r"(hpv|tct|ascus|lsil|hsil|cytology)", re.I)


def _specific(text: str, pat: re.Pattern) -> bool:
    return bool(pat.search(text or ""))


def hallucination_flags(
    sections: dict[str, str],
    condition: str,
) -> list[str]:
    flags = []
    oct_t = sections.get("oct_findings", "")
    col_t = sections.get("colposcopy_findings", "")
    ctx_t = sections.get("clinical_context", "")

    if "mask_oct" in condition or condition == "mask_visual" or condition == "label_only_inference":
        if _specific(oct_t, SPECIFIC_OCT) and "unavailable" not in oct_t.lower():
            flags.append("oct_missing_hallucination")
    if "mask_colposcopy" in condition or condition == "mask_visual" or condition == "label_only_inference":
        if _specific(col_t, SPECIFIC_COLPO) and "unavailable" not in col_t.lower():
            flags.append("colposcopy_missing_hallucination")
    if "mask_instruction" in condition or condition == "label_only_inference":
        if _specific(ctx_t, SPECIFIC_INSTR) and "unavailable" not in ctx_t.lower():
            flags.append("instruction_missing_hallucination")
    if re.search(r"cin\s*3|invasive cancer|鳞癌", oct_t + col_t, re.I):
        flags.append("pathology_hallucination")
    if flags:
        flags.append("unsupported_specific_finding")
    return flags


def hallucination_rates(case_flags: list[list[str]]) -> dict[str, float]:
    n = max(len(case_flags), 1)
    keys = [
        "oct_missing_hallucination_rate",
        "colposcopy_missing_hallucination_rate",
        "instruction_missing_hallucination_rate",
        "pathology_hallucination_rate",
        "unsupported_specific_finding_rate",
    ]
    out = {}
    for k in keys:
        short = k.replace("_rate", "")
        out[k] = sum(1 for fl in case_flags if short in fl) / n
    return out
