"""Heuristic section parsing for normalized reference reports."""

from __future__ import annotations

import re
from typing import Any


SECTION_KEYS = (
    "diagnostic_summary",
    "oct_findings",
    "colposcopy_findings",
    "clinical_context",
    "impression",
    "recommendation",
)


def normalize_sections(raw_text: str, row: dict[str, Any] | None = None) -> dict[str, Any]:
    text = (raw_text or "").strip()
    flags: list[str] = []
    if len(text) < 30 and row:
        text = str(row.get("pathology_raw", ""))[:800]
        flags.append("fallback_composite_short")

    sections = {k: "" for k in SECTION_KEYS}
    if not text:
        return {"normalized_report": sections, "normalization_flags": ["empty_text"], "extraction_confidence": 0.0}

    # Keyword-based Chinese/English splits
    patterns = {
        "oct_findings": r"(OCT|光学相干|断层)",
        "colposcopy_findings": r"(阴道镜|colposcop)",
        "impression": r"(印象|诊断意见|impression|结论)",
        "recommendation": r"(建议|recommend|随访)",
        "clinical_context": r"(临床|HPV|TCT|年龄|病史)",
    }
    sections["diagnostic_summary"] = text[:400]
    for key, pat in patterns.items():
        m = re.search(pat, text, re.I)
        if m:
            start = m.start()
            chunk = text[start : start + 500]
            sections[key] = chunk.strip()

    if row:
        ctx = []
        for k in ("age", "hpv", "tct"):
            v = row.get(k, "")
            if v and str(v) != "nan":
                ctx.append(f"{k}:{v}")
        if ctx:
            sections["clinical_context"] = (sections["clinical_context"] + " " + " ".join(ctx)).strip()

    conf = min(1.0, 0.2 + 0.1 * sum(1 for v in sections.values() if len(v) > 20))
    return {
        "normalized_report": sections,
        "normalization_flags": flags,
        "extraction_confidence": conf,
        "reference_report_text": " ".join(sections[k] for k in SECTION_KEYS if sections[k]),
    }
