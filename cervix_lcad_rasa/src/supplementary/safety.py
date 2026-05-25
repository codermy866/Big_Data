"""Report safety metrics (Prompt 9)."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.evaluation_publishable.hallucination import hallucination_flags


def safety_case_metrics(sections: dict[str, str], row: dict, label: int) -> dict[str, Any]:
    flags = hallucination_flags(sections, "normal")
    text = " ".join(sections.values()).lower()
    imp = sections.get("impression", "").lower()
    rec = sections.get("recommendation", "").lower()
    required = ("oct_findings", "colposcopy_findings", "clinical_context", "impression", "recommendation")
    incomplete = sum(1 for s in required if len(str(sections.get(s, "")).strip()) < 15) / len(required)
    label_contra = 0.0
    if label == 1 and any(p in imp for p in ("no definitive", "negative", "nil")):
        label_contra = 1.0
    if label == 0 and re.search(r"cin\s*2\+|suspicious for cin2|high-grade", imp):
        label_contra = 1.0
    rec_contra = 0.0
    if "high risk" in imp and "routine follow" in rec and "histopath" not in rec:
        rec_contra = 1.0
    if "no definitive" in imp and "immediate" in rec:
        rec_contra = 1.0
    mod_hall = 1.0 if flags else 0.0
    if int(row.get("missing_oct", 0)) and "microstructural" in sections.get("oct_findings", "").lower():
        mod_hall = 1.0
    template_rep = 1.0 if len(set(text.split())) < 25 else 0.0
    return {
        "modality_missing_hallucination": mod_hall,
        "label_report_contradiction": label_contra,
        "recommendation_contradiction": rec_contra,
        "section_incompleteness": incomplete,
        "template_repetition": template_rep,
        "hallucination_any": float(bool(flags)),
    }


def aggregate_safety(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "experiment_id"]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}
