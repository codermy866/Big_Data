"""Pseudo-report QC and training weights."""

from __future__ import annotations

import json
from typing import Any

from src.utils.privacy import sanitize_text

FORBIDDEN_DIAG = ["cin3", "cin 3", "invasive cancer", "鳞癌", "浸润癌"]


def qc_pseudo_report(report: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    flags = []
    score = 1.0
    label = int(row.get("binary_label", 0))
    text_blob = json.dumps(report, ensure_ascii=False).lower()

    required = ["diagnostic_summary", "oct_findings", "colposcopy_findings", "clinical_context", "impression", "recommendation"]
    for f in required:
        if not str(report.get(f, "")).strip():
            flags.append(f"empty_{f}")
            score -= 0.1

    if len(text_blob) < 80:
        flags.append("overly_short")
        score -= 0.15

    imp = str(report.get("impression", "")).lower()
    if label == 0 and "suspicious for cin2+" in imp:
        flags.append("negative_label_positive_impression")
        score -= 0.25
    if label == 1 and "no definitive evidence for cin2+" in imp:
        flags.append("positive_label_negative_impression")
        score -= 0.2

    for term in FORBIDDEN_DIAG:
        if term in text_blob and term not in str(row.get("pathology_raw", "")).lower():
            flags.append("pathology_hallucination")
            score -= 0.4
            break

    if row.get("missing_oct") and "oct evidence unavailable" not in text_blob:
        if report.get("evidence_support", {}).get("oct_supported"):
            flags.append("oct_missing_hallucination")
            score -= 0.25

    if not report.get("sentence_level_evidence"):
        flags.append("missing_sentence_evidence")
        score -= 0.1

    conf = float(report.get("confidence", 0.5))
    if conf < 0 or conf > 1:
        flags.append("confidence_out_of_range")
        score -= 0.2

    raw = sanitize_text(text_blob)
    if raw != text_blob:
        flags.append("privacy_sanitized")

    severe = {"invalid_json", "privacy_leakage", "pathology_hallucination", "negative_label_positive_impression"}
    passed = score >= 0.45 and not (severe & set(flags))
    score = max(0.0, min(1.0, score))
    weight = conf * score if passed else 0.0
    return {
        "pseudo_report_pass_qc": int(passed),
        "qc_score": score,
        "qc_flags": ";".join(flags),
        "pseudo_report_confidence": conf,
        "pseudo_training_weight": weight,
    }
