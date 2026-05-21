"""De-identification helpers for logs and exports."""

from __future__ import annotations

import re

_PII_PATTERNS = [
    re.compile(r"\b1[3-9]\d{9}\b"),
    re.compile(r"\b\d{15,18}[xX]?\b"),
]


def sanitize_text(text: str) -> str:
    if not text:
        return text
    out = text
    for pat in _PII_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


def safe_case_id(case_id: str) -> str:
    return str(case_id).split("_")[0] if case_id else "unknown"
