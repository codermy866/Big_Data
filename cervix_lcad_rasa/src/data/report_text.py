"""Build de-identified reference report text for masking / eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.privacy import sanitize_text

STD_ROOT = Path(__file__).resolve().parents[3] / "data" / "standardized_reports"


def _extract_pdf(path: Path, max_chars: int) -> str:
    try:
        import fitz  # pymupdf

        doc = fitz.open(str(path))
        parts = [page.get_text() for page in doc]
        return sanitize_text("\n".join(parts))[:max_chars]
    except ImportError:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(path))
            parts = [pg.extract_text() or "" for pg in reader.pages[:20]]
            return sanitize_text("\n".join(parts))[:max_chars]
        except Exception:
            return ""
    except Exception:
        return ""


def _read_text_file(path: Path, max_chars: int = 2000) -> str:
    if not path.is_file():
        return ""
    try:
        suf = path.suffix.lower()
        if suf == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                return sanitize_text(" ".join(str(v) for v in data.values() if v))[:max_chars]
        if suf == ".pdf":
            text = _extract_pdf(path, max_chars)
            if text:
                return text
        if suf == ".xml":
            import re

            raw = path.read_text(encoding="utf-8", errors="ignore")
            return sanitize_text(re.sub(r"<[^>]+>", " ", raw))[:max_chars]
        return sanitize_text(path.read_text(encoding="utf-8", errors="ignore"))[:max_chars]
    except Exception:
        return ""


def reference_report_from_row(row: dict[str, Any] | Any) -> str:
    """Priority: standardized JSON > raw text file > clinical composite."""
    get = row.get if hasattr(row, "get") else lambda k, d="": getattr(row, k, d)

    std = str(get("standardized_report_path", "") or "").strip()
    if std:
        text = _read_text_file(Path(std))
        if text:
            return text
    if not std:
        eid = str(get("case_id", get("exam_id", "")))
        cand = STD_ROOT / f"{eid}.json"
        text = _read_text_file(cand)
        if text:
            return text

    raw = str(get("real_report_path", "") or "").strip()
    if raw and Path(raw).is_file():
        text = _read_text_file(Path(raw))
        if text:
            return text

    parts = []
    for key in (
        "pathology_raw",
        "other_clinical_attributes",
        "tct",
        "hpv",
        "tct_class",
        "hpv_class",
        "cin_grade",
        "treatment_text",
        "binary_label_text",
    ):
        val = get(key, "")
        if val is None or str(val).strip() in ("", "nan", "missing", "unclassifiable"):
            continue
        parts.append(f"{key}: {sanitize_text(str(val)[:400])}")
    return " ".join(parts) if parts else ""
