"""Extract text from archived cervical reports (PDF/XML/INI/image/txt)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.utils.privacy import sanitize_text


def extract_raw_text(path: Path, max_chars: int = 12000) -> tuple[str, str, list[str]]:
    """Return (text, source_type, flags)."""
    flags: list[str] = []
    if not path or not Path(path).is_file():
        return "", "missing", ["file_not_found"]
    p = Path(path)
    suf = p.suffix.lower()
    try:
        if suf == ".pdf":
            text = _pdf_text(p, max_chars)
            return sanitize_text(text), "pdf", flags if text else ["needs_ocr"]
        if suf == ".xml":
            raw = p.read_text(encoding="utf-8", errors="ignore")
            text = sanitize_text(re.sub(r"<[^>]+>", " ", raw))[:max_chars]
            return text, "xml", flags
        if suf in {".txt", ".ini"}:
            return sanitize_text(p.read_text(encoding="utf-8", errors="ignore"))[:max_chars], suf.lstrip("."), flags
        if suf == ".json":
            import json

            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                text = sanitize_text(" ".join(str(v) for v in data.values() if v))[:max_chars]
                return text, "json", flags
        if suf in {".jpg", ".jpeg", ".png", ".bmp"}:
            text = _ocr_image(p)
            st = "image_ocr" if text else "image"
            if not text:
                flags.append("needs_ocr")
            return sanitize_text(text)[:max_chars], st, flags
    except Exception as exc:
        flags.append(f"read_error:{exc}")
    return "", suf.lstrip(".") or "unknown", flags


def _pdf_text(p: Path, max_chars: int) -> str:
    try:
        import fitz

        doc = fitz.open(str(p))
        parts = [page.get_text() for page in doc]
        return "\n".join(parts)[:max_chars]
    except ImportError:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(p))
            parts = [pg.extract_text() or "" for pg in reader.pages[:30]]
            return "\n".join(parts)[:max_chars]
        except Exception:
            return ""
    except Exception:
        return ""


def _ocr_image(p: Path) -> str:
    try:
        import pytesseract
        from PIL import Image

        with Image.open(p) as im:
            return pytesseract.image_to_string(im, lang="chi_sim+eng")
    except Exception:
        return ""


def composite_from_row(row: dict[str, Any]) -> tuple[str, str]:
    parts = []
    for k in ("pathology_raw", "hpv", "tct", "other_clinical_attributes"):
        v = row.get(k, "")
        if v and str(v).strip() not in ("", "nan"):
            parts.append(f"{k}: {sanitize_text(str(v)[:500])}")
    text = " ".join(parts)
    return text, "composite"
