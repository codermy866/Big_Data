#!/usr/bin/env python3
"""Clinical field harmonisation (shared by Exp0 ledger and manifests)."""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

HR_HPV_GENOTYPES = {
    "16", "18", "31", "33", "35", "39", "45", "51", "52", "53", "56", "58", "59", "66", "68", "73", "82", "81",
}

CENTER_ID = {
    "武大人民医院": "wuda",
    "恩施州中心医院": "enshi",
    "襄阳市中心医院": "xiangyang",
    "十堰市人民医院": "shiyan",
    "荆州市第一人民医院": "jingzhou",
}

REPORT_ARCHIVE = {
    "wuda": "none",
    "enshi": "full",
    "xiangyang": "sparse",
    "shiyan": "none",
    "jingzhou": "archive_yes_low_events",
}


def classify_hpv(val) -> str:
    if pd.isna(val) or str(val).strip() in ("", "nan", "-", "—"):
        return "unclassifiable"
    s = str(val).strip()
    if s in {"阴性", "negative", "Negative", "NEGATIVE"}:
        return "negative"
    if "高危" in s and not re.search(r"\d", s):
        return "hr_positive"
    nums = re.findall(r"\d+", s)
    if nums and any(n in HR_HPV_GENOTYPES for n in nums):
        return "hr_positive"
    if nums:
        return "unclassifiable"
    return "unclassifiable"


def classify_tct(val) -> str:
    if pd.isna(val) or str(val).strip() in ("", "nan", "-", "—"):
        return "missing"
    s = str(val).strip().upper()
    if s in {"NILM", "NORMAL", "WNL"} or "NILM" in s:
        return "negative"
    if "ASC-US" in s:
        return "asc_us"
    if any(k in s for k in ("LSIL", "ASC-H", "HSIL", "AGC", "SCC", "癌", "MALIGNANT", "CA")):
        return "lsil_or_worse"
    return "missing"


def infer_oct_abnormal(oct_reading) -> Optional[int]:
    if pd.isna(oct_reading) or str(oct_reading).strip() in ("", "nan"):
        return None
    s = str(oct_reading)
    if "高级别" in s or "疑似" in s:
        return 1
    if "未发现" in s or "低级别" in s:
        return 0
    return None


def _pathology_text(*fields) -> str:
    return " ".join(
        str(x).strip() for x in fields if pd.notna(x) and str(x).strip() not in ("", "nan", "None")
    )


def _strip_negated_cancer_phrases(text: str) -> str:
    cleaned = text
    for pat in (
        r"未见[^。；;，,]{0,40}癌[^。；;，,]*",
        r"无[^。；;，,]{0,30}癌[^。；;，,]*",
    ):
        cleaned = re.sub(pat, " ", cleaned)
    return cleaned


def classify_histology(path_result, path_grade=None, treatment=None, oct_abnormal: Optional[int] = None) -> str:
    primary = _pathology_text(path_result, path_grade)
    follow = _pathology_text(treatment)
    text = _pathology_text(path_result, path_grade, treatment)
    if not text:
        return "missing"
    primary = _strip_negated_cancer_phrases(primary)
    text = _strip_negated_cancer_phrases(text)

    if re.search(r"浸润|鳞状细胞癌|鳞癌|腺癌|恶性肿瘤|宫颈癌", primary) or re.search(
        r"浸润性鳞状细胞癌|宫颈癌", follow
    ):
        return "invasive"
    if re.search(r"CIN\s*(?:3|III|Ⅲ|三)|上皮内癌|原位癌", text, re.I):
        return "cin3"
    if re.search(r"CIN\s*(?:2|II|Ⅱ|二)", text, re.I) or re.search(r"HSIL|高度.*上皮内", text, re.I):
        return "cin2"
    if re.search(r"CIN\s*(?:1|I|一)|低级别|LSIL|湿疣|慢性", text, re.I):
        return "cin0_1"
    if oct_abnormal == 1:
        return "cin0_1"
    if oct_abnormal == 0:
        return "cin0_1"
    return "missing"


def hist_to_endpoints(hist: str) -> tuple[int, int, int]:
    cin2plus = int(hist in {"cin2", "cin3", "invasive"})
    cin3plus = int(hist in {"cin3", "invasive"})
    return cin2plus, cin3plus, int(hist == "invasive")
