"""Case-level report supervision (dual archive centres: Enshi + Jingzhou)."""

from __future__ import annotations

from typing import Any

import pandas as pd

REAL_REPORT_CENTERS_PRIMARY = ("enshi", "jingzhou")
SPARSE_REPORT_CENTER = "xiangyang"


def case_has_real_report(row: dict[str, Any] | pd.Series) -> bool:
    """Case-level rule overrides centre-level assumptions (revised LCAD-RASA method)."""
    get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    center = str(get("center", get("center_id", ""))).strip()
    tier = str(get("report_archive_tier", "") or "").strip()
    rep_flag = int(get("report_available", 0) or 0) == 1

    if center == "enshi" and tier == "full":
        return True
    if center == "jingzhou" and rep_flag:
        return True
    if center == SPARSE_REPORT_CENTER and rep_flag:
        return True
    return rep_flag and tier not in ("none", "")


def report_supervision_class(row: dict[str, Any] | pd.Series) -> str:
    get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    if int(get("missing_label", 0) or 0) == 1:
        return "no_report_no_label_case"
    if case_has_real_report(row):
        center = str(get("center", get("center_id", "")))
        if center == SPARSE_REPORT_CENTER:
            return "sparse_report_case"
        return "real_report_case"
    return "pseudo_report_candidate"


def needs_pseudo_report(row: dict[str, Any] | pd.Series) -> bool:
    return report_supervision_class(row) == "pseudo_report_candidate"


def training_report_type(row: dict[str, Any] | pd.Series) -> str:
    return "real" if case_has_real_report(row) else "pseudo"
