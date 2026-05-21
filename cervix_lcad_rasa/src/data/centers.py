"""Centre roles: dual report archive (Enshi full + Jingzhou partial)."""

from __future__ import annotations

import pandas as pd

from src.data.report_supervision import REAL_REPORT_CENTERS_PRIMARY

SEMANTIC_ANCHOR_CENTER = "enshi"
REPORT_ARCHIVE_CENTERS = list(REAL_REPORT_CENTERS_PRIMARY)


def identify_report_archive_center(df: pd.DataFrame) -> str:
    """Primary archive centre by real-report count (Enshi expected when tier=full)."""
    if "has_real_report" not in df.columns:
        return SEMANTIC_ANCHOR_CENTER
    sub = df[df["has_real_report"] == 1]
    if sub.empty:
        return SEMANTIC_ANCHOR_CENTER
    counts = sub["center_id"].value_counts()
    return str(counts.index[0])


def masking_validation_mask(df: pd.DataFrame) -> pd.Series:
    """Enshi + Jingzhou real-report cases (+ optional sparse Xiangyang)."""
    if "report_supervision_class" in df.columns:
        base = df["report_supervision_class"].isin(
            ["real_report_case", "sparse_report_case"]
        )
    else:
        base = df["has_real_report"] == 1
    primary = df["center_id"].isin(REPORT_ARCHIVE_CENTERS) & (df["has_real_report"] == 1)
    sparse = (df["center_id"] == "xiangyang") & (df["has_real_report"] == 1)
    return primary | sparse if "center_id" in df.columns else base


def weak_label_mask(df: pd.DataFrame) -> pd.Series:
    """Report-missing cases → LCAD pseudo-report targets."""
    if "needs_pseudo_report" in df.columns:
        return df["needs_pseudo_report"].astype(bool)
    return df["has_real_report"] == 0
