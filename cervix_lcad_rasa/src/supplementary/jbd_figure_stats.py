"""Paired-bootstrap lookups and significance annotation helpers for JBD figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MANUSCRIPT_REL = "outputs/publishable/tables/manuscript"
REVISION_REL = "outputs/publishable/mosaic_revision_audit/tables"


def _read(project: Path, rel: str) -> pd.DataFrame | None:
    p = project / rel
    if not p.is_file():
        return None
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None


def format_pvalue(p: float | None, *, style: str = "journal") -> str:
    """Format two-sided bootstrap p for figure annotations."""
    if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
        return "ns"
    p = float(p)
    if style == "sci":
        if p >= 0.05:
            return "ns"
        if p < 1e-4:
            return r"$P < 1.0 \times 10^{-4}$"
        if p < 0.001:
            return r"$P < 0.001$"
        return rf"$P = {p:.4g}$"
    if p >= 0.05:
        return "ns"
    if p < 0.0005:
        return "p < 0.001"
    if p < 0.001:
        return "p = 0.001"
    return f"p = {p:.3f}"


def load_comparator_pvals(project: Path) -> dict[str, float]:
    """Map comparator label -> paired bootstrap p (primary vs comparator)."""
    out: dict[str, float] = {}
    sources = [
        (f"{MANUSCRIPT_REL}/T_external_baseline_paired_bootstrap_recheck.csv", "comparator", "paired_bootstrap_p_two_sided"),
        (f"{REVISION_REL}/mosaic_vs_contrastive_paired_bootstrap.csv", "comparison", "paired_bootstrap_p_two_sided"),
        (f"{MANUSCRIPT_REL}/T2_pairwise_statistical_tests.csv", "comparator", "bootstrap_p_auc"),
    ]
    for rel, key_col, p_col in sources:
        df = _read(project, rel)
        if df is None or key_col not in df.columns or p_col not in df.columns:
            continue
        for _, row in df.iterrows():
            key = str(row[key_col])
            if key_col == "comparison" and "contrastive" in key.lower():
                key = "CLIP-style contrastive multimodal baseline"
            try:
                p = float(row[p_col])
            except (TypeError, ValueError):
                continue
            if key_col == "comparator" and p_col == "bootstrap_p_auc" and p > 0.49:
                delta = abs(float(row.get("delta_auc", 0.0)))
                if delta > 0.03:
                    continue
            if key not in out:
                out[key] = p
    mosaic_pb = _read(project, f"{MANUSCRIPT_REL}/T_mosaic_paired_bootstrap.csv")
    if mosaic_pb is not None and "paired_bootstrap_p_two_sided" in mosaic_pb.columns:
        out["MOSAIC (full) vs MOSAIC--RASA backbone"] = float(mosaic_pb["paired_bootstrap_p_two_sided"].iloc[0])
        out["MOSAIC--RASA backbone"] = float(mosaic_pb["paired_bootstrap_p_two_sided"].iloc[0])
    return out


def add_significance_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    p: float | None,
    *,
    h: float = 0.025,
    fontsize: float = 9.0,
    style: str = "journal",
) -> None:
    text = format_pvalue(p, style=style)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.25, c="#343434", clip_on=False, zorder=6)
    ax.text(
        (x1 + x2) / 2,
        y + h * 1.15,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="#343434",
        fontfamily="Arial",
        clip_on=False,
        zorder=7,
    )


def annotate_p_at_xy(
    ax: plt.Axes,
    x: float,
    y: float,
    p: float | None,
    *,
    ha: str = "left",
    va: str = "center",
    fontsize: float = 9.0,
    style: str = "journal",
) -> None:
    ax.text(
        x,
        y,
        format_pvalue(p, style=style),
        ha=ha,
        va=va,
        fontsize=fontsize,
        color="#343434",
        fontfamily="Arial",
        clip_on=False,
        zorder=7,
    )
