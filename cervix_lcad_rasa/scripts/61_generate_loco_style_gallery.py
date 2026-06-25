#!/usr/bin/env python3
"""Generate Seaborn-reference style variants for the strict LOCO figure."""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.supplementary.jbd_figure_typography import (  # noqa: E402
    FONT_ARIAL,
    FONT_TIMES,
    apply_arial_to_figure,
    apply_mixed_en_typography,
    setup_arial_rcparams,
)

TABLES = ROOT / "outputs/publishable/tables/manuscript"
CENTRE_TABLE = ROOT / "outputs/publishable/tables/table_loco_center_characteristics.csv"
SEABORN_DIR = PROJECT / "Seaborn"
FINAL_DIR = PROJECT / "final_Fig/fig_loco_heatmap_style_gallery"
PUB_DIR = ROOT / "outputs/publishable/figures/jbd_final/fig_loco_heatmap_style_gallery"

TEXT = "#17212B"
GRID = "#E1E7EF"
BLUE = "#254B6D"
RUST = "#C65A46"
MID = "#557A95"
REF = "#95A1B2"
LIGHT = "#D6DEE8"
GOLD = "#D2AE76"
MODEL_ORDER = ["Real-report only", "No section alignment", "Full LCAD-RASA"]
MODEL_COLORS = {
    "Real-report only": REF,
    "No section alignment": MID,
    "Full LCAD-RASA": BLUE,
}
MODEL_MARKERS = {
    "Real-report only": "s",
    "No section alignment": "^",
    "Full LCAD-RASA": "D",
}
SEQ = LinearSegmentedColormap.from_list("loco_seq", ["#F7F9FC", "#D8E4EC", "#7F9AAC", BLUE], N=256)
DIV = LinearSegmentedColormap.from_list("loco_div", [RUST, "#F2D8BD", "#F7F9FC", "#A8BBC8", BLUE], N=256)


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def setup_theme() -> None:
    setup_arial_rcparams(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_ARIAL, "Helvetica", "DejaVu Sans", "sans-serif"],
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.80,
            "axes.titlesize": 13.2,
            "axes.labelsize": 11.4,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
            "legend.fontsize": 9.8,
            "legend.title_fontsize": 10.0,
            "font.size": 10.2,
            "mathtext.rm": FONT_TIMES,
            "mathtext.it": f"{FONT_TIMES}:italic",
            "mathtext.bf": f"{FONT_TIMES}:bold",
        }
    )
    sns.set_theme(style="whitegrid", context="paper", font=FONT_ARIAL)


def apply_style(fig: plt.Figure) -> None:
    fig._jbd_min_font_size_override = 8.9
    fig._jbd_max_font_size_override = 15.2
    apply_arial_to_figure(fig)
    apply_mixed_en_typography(fig)


def polish(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.set_facecolor("white")
    if grid_axis in {"x", "both"}:
        ax.grid(True, axis="x", color=GRID, linewidth=0.90, alpha=0.84)
    else:
        ax.grid(False, axis="x")
    if grid_axis in {"y", "both"}:
        ax.grid(True, axis="y", color=GRID, linewidth=0.90, alpha=0.84)
    else:
        ax.grid(False, axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C8D2DD")
        ax.spines[side].set_linewidth(0.9)


def save_fig(fig: plt.Figure, stem: str) -> None:
    apply_style(fig)
    for out_dir in (FINAL_DIR, PUB_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / stem
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def load_loco() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(TABLES / "S2_loco_strict_retrain.csv")
    labels = {
        "full_lcad_rasa": "Full LCAD-RASA",
        "real_report_only_decoder": "Real-report only",
        "report_generation_without_section_alignment": "No section alignment",
    }
    df = df.copy()
    df["Model"] = df["model"].map(labels).fillna(df["model"].astype(str))
    if CENTRE_TABLE.is_file():
        centre = pd.read_csv(CENTRE_TABLE).rename(columns={"center": "held_out_center"})
        df = df.merge(centre, on="held_out_center", how="left")
    df["report_supervision_density"] = pd.to_numeric(df.get("report_supervision_density"), errors="coerce")
    df["test_cases"] = pd.to_numeric(df["test_cases"], errors="coerce")
    df["auc"] = pd.to_numeric(df["auc"], errors="coerce")
    df["label_consistency"] = pd.to_numeric(df["label_consistency"], errors="coerce")
    df["delta_vs_chance"] = df["auc"] - 0.50
    df["supervision_pct"] = 100.0 * df["report_supervision_density"].fillna(0.0)
    df["positive_rate"] = pd.to_numeric(df.get("CIN2_positive_cases"), errors="coerce") / pd.to_numeric(df.get("total_cases"), errors="coerce")
    full = df[df["Model"].eq("Full LCAD-RASA")][["center_label", "auc"]].rename(columns={"auc": "full_auc"})
    summary = (
        df.groupby("center_label", as_index=False)
        .agg(test_cases=("test_cases", "max"), label_consistency=("label_consistency", "max"))
        .merge(full, on="center_label", how="left")
        .sort_values(["full_auc", "center_label"], ascending=[True, True])
        .reset_index(drop=True)
    )
    center_order = summary["center_label"].tolist()
    df["center_label"] = pd.Categorical(df["center_label"], categories=center_order, ordered=True)
    df["Model"] = pd.Categorical(df["Model"], categories=MODEL_ORDER, ordered=True)
    df["center_rank"] = df["center_label"].cat.codes.astype(float)
    df["model_rank"] = df["Model"].cat.codes.astype(float)
    return df.sort_values(["center_label", "Model"]).reset_index(drop=True), center_order


def auc_matrix(df: pd.DataFrame, center_order: list[str]) -> pd.DataFrame:
    return df.pivot_table(index="center_label", columns="Model", values="auc", observed=False).reindex(index=center_order, columns=MODEL_ORDER)


def base_caption() -> str:
    return (
        "Strict leave-one-centre-out (LOCO) evaluation under fixed quick-budget retraining. "
        "Tiles or points show AUROC for each held-out centre and model; the 0.50 reference marks chance-level discrimination. "
        "Point size encodes the held-out sample size where shown."
    )


def plot_heatmap(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    mat = auc_matrix(df, center_order)
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap=DIV, vmin=0.25, vmax=1.0, center=0.50, linewidths=0.9, linecolor="white", cbar_kws={"label": "AUROC"}, ax=ax)
    ax.set_title("Strict LOCO AUROC by held-out centre", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Held-out centre")
    ax.tick_params(axis="x", rotation=28)
    ax.tick_params(axis="y", rotation=0)
    return fig


def plot_cluster_heatmap(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    mat = auc_matrix(df, center_order)
    fig, ax = plt.subplots(figsize=(7.9, 5.8))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap=SEQ, vmin=0.25, vmax=1.0, linewidths=0.9, linecolor="white", cbar_kws={"label": "AUROC"}, ax=ax)
    ax.set_title("Centre-ordered LOCO AUROC structure", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Held-out centre")
    ax.tick_params(axis="x", rotation=28)
    ax.tick_params(axis="y", rotation=0)
    return fig


def plot_lollipop_facets(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.6), sharey=True)
    ypos = np.arange(len(center_order))
    for ax, model in zip(axes, MODEL_ORDER):
        sub = df[df["Model"].eq(model)].set_index("center_label").reindex(center_order).reset_index()
        color = MODEL_COLORS[model]
        ax.axvspan(0.25, 0.50, color=RUST, alpha=0.055, zorder=0)
        ax.axvline(0.50, color=TEXT, linewidth=1.15, linestyle=(0, (2, 2)), alpha=0.72)
        ax.hlines(ypos, 0.25, sub["auc"], color=color, linewidth=4.8, alpha=0.90)
        ax.scatter(sub["auc"], ypos, s=138, marker=MODEL_MARKERS[model], color=color, edgecolor=TEXT, linewidth=0.85, zorder=3)
        for y, auc in zip(ypos, sub["auc"]):
            ax.text(min(float(auc) + 0.026, 1.03), y, f"{float(auc):.3f}", ha="left", va="center", fontsize=9.4, fontweight="bold" if model == "Full LCAD-RASA" else "normal")
        ax.set_title(str(model), fontweight="bold", fontsize=12.4)
        ax.set_xlim(0.25, 1.08)
        ax.set_xlabel("AUROC", fontweight="bold")
        ax.set_yticks(ypos)
        ax.set_ylim(len(center_order) - 0.5, -0.5)
        polish(ax, "x")
    labels = []
    for center in center_order:
        n = int(df.loc[df["center_label"].eq(center), "test_cases"].max())
        labels.append(f"{center}  (n={n})")
    axes[0].set_yticklabels(labels)
    axes[0].set_ylabel("Held-out centre", fontweight="bold")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Strict LOCO centre-wise AUROC profile", fontsize=15.0, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.18, right=0.985, top=0.82, bottom=0.16, wspace=0.13)
    return fig


def plot_grouped_bar(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.4, 5.7))
    sns.barplot(data=df, y="center_label", x="auc", hue="Model", order=center_order, hue_order=MODEL_ORDER, palette=MODEL_COLORS, width=0.82, edgecolor=TEXT, linewidth=0.45, ax=ax)
    ax.axvline(0.50, color=TEXT, linewidth=1.05, linestyle=(0, (2, 2)), alpha=0.72)
    ax.set_title("Grouped AUROC bars across held-out centres", fontweight="bold", pad=12)
    ax.set_xlabel("AUROC", fontweight="bold")
    ax.set_ylabel("Held-out centre")
    ax.set_xlim(0.25, 1.04)
    polish(ax, "x")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", title="Model")
    return fig


def plot_horizontal_bar(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    sub = df[df["Model"].eq("Full LCAD-RASA")].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    color_map = {
        center: BLUE if float(sub.loc[sub["center_label"].eq(center), "auc"].iloc[0]) >= 0.50 else RUST
        for center in center_order
    }
    sns.barplot(
        data=sub,
        y="center_label",
        x="auc",
        hue="center_label",
        order=center_order,
        hue_order=center_order,
        palette=color_map,
        legend=False,
        edgecolor=TEXT,
        linewidth=0.55,
        ax=ax,
    )
    ax.axvline(0.50, color=TEXT, linewidth=1.10, linestyle=(0, (2, 2)), alpha=0.72)
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(min(width + 0.018, 1.02), patch.get_y() + patch.get_height() / 2, f"{width:.3f}", va="center", ha="left", fontsize=9.6, fontweight="bold")
    ax.set_title("Full LCAD-RASA strict LOCO AUROC", fontweight="bold", pad=12)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Held-out centre")
    ax.set_xlim(0.25, 1.05)
    polish(ax, "x")
    return fig


def plot_box_strip(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    sns.boxplot(data=df, y="Model", x="auc", order=MODEL_ORDER, color=LIGHT, fliersize=0, linewidth=1.0, ax=ax)
    sns.stripplot(data=df, y="Model", x="auc", order=MODEL_ORDER, hue="center_label", palette=sns.color_palette("crest", n_colors=len(center_order)), size=7.0, jitter=0.17, edgecolor=TEXT, linewidth=0.55, ax=ax)
    ax.axvline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_title("Centre observations within each LOCO model", fontweight="bold", pad=12)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("")
    ax.set_xlim(0.25, 1.04)
    polish(ax, "x")
    ax.legend(frameon=False, title="Held-out centre", bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig


def plot_violin(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    sns.violinplot(
        data=df,
        y="Model",
        x="auc",
        hue="Model",
        order=MODEL_ORDER,
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        legend=False,
        linewidth=1.1,
        inner=None,
        cut=0,
        alpha=0.55,
        ax=ax,
    )
    sns.stripplot(data=df, y="Model", x="auc", order=MODEL_ORDER, color="white", edgecolor=TEXT, linewidth=0.65, size=6.8, jitter=0.13, ax=ax)
    ax.axvline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_title("Distribution-style LOCO AUROC profile", fontweight="bold", pad=12)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("")
    ax.set_xlim(0.25, 1.04)
    polish(ax, "x")
    return fig


def plot_ecdf(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for model in MODEL_ORDER:
        vals = np.sort(df.loc[df["Model"].eq(model), "auc"].to_numpy(dtype=float))
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=MODEL_COLORS[model], linewidth=2.4, label=model)
        ax.scatter(vals, y, s=42, color=MODEL_COLORS[model], edgecolor=TEXT, linewidth=0.45)
    ax.axvline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_title("ECDF of centre-wise strict LOCO AUROC", fontweight="bold", pad=12)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Cumulative proportion")
    ax.set_xlim(0.25, 1.04)
    ax.legend(frameon=False)
    polish(ax)
    return fig


def plot_line_wide(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for model in MODEL_ORDER:
        sub = df[df["Model"].eq(model)].set_index("center_label").reindex(center_order).reset_index()
        ax.plot(sub["center_label"], sub["auc"], marker=MODEL_MARKERS[model], color=MODEL_COLORS[model], linewidth=2.2, markersize=8.0, label=model)
    ax.axhline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_title("Wide-form centre trend across LOCO models", fontweight="bold", pad=12)
    ax.set_xlabel("Held-out centre")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.25, 1.04)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    polish(ax)
    return fig


def plot_facet_lines(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 5, figsize=(14.8, 4.8), sharey=True)
    for ax, center in zip(axes, center_order):
        sub = df[df["center_label"].eq(center)].sort_values("Model")
        ax.axhline(0.50, color=TEXT, linewidth=0.95, linestyle=(0, (2, 2)), alpha=0.65)
        ax.plot(sub["Model"].astype(str), sub["auc"], color=BLUE, linewidth=2.1, marker="o", markersize=7.2)
        ax.set_title(str(center), fontweight="bold", fontsize=11.6)
        ax.set_ylim(0.25, 1.04)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=70)
        polish(ax)
    axes[0].set_ylabel("AUROC")
    fig.suptitle("Centre facets for strict LOCO model comparison", fontsize=14.8, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.35, wspace=0.20)
    return fig


def plot_scatter_semantics(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    sns.scatterplot(data=df, x="supervision_pct", y="auc", hue="Model", style="Model", size="test_cases", sizes=(80, 420), hue_order=MODEL_ORDER, palette=MODEL_COLORS, edgecolor=TEXT, linewidth=0.70, ax=ax)
    ax.axhline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_title("AUROC against report-supervision density", fontweight="bold", pad=12)
    ax.set_xlabel("Report-supervision density (%)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.25, 1.04)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    polish(ax)
    return fig


def plot_bubble_consistency(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    sns.scatterplot(data=df, x="label_consistency", y="auc", hue="Model", style="Model", size="test_cases", sizes=(80, 430), hue_order=MODEL_ORDER, palette=MODEL_COLORS, edgecolor=TEXT, linewidth=0.70, ax=ax)
    ax.axhline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_title("LOCO AUROC versus label consistency", fontweight="bold", pad=12)
    ax.set_xlabel("Label consistency")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.25, 1.04)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    polish(ax)
    return fig


def plot_regression(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    sns.regplot(data=df, x="supervision_pct", y="auc", scatter=False, color=TEXT, line_kws={"linewidth": 1.6, "alpha": 0.70}, ax=ax)
    sns.scatterplot(data=df, x="supervision_pct", y="auc", hue="Model", style="Model", s=115, hue_order=MODEL_ORDER, palette=MODEL_COLORS, edgecolor=TEXT, linewidth=0.65, ax=ax)
    ax.axhline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_title("Regression-style relation between supervision and AUROC", fontweight="bold", pad=12)
    ax.set_xlabel("Report-supervision density (%)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.25, 1.04)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    polish(ax)
    return fig


def plot_residuals(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    clean = df.dropna(subset=["supervision_pct", "auc"]).copy()
    x = clean["supervision_pct"].to_numpy(dtype=float)
    y = clean["auc"].to_numpy(dtype=float)
    coeff = np.polyfit(x, y, deg=1) if len(np.unique(x)) > 1 else np.array([0.0, np.nanmean(y)])
    clean["residual"] = y - (coeff[0] * x + coeff[1])
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    sns.scatterplot(data=clean, x="supervision_pct", y="residual", hue="Model", style="Model", s=120, hue_order=MODEL_ORDER, palette=MODEL_COLORS, edgecolor=TEXT, linewidth=0.65, ax=ax)
    ax.axhline(0, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_title("Residual profile after supervision-density trend", fontweight="bold", pad=12)
    ax.set_xlabel("Report-supervision density (%)")
    ax.set_ylabel("AUROC residual")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    polish(ax)
    return fig


def plot_distribution(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for model in MODEL_ORDER:
        vals = df.loc[df["Model"].eq(model), "auc"].to_numpy(dtype=float)
        ax.hist(vals, bins=np.linspace(0.25, 1.0, 9), histtype="stepfilled", alpha=0.18, color=MODEL_COLORS[model], edgecolor=MODEL_COLORS[model], linewidth=1.5, label=model)
        ax.scatter(vals, np.full_like(vals, 0.02), color=MODEL_COLORS[model], edgecolor=TEXT, linewidth=0.45, s=42)
    ax.axvline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_title("Distribution-style strict LOCO AUROC", fontweight="bold", pad=12)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Centre count")
    ax.set_xlim(0.25, 1.04)
    ax.legend(frameon=False)
    polish(ax)
    return fig


def plot_ridge(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    bins = np.linspace(0.25, 1.0, 18)
    for i, model in enumerate(MODEL_ORDER):
        vals = df.loc[df["Model"].eq(model), "auc"].to_numpy(dtype=float)
        hist, edges = np.histogram(vals, bins=bins, density=False)
        xs = (edges[:-1] + edges[1:]) / 2
        heights = hist / max(hist.max(), 1) * 0.55
        base = i
        ax.fill_between(xs, base, base + heights, color=MODEL_COLORS[model], alpha=0.42, linewidth=0)
        ax.plot(xs, base + heights, color=MODEL_COLORS[model], linewidth=1.8)
        ax.scatter(vals, np.full_like(vals, base), color=MODEL_COLORS[model], edgecolor=TEXT, linewidth=0.45, s=42, zorder=3)
    ax.axvline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_title("Ridgeline-style AUROC distributions", fontweight="bold", pad=12)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("")
    ax.set_xlim(0.25, 1.04)
    polish(ax, "x")
    return fig


def plot_pair_matrix(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    vars_ = ["auc", "label_consistency", "supervision_pct"]
    labels = ["AUROC", "Label consistency", "Report density (%)"]
    fig, axes = plt.subplots(3, 3, figsize=(8.6, 8.0))
    for i, yvar in enumerate(vars_):
        for j, xvar in enumerate(vars_):
            ax = axes[i, j]
            if i == j:
                for model in MODEL_ORDER:
                    vals = df.loc[df["Model"].eq(model), xvar]
                    ax.hist(vals, bins=6, color=MODEL_COLORS[model], alpha=0.28, edgecolor="white")
            else:
                sns.scatterplot(data=df, x=xvar, y=yvar, hue="Model", hue_order=MODEL_ORDER, palette=MODEL_COLORS, s=42, edgecolor=TEXT, linewidth=0.35, legend=False, ax=ax)
            if i == 2:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            polish(ax)
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS[m], markeredgecolor=TEXT, markersize=7, label=m) for m in MODEL_ORDER]
    fig.legend(handles=handles, labels=MODEL_ORDER, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Pairwise LOCO metric relationships", fontsize=14.6, fontweight="bold", y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_joint_marginal(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig = plt.figure(figsize=(8.0, 6.2))
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1:, :-1])
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax)
    sns.scatterplot(data=df, x="label_consistency", y="auc", hue="Model", style="Model", size="test_cases", sizes=(70, 320), hue_order=MODEL_ORDER, palette=MODEL_COLORS, edgecolor=TEXT, linewidth=0.60, ax=ax)
    sns.histplot(data=df, x="label_consistency", hue="Model", hue_order=MODEL_ORDER, palette=MODEL_COLORS, bins=8, alpha=0.25, legend=False, ax=ax_top)
    sns.histplot(data=df, y="auc", hue="Model", hue_order=MODEL_ORDER, palette=MODEL_COLORS, bins=8, alpha=0.25, legend=False, ax=ax_right)
    ax.axhline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_xlabel("Label consistency")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.25, 1.04)
    ax.legend(frameon=False, loc="lower right")
    ax_top.set_xlabel("")
    ax_top.set_ylabel("")
    ax_right.set_xlabel("")
    ax_right.set_ylabel("")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    polish(ax)
    polish(ax_top)
    polish(ax_right)
    fig.suptitle("Joint-style LOCO behaviour with marginal distributions", fontsize=14.4, fontweight="bold", y=0.99)
    return fig


def plot_conditional_means(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    sns.pointplot(data=df, x="auc", y="Model", order=MODEL_ORDER, color=TEXT, errorbar=None, markers="D", linestyle="none", ax=ax)
    sns.stripplot(data=df, x="auc", y="Model", order=MODEL_ORDER, hue="center_label", palette=sns.color_palette("crest", n_colors=len(center_order)), size=6.8, jitter=0.18, edgecolor=TEXT, linewidth=0.50, ax=ax)
    ax.axvline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_title("Conditional means with centre observations", fontweight="bold", pad=12)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("")
    ax.set_xlim(0.25, 1.04)
    polish(ax, "x")
    ax.legend(frameon=False, title="Held-out centre", bbox_to_anchor=(1.02, 1), loc="upper left")
    return fig


def plot_paired(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for center in center_order:
        sub = df[df["center_label"].eq(center)].sort_values("Model")
        ax.plot(sub["Model"].astype(str), sub["auc"], color="#A8B6C5", linewidth=1.5, alpha=0.75, zorder=1)
        ax.scatter(sub["Model"].astype(str), sub["auc"], s=92, c=[MODEL_COLORS[str(m)] for m in sub["Model"].astype(str)], edgecolor=TEXT, linewidth=0.65, zorder=2)
        ax.text(2.08, float(sub[sub["Model"].eq("Full LCAD-RASA")]["auc"].iloc[0]), str(center), va="center", fontsize=8.8, color=TEXT)
    ax.axhline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_title("Paired centre trajectories across LOCO models", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.25, 1.04)
    ax.set_xlim(-0.35, 2.65)
    ax.tick_params(axis="x", rotation=18)
    polish(ax)
    return fig


def plot_tile_scatter(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    sns.scatterplot(data=df, x="Model", y="center_label", hue="auc", size="test_cases", sizes=(170, 620), palette=DIV, hue_norm=(0.25, 1.0), edgecolor=TEXT, linewidth=0.70, ax=ax)
    for _, row in df.iterrows():
        ax.text(row["Model"], row["center_label"], f"{float(row['auc']):.2f}", ha="center", va="center", fontsize=8.8, fontweight="bold", color="white" if row["auc"] > 0.62 else TEXT)
    ax.set_title("Scatterplot heatmap of strict LOCO AUROC", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Held-out centre")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    polish(ax)
    return fig


def plot_anova(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    sns.pointplot(data=df, x="center_label", y="auc", hue="Model", order=center_order, hue_order=MODEL_ORDER, palette=MODEL_COLORS, dodge=0.35, errorbar=None, markers=["s", "^", "D"], linestyles=["-", "-", "-"], ax=ax)
    ax.axhline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_title("Two-factor profile: held-out centre by model", fontweight="bold", pad=12)
    ax.set_xlabel("Held-out centre")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.25, 1.04)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", title="Model")
    polish(ax)
    return fig


def plot_palette(df: pd.DataFrame, center_order: list[str], ref: str) -> plt.Figure:
    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(8.5, 6.0), height_ratios=[0.55, 4.0])
    for i, model in enumerate(MODEL_ORDER):
        ax0.barh(0, 1, left=i, color=MODEL_COLORS[model], edgecolor="white")
        ax0.text(i + 0.5, 0, model, ha="center", va="center", fontsize=9.5, fontweight="bold", color="white" if model != "Real-report only" else TEXT)
    ax0.set_xlim(0, len(MODEL_ORDER))
    ax0.set_axis_off()
    sub = df[df["Model"].eq("Full LCAD-RASA")]
    sns.barplot(data=sub, y="center_label", x="auc", order=center_order, color=BLUE, edgecolor=TEXT, linewidth=0.55, ax=ax)
    ax.axvline(0.50, color=TEXT, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.70)
    ax.set_title("Palette-aligned Full LCAD-RASA LOCO summary", fontweight="bold", pad=10)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Held-out centre")
    ax.set_xlim(0.25, 1.05)
    polish(ax, "x")
    return fig


def choose_plot(slug: str) -> Callable[[pd.DataFrame, list[str], str], plt.Figure]:
    if "annotated_heatmaps" in slug:
        return plot_heatmap
    if "discovering_structure" in slug or "cubehelix" in slug:
        return plot_cluster_heatmap
    if "diagonal_correlation" in slug:
        return plot_pair_matrix
    if "heatmap" in slug or "trivariate_histogram" in slug:
        return plot_tile_scatter
    if "boxplot" in slug:
        return plot_box_strip
    if "violin" in slug:
        return plot_violin
    if "horizontal_bar" in slug:
        return plot_horizontal_bar
    if "barplots" in slug:
        return plot_grouped_bar
    if "dot_plot" in slug:
        return plot_lollipop_facets
    if "ecdf" in slug:
        return plot_ecdf
    if "ridge" in slug:
        return plot_ridge
    if "line_plots" in slug or "large_number_of_facets" in slug or "facetgrid" in slug or "small_multiple" in slug:
        return plot_facet_lines
    if "lineplot_from" in slug or "timeseries" in slug:
        return plot_line_wide
    if "regression" in slug or "linear" in slug or "logistic" in slug:
        return plot_regression
    if "residuals" in slug:
        return plot_residuals
    if "conditional_means" in slug:
        return plot_conditional_means
    if "paired_categorical" in slug:
        return plot_paired
    if "joint" in slug or "marginal" in slug or "hexbin" in slug:
        return plot_joint_marginal
    if "scatterplot_matrix" in slug or "paired_density" in slug or "anscombe" in slug:
        return plot_pair_matrix
    if "histogram" in slug or "distributions" in slug or "kde" in slug or "kernel_density" in slug:
        return plot_distribution
    if "continuous_hues" in slug or "varying_point_sizes" in slug:
        return plot_bubble_consistency
    if "scatterplot" in slug:
        return plot_scatter_semantics
    if "anova" in slug:
        return plot_anova
    if "color_palette" in slug:
        return plot_palette
    return plot_lollipop_facets


def recommendation(slug: str) -> str:
    if any(key in slug for key in ("dot_plot", "scatterplot_with_categorical", "varying_point_sizes", "conditional_means", "horizontal_boxplot", "annotated_heatmaps", "line_plots")):
        return "recommended"
    if any(key in slug for key in ("kde", "histogram", "violin", "ridge", "hexbin")):
        return "exploratory only; n is small for distribution inference"
    return "candidate"


def main() -> None:
    setup_theme()
    df, center_order = load_loco()
    refs = [p.stem for p in sorted(SEABORN_DIR.glob("*.png"))]
    rows = []
    for ref in refs:
        slug = slugify(ref)
        stem = f"fig_loco_heatmap_{slug}"
        func = choose_plot(slug)
        fig = func(df.copy(), center_order, ref)
        save_fig(fig, stem)
        rows.append(
            {
                "stem": stem,
                "seaborn_reference": ref,
                "plot_function": func.__name__,
                "recommendation": recommendation(slug),
                "caption": base_caption(),
            }
        )
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    PUB_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    for out_dir in (FINAL_DIR, PUB_DIR):
        manifest.to_csv(out_dir / "STYLE_GALLERY_MANIFEST.csv", index=False)
        lines = ["# fig_loco_heatmap style gallery", ""]
        lines.append("Recommended main-text candidates: dot plot with several variables, scatterplot with categorical variables, conditional means with observations, and annotated heatmaps.")
        lines.append("")
        for row in rows:
            lines.extend(
                [
                    f"## {row['stem']}",
                    "",
                    f"**Seaborn reference.** {row['seaborn_reference']}.",
                    "",
                    f"**Recommendation.** {row['recommendation']}.",
                    "",
                    f"**Caption.** {row['caption']}",
                    "",
                ]
            )
        (out_dir / "SCI_CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {len(rows)} LOCO style variants in {FINAL_DIR}")


if __name__ == "__main__":
    main()
