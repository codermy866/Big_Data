#!/usr/bin/env python3
"""Generate Seaborn-inspired style candidates for the pseudo-report source figure."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

TABLE = ROOT / "outputs/publishable/tables/manuscript/T_theme1_llm_vs_template_rule_pseudo_report.csv"
OUT_DIRS = [
    ROOT / "outputs/publishable/theme1_alignment/figures/Figure_theme1_pseudo_report_source_comparison_style_gallery",
    PROJECT / "final_Fig/Figure_theme1_pseudo_report_source_comparison_style_gallery",
]

TEXT = "#17212B"
GRID = "#E2E7EE"
REF = "#95A1B2"
BLUE = "#254B6D"
SLATE = "#7D8793"
LIGHT = "#D6DEE8"
RED = "#C65A46"
MID = "#557A95"
PALETTE = [BLUE, REF, LIGHT, RED, SLATE, MID]
SOURCE_ORDER = ["Template", "Rule-based", "Local LLM"]
SOURCE_PALETTE = {"Template": SLATE, "Rule-based": BLUE, "Local LLM": RED}
GROUP_ORDER = ["Schema and label", "Modality grounding", "Text diversity", "Latent alignment"]
GROUP_PALETTE = {
    "Schema and label": REF,
    "Modality grounding": BLUE,
    "Text diversity": RED,
    "Latent alignment": SLATE,
}


METRIC_SPECS = [
    ("Schema and label", "Section complete", "section_complete_rate", True),
    ("Schema and label", "Label consistency", "label_consistency_mean", True),
    ("Modality grounding", "OCT support", "oct_supported_rate", True),
    ("Modality grounding", "Colposcopy support", "colposcopy_supported_rate", True),
    ("Modality grounding", "Clinical support", "instruction_supported_rate", True),
    ("Modality grounding", "Mean support", "mean_modality_support_rate", True),
    ("Text diversity", "Unique text", "unique_text_rate", True),
    ("Text diversity", "Duplicate fraction", "max_duplicate_fraction", False),
    ("Latent alignment", "Alignment MRR", "latent_alignment_mrr_full_model", True),
    ("Latent alignment", "Alignment gap", "latent_alignment_gap_full_model", True),
]


def setup_theme() -> None:
    setup_arial_rcparams(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_ARIAL, "Helvetica", "DejaVu Sans", "sans-serif"],
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.82,
            "axes.titlesize": 13.5,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
            "legend.fontsize": 10.2,
            "legend.title_fontsize": 10.2,
            "font.size": 10.8,
        }
    )
    sns.set_theme(style="whitegrid", context="talk", font=FONT_ARIAL, palette=PALETTE)


def sanitize(name: str) -> str:
    name = name.lower().replace("&", "and")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def source_label(raw: str) -> str:
    return {
        "label_template": "Template",
        "rule_based": "Rule-based",
        "local_llm": "Local LLM",
    }.get(raw, raw)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = pd.read_csv(TABLE)
    wide["Source"] = wide["pseudo_report_source"].map(source_label)
    wide["source_idx"] = wide["Source"].map({s: i for i, s in enumerate(SOURCE_ORDER)})
    align_max = max(float(wide["latent_alignment_mrr_full_model"].max()), 1e-9)
    wide["alignment_norm"] = wide["latent_alignment_mrr_full_model"] / align_max
    wide["inverse_duplicate"] = 1.0 - wide["max_duplicate_fraction"]
    wide["semantic_utility"] = wide[
        ["label_consistency_mean", "mean_modality_support_rate", "unique_text_rate", "alignment_norm"]
    ].mean(axis=1)
    wide["risk_index"] = wide["max_duplicate_fraction"]

    rows: list[dict[str, object]] = []
    for group, metric, col, higher_good in METRIC_SPECS:
        if col not in wide.columns:
            continue
        for _, row in wide.iterrows():
            val = float(row[col])
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "column": col,
                    "Source": row["Source"],
                    "source_idx": int(row["source_idx"]),
                    "value": val,
                    "display_value": val,
                    "higher_good": higher_good,
                    "benefit_value": val if higher_good else 1.0 - val,
                }
            )
    long = pd.DataFrame(rows)
    long["Source"] = pd.Categorical(long["Source"], SOURCE_ORDER, ordered=True)
    long["group"] = pd.Categorical(long["group"], GROUP_ORDER, ordered=True)

    key = wide[
        [
            "Source",
            "source_idx",
            "label_consistency_mean",
            "mean_modality_support_rate",
            "unique_text_rate",
            "max_duplicate_fraction",
            "latent_alignment_mrr_full_model",
            "latent_alignment_gap_full_model",
            "alignment_norm",
            "inverse_duplicate",
            "semantic_utility",
            "risk_index",
        ]
    ].copy()
    return wide, long, key


def apply_style(fig: plt.Figure, *, min_size: float = 9.4, max_size: float = 16.0) -> None:
    fig._jbd_mixed_en_typography = True
    fig._jbd_min_font_size_override = min_size
    fig._jbd_max_font_size_override = max_size
    apply_arial_to_figure(fig)
    apply_mixed_en_typography(fig)


def save_fig(fig: plt.Figure, stem: str) -> list[Path]:
    apply_style(fig)
    written = []
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / stem
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.08)
        written.append(base.with_suffix(".pdf"))
    plt.close(fig)
    return written


def set_numeric_ticks(ax: plt.Axes) -> None:
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        text = tick.get_text().strip()
        if any(ch.isdigit() for ch in text):
            tick.set_fontfamily(FONT_TIMES)
        else:
            tick.set_fontfamily(FONT_ARIAL)


def annotate_sources(ax: plt.Axes, data: pd.DataFrame, x: str, y: str, dx: float = 0.012, dy: float = 0.0) -> None:
    for _, row in data.iterrows():
        xv = float(row[x])
        yv = float(row[y])
        x_text = xv + dx
        ha = "left"
        if xv >= 0.92:
            x_text = xv - abs(dx) * 1.35
            ha = "right"
        ax.text(
            x_text,
            yv + dy,
            str(row["Source"]),
            ha=ha,
            va="center",
            fontsize=10.4,
            fontweight="bold",
            fontfamily=FONT_ARIAL,
            color=TEXT,
            clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 1.0},
        )


def plot_novel_quadrant(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.9, 6.5))
    sc = ax.scatter(
        key["mean_modality_support_rate"],
        key["unique_text_rate"],
        s=680 * key["label_consistency_mean"],
        c=key["max_duplicate_fraction"],
        cmap=sns.light_palette(RED, as_cmap=True),
        edgecolor=TEXT,
        linewidth=1.0,
        zorder=4,
    )
    annotate_sources(ax, key, "mean_modality_support_rate", "unique_text_rate", dx=0.045)
    ax.axvline(0.5, color=GRID, linewidth=1.2, zorder=1)
    ax.axhline(0.10, color=GRID, linewidth=1.2, zorder=1)
    ax.set_xlim(-0.06, 1.18)
    ax.set_ylim(-0.025, 0.235)
    ax.set_xlabel("Mean modality support", fontweight="bold", fontfamily=FONT_ARIAL)
    ax.set_ylabel("Unique text rate", fontweight="bold", fontfamily=FONT_ARIAL)
    ax.set_title("Novel semantic-risk quadrant for pseudo-report sources", fontweight="bold", pad=12)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.055, pad=0.035)
    cbar.set_label("Duplicate fraction", fontfamily=FONT_ARIAL, fontweight="bold")
    for _, row in key.iterrows():
        ax.text(
            float(row["mean_modality_support_rate"]),
            float(row["unique_text_rate"]) - 0.014,
            f"MRR {float(row['latent_alignment_mrr_full_model']):.3f}",
            ha="center",
            va="top",
            fontsize=9.2,
            fontfamily=FONT_ARIAL,
            color=TEXT,
        )
    ax.grid(True, color=GRID, linewidth=0.9, alpha=0.82)
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.12, right=0.90, top=0.88, bottom=0.15)
    return fig


def plot_annotated_heatmap(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    matrix = long.pivot_table(index="Source", columns="metric", values="benefit_value", aggfunc="mean").reindex(SOURCE_ORDER)
    metrics = [m for _, m, col, _ in METRIC_SPECS if m in matrix.columns]
    matrix = matrix[metrics]
    display_labels = [
        {
            "Section complete": "Section",
            "Label consistency": "Label",
            "OCT support": "OCT",
            "Colposcopy support": "Colpo.",
            "Clinical support": "Clinical",
            "Mean support": "Mean",
            "Unique text": "Unique",
            "Duplicate fraction": "Low dup.\n(1-risk)",
            "Alignment MRR": "MRR",
            "Alignment gap": "Gap",
        }.get(metric, metric)
        for metric in metrics
    ]
    fig, ax = plt.subplots(figsize=(14.4, 5.7))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=sns.light_palette(BLUE, as_cmap=True),
        vmin=0,
        vmax=max(1.0, float(np.nanmax(matrix.to_numpy()))),
        annot=True,
        fmt=".3f",
        linewidths=0.85,
        linecolor="white",
        cbar_kws={"label": "Benefit-oriented metric value"},
        annot_kws={"fontsize": 9.2, "fontfamily": FONT_TIMES, "color": TEXT},
    )
    ax.set_title("Annotated heatmap: benefit-oriented pseudo-report source profile", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(len(metrics)) + 0.5)
    ax.set_xticklabels(display_labels, rotation=0, ha="center", va="top", fontfamily=FONT_ARIAL, fontsize=10.8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontfamily=FONT_ARIAL, fontweight="bold", fontsize=12.2)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=0, pad=8)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.25)
    return fig


def plot_grouped_heatmap_facets(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), constrained_layout=True)
    for ax, group in zip(axes.flat, GROUP_ORDER):
        sub = long[long["group"].eq(group)]
        matrix = sub.pivot_table(index="Source", columns="metric", values="display_value", aggfunc="mean").reindex(SOURCE_ORDER)
        vmax = max(0.10, float(np.nanmax(matrix.to_numpy())) * 1.05)
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=sns.light_palette(GROUP_PALETTE[group], as_cmap=True),
            vmin=0,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            linewidths=0.85,
            linecolor="white",
            cbar=False,
            annot_kws={"fontsize": 8.8, "fontfamily": FONT_TIMES, "color": TEXT},
        )
        ax.set_title(group, fontweight="bold", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=24)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Trivariate categorical heatmap by metric family", fontsize=15.2, fontweight="bold")
    return fig


def plot_clustered_heatmap(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    matrix = long.pivot_table(index="Source", columns="metric", values="benefit_value", aggfunc="mean").reindex(SOURCE_ORDER)
    matrix = matrix[[m for _, m, _, _ in METRIC_SPECS if m in matrix.columns]]
    grid = sns.clustermap(
        matrix,
        cmap=sns.light_palette(BLUE, as_cmap=True),
        linewidths=0.75,
        linecolor="white",
        figsize=(12.0, 5.9),
        row_cluster=False,
        col_cluster=True,
        cbar_pos=(0.91, 0.22, 0.018, 0.48),
        dendrogram_ratio=(0.08, 0.18),
    )
    grid.fig.suptitle("Clustered heatmap: benefit-oriented pseudo-report profile", y=1.03, fontweight="bold")
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    grid.ax_heatmap.tick_params(axis="x", rotation=35)
    grid.ax_heatmap.tick_params(axis="y", rotation=0)
    return grid.fig


def plot_dotplot(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 5.6), sharex=False)
    for ax, group in zip(axes, GROUP_ORDER):
        sub = long[long["group"].eq(group)].copy()
        sns.scatterplot(
            data=sub,
            x="display_value",
            y="metric",
            hue="Source",
            hue_order=SOURCE_ORDER,
            palette=SOURCE_PALETTE,
            s=112,
            edgecolor=TEXT,
            linewidth=0.75,
            ax=ax,
        )
        ax.set_title(group, fontweight="bold", pad=10)
        ax.set_xlabel("Metric value", fontweight="bold")
        ax.set_ylabel("")
        ax.set_xlim(min(-0.08, float(sub["display_value"].min()) - 0.04), max(1.05, float(sub["display_value"].max()) + 0.03))
        ax.grid(True, axis="x", color=GRID, linewidth=0.85)
        ax.grid(False, axis="y")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        set_numeric_ticks(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Dot plot with several variables: pseudo-report source comparison", fontweight="bold", y=0.99)
    sns.despine(fig=fig)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.22, wspace=0.50)
    return fig


def plot_grouped_barplots(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4), constrained_layout=True)
    for ax, group in zip(axes.flat, GROUP_ORDER):
        sub = long[long["group"].eq(group)].copy()
        sns.barplot(
            data=sub,
            x="metric",
            y="display_value",
            hue="Source",
            hue_order=SOURCE_ORDER,
            palette=SOURCE_PALETTE,
            edgecolor=TEXT,
            linewidth=0.65,
            ax=ax,
        )
        ax.set_title(group, fontweight="bold", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0, max(1.05, float(sub["display_value"].max()) * 1.14))
        ax.tick_params(axis="x", rotation=22)
        ax.grid(True, axis="y", color=GRID, linewidth=0.85)
        ax.grid(False, axis="x")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Grouped barplots: structured pseudo-report source metrics", fontweight="bold")
    return fig


def plot_horizontal_bars(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.5), sharex=True, sharey=True)
    metric_order = [
        "Mean support",
        "Label consistency",
        "Unique text",
        "Duplicate fraction",
        "Alignment MRR",
    ]
    sub = long[long["metric"].isin(metric_order)].copy()
    for i, (ax, source) in enumerate(zip(axes, SOURCE_ORDER)):
        s = sub[sub["Source"].eq(source)].copy()
        s["metric"] = pd.Categorical(s["metric"], metric_order[::-1], ordered=True)
        sns.barplot(data=s, x="display_value", y="metric", color=SOURCE_PALETTE[source], edgecolor=TEXT, linewidth=0.8, ax=ax)
        for _, row in s.iterrows():
            ax.text(float(row["display_value"]) + 0.015, str(row["metric"]), f"{float(row['display_value']):.3f}", va="center", ha="left", fontfamily=FONT_TIMES, fontsize=9.4)
        ax.set_title(source, fontweight="bold", pad=10)
        ax.set_xlabel("Metric value")
        ax.set_ylabel("")
        if i > 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xlim(0, 1.08)
        ax.grid(True, axis="x", color=GRID, linewidth=0.85)
        ax.grid(False, axis="y")
    fig.suptitle("Horizontal bar plots: source-specific metric profile", fontweight="bold", y=0.98)
    sns.despine(fig=fig)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.82, bottom=0.15, wspace=0.10)
    return fig


def plot_box_strip(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    sns.boxplot(data=long, x="value", y="Source", order=SOURCE_ORDER, hue="Source", palette=SOURCE_PALETTE, showfliers=False, width=0.48, linewidth=1.0, legend=False, ax=ax)
    sns.stripplot(data=long, x="value", y="Source", order=SOURCE_ORDER, hue="group", palette=GROUP_PALETTE, dodge=False, size=7, edgecolor=TEXT, linewidth=0.5, alpha=0.88, ax=ax)
    ax.set_title("Horizontal boxplot with observations across all metrics", fontweight="bold", pad=12)
    ax.set_xlabel("Metric value")
    ax.set_ylabel("")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, axis="x", color=GRID, linewidth=0.85)
    ax.grid(False, axis="y")
    leg = ax.legend(frameon=False, loc="lower right", title="Metric family")
    if leg is not None:
        leg.get_title().set_fontfamily(FONT_ARIAL)
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.16, right=0.95, top=0.88, bottom=0.14)
    return fig


def plot_violin(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.4, 6.1))
    sns.violinplot(data=long, x="Source", y="value", order=SOURCE_ORDER, hue="Source", palette=SOURCE_PALETTE, inner=None, cut=0, linewidth=1.0, legend=False, ax=ax)
    sns.stripplot(data=long, x="Source", y="value", order=SOURCE_ORDER, color=TEXT, size=5.5, alpha=0.70, jitter=0.14, ax=ax)
    ax.set_title("Grouped violinplot from metric-value distribution", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", color=GRID, linewidth=0.85)
    ax.grid(False, axis="x")
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.13)
    return fig


def plot_ecdf(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.9, 6.0))
    for source in SOURCE_ORDER:
        vals = np.sort(long[long["Source"].eq(source)]["value"].to_numpy(dtype=float))
        y = np.arange(1, len(vals) + 1) / max(len(vals), 1)
        ax.step(vals, y, where="post", linewidth=2.5, color=SOURCE_PALETTE[source], label=source)
        ax.scatter(vals, y, s=42, color=SOURCE_PALETTE[source], edgecolor=TEXT, linewidth=0.45, zorder=4)
    ax.set_title("Facetted ECDF style: distribution of metric values by source", fontweight="bold", pad=12)
    ax.set_xlabel("Metric value")
    ax.set_ylabel("Cumulative proportion")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, color=GRID, linewidth=0.85)
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    return fig


def plot_hist_facets(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.7), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 8)
    for ax, source in zip(axes, SOURCE_ORDER):
        vals = long[long["Source"].eq(source)]["value"].to_numpy(dtype=float)
        ax.hist(vals, bins=bins, color=SOURCE_PALETTE[source], edgecolor=TEXT, linewidth=0.75, alpha=0.86)
        ax.set_title(source, fontweight="bold", pad=9)
        ax.set_xlabel("Metric value")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", color=GRID, linewidth=0.85)
        ax.grid(False, axis="x")
    fig.suptitle("Facetting histograms by pseudo-report source", fontweight="bold", y=0.99)
    sns.despine(fig=fig)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.18, wspace=0.18)
    return fig


def plot_ridge(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    y_positions = np.arange(len(SOURCE_ORDER))[::-1]
    x_grid = np.linspace(0, 1, 250)
    for ypos, source in zip(y_positions, SOURCE_ORDER):
        vals = long[long["Source"].eq(source)]["value"].to_numpy(dtype=float)
        hist, edges = np.histogram(vals, bins=np.linspace(0, 1, 13), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        density = np.interp(x_grid, centers, hist, left=0, right=0)
        if density.max() > 0:
            density = density / density.max() * 0.65
        ax.fill_between(x_grid, ypos, ypos + density, color=SOURCE_PALETTE[source], alpha=0.72, linewidth=0)
        ax.plot(x_grid, ypos + density, color=SOURCE_PALETTE[source], linewidth=2.0)
        ax.text(-0.04, ypos + 0.18, source, ha="right", va="center", fontweight="bold", fontfamily=FONT_ARIAL)
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.35, len(SOURCE_ORDER) - 0.15)
    ax.set_xlabel("Metric value")
    ax.set_title("Overlapping densities ridge-style metric distribution", fontweight="bold", pad=12)
    ax.grid(True, axis="x", color=GRID, linewidth=0.85)
    ax.grid(False, axis="y")
    sns.despine(fig=fig, ax=ax, left=True)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.15)
    return fig


def plot_line_facets(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), sharex=True, constrained_layout=True)
    for ax, group in zip(axes.flat, GROUP_ORDER):
        sub = long[long["group"].eq(group)].copy()
        for metric in sub["metric"].drop_duplicates():
            m = sub[sub["metric"].eq(metric)].sort_values("source_idx")
            ax.plot(m["source_idx"], m["value"], marker="o", linewidth=2.1, markersize=6.5, label=metric)
        ax.set_title(group, fontweight="bold", pad=8)
        ax.set_xticks(range(len(SOURCE_ORDER)))
        ax.set_xticklabels(SOURCE_ORDER, fontweight="bold")
        ax.set_ylabel("Metric value")
        ax.set_ylim(-0.05, max(1.05, float(sub["value"].max()) * 1.18))
        ax.grid(True, axis="y", color=GRID, linewidth=0.85)
        ax.grid(False, axis="x")
        ax.legend(frameon=False, fontsize=8.8, loc="best")
    fig.suptitle("Line plots on multiple facets: source-ordered metric profile", fontweight="bold")
    return fig


def plot_wide_lineplot(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    selected = long[long["metric"].isin(["Label consistency", "Mean support", "Unique text", "Duplicate fraction", "Alignment MRR"])].copy()
    pivot = selected.pivot_table(index="Source", columns="metric", values="value", aggfunc="mean").reindex(SOURCE_ORDER)
    fig, ax = plt.subplots(figsize=(9.8, 6.3))
    colors = [BLUE, RED, REF, SLATE, MID]
    for color, metric in zip(colors, pivot.columns):
        ax.plot(np.arange(len(SOURCE_ORDER)), pivot[metric], marker="o", linewidth=2.5, markersize=7.0, color=color, label=metric)
    ax.set_xticks(range(len(SOURCE_ORDER)))
    ax.set_xticklabels(SOURCE_ORDER, fontweight="bold")
    ax.set_ylabel("Metric value")
    ax.set_title("Wide-form lineplot across selected pseudo-report metrics", fontweight="bold", pad=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", color=GRID, linewidth=0.85)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.11, right=0.75, top=0.88, bottom=0.15)
    return fig


def plot_pair_matrix(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    data = key.rename(
        columns={
            "label_consistency_mean": "Label consistency",
            "mean_modality_support_rate": "Modality support",
            "unique_text_rate": "Unique text",
            "max_duplicate_fraction": "Duplicate fraction",
            "latent_alignment_mrr_full_model": "Alignment MRR",
        }
    )[["Source", "Label consistency", "Modality support", "Unique text", "Duplicate fraction", "Alignment MRR"]]
    grid = sns.pairplot(
        data,
        vars=["Label consistency", "Modality support", "Unique text", "Duplicate fraction", "Alignment MRR"],
        hue="Source",
        hue_order=SOURCE_ORDER,
        palette=SOURCE_PALETTE,
        corner=True,
        plot_kws={"s": 72, "edgecolor": TEXT, "linewidth": 0.6},
        diag_kind="hist",
        height=1.85,
    )
    grid.fig.suptitle("Scatterplot matrix: source-level pseudo-report profile", y=1.02, fontweight="bold")
    return grid.fig


def plot_corr_heatmap(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    selected = long.pivot_table(index="Source", columns="metric", values="value", aggfunc="mean").reindex(SOURCE_ORDER)
    metric_order = [metric for _, metric, _, _ in METRIC_SPECS if metric in selected.columns]
    selected = selected[metric_order]
    corr = selected.corr(numeric_only=True).reindex(index=metric_order, columns=metric_order).fillna(0)
    tick_labels = {
        "Section complete": "Section\ncomplete",
        "Label consistency": "Label\nconsistency",
        "OCT support": "OCT\nsupport",
        "Colposcopy support": "Colposcopy\nsupport",
        "Clinical support": "Clinical\nsupport",
        "Mean support": "Mean\nsupport",
        "Unique text": "Unique\ntext",
        "Duplicate fraction": "Duplicate\nfraction",
        "Alignment MRR": "Alignment\nMRR",
        "Alignment gap": "Alignment\ngap",
    }
    x_labels = {
        "Section complete": "Sec",
        "Label consistency": "Lab",
        "OCT support": "OCT",
        "Colposcopy support": "Col",
        "Clinical support": "Clin",
        "Mean support": "Mean",
        "Unique text": "Uniq",
        "Duplicate fraction": "Dup",
        "Alignment MRR": "MRR",
        "Alignment gap": "Gap",
    }
    y_labels = [tick_labels.get(metric, metric) for metric in metric_order]
    x_short_labels = [x_labels.get(metric, metric) for metric in metric_order]
    fig, ax = plt.subplots(figsize=(11.4, 9.4))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.75,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8.4, "fontfamily": FONT_TIMES},
        cbar_kws={"label": "Correlation", "shrink": 0.86, "pad": 0.035},
        ax=ax,
    )
    ax.set_title("Diagonal correlation matrix of pseudo-report metrics", fontweight="bold", pad=12)
    centers = np.arange(len(metric_order)) + 0.5
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(x_short_labels, rotation=0, ha="center", va="top")
    ax.set_yticklabels(y_labels, rotation=0, ha="right", va="center")
    ax.tick_params(axis="x", pad=9)
    ax.tick_params(axis="y", pad=6)
    ax.set_xlabel("Column abbreviations follow the same order as row metrics")
    ax.set_ylabel("Metric")
    fig.subplots_adjust(left=0.23, right=0.92, top=0.89, bottom=0.14)
    return fig


def plot_regression_strip(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.4, 6.1))
    sns.stripplot(data=long, x="source_idx", y="value", hue="group", palette=GROUP_PALETTE, size=7.0, jitter=0.18, edgecolor=TEXT, linewidth=0.45, ax=ax)
    sns.regplot(data=long, x="source_idx", y="value", scatter=False, color=TEXT, line_kws={"linewidth": 2.0, "alpha": 0.82}, ax=ax)
    ax.set_xticks(range(len(SOURCE_ORDER)))
    ax.set_xticklabels(SOURCE_ORDER, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Regression fit over a strip plot: source progression candidate", fontweight="bold", pad=12)
    ax.grid(True, axis="y", color=GRID, linewidth=0.85)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, title="Metric family", loc="lower right")
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.15)
    return fig


def plot_joint_marginals(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    grid = sns.JointGrid(data=key, x="mean_modality_support_rate", y="unique_text_rate", height=6.5, ratio=4, space=0.05)
    for source in SOURCE_ORDER:
        sub = key[key["Source"].eq(source)]
        grid.ax_joint.scatter(
            sub["mean_modality_support_rate"],
            sub["unique_text_rate"],
            s=230,
            color=SOURCE_PALETTE[source],
            edgecolor=TEXT,
            linewidth=0.9,
            label=source,
            zorder=4,
        )
    sns.histplot(data=key, x="mean_modality_support_rate", bins=np.linspace(0, 1, 7), color=BLUE, alpha=0.55, ax=grid.ax_marg_x)
    sns.histplot(data=key, y="unique_text_rate", bins=np.linspace(0, 0.24, 7), color=RED, alpha=0.55, ax=grid.ax_marg_y)
    annotate_sources(grid.ax_joint, key, "mean_modality_support_rate", "unique_text_rate", dx=0.018)
    grid.ax_joint.set_xlim(-0.06, 1.10)
    grid.ax_joint.set_ylim(-0.02, 0.235)
    grid.ax_joint.set_xlabel("Mean modality support")
    grid.ax_joint.set_ylabel("Unique text rate")
    grid.ax_joint.legend(frameon=False, loc="upper left")
    grid.fig.suptitle("Joint and marginal histograms: grounding versus diversity", y=1.02, fontweight="bold")
    return grid.fig


def plot_bubble_semantics(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    sc = ax.scatter(
        key["label_consistency_mean"],
        key["alignment_norm"],
        s=2600 * key["unique_text_rate"] + 70,
        c=key["risk_index"],
        cmap=sns.light_palette(RED, as_cmap=True),
        edgecolor=TEXT,
        linewidth=1.0,
        zorder=4,
    )
    annotate_sources(ax, key, "label_consistency_mean", "alignment_norm", dx=0.006)
    ax.set_xlabel("Label consistency")
    ax.set_ylabel("Normalized alignment MRR")
    ax.set_title("Scatterplot with continuous hue and size semantics", fontweight="bold", pad=12)
    ax.set_xlim(0.54, 0.65)
    ax.set_ylim(0.68, 1.06)
    ax.grid(True, color=GRID, linewidth=0.85)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.055, pad=0.035)
    cbar.set_label("Duplicate fraction", fontfamily=FONT_ARIAL, fontweight="bold")
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.12, right=0.91, top=0.88, bottom=0.15)
    return fig


def plot_polar_radar(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    labels = ["Label", "Modality", "Diversity", "Low duplicate", "Alignment"]
    cols = ["label_consistency_mean", "mean_modality_support_rate", "unique_text_rate", "inverse_duplicate", "alignment_norm"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7.6, 7.2), subplot_kw={"projection": "polar"})
    for _, row in key.iterrows():
        vals = [float(row[c]) for c in cols]
        vals += vals[:1]
        ax.plot(angles, vals, color=SOURCE_PALETTE[row["Source"]], linewidth=2.4, label=row["Source"])
        ax.fill(angles, vals, color=SOURCE_PALETTE[row["Source"]], alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight="bold", fontfamily=FONT_ARIAL)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontfamily=FONT_TIMES)
    ax.set_title("FacetGrid custom projection candidate: radar profile", fontweight="bold", pad=22)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=3)
    return fig


def plot_residuals(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    df = long.copy()
    x = df["source_idx"].to_numpy(dtype=float)
    y = df["value"].to_numpy(dtype=float)
    coef = np.polyfit(x, y, 1)
    df["fitted"] = coef[0] * x + coef[1]
    df["residual"] = y - df["fitted"]
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    sns.scatterplot(data=df, x="fitted", y="residual", hue="group", style="Source", palette=GROUP_PALETTE, s=96, edgecolor=TEXT, linewidth=0.6, ax=ax)
    ax.axhline(0, color=TEXT, linewidth=1.3, alpha=0.75)
    ax.set_xlabel("Fitted metric value from source order")
    ax.set_ylabel("Residual")
    ax.set_title("Plotting model residuals style candidate", fontweight="bold", pad=12)
    ax.grid(True, color=GRID, linewidth=0.85)
    ax.legend(frameon=False, loc="best", fontsize=8.8)
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    return fig


def plot_timeseries_errorbands(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    stats = long.groupby(["group", "Source", "source_idx"], as_index=False)["value"].agg(["mean", "std"]).reset_index()
    stats["std"] = stats["std"].fillna(0)
    fig, ax = plt.subplots(figsize=(9.7, 6.1))
    for group in GROUP_ORDER:
        sub = stats[stats["group"].eq(group)].sort_values("source_idx")
        ax.plot(sub["source_idx"], sub["mean"], marker="o", linewidth=2.4, markersize=7.0, color=GROUP_PALETTE[group], label=group)
        ax.fill_between(
            sub["source_idx"].to_numpy(dtype=float),
            (sub["mean"] - sub["std"]).clip(lower=0).to_numpy(dtype=float),
            (sub["mean"] + sub["std"]).clip(upper=1).to_numpy(dtype=float),
            color=GROUP_PALETTE[group],
            alpha=0.12,
        )
    ax.set_xticks(range(len(SOURCE_ORDER)))
    ax.set_xticklabels(SOURCE_ORDER, fontweight="bold")
    ax.set_ylabel("Mean metric value")
    ax.set_title("Timeseries plot with error bands style candidate", fontweight="bold", pad=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", color=GRID, linewidth=0.85)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.12, right=0.76, top=0.88, bottom=0.15)
    return fig


def plot_anova_style(wide: pd.DataFrame, long: pd.DataFrame, key: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    sns.pointplot(
        data=long,
        x="group",
        y="value",
        hue="Source",
        order=GROUP_ORDER,
        hue_order=SOURCE_ORDER,
        palette=SOURCE_PALETTE,
        dodge=0.38,
        errorbar="sd",
        markers="o",
        linestyles="",
        capsize=0.10,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.set_title("Plotting a three-way ANOVA style: group x source summary", fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=16)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", color=GRID, linewidth=0.85)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.20)
    return fig


STYLE_BUILDERS = [
    ("novel_semantic_risk_quadrant", "Scatterplot with continuous hues and sizes", plot_novel_quadrant, "Recommended: clear semantic claim and risk trade-off."),
    ("annotated_heatmap", "Annotated heatmaps", plot_annotated_heatmap, "Good for compact metric audit."),
    ("trivariate_heatmap_facets", "Trivariate histogram with two categorical variables", plot_grouped_heatmap_facets, "Good for family-level metric reading."),
    ("clustered_heatmap", "Discovering structure in heatmap data", plot_clustered_heatmap, "Exploratory; clustering is fragile with three sources."),
    ("dotplot_several_variables", "Dot plot with several variables", plot_dotplot, "Recommended: source-level metric comparison without implying continuity."),
    ("grouped_barplots", "Grouped barplots", plot_grouped_barplots, "Readable but less elegant for dense metric audit."),
    ("horizontal_bar_profiles", "Horizontal bar plots", plot_horizontal_bars, "Good source-by-source profile."),
    ("horizontal_boxplot_observations", "Horizontal boxplot with observations", plot_box_strip, "Candidate; box summarizes metrics, not biological samples."),
    ("grouped_violin_metric_distribution", "Grouped violinplots with split violins", plot_violin, "Exploratory; metric distribution is synthetic."),
    ("facetted_ecdf", "Facetted ECDF plots", plot_ecdf, "Exploratory distribution view of metric values."),
    ("facetted_histograms", "Facetting histograms by subsets of data", plot_hist_facets, "Exploratory distribution view."),
    ("ridge_density", "Overlapping densities ridge plot", plot_ridge, "Visual style candidate only."),
    ("line_facets", "Line plots on multiple facets", plot_line_facets, "Use cautiously; source order is conceptual, not temporal."),
    ("wide_lineplot", "Lineplot from a wide-form dataset", plot_wide_lineplot, "Use cautiously; connected source categories may imply trajectory."),
    ("scatter_matrix", "Scatterplot Matrix", plot_pair_matrix, "Exploratory; only three source-level points."),
    ("diagonal_correlation_matrix", "Plotting a diagonal correlation matrix", plot_corr_heatmap, "Exploratory; correlations from three sources are unstable."),
    ("regression_strip", "Regression fit over a strip plot", plot_regression_strip, "Style candidate; regression over source order is not primary evidence."),
    ("joint_marginal_histograms", "Joint and marginal histograms", plot_joint_marginals, "Good for grounding-diversity relation."),
    ("bubble_semantics", "Scatterplot with varying point sizes and hues", plot_bubble_semantics, "Good for compact source-level trade-off."),
    ("polar_radar", "FacetGrid with custom projection", plot_polar_radar, "Novel overview; avoid as sole statistical evidence."),
    ("residual_plot", "Plotting model residuals", plot_residuals, "Diagnostic-style candidate only."),
    ("timeseries_errorbands", "Timeseries plot with error bands", plot_timeseries_errorbands, "Candidate only; source order is not time."),
    ("anova_pointplot", "Plotting a three-way ANOVA", plot_anova_style, "Candidate summary of metric-family x source."),
]


def write_manifest(rows: list[dict[str, str]]) -> None:
    manifest = pd.DataFrame(rows)
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out_dir / "STYLE_GALLERY_MANIFEST.csv", index=False)


def main() -> None:
    setup_theme()
    wide, long, key = load_data()
    manifest_rows: list[dict[str, str]] = []
    for stem, reference, builder, note in STYLE_BUILDERS:
        fig = builder(wide, long, key)
        file_stem = f"Figure_theme1_pseudo_report_source_comparison_{stem}"
        written = save_fig(fig, file_stem)
        manifest_rows.append(
            {
                "stem": file_stem,
                "seaborn_reference": reference,
                "note": note,
                "final_fig_pdf": str((OUT_DIRS[-1] / file_stem).with_suffix(".pdf")),
                "final_fig_png": str((OUT_DIRS[-1] / file_stem).with_suffix(".png")),
            }
        )
        print(f"Wrote {written[-1]}")
    write_manifest(manifest_rows)
    print(f"Wrote manifest to {OUT_DIRS[-1] / 'STYLE_GALLERY_MANIFEST.csv'}")


if __name__ == "__main__":
    main()
