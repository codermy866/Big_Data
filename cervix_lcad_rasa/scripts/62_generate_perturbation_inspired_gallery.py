#!/usr/bin/env python3
"""Generate reference-inspired perturbation figures beyond the standard heatmap gallery."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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

THEME_TAB = ROOT / "outputs/publishable/theme1_alignment/tables"
MANUSCRIPT = ROOT / "outputs/publishable/tables/manuscript"
OUT_JBD = ROOT / "outputs/publishable/figures/jbd_final"
FINAL = PROJECT / "final_Fig"

TEXT = "#17212B"
GRID = "#E1E7EF"
BLUE = "#254B6D"
RUST = "#C65A46"
MID = "#557A95"
TEAL = "#436E6F"
REF = "#95A1B2"
LIGHT = "#D6DEE8"
GOLD = "#D2AE76"
PURPLE = "#6F5B85"
SEQ = LinearSegmentedColormap.from_list("mosaic_seq", ["#F7F9FC", "#D8E4EC", "#7F9AAC", BLUE], N=256)
RISK = LinearSegmentedColormap.from_list("mosaic_risk", ["#FBF8F4", "#E7CFC4", RUST, "#7E3026"], N=256)
DIV = LinearSegmentedColormap.from_list("mosaic_div", [RUST, "#F2D8BD", "#F7F9FC", "#A8BBC8", BLUE], N=256)

COND_ORDER = [
    "Normal",
    "Mask OCT",
    "Mask colposcopy",
    "Mask clinical",
    "Shuffle OCT",
    "Shuffle colposcopy",
    "Shuffle clinical",
    "Mask visual",
    "Label-only",
    "Randomize label",
]
FOCUS_ORDER = ["Normal", "Mask OCT", "Mask colposcopy", "Mask clinical", "Mask visual", "Label-only"]
SECTION_ORDER = ["OCT findings", "Colposcopy findings", "Clinical context", "Impression"]
DROP_ORDER = ["OCT findings", "Colposcopy findings", "Clinical context", "Overall report", "Risk shift"]
FAMILY_PALETTE = {
    "Reference": BLUE,
    "Mask": RUST,
    "Shuffle": GOLD,
    "Label-only": PURPLE,
    "Label control": REF,
}
FAMILY_MARKERS = {"Reference": "o", "Mask": "s", "Shuffle": "^", "Label-only": "D", "Label control": "X"}
FAMILY_LABELS = {
    "Reference": "Normal / reference",
    "Mask": "Masked evidence",
    "Shuffle": "Shuffled evidence",
    "Label-only": "Label-only",
    "Label control": "Label control",
}


def label_condition(v: str) -> str:
    return {
        "normal": "Normal",
        "mask_oct": "Mask OCT",
        "mask_colposcopy": "Mask colposcopy",
        "mask_instruction": "Mask clinical",
        "shuffle_oct": "Shuffle OCT",
        "shuffle_colposcopy": "Shuffle colposcopy",
        "shuffle_instruction": "Shuffle clinical",
        "mask_visual": "Mask visual",
        "label_only_inference": "Label-only",
        "randomize_label": "Randomize label",
    }.get(v, v.replace("_", " ").title())


def label_section(v: str) -> str:
    return {
        "oct_findings": "OCT findings",
        "colposcopy_findings": "Colposcopy findings",
        "clinical_context": "Clinical context",
        "impression": "Impression",
        "oct_findings_drop": "OCT findings",
        "colposcopy_findings_drop": "Colposcopy findings",
        "clinical_context_drop": "Clinical context",
        "impression_drop": "Impression",
        "report_drop": "Overall report",
        "risk_abs_delta": "Risk shift",
    }.get(str(v).replace("_similarity_to_normal", ""), str(v).replace("_", " ").title())


def condition_family(condition: str) -> str:
    if condition == "Normal":
        return "Reference"
    if condition.startswith("Mask"):
        return "Mask"
    if condition.startswith("Shuffle"):
        return "Shuffle"
    if condition == "Label-only":
        return "Label-only"
    return "Label control"


def setup_theme() -> None:
    setup_arial_rcparams(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_ARIAL, "Helvetica", "DejaVu Sans", "sans-serif"],
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.78,
            "axes.titlesize": 17.8,
            "axes.labelsize": 15.8,
            "xtick.labelsize": 14.3,
            "ytick.labelsize": 14.3,
            "legend.fontsize": 13.2,
            "legend.title_fontsize": 13.6,
            "font.size": 14.4,
            "mathtext.rm": FONT_TIMES,
            "mathtext.it": f"{FONT_TIMES}:italic",
            "mathtext.bf": f"{FONT_TIMES}:bold",
        }
    )
    sns.set_theme(style="whitegrid", context="paper", font=FONT_ARIAL)


def apply_style(fig: plt.Figure) -> None:
    fig._jbd_min_font_size_override = 13.0
    fig._jbd_max_font_size_override = 23.5
    apply_arial_to_figure(fig)
    apply_mixed_en_typography(fig)


def save_fig(fig: plt.Figure, stem: str) -> None:
    apply_style(fig)
    for out_dir in [OUT_JBD / "Figure3_modality_perturbation_heatmap_inspired_gallery", FINAL / "Figure3_modality_perturbation_heatmap_inspired_gallery"]:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / stem
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def polish(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.set_facecolor("white")
    if grid_axis in {"x", "both"}:
        ax.grid(True, axis="x", color=GRID, linewidth=0.85, alpha=0.82)
    else:
        ax.grid(False, axis="x")
    if grid_axis in {"y", "both"}:
        ax.grid(True, axis="y", color=GRID, linewidth=0.85, alpha=0.82)
    else:
        ax.grid(False, axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C8D2DD")
        ax.spines[side].set_linewidth(1.05)
    ax.tick_params(axis="both", labelsize=14.3, colors=TEXT)


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.105, y: float = 1.095) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=19.0,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.17,rounding_size=0.035", "facecolor": BLUE, "edgecolor": "none"},
    )


def family_legend_handles() -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            color="none",
            markerfacecolor=FAMILY_PALETTE[family],
            markeredgecolor=TEXT,
            markeredgewidth=0.78,
            markersize=8.6,
            label=FAMILY_LABELS[family],
        )
        for family in ["Reference", "Mask", "Shuffle", "Label-only", "Label control"]
    ]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    text = pd.read_csv(MANUSCRIPT / "S6_modality_perturbation_text_decoding.csv")
    matrix = pd.read_csv(THEME_TAB / "T_theme1_upgraded_perturbation_sensitivity_matrix.csv")
    text = text.copy()
    matrix = matrix.copy()
    text["Condition"] = text["condition"].map(label_condition)
    matrix["Condition"] = matrix["condition"].map(label_condition)
    text["Family"] = text["Condition"].map(condition_family)
    sim_cols = [
        "oct_findings_similarity_to_normal",
        "colposcopy_findings_similarity_to_normal",
        "clinical_context_similarity_to_normal",
        "impression_similarity_to_normal",
    ]
    sim = text.melt(id_vars=["condition", "Condition", "Family"], value_vars=sim_cols, var_name="Section", value_name="Similarity")
    sim["Section"] = sim["Section"].map(label_section)
    drop_cols = ["oct_findings_drop", "colposcopy_findings_drop", "clinical_context_drop", "impression_drop", "report_drop", "risk_abs_delta"]
    drop = matrix.melt(id_vars=["condition", "Condition"], value_vars=drop_cols, var_name="Measure", value_name="Drop")
    drop["Measure"] = drop["Measure"].map(label_section)
    return text, matrix, sim, drop


def embedding_frame(text: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "oct_findings_similarity_to_normal",
        "colposcopy_findings_similarity_to_normal",
        "clinical_context_similarity_to_normal",
        "impression_similarity_to_normal",
        "report_similarity_to_normal",
        "risk_score_absolute_delta_vs_normal",
    ]
    x = text[cols].to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=np.nanmean(x))
    x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) == 0, 1, x.std(axis=0))
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    out = text.copy()
    out["PC1"] = u[:, 0] * s[0]
    out["PC2"] = u[:, 1] * s[1]
    return out


def draw_scatter(ax: plt.Axes, text: pd.DataFrame, *, legend: bool = True) -> None:
    emb = embedding_frame(text)
    for family, sub in emb.groupby("Family", sort=False):
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=120 + 560 * sub["risk_score_absolute_delta_vs_normal"].to_numpy(dtype=float),
            color=FAMILY_PALETTE[family],
            marker=FAMILY_MARKERS[family],
            edgecolor=TEXT,
            linewidth=0.82,
            alpha=0.94,
            label=FAMILY_LABELS[family],
        )
    label_offsets = {
        "Normal": (0.075, 0.045),
        "Mask OCT": (0.075, 0.065),
        "Mask colposcopy": (0.075, -0.015),
        "Mask clinical": (0.075, 0.070),
        "Mask visual": (0.075, 0.020),
        "Label-only": (-0.105, 0.055),
    }
    for _, row in emb.iterrows():
        if row["Condition"] in {"Normal", "Mask visual", "Label-only", "Mask OCT", "Mask colposcopy", "Mask clinical"}:
            dx, dy = label_offsets.get(row["Condition"], (0.065, 0.045))
            ha = "right" if row["Condition"] == "Label-only" else "left"
            ax.text(
                row["PC1"] + dx,
                row["PC2"] + dy,
                row["Condition"].replace("Mask ", ""),
                fontsize=13.0,
                color=TEXT,
                ha=ha,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.60, "pad": 1.3},
                zorder=6,
            )
    ax.axhline(0, color=GRID, linewidth=1.0)
    ax.axvline(0, color=GRID, linewidth=1.0)
    ax.set_title("Perturbation response embedding (SVD)", fontweight="bold", pad=10)
    ax.set_xlabel("Response component 1")
    ax.set_ylabel("Response component 2")
    polish(ax)
    if legend:
        ax.legend(handles=family_legend_handles(), frameon=False, title="Marker shape", loc="best")


def draw_confusion(ax: plt.Axes, matrix: pd.DataFrame) -> None:
    specific = matrix.dropna(subset=["expected_primary_drop"]).copy()
    specific["Expected"] = specific["expected_primary_drop"].map(label_section)
    specific["Observed"] = specific["max_drop_section"].map(label_section)
    labels = ["OCT findings", "Colposcopy findings", "Clinical context"]
    cm = pd.crosstab(specific["Expected"], specific["Observed"]).reindex(index=labels, columns=labels, fill_value=0)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cmap=SEQ,
        vmin=0,
        vmax=max(1, int(cm.to_numpy().max())),
        linewidths=1.05,
        linecolor="white",
        annot_kws={"fontsize": 15.6, "fontweight": "bold"},
        cbar_kws={"label": "Count", "shrink": 0.88},
        ax=ax,
    )
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=14.0, colors=TEXT)
        cbar.ax.yaxis.label.set_size(15.4)
    ax.set_title("Expected versus observed degraded section", fontweight="bold", pad=9)
    ax.set_xlabel("Observed largest drop")
    ax.set_ylabel("Expected primary drop")
    ax.tick_params(axis="x", rotation=30, labelsize=14.0)
    ax.tick_params(axis="y", rotation=0, labelsize=14.0)


def draw_ridgeline(ax: plt.Axes, sim: pd.DataFrame) -> None:
    colors = [BLUE, RUST, MID, TEAL, GOLD, PURPLE]
    xs = np.linspace(0, 1.02, 240)
    for ypos, condition, color in zip(np.arange(len(FOCUS_ORDER))[::-1], FOCUS_ORDER, colors):
        vals = sim.loc[sim["Condition"].eq(condition), "Similarity"].to_numpy(dtype=float)
        density = np.zeros_like(xs)
        for val in vals:
            density += np.exp(-0.5 * ((xs - val) / 0.045) ** 2)
        density = density / max(density.max(), 1e-9) * 0.62
        ax.fill_between(xs, ypos, ypos + density, color=color, alpha=0.66, linewidth=0)
        ax.plot(xs, ypos + density, color=color, linewidth=2.35)
        ax.scatter(vals, np.full_like(vals, ypos), s=32, color=color, edgecolor=TEXT, linewidth=0.42, zorder=3)
    ax.axvline(1.0, color=TEXT, linestyle=(0, (2, 2)), linewidth=1.0, alpha=0.72)
    ax.set_yticks(np.arange(len(FOCUS_ORDER))[::-1])
    ax.set_yticklabels(FOCUS_ORDER)
    ax.set_xlim(0, 1.03)
    ax.set_title("Section similarity distributions", fontweight="bold", pad=9)
    ax.set_xlabel("Similarity to normal")
    ax.set_ylabel("")
    polish(ax, "x")


def draw_dual_track(fig: plt.Figure, spec, text: pd.DataFrame, *, title: str | None = "Report similarity and risk displacement") -> tuple[plt.Axes, plt.Axes]:
    sub = text[text["Condition"].isin(FOCUS_ORDER)].set_index("Condition").reindex(FOCUS_ORDER).reset_index()
    inner = spec.subgridspec(2, 1, height_ratios=[1, 1], hspace=0.18)
    ax_top = fig.add_subplot(inner[0, 0])
    ax_bot = fig.add_subplot(inner[1, 0], sharex=ax_top)
    xpos = np.arange(len(sub))
    ax_top.plot(xpos, sub["report_similarity_to_normal"], color=BLUE, marker="o", linewidth=2.75, markersize=7.8)
    ax_top.fill_between(xpos, sub["report_similarity_to_normal"], 1, color=BLUE, alpha=0.14)
    ax_top.set_ylim(0.45, 1.04)
    ax_top.set_ylabel("Report sim.")
    if title:
        ax_top.set_title(title, fontweight="bold", pad=8)
    ax_bot.bar(xpos, sub["risk_score_absolute_delta_vs_normal"], color=RUST, edgecolor=TEXT, linewidth=0.68, alpha=0.91)
    ax_bot.plot(xpos, sub["risk_score_absolute_delta_vs_normal"], color=TEXT, marker="D", linewidth=1.55, markersize=5.4)
    ax_bot.set_ylim(0, max(0.42, float(sub["risk_score_absolute_delta_vs_normal"].max()) * 1.20))
    ax_bot.set_ylabel("Risk shift")
    ax_bot.set_xticks(xpos)
    ax_bot.set_xticklabels([c.replace("Mask ", "M. ").replace("Label-only", "Label") for c in sub["Condition"]], rotation=27, ha="right")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    for ax in (ax_top, ax_bot):
        polish(ax)
    return ax_top, ax_bot


def plot_composite(text: pd.DataFrame, matrix: pd.DataFrame, sim: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16.8, 12.8))
    gs = fig.add_gridspec(2, 2, hspace=0.95, wspace=0.46)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    draw_scatter(ax_a, text, legend=False)
    leg = ax_a.legend(
        handles=family_legend_handles(),
        title="Marker key: shape/color = perturbation family; size = risk shift",
        loc="upper center",
        bbox_to_anchor=(0.50, -0.165),
        ncol=3,
        frameon=True,
        borderpad=0.52,
        labelspacing=0.34,
        handletextpad=0.56,
        columnspacing=1.18,
        fontsize=12.8,
        title_fontsize=13.2,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_linewidth(0.75)
    leg.get_frame().set_alpha(0.88)
    draw_confusion(ax_b, matrix)
    draw_ridgeline(ax_c, sim)
    ax_d_top, _ = draw_dual_track(fig, gs[1, 1], text, title="Report-risk profile")
    panel_label(ax_a, "A")
    panel_label(ax_b, "B")
    panel_label(ax_c, "C")
    panel_label(ax_d_top, "D", x=-0.125, y=1.19)
    fig.suptitle("Modality perturbation response signatures", fontsize=22.4, fontweight="bold", y=0.986)
    save_fig(fig, "Figure3_modality_perturbation_inspired_composite_grid")


def plot_embedding(text: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    draw_scatter(ax, text, legend=True)
    save_fig(fig, "Figure3_modality_perturbation_inspired_embedding_scatter")


def plot_confusion(matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    draw_confusion(ax, matrix)
    save_fig(fig, "Figure3_modality_perturbation_inspired_confusion_matrix")


def plot_ridgeline(sim: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    draw_ridgeline(ax, sim)
    save_fig(fig, "Figure3_modality_perturbation_inspired_ridgeline")


def plot_dual_track(text: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8.4, 5.8))
    draw_dual_track(fig, fig.add_gridspec(1, 1)[0, 0], text)
    fig.suptitle("Dual-track perturbation profile", fontsize=14.2, fontweight="bold", y=0.99)
    save_fig(fig, "Figure3_modality_perturbation_inspired_dual_track_profile")


def plot_violin(sim: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    section_palette = dict(zip(SECTION_ORDER, [BLUE, RUST, MID, REF]))
    sns.violinplot(
        data=sim,
        x="Section",
        y="Similarity",
        hue="Section",
        order=SECTION_ORDER,
        hue_order=SECTION_ORDER,
        palette=section_palette,
        legend=False,
        inner=None,
        cut=0,
        linewidth=1.0,
        ax=ax,
    )
    sns.stripplot(data=sim, x="Section", y="Similarity", order=SECTION_ORDER, hue="Family", palette={"Reference": BLUE, "Mask": RUST, "Shuffle": GOLD, "Label-only": PURPLE, "Label control": REF}, jitter=0.18, size=5.0, edgecolor=TEXT, linewidth=0.45, ax=ax)
    ax.axhline(1.0, color=TEXT, linestyle=(0, (2, 2)), linewidth=1.0, alpha=0.72)
    ax.set_title("Section-level response distribution across perturbations", fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Similarity to normal")
    ax.set_ylim(-0.03, 1.08)
    ax.legend(frameon=False, title="Perturbation family", bbox_to_anchor=(1.02, 1), loc="upper left")
    polish(ax)
    save_fig(fig, "Figure3_modality_perturbation_inspired_violin_response")


def plot_stacked(matrix: pd.DataFrame) -> None:
    rows = matrix[matrix["Condition"].isin(["Mask OCT", "Mask colposcopy", "Mask clinical", "Mask visual", "Label-only"])].copy()
    cols = ["oct_findings_drop", "colposcopy_findings_drop", "clinical_context_drop", "report_drop", "risk_abs_delta"]
    labels = ["OCT", "Colposcopy", "Clinical", "Report", "Risk"]
    values = rows[cols].to_numpy(dtype=float)
    totals = values.sum(axis=1)
    proportions = values / np.where(totals[:, None] == 0, 1, totals[:, None]) * 100.0
    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    bottom = np.zeros(len(rows))
    colors = [BLUE, RUST, MID, REF, GOLD]
    xpos = np.arange(len(rows))
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(xpos, proportions[:, i], bottom=bottom, color=color, edgecolor="white", linewidth=0.8, label=label, alpha=0.92)
        bottom += proportions[:, i]
    ax.set_xticks(xpos)
    ax.set_xticklabels(rows["Condition"], rotation=25, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Relative contribution (%)")
    ax.set_title("Composition of perturbation-induced degradation", fontweight="bold", pad=10)
    ax.legend(frameon=False, title="Component", bbox_to_anchor=(1.02, 1), loc="upper left")
    polish(ax, "y")
    save_fig(fig, "Figure3_modality_perturbation_inspired_stacked_composition")


def write_manifest() -> None:
    rows = [
        ("Figure3_modality_perturbation_inspired_composite_grid", "Multi-panel reference layout", "Composite perturbation-response view. Panel A projects section-similarity and risk-shift signatures into a two-dimensional response space; marker shape and color denote the perturbation family, and marker size indicates absolute risk-score displacement. Panel B compares the expected evidence stream with the report section showing the largest degradation. Panel C displays section-similarity distributions relative to the normal-input report, and Panel D pairs report-level similarity with downstream risk-score displacement."),
        ("Figure3_modality_perturbation_inspired_embedding_scatter", "Multi-series scatter plot", "Low-dimensional embedding of perturbation response signatures. Point size reflects absolute risk displacement."),
        ("Figure3_modality_perturbation_inspired_confusion_matrix", "Confusion matrix heatmap", "Expected primary evidence stream versus the report section showing the largest degradation."),
        ("Figure3_modality_perturbation_inspired_ridgeline", "Ridgeline distribution plot", "Distribution-style view of section similarity values under representative perturbations."),
        ("Figure3_modality_perturbation_inspired_dual_track_profile", "Dual-track performance profile", "Shared-x profile of report similarity and absolute risk-score displacement. This avoids a true dual-y axis while preserving the reference visual logic."),
        ("Figure3_modality_perturbation_inspired_violin_response", "Violin plot with observations", "Section-level similarity distributions with individual perturbation observations overlaid."),
        ("Figure3_modality_perturbation_inspired_stacked_composition", "Stacked composition chart", "Relative contribution of section degradation, report degradation, and risk displacement within each major perturbation."),
    ]
    for out_dir in [OUT_JBD / "Figure3_modality_perturbation_heatmap_inspired_gallery", FINAL / "Figure3_modality_perturbation_heatmap_inspired_gallery"]:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["stem", "reference_style", "caption"]).to_csv(out_dir / "INSPIRED_GALLERY_MANIFEST.csv", index=False)
        lines = ["# Figure3 Modality Perturbation Inspired Gallery", ""]
        for stem, ref, caption in rows:
            lines.extend([f"## {stem}", "", f"**Reference style.** {ref}.", "", f"**Caption.** {caption}", ""])
        (out_dir / "SCI_CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_theme()
    text, matrix, sim, _ = load_data()
    plot_composite(text, matrix, sim)
    plot_embedding(text)
    plot_confusion(matrix)
    plot_ridgeline(sim)
    plot_dual_track(text)
    plot_violin(sim)
    plot_stacked(matrix)
    write_manifest()
    print("Generated perturbation inspired gallery")


if __name__ == "__main__":
    main()
