#!/usr/bin/env python3
"""Generate Seaborn-style galleries for key Theme-1 and perturbation figures."""

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
OUT_THEME = ROOT / "outputs/publishable/theme1_alignment/figures"
OUT_JBD = ROOT / "outputs/publishable/figures/jbd_final"
FINAL = PROJECT / "final_Fig"

TEXT = "#17212B"
GRID = "#E1E7EF"
BLUE = "#254B6D"
TEAL = "#436E6F"
SLATE = "#8796A5"
LIGHT = "#D8E0E8"
GOLD = "#D2AE76"
RUST = "#C65A46"
PALETTE = [BLUE, RUST, TEAL, GOLD, SLATE, "#6F5B85", "#9C7A52", "#5A748A", "#B88D72", "#6D8F84"]
SEQ = LinearSegmentedColormap.from_list("mosaic_seq", ["#F7F9FC", "#D8E4EC", "#7F9AAC", BLUE], N=256)
SEQ_RUST = LinearSegmentedColormap.from_list("mosaic_rust", ["#FBF8F4", "#E7CFC4", "#C65A46", "#7E3026"], N=256)
DIV = LinearSegmentedColormap.from_list("mosaic_div", ["#C65A46", "#F2D8BD", "#F7F9FC", "#A8BBC8", "#254B6D"], N=256)


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
            "axes.titlesize": 13.0,
            "axes.labelsize": 11.2,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.4,
            "legend.title_fontsize": 9.6,
            "font.size": 10.0,
            "mathtext.rm": FONT_TIMES,
            "mathtext.it": f"{FONT_TIMES}:italic",
            "mathtext.bf": f"{FONT_TIMES}:bold",
        }
    )
    sns.set_theme(style="whitegrid", context="paper", font=FONT_ARIAL, palette=PALETTE)


def apply_style(fig: plt.Figure, *, min_size: float = 8.8, max_size: float = 15.0) -> None:
    fig._jbd_min_font_size_override = min_size
    fig._jbd_max_font_size_override = max_size
    apply_arial_to_figure(fig)
    apply_mixed_en_typography(fig)


def save_fig(fig: plt.Figure, out_dirs: list[Path], stem: str) -> list[Path]:
    apply_style(fig)
    written = []
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / stem
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.08)
        written.append(base.with_suffix(".pdf"))
    plt.close(fig)
    return written


def write_manifest(out_dir: Path, rows: list[dict[str, str]], title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "STYLE_GALLERY_MANIFEST.csv", index=False)
    lines = [f"# {title}", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['stem']}",
                "",
                f"**Seaborn reference.** {row['seaborn_reference']}.",
                "",
                f"**Caption.** {row['caption']}",
                "",
            ]
        )
    (out_dir / "SCI_CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")


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
        ax.spines[side].set_linewidth(0.9)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.06, 1.05, label, transform=ax.transAxes, ha="left", va="top", fontsize=13, fontweight="bold")


def label_setup(v: str) -> str:
    return {
        "lcad_augmented_surrogate": "LCAD-augmented",
        "real_report_only_surrogate": "Real-report only",
    }.get(v, v.replace("_", " ").title())


def label_model(v: str) -> str:
    return {
        "full_lcad_rasa": "Full LCAD-RASA",
        "pseudo_augmented_lcad": "Pseudo-augmented LCAD",
        "real_report_only": "Real-report only",
        "no_section_alignment": "No section alignment",
        "no_label_consistency_loss": "No label consistency",
        "risk_head_only_auxiliary": "Risk-head auxiliary",
        "no_risk_head": "No risk head",
        "report_loss_only": "Report loss only",
        "simple_concat_fusion": "Simple concat fusion",
    }.get(v, v.replace("_", " ").title())


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
    }.get(v, v.replace("_", " ").title())


def load_scarcity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(THEME_TAB / "T_theme1_report_supervision_scarcity_curve.csv")
    raw = pd.read_csv(THEME_TAB / "T_theme1_report_supervision_scarcity_curve_raw.csv")
    for df in (summary, raw):
        df["Setup"] = df["setup"].map(label_setup)
        df["Real-report fraction"] = df["real_report_fraction"].astype(float)
        df["Real-report fraction label"] = (df["Real-report fraction"] * 100).round().astype(int).astype(str) + "%"
    wide = summary.pivot(index="Setup", columns="Real-report fraction label", values="auc_mean")
    order = ["10%", "25%", "50%", "100%"]
    wide = wide[[c for c in order if c in wide.columns]]
    return summary, raw, wide


def load_alignment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail = pd.read_csv(THEME_TAB / "T_theme1_modality_section_retrieval_alignment.csv")
    macro = pd.read_csv(THEME_TAB / "T_theme1_rasa_direct_alignment_ablation.csv")
    detail["Model"] = detail["model"].map(label_model)
    detail["Section"] = detail["section"].map(label_section)
    macro["Model"] = macro["model"].map(label_model)
    metric_long = macro.melt(
        id_vars=["model", "Model"],
        value_vars=["macro_recall_at_1", "macro_recall_at_5", "macro_mrr", "macro_positive_minus_cross_case", "macro_positive_minus_wrong_section"],
        var_name="Metric",
        value_name="Value",
    )
    metric_long["Metric"] = metric_long["Metric"].map(
        {
            "macro_recall_at_1": "Recall@1",
            "macro_recall_at_5": "Recall@5",
            "macro_mrr": "MRR",
            "macro_positive_minus_cross_case": "Cross-case gap",
            "macro_positive_minus_wrong_section": "Wrong-section gap",
        }
    )
    return detail, macro, metric_long


def load_perturbation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    text = pd.read_csv(MANUSCRIPT / "S6_modality_perturbation_text_decoding.csv")
    matrix = pd.read_csv(THEME_TAB / "T_theme1_upgraded_perturbation_sensitivity_matrix.csv")
    sec_cols = [
        "oct_findings_similarity_to_normal",
        "colposcopy_findings_similarity_to_normal",
        "clinical_context_similarity_to_normal",
        "impression_similarity_to_normal",
    ]
    sim = text.melt(id_vars=["condition"], value_vars=sec_cols, var_name="Section", value_name="Similarity")
    sim["Section"] = sim["Section"].str.replace("_similarity_to_normal", "", regex=False).map(label_section)
    sim["Condition"] = sim["condition"].map(label_condition)
    drop_cols = ["oct_findings_drop", "colposcopy_findings_drop", "clinical_context_drop", "impression_drop", "report_drop", "risk_abs_delta"]
    drop = matrix.melt(id_vars=["condition"], value_vars=drop_cols, var_name="Measure", value_name="Drop")
    drop["Measure"] = drop["Measure"].map(label_section)
    drop["Condition"] = drop["condition"].map(label_condition)
    text["Condition"] = text["condition"].map(label_condition)
    matrix["Condition"] = matrix["condition"].map(label_condition)
    return text, matrix, sim, drop


SCARCITY_CAPTION = (
    "Report-supervision scarcity analysis comparing a real-report-only surrogate with an LCAD-augmented surrogate across decreasing fractions of available physician-authored reports. "
    "Points or tiles show held-out AUROC on the locked test set; where shown, error bars or distributions summarise five seeded resampling runs. "
    "The LCAD-augmented setting retains pseudo-report supervision while real-report availability is reduced, testing whether quality-controlled weak-oracle priors buffer representation learning under report scarcity."
)
ALIGNMENT_CAPTION = (
    "Modality-section retrieval alignment audit for RASA and its ablations. The plots summarise section-level and macro-level mean reciprocal rank (MRR), recall, and positive-minus-negative cosine separation, thereby testing whether multimodal representations are organised around clinically meaningful report sections rather than only around diagnostic labels."
)
PERTURB_CAPTION = (
    "Modality perturbation audit over 128 held-out cases. Similarity values compare each perturbed report section with the normal-input report, whereas drop and risk-shift metrics quantify section-specific semantic degradation and downstream score displacement. A modality-grounded model should show the largest degradation in the report section linked to the perturbed evidence stream."
)


def scarcity_gallery() -> None:
    setup_theme()
    summary, raw, wide = load_scarcity()
    out_final = FINAL / "Figure_theme1_report_supervision_scarcity_curve_style_gallery"
    out_pub = OUT_THEME / "Figure_theme1_report_supervision_scarcity_curve_style_gallery"
    out_dirs = [out_pub, out_final]
    rows: list[dict[str, str]] = []

    def record(stem: str, ref: str, caption: str = SCARCITY_CAPTION) -> None:
        rows.append({"stem": stem, "seaborn_reference": ref, "caption": caption})

    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    sns.heatmap(wide, annot=True, fmt=".3f", cmap=SEQ, linewidths=0.8, linecolor="white", cbar_kws={"label": "AUROC"}, ax=ax)
    ax.set_title("Annotated heatmap: AUROC under report scarcity", fontweight="bold", pad=10)
    ax.set_xlabel("Available real-report supervision")
    ax.set_ylabel("")
    save_fig(fig, out_dirs, "Figure_theme1_report_supervision_scarcity_curve_annotated_heatmap")
    record("Figure_theme1_report_supervision_scarcity_curve_annotated_heatmap", "Annotated heatmaps")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for setup, color in zip(["LCAD-augmented", "Real-report only"], [BLUE, RUST]):
        d = summary[summary["Setup"].eq(setup)].sort_values("Real-report fraction")
        ax.errorbar(d["Real-report fraction"], d["auc_mean"], yerr=d["auc_std"], color=color, marker="o", linewidth=2.4, capsize=4, label=setup)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.set_ylabel("Held-out AUROC")
    ax.set_xlabel("Available real-report supervision")
    ax.set_title("Lineplot with error bands: scarcity response", fontweight="bold")
    ax.legend(frameon=False)
    polish(ax)
    save_fig(fig, out_dirs, "Figure_theme1_report_supervision_scarcity_curve_line_errorbands")
    record("Figure_theme1_report_supervision_scarcity_curve_line_errorbands", "Timeseries plot with error bands")

    fig, ax = plt.subplots(figsize=(8.7, 4.9))
    sns.pointplot(data=raw, x="Real-report fraction label", y="auc", hue="Setup", errorbar="sd", dodge=0.35, palette=[BLUE, RUST], markers=["o", "D"], linestyles=["-", "--"], ax=ax)
    sns.stripplot(data=raw, x="Real-report fraction label", y="auc", hue="Setup", dodge=True, palette=[BLUE, RUST], alpha=0.42, size=4.5, ax=ax, legend=False)
    ax.set_title("Conditional means with seeded observations", fontweight="bold")
    ax.set_xlabel("Available real-report supervision")
    ax.set_ylabel("AUROC")
    polish(ax, "y")
    ax.legend(frameon=False, title="")
    save_fig(fig, out_dirs, "Figure_theme1_report_supervision_scarcity_curve_point_observations")
    record("Figure_theme1_report_supervision_scarcity_curve_point_observations", "Conditional means with observations")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.boxplot(data=raw, y="Real-report fraction label", x="auc", hue="Setup", palette=[BLUE, RUST], fliersize=0, linewidth=1.0, ax=ax)
    sns.stripplot(data=raw, y="Real-report fraction label", x="auc", hue="Setup", dodge=True, palette=[BLUE, RUST], size=4.2, alpha=0.50, ax=ax, legend=False)
    ax.set_title("Horizontal boxplot with seeded AUROC observations", fontweight="bold")
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Available real-report supervision")
    polish(ax, "x")
    ax.legend(frameon=False, title="")
    save_fig(fig, out_dirs, "Figure_theme1_report_supervision_scarcity_curve_box_observations")
    record("Figure_theme1_report_supervision_scarcity_curve_box_observations", "Horizontal boxplot with observations")

    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    for setup, color in zip(["LCAD-augmented", "Real-report only"], [BLUE, RUST]):
        vals = np.sort(raw.loc[raw["Setup"].eq(setup), "auc"].to_numpy(dtype=float))
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=color, linewidth=2.2, label=setup)
        ax.scatter(vals, y, s=34, color=color, edgecolor=TEXT, linewidth=0.4)
    ax.set_title("Facetted ECDF style: seeded AUROC distribution", fontweight="bold")
    ax.set_xlabel("AUROC")
    ax.set_ylabel("Cumulative proportion")
    ax.legend(frameon=False)
    polish(ax)
    save_fig(fig, out_dirs, "Figure_theme1_report_supervision_scarcity_curve_ecdf")
    record("Figure_theme1_report_supervision_scarcity_curve_ecdf", "Facetted ECDF plots")

    delta = summary.pivot(index="Real-report fraction", columns="Setup", values="auc_mean")
    delta["Delta AUROC"] = delta["LCAD-augmented"] - delta["Real-report only"]
    d = delta.reset_index()
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.bar(d["Real-report fraction label"] if "Real-report fraction label" in d.columns else [f"{int(x*100)}%" for x in d["Real-report fraction"]], d["Delta AUROC"], color=GOLD, edgecolor=TEXT, linewidth=0.8)
    ax.axhline(0, color=TEXT, linewidth=0.9)
    ax.set_title("Paired delta view: benefit of LCAD augmentation", fontweight="bold")
    ax.set_xlabel("Available real-report supervision")
    ax.set_ylabel("Delta AUROC")
    polish(ax, "y")
    save_fig(fig, out_dirs, "Figure_theme1_report_supervision_scarcity_curve_delta_bar")
    record("Figure_theme1_report_supervision_scarcity_curve_delta_bar", "Horizontal bar plots / paired categorical plots")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharey=True)
    for ax, metric, label in zip(axes, ["auc", "f1"], ["AUROC", "F1"]):
        sns.lineplot(data=raw, x="Real-report fraction", y=metric, hue="Setup", style="Setup", markers=True, dashes=False, palette=[BLUE, RUST], ax=ax)
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.25, 0.5, 1.0])
        ax.set_xticklabels(["10%", "25%", "50%", "100%"])
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Real-report fraction")
        ax.set_ylabel(label)
        polish(ax)
        ax.legend_.remove()
    axes[1].legend(*axes[1].get_legend_handles_labels(), frameon=False, loc="lower right")
    fig.suptitle("Line plots on multiple facets: discrimination and threshold profile", fontweight="bold", y=1.02)
    save_fig(fig, out_dirs, "Figure_theme1_report_supervision_scarcity_curve_metric_facets")
    record("Figure_theme1_report_supervision_scarcity_curve_metric_facets", "Line plots on multiple facets")

    fig, ax = plt.subplots(figsize=(8.1, 4.8))
    sns.scatterplot(data=summary, x="mean_n_real_train", y="auc_mean", hue="Setup", size="mean_n_pseudo_train", sizes=(80, 520), palette=[BLUE, RUST], edgecolor=TEXT, linewidth=0.75, ax=ax)
    ax.set_title("Bubble scatter: training evidence versus AUROC", fontweight="bold")
    ax.set_xlabel("Mean real-report training cases")
    ax.set_ylabel("Mean AUROC")
    polish(ax)
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, out_dirs, "Figure_theme1_report_supervision_scarcity_curve_bubble_evidence")
    record("Figure_theme1_report_supervision_scarcity_curve_bubble_evidence", "Scatterplot with varying point sizes and hues")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    wide_line = summary.pivot(index="Real-report fraction label", columns="Setup", values="auc_mean").reindex(["10%", "25%", "50%", "100%"])
    for setup, color in zip(wide_line.columns, [BLUE, RUST]):
        ax.plot(wide_line.index, wide_line[setup], marker="o", linewidth=2.4, color=color, label=setup)
    ax.set_title("Wide-form lineplot: AUROC trajectory", fontweight="bold")
    ax.set_xlabel("Available real-report supervision")
    ax.set_ylabel("Mean AUROC")
    ax.legend(frameon=False)
    polish(ax, "y")
    save_fig(fig, out_dirs, "Figure_theme1_report_supervision_scarcity_curve_wide_lineplot")
    record("Figure_theme1_report_supervision_scarcity_curve_wide_lineplot", "Lineplot from a wide-form dataset")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for ypos, (setup, color) in enumerate(zip(["LCAD-augmented", "Real-report only"], [BLUE, RUST])):
        vals = raw.loc[raw["Setup"].eq(setup), "auc"].to_numpy(dtype=float)
        hist, edges = np.histogram(vals, bins=np.linspace(0.55, 0.86, 13), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        dens = hist / max(hist.max(), 1e-9) * 0.62
        ax.fill_between(centers, ypos, ypos + dens, color=color, alpha=0.70, linewidth=0)
        ax.plot(centers, ypos + dens, color=color, linewidth=2.0)
        ax.text(0.548, ypos + 0.22, setup, ha="right", va="center", fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0.55, 0.87)
    ax.set_ylim(-0.15, 1.85)
    ax.set_xlabel("AUROC")
    ax.set_title("Overlapping densities: seeded AUROC stability", fontweight="bold")
    polish(ax, "x")
    save_fig(fig, out_dirs, "Figure_theme1_report_supervision_scarcity_curve_ridge_density")
    record("Figure_theme1_report_supervision_scarcity_curve_ridge_density", "Overlapping densities (ridge plot)")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for setup, color, marker in zip(["LCAD-augmented", "Real-report only"], [BLUE, RUST], ["o", "D"]):
        d = raw[raw["Setup"].eq(setup)]
        sns.regplot(data=d, x="Real-report fraction", y="auc", scatter=True, marker=marker, color=color, ci=None, ax=ax, label=setup, scatter_kws={"s": 48, "edgecolor": TEXT, "linewidth": 0.5, "alpha": 0.78})
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.set_title("Regression fit over seeded scarcity observations", fontweight="bold")
    ax.set_xlabel("Available real-report supervision")
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False)
    polish(ax)
    save_fig(fig, out_dirs, "Figure_theme1_report_supervision_scarcity_curve_regression_strip")
    record("Figure_theme1_report_supervision_scarcity_curve_regression_strip", "Regression fit over a strip plot")

    fig = plt.figure(figsize=(8.2, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.15], height_ratios=[1.1, 4], hspace=0.05, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    sns.scatterplot(data=summary, x="mean_n_real_train", y="auc_mean", hue="Setup", size="mean_n_pseudo_train", sizes=(80, 430), palette=[BLUE, RUST], edgecolor=TEXT, linewidth=0.65, ax=ax, legend=False)
    ax_top.hist(summary["mean_n_real_train"], bins=6, color=SLATE, alpha=0.72, edgecolor="white")
    ax_right.hist(summary["auc_mean"], bins=6, orientation="horizontal", color=GOLD, alpha=0.72, edgecolor="white")
    ax.set_title("Joint and marginal view: evidence scale and AUROC", fontweight="bold", pad=18)
    ax.set_xlabel("Mean real-report training cases")
    ax.set_ylabel("Mean AUROC")
    ax_top.axis("off")
    ax_right.axis("off")
    polish(ax)
    save_fig(fig, out_dirs, "Figure_theme1_report_supervision_scarcity_curve_joint_marginals")
    record("Figure_theme1_report_supervision_scarcity_curve_joint_marginals", "Joint and marginal histograms")

    fig, ax = plt.subplots(figsize=(6.6, 6.2), subplot_kw={"projection": "polar"})
    cats = ["10%", "25%", "50%", "100%"]
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]
    for setup, color in zip(["LCAD-augmented", "Real-report only"], [BLUE, RUST]):
        vals = wide_line.loc[cats, setup].to_list()
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2.3, label=setup)
        ax.fill(angles, vals, color=color, alpha=0.13)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats)
    ax.set_ylim(0.55, 0.86)
    ax.set_title("Polar profile: AUROC across scarcity levels", fontweight="bold", pad=16)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    save_fig(fig, out_dirs, "Figure_theme1_report_supervision_scarcity_curve_polar_profile")
    record("Figure_theme1_report_supervision_scarcity_curve_polar_profile", "FacetGrid with custom projection")

    write_manifest(out_final, rows, "Report-Supervision Scarcity Style Gallery")
    write_manifest(out_pub, rows, "Report-Supervision Scarcity Style Gallery")


def alignment_gallery() -> None:
    setup_theme()
    detail, macro, metric_long = load_alignment()
    out_final = FINAL / "Figure_theme1_alignment_retrieval_mrr_style_gallery"
    out_pub = OUT_THEME / "Figure_theme1_alignment_retrieval_mrr_style_gallery"
    out_dirs = [out_pub, out_final]
    rows: list[dict[str, str]] = []

    def record(stem: str, ref: str, caption: str = ALIGNMENT_CAPTION) -> None:
        rows.append({"stem": stem, "seaborn_reference": ref, "caption": caption})

    matrix = detail.pivot_table(index="Model", columns="Section", values="mrr", aggfunc="mean")
    model_order = macro.sort_values("macro_mrr", ascending=False)["Model"].tolist()
    matrix = matrix.reindex(model_order)
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap=SEQ, linewidths=0.8, linecolor="white", cbar_kws={"label": "MRR"}, ax=ax)
    ax.set_title("Annotated heatmap: section-wise retrieval MRR", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, out_dirs, "Figure_theme1_alignment_retrieval_mrr_annotated_heatmap")
    record("Figure_theme1_alignment_retrieval_mrr_annotated_heatmap", "Annotated heatmaps")

    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    plot = macro.sort_values("macro_mrr", ascending=True)
    ax.hlines(plot["Model"], 0, plot["macro_mrr"], color=LIGHT, linewidth=4)
    ax.scatter(plot["macro_mrr"], plot["Model"], s=110, color=BLUE, edgecolor=TEXT, linewidth=0.7)
    ax.set_title("Dot plot: macro retrieval alignment by model", fontweight="bold")
    ax.set_xlabel("Macro MRR")
    ax.set_ylabel("")
    polish(ax, "x")
    save_fig(fig, out_dirs, "Figure_theme1_alignment_retrieval_mrr_dotplot")
    record("Figure_theme1_alignment_retrieval_mrr_dotplot", "Dot plot with several variables")

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    model_palette = {model: PALETTE[i % len(PALETTE)] for i, model in enumerate(macro["Model"].drop_duplicates())}
    sns.scatterplot(data=macro, x="macro_recall_at_5", y="macro_mrr", hue="Model", size="macro_positive_cosine", sizes=(90, 460), palette=model_palette, edgecolor=TEXT, linewidth=0.65, ax=ax, legend=False)
    label_offsets = {
        "Real-report only": (10, 10, "left"),
        "Pseudo-augmented LCAD": (10, -8, "left"),
        "Full LCAD-RASA": (-14, -18, "right"),
        "No section alignment": (12, 2, "left"),
        "Risk-head auxiliary": (12, 2, "left"),
    }
    for _, row in macro.iterrows():
        if row["Model"] in label_offsets:
            dx, dy, ha = label_offsets[row["Model"]]
            ax.annotate(
                row["Model"],
                xy=(row["macro_recall_at_5"], row["macro_mrr"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=TEXT,
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )
    ax.set_title("Bubble scatter: recall and rank-based alignment", fontweight="bold")
    ax.set_xlabel("Macro recall@5")
    ax.set_ylabel("Macro MRR")
    polish(ax)
    ax.text(0.98, 0.04, "Bubble size encodes positive-pair cosine.", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=SLATE)
    save_fig(fig, out_dirs, "Figure_theme1_alignment_retrieval_mrr_bubble_recall")
    record("Figure_theme1_alignment_retrieval_mrr_bubble_recall", "Scatterplot with varying point sizes and hues")

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.9), sharey=True)
    for ax, section in zip(axes, ["OCT findings", "Colposcopy findings", "Clinical context", "Impression"]):
        sub = detail[detail["Section"].eq(section)].sort_values("mrr", ascending=True)
        ax.hlines(sub["Model"], 0, sub["mrr"], color=GRID, linewidth=3)
        ax.scatter(sub["mrr"], sub["Model"], s=70, color=TEAL, edgecolor=TEXT, linewidth=0.55)
        ax.set_title(section, fontweight="bold", fontsize=10.2)
        ax.set_xlabel("MRR")
        polish(ax, "x")
    axes[0].set_ylabel("")
    fig.suptitle("Line plots on multiple facets: section-specific MRR", fontweight="bold", y=1.02)
    save_fig(fig, out_dirs, "Figure_theme1_alignment_retrieval_mrr_section_facets")
    record("Figure_theme1_alignment_retrieval_mrr_section_facets", "Line plots on multiple facets")

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    sns.boxplot(data=detail, y="Model", x="mrr", order=model_order, color=LIGHT, fliersize=0, linewidth=1.0, ax=ax)
    sns.stripplot(data=detail, y="Model", x="mrr", order=model_order, hue="Section", palette=[BLUE, TEAL, GOLD, RUST], size=6.2, jitter=0.18, edgecolor=TEXT, linewidth=0.45, ax=ax)
    ax.set_title("Horizontal boxplot with section observations", fontweight="bold")
    ax.set_xlabel("Section-level MRR")
    ax.set_ylabel("")
    polish(ax, "x")
    ax.legend(frameon=False, title="Section", loc="lower right")
    save_fig(fig, out_dirs, "Figure_theme1_alignment_retrieval_mrr_box_observations")
    record("Figure_theme1_alignment_retrieval_mrr_box_observations", "Horizontal boxplot with observations")

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    metric_palette = {
        "Recall@1": BLUE,
        "Recall@5": "#95A1B2",
        "MRR": RUST,
        "Cross-case gap": "#D6DEE8",
        "Wrong-section gap": "#557A95",
    }
    sns.barplot(
        data=metric_long,
        y="Model",
        x="Value",
        hue="Metric",
        palette=metric_palette,
        width=0.92,
        edgecolor=TEXT,
        linewidth=0.45,
        ax=ax,
    )
    for patch in ax.patches:
        patch.set_alpha(0.86)
    ax.set_title("Retrieval-calibrated semantic alignment across model variants", fontweight="bold", fontsize=14.4, pad=12)
    ax.set_xlabel("Metric value", fontsize=12.4, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=10.9)
    polish(ax, "x")
    ax.legend(frameon=False, bbox_to_anchor=(1.015, 1), loc="upper left", fontsize=10.2, title_fontsize=10.4)
    fig.subplots_adjust(left=0.26, right=0.78, top=0.88, bottom=0.14)
    save_fig(fig, out_dirs, "Figure_theme1_alignment_retrieval_mrr_grouped_metrics")
    record("Figure_theme1_alignment_retrieval_mrr_grouped_metrics", "Grouped barplots")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for section, color in zip(detail["Section"].drop_duplicates(), [BLUE, TEAL, GOLD, RUST]):
        vals = np.sort(detail.loc[detail["Section"].eq(section), "mrr"].to_numpy(dtype=float))
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=color, linewidth=2.0, label=section)
    ax.set_title("Facetted ECDF style: section-level MRR distribution", fontweight="bold")
    ax.set_xlabel("MRR")
    ax.set_ylabel("Cumulative proportion")
    ax.legend(frameon=False)
    polish(ax)
    save_fig(fig, out_dirs, "Figure_theme1_alignment_retrieval_mrr_ecdf")
    record("Figure_theme1_alignment_retrieval_mrr_ecdf", "Facetted ECDF plots")

    gap_cols = ["macro_positive_minus_cross_case", "macro_positive_minus_wrong_section"]
    gap = macro.melt(id_vars=["Model"], value_vars=gap_cols, var_name="Gap type", value_name="Gap")
    gap["Gap type"] = gap["Gap type"].map({"macro_positive_minus_cross_case": "Cross-case", "macro_positive_minus_wrong_section": "Wrong-section"})
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    sns.scatterplot(data=gap, x="Gap", y="Model", hue="Gap type", style="Gap type", s=95, palette=[BLUE, RUST], edgecolor=TEXT, linewidth=0.6, ax=ax)
    ax.axvline(0, color=TEXT, linewidth=0.9, linestyle=(0, (2, 2)))
    ax.set_title("Alignment gap audit: positive versus negative pairs", fontweight="bold")
    ax.set_xlabel("Positive-minus-negative cosine gap")
    ax.set_ylabel("")
    polish(ax, "x")
    ax.legend(frameon=False)
    save_fig(fig, out_dirs, "Figure_theme1_alignment_retrieval_mrr_gap_audit")
    record("Figure_theme1_alignment_retrieval_mrr_gap_audit", "Scatterplot with multiple semantics")

    metric_matrix = macro.set_index("Model")[["macro_recall_at_1", "macro_recall_at_5", "macro_mrr", "macro_positive_minus_cross_case", "macro_positive_minus_wrong_section"]]
    metric_matrix = metric_matrix.rename(
        columns={
            "macro_recall_at_1": "Recall@1",
            "macro_recall_at_5": "Recall@5",
            "macro_mrr": "MRR",
            "macro_positive_minus_cross_case": "Cross-case gap",
            "macro_positive_minus_wrong_section": "Wrong-section gap",
        }
    ).reindex(model_order)
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    sns.heatmap(metric_matrix, annot=True, fmt=".3f", cmap=DIV, center=0, linewidths=0.8, linecolor="white", cbar_kws={"label": "Metric value"}, ax=ax)
    ax.set_title("Clustered-heatmap style: macro alignment evidence", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, out_dirs, "Figure_theme1_alignment_retrieval_mrr_macro_heatmap")
    record("Figure_theme1_alignment_retrieval_mrr_macro_heatmap", "Discovering structure in heatmap data")

    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    sns.violinplot(data=detail, x="Section", y="mrr", hue="Section", palette=[BLUE, TEAL, GOLD, RUST], inner=None, cut=0, linewidth=1.0, ax=ax, legend=False)
    sns.stripplot(data=detail, x="Section", y="mrr", color=TEXT, alpha=0.55, size=4.5, jitter=0.18, ax=ax)
    ax.set_title("Grouped violinplots: MRR distribution by report section", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("MRR")
    polish(ax, "y")
    save_fig(fig, out_dirs, "Figure_theme1_alignment_retrieval_mrr_violin_sections")
    record("Figure_theme1_alignment_retrieval_mrr_violin_sections", "Grouped violinplots with split violins")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    sns.regplot(data=detail, x="positive_minus_cross_case", y="mrr", scatter=False, color=TEXT, ci=None, ax=ax)
    sns.scatterplot(data=detail, x="positive_minus_cross_case", y="mrr", hue="Section", style="Section", palette=[BLUE, TEAL, GOLD, RUST], s=78, edgecolor=TEXT, linewidth=0.55, ax=ax)
    ax.axvline(0, color=TEXT, linestyle=(0, (2, 2)), linewidth=0.85)
    ax.set_title("Regression fit: cosine separation versus MRR", fontweight="bold")
    ax.set_xlabel("Positive-minus-cross-case cosine gap")
    ax.set_ylabel("MRR")
    polish(ax)
    ax.legend(frameon=False, title="Section")
    save_fig(fig, out_dirs, "Figure_theme1_alignment_retrieval_mrr_regression_gap")
    record("Figure_theme1_alignment_retrieval_mrr_regression_gap", "Regression fit over a strip plot")

    pair = macro[["macro_recall_at_1", "macro_recall_at_5", "macro_mrr", "macro_positive_minus_cross_case"]].rename(
        columns={
            "macro_recall_at_1": "Recall@1",
            "macro_recall_at_5": "Recall@5",
            "macro_mrr": "MRR",
            "macro_positive_minus_cross_case": "Gap",
        }
    )
    g = sns.pairplot(pair, corner=True, diag_kind="hist", plot_kws={"color": BLUE, "edgecolor": TEXT, "s": 58}, diag_kws={"color": SLATE, "edgecolor": "white"})
    g.fig.suptitle("Scatterplot matrix: macro alignment metrics", fontweight="bold", y=1.02)
    save_fig(g.fig, out_dirs, "Figure_theme1_alignment_retrieval_mrr_scatter_matrix")
    record("Figure_theme1_alignment_retrieval_mrr_scatter_matrix", "Scatterplot Matrix")

    fig, ax = plt.subplots(figsize=(7.0, 6.4), subplot_kw={"projection": "polar"})
    sections = ["OCT findings", "Colposcopy findings", "Clinical context", "Impression"]
    angles = np.linspace(0, 2 * np.pi, len(sections), endpoint=False).tolist()
    angles += angles[:1]
    top_models = macro.sort_values("macro_mrr", ascending=False).head(3)["Model"].tolist()
    for model, color in zip(top_models, [BLUE, RUST, TEAL]):
        vals = detail[detail["Model"].eq(model)].set_index("Section").reindex(sections)["mrr"].to_list()
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2.2, label=model)
        ax.fill(angles, vals, color=color, alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["OCT", "Colpo.", "Clinical", "Impression"])
    ax.set_ylim(0, max(0.09, float(detail["mrr"].max()) * 1.18))
    ax.set_title("Polar profile: section-wise MRR in top models", fontweight="bold", pad=16)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    save_fig(fig, out_dirs, "Figure_theme1_alignment_retrieval_mrr_polar_profile")
    record("Figure_theme1_alignment_retrieval_mrr_polar_profile", "FacetGrid with custom projection")

    write_manifest(out_final, rows, "Modality-Section Alignment Style Gallery")
    write_manifest(out_pub, rows, "Modality-Section Alignment Style Gallery")


def perturbation_gallery() -> None:
    setup_theme()
    text, matrix, sim, drop = load_perturbation()
    out_final = FINAL / "Figure3_modality_perturbation_heatmap_style_gallery"
    out_pub = OUT_JBD / "Figure3_modality_perturbation_heatmap_style_gallery"
    out_dirs = [out_pub, out_final]
    rows: list[dict[str, str]] = []

    def record(stem: str, ref: str, caption: str = PERTURB_CAPTION) -> None:
        rows.append({"stem": stem, "seaborn_reference": ref, "caption": caption})

    cond_order = ["Normal", "Mask OCT", "Mask colposcopy", "Mask clinical", "Mask visual", "Label-only"]
    sim_sub = sim[sim["Condition"].isin(cond_order)]
    piv = sim_sub.pivot_table(index="Condition", columns="Section", values="Similarity").reindex(cond_order)
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap=SEQ, vmin=0, vmax=1, linewidths=0.8, linecolor="white", cbar_kws={"label": "Similarity to normal"}, ax=ax)
    ax.set_title("Annotated heatmap: section similarity under perturbation", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, out_dirs, "Figure3_modality_perturbation_heatmap_annotated_similarity")
    record("Figure3_modality_perturbation_heatmap_annotated_similarity", "Annotated heatmaps")

    drop_sub = drop[drop["Condition"].isin(cond_order) & drop["Measure"].isin(["OCT findings", "Colposcopy findings", "Clinical context", "Impression", "Overall report", "Risk shift"])]
    drop_piv = drop_sub.pivot_table(index="Condition", columns="Measure", values="Drop").reindex(cond_order)
    fig, ax = plt.subplots(figsize=(9.6, 5.5))
    sns.heatmap(drop_piv, annot=True, fmt=".2f", cmap=SEQ_RUST, linewidths=0.8, linecolor="white", cbar_kws={"label": "Drop / shift"}, ax=ax)
    ax.set_title("Drop matrix: section degradation and risk shift", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, out_dirs, "Figure3_modality_perturbation_heatmap_drop_matrix")
    record("Figure3_modality_perturbation_heatmap_drop_matrix", "Discovering structure in heatmap data")

    fig, axes = plt.subplots(1, 4, figsize=(14.6, 5.8), sharey=True)
    display_order = ["Normal", "Mask OCT", "Mask colposcopy", "Mask clinical", "Mask visual", "Label-only"]
    y_positions = np.arange(len(display_order))
    section_palette = {
        "OCT findings": BLUE,
        "Colposcopy findings": RUST,
        "Clinical context": "#557A95",
        "Impression": "#6F7F91",
    }
    for ax, section in zip(axes, ["OCT findings", "Colposcopy findings", "Clinical context", "Impression"]):
        sub = sim_sub[sim_sub["Section"].eq(section)].set_index("Condition").reindex(display_order).reset_index()
        color = section_palette[section]
        ax.axvspan(0.0, 0.50, color=RUST, alpha=0.055, zorder=0)
        ax.hlines(y_positions, 0, sub["Similarity"], color=color, linewidth=4.8, alpha=0.88)
        ax.scatter(sub["Similarity"], y_positions, s=152, color=color, edgecolor=TEXT, linewidth=0.85, zorder=3)
        ax.axvline(1.0, color=TEXT, linestyle=(0, (2, 2)), linewidth=1.15, alpha=0.72)
        for y, value in zip(y_positions, sub["Similarity"]):
            if pd.notna(value) and value <= 0.38:
                ax.text(min(value + 0.035, 0.97), y, f"{value:.2f}", va="center", ha="left", fontsize=9.4, fontweight="bold", color=TEXT)
        ax.set_title(section, fontweight="bold", fontsize=12.8, pad=9)
        ax.set_xlabel("Similarity", fontsize=11.7, fontweight="bold")
        ax.set_xlim(0, 1.04)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(display_order, fontsize=10.9)
        ax.set_ylim(len(display_order) - 0.5, -0.5)
        ax.tick_params(axis="x", labelsize=10.6)
        polish(ax, "x")
    axes[0].set_ylabel("Perturbation condition", fontsize=11.7, fontweight="bold")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
        ax.set_ylabel("")
    fig.suptitle("Section-wise modality response under perturbation", fontweight="bold", fontsize=15.2, y=0.98)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.82, bottom=0.16, wspace=0.13)
    save_fig(fig, out_dirs, "Figure3_modality_perturbation_heatmap_section_facets")
    record("Figure3_modality_perturbation_heatmap_section_facets", "Line plots on multiple facets")

    text_sub = text[text["Condition"].isin(cond_order)].copy()
    fig, ax = plt.subplots(figsize=(8.7, 5.1))
    condition_palette = {condition: PALETTE[i % len(PALETTE)] for i, condition in enumerate(text_sub["Condition"].drop_duplicates())}
    sns.scatterplot(data=text_sub, x="report_similarity_to_normal", y="mean_risk_score", hue="Condition", size="risk_score_absolute_delta_vs_normal", sizes=(70, 470), palette=condition_palette, edgecolor=TEXT, linewidth=0.65, ax=ax)
    ax.set_title("Bubble scatter: report degradation versus risk shift", fontweight="bold")
    ax.set_xlabel("Report similarity to normal")
    ax.set_ylabel("Mean risk score")
    polish(ax)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, out_dirs, "Figure3_modality_perturbation_heatmap_bubble_risk")
    record("Figure3_modality_perturbation_heatmap_bubble_risk", "Scatterplot with continuous hues and sizes")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    sns.boxplot(data=sim_sub, y="Condition", x="Similarity", order=cond_order, color=LIGHT, fliersize=0, linewidth=1.0, ax=ax)
    sns.stripplot(data=sim_sub, y="Condition", x="Similarity", order=cond_order, hue="Section", palette=[BLUE, TEAL, GOLD, RUST], size=6.5, jitter=0.16, edgecolor=TEXT, linewidth=0.45, ax=ax)
    ax.set_title("Horizontal boxplot with section observations", fontweight="bold")
    ax.set_xlabel("Similarity to normal")
    ax.set_ylabel("")
    polish(ax, "x")
    ax.legend(frameon=False, title="Section", loc="lower left")
    save_fig(fig, out_dirs, "Figure3_modality_perturbation_heatmap_box_observations")
    record("Figure3_modality_perturbation_heatmap_box_observations", "Horizontal boxplot with observations")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    sns.barplot(data=drop_sub, y="Condition", x="Drop", hue="Measure", order=cond_order, palette=[BLUE, TEAL, GOLD, RUST, SLATE, "#6F5B85"], ax=ax)
    ax.set_title("Grouped barplots: perturbation drop profile", fontweight="bold")
    ax.set_xlabel("Drop / shift")
    ax.set_ylabel("")
    polish(ax, "x")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, out_dirs, "Figure3_modality_perturbation_heatmap_grouped_drop")
    record("Figure3_modality_perturbation_heatmap_grouped_drop", "Grouped barplots")

    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    for section, color in zip(sim_sub["Section"].drop_duplicates(), [BLUE, TEAL, GOLD, RUST]):
        vals = np.sort(sim_sub.loc[sim_sub["Section"].eq(section), "Similarity"].to_numpy(dtype=float))
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=color, linewidth=2.0, label=section)
    ax.set_title("ECDF style: section similarity distribution", fontweight="bold")
    ax.set_xlabel("Similarity to normal")
    ax.set_ylabel("Cumulative proportion")
    ax.legend(frameon=False)
    polish(ax)
    save_fig(fig, out_dirs, "Figure3_modality_perturbation_heatmap_ecdf")
    record("Figure3_modality_perturbation_heatmap_ecdf", "Facetted ECDF plots")

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    risk = text_sub[~text_sub["condition"].eq("normal")].sort_values("risk_score_absolute_delta_vs_normal")
    ax.hlines(risk["Condition"], 0, risk["risk_score_absolute_delta_vs_normal"], color=LIGHT, linewidth=4)
    ax.scatter(risk["risk_score_absolute_delta_vs_normal"], risk["Condition"], s=110, color=RUST, edgecolor=TEXT, linewidth=0.7)
    ax.set_title("Dot plot: absolute risk displacement", fontweight="bold")
    ax.set_xlabel("Absolute risk-score shift")
    ax.set_ylabel("")
    polish(ax, "x")
    save_fig(fig, out_dirs, "Figure3_modality_perturbation_heatmap_risk_dotplot")
    record("Figure3_modality_perturbation_heatmap_risk_dotplot", "Dot plot with several variables")

    fig, ax = plt.subplots(figsize=(9.6, 5.5))
    ordered = drop_piv.loc[[idx for idx in drop_piv.index if idx != "Normal"]].sort_values("Risk shift", ascending=False)
    sns.heatmap(ordered, annot=True, fmt=".2f", cmap=SEQ_RUST, linewidths=0.8, linecolor="white", cbar_kws={"label": "Drop / shift"}, ax=ax)
    ax.set_title("Clustered-heatmap style: perturbations ordered by risk shift", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_fig(fig, out_dirs, "Figure3_modality_perturbation_heatmap_ordered_drop")
    record("Figure3_modality_perturbation_heatmap_ordered_drop", "Discovering structure in heatmap data")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    y_positions = np.arange(4)[::-1]
    for ypos, section, color in zip(y_positions, ["OCT findings", "Colposcopy findings", "Clinical context", "Impression"], [BLUE, TEAL, GOLD, RUST]):
        vals = sim_sub.loc[sim_sub["Section"].eq(section), "Similarity"].to_numpy(dtype=float)
        hist, edges = np.histogram(vals, bins=np.linspace(0, 1, 14), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        dens = hist / max(hist.max(), 1e-9) * 0.62
        ax.fill_between(centers, ypos, ypos + dens, color=color, alpha=0.68, linewidth=0)
        ax.plot(centers, ypos + dens, color=color, linewidth=2.0)
        ax.text(-0.03, ypos + 0.20, section, ha="right", va="center", fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Similarity to normal")
    ax.set_title("Overlapping densities: section similarity distributions", fontweight="bold")
    polish(ax, "x")
    save_fig(fig, out_dirs, "Figure3_modality_perturbation_heatmap_ridge_density")
    record("Figure3_modality_perturbation_heatmap_ridge_density", "Overlapping densities (ridge plot)")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    sns.regplot(data=text_sub, x="report_similarity_to_normal", y="risk_score_absolute_delta_vs_normal", color=TEXT, scatter=False, ci=None, ax=ax)
    sns.scatterplot(data=text_sub, x="report_similarity_to_normal", y="risk_score_absolute_delta_vs_normal", hue="Condition", palette=condition_palette, s=98, edgecolor=TEXT, linewidth=0.65, ax=ax)
    ax.set_title("Regression fit: report similarity versus risk displacement", fontweight="bold")
    ax.set_xlabel("Report similarity to normal")
    ax.set_ylabel("Absolute risk-score shift")
    polish(ax)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, out_dirs, "Figure3_modality_perturbation_heatmap_regression_risk")
    record("Figure3_modality_perturbation_heatmap_regression_risk", "Regression fit over a strip plot")

    fig = plt.figure(figsize=(8.2, 6.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.15], height_ratios=[1.1, 4], hspace=0.05, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    sns.scatterplot(data=text_sub, x="report_similarity_to_normal", y="risk_score_absolute_delta_vs_normal", hue="Condition", palette=condition_palette, s=95, edgecolor=TEXT, linewidth=0.65, ax=ax, legend=False)
    ax_top.hist(text_sub["report_similarity_to_normal"], bins=7, color=SLATE, alpha=0.72, edgecolor="white")
    ax_right.hist(text_sub["risk_score_absolute_delta_vs_normal"], bins=7, orientation="horizontal", color=RUST, alpha=0.70, edgecolor="white")
    ax.set_title("Joint and marginal view: report degradation and risk shift", fontweight="bold", pad=18)
    ax.set_xlabel("Report similarity to normal")
    ax.set_ylabel("Absolute risk-score shift")
    ax_top.axis("off")
    ax_right.axis("off")
    polish(ax)
    save_fig(fig, out_dirs, "Figure3_modality_perturbation_heatmap_joint_marginals")
    record("Figure3_modality_perturbation_heatmap_joint_marginals", "Joint and marginal histograms")

    radar_cols = ["OCT findings", "Colposcopy findings", "Clinical context", "Overall report", "Risk shift"]
    radar_conditions = ["Mask OCT", "Mask colposcopy", "Mask clinical", "Mask visual", "Label-only"]
    fig, ax = plt.subplots(figsize=(7.2, 6.5), subplot_kw={"projection": "polar"})
    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]
    for condition, color in zip(radar_conditions, [BLUE, TEAL, GOLD, RUST, "#6F5B85"]):
        vals = drop_piv.loc[condition, radar_cols].to_list()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.0, color=color, label=condition)
        ax.fill(angles, vals, alpha=0.08, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["OCT", "Colpo.", "Clinical", "Report", "Risk"])
    ax.set_ylim(0, max(1.02, float(drop_piv[radar_cols].max().max()) * 1.08))
    ax.set_title("Polar profile: modality-specific degradation signature", fontweight="bold", pad=16)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=2)
    save_fig(fig, out_dirs, "Figure3_modality_perturbation_heatmap_polar_profile")
    record("Figure3_modality_perturbation_heatmap_polar_profile", "FacetGrid with custom projection")

    write_manifest(out_final, rows, "Modality Perturbation Style Gallery")
    write_manifest(out_pub, rows, "Modality Perturbation Style Gallery")


def redraw_sensitivity_matrix() -> None:
    setup_theme()
    _, matrix, _, _ = load_perturbation()
    cols = ["oct_findings_drop", "colposcopy_findings_drop", "clinical_context_drop", "impression_drop", "report_drop", "risk_abs_delta"]
    view = matrix.set_index("Condition")[cols].rename(columns={c: label_section(c) for c in cols})
    order = ["Normal", "Mask OCT", "Mask colposcopy", "Mask clinical", "Shuffle OCT", "Shuffle colposcopy", "Shuffle clinical", "Mask visual", "Label-only", "Randomize label"]
    view = view.reindex([x for x in order if x in view.index])
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    sns.heatmap(view, annot=True, fmt=".2f", cmap=SEQ_RUST, linewidths=0.8, linecolor="white", cbar_kws={"label": "Semantic drop / risk shift"}, ax=ax)
    ax.set_title("Perturbation sensitivity across report sections", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=22)
    ax.tick_params(axis="y", rotation=0)
    fig.subplots_adjust(left=0.21, right=0.98, top=0.88, bottom=0.25)
    save_fig(fig, [OUT_THEME, FINAL], "Figure_theme1_perturbation_sensitivity_matrix")


def main() -> None:
    scarcity_gallery()
    alignment_gallery()
    perturbation_gallery()
    redraw_sensitivity_matrix()
    print("Generated style galleries for scarcity, alignment, and perturbation figures.")


if __name__ == "__main__":
    main()
