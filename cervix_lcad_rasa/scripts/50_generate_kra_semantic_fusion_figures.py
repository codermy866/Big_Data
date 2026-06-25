#!/usr/bin/env python3
"""Generate manuscript-ready figures for KRA semantic fusion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-y_score)
    y = y_true[order].astype(int)
    scores = y_score[order]
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = np.concatenate([[0.0], tps / pos])
    fpr = np.concatenate([[0.0], fps / neg])
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROJECT = ROOT.parent
ANALYSIS = ROOT / "outputs/publishable/kra_semantic_fusion_analysis"
FIG_DIR = ROOT / "outputs/publishable/figures/jbd_final"
FIG_ROOT = ROOT / "outputs/publishable/figures"
FIG_MAIN = ROOT / "outputs/publishable/figures/main"
FIG_SUBMISSION = ROOT / "outputs/publishable_jbd_submission_v2/figures"
FINAL_FIG = PROJECT / "final_Fig"
TABLE_DIR = ROOT / "outputs/publishable/tables/manuscript"

TEXT = "#17212B"
GRID = "#E2E7EE"
REF = "#95A1B2"
EDGE = "#17212B"
PANEL_BG = "#FFFFFF"
PSEUDO_FILL = "#F0F3F7"
NEGATIVE_FILL = "#D6DEE8"
ACCENT = "#C65A46"
PALETTE = {
    "full_lcad_rasa_stablehash": "#254B6D",
    "semantic_retrieval_positive_ratio": "#95A1B2",
    "kra_semantic_fusion": ACCENT,
}
LABELS = {
    "full_lcad_rasa_stablehash": "MOSAIC-RASA backbone",
    "semantic_retrieval_positive_ratio": "Semantic retrieval only",
    "kra_semantic_fusion": "MOSAIC (full)",
}
MOSAIC_LABELS = LABELS
MODEL_MARKERS = {
    "full_lcad_rasa_stablehash": "o",
    "semantic_retrieval_positive_ratio": "D",
    "kra_semantic_fusion": "o",
}
METRIC_LABELS = {
    "auc": "AUROC",
    "auprc": "AUPRC",
    "f1": "F1",
    "sensitivity": "Sensitivity",
    "precision": "Precision",
    "balanced_accuracy": "Balanced acc.",
}


def _math_num(value: float, fmt: str = ".3f", *, bold: bool = False) -> str:
    text = format(float(value), fmt)
    cmd = "mathbf" if bold else "mathrm"
    return rf"$\{cmd}{{{text}}}$"


def _math_text(text: str, *, bold: bool = False) -> str:
    cmd = "mathbf" if bold else "mathrm"
    escaped = text.replace("%", r"\%")
    return rf"$\{cmd}{{{escaped}}}$"


def _mixed_pvalue(p_value: float, *, bold: bool = False) -> str:
    if float(p_value) < 0.001:
        return f"p < {_math_num(0.001, '.3f', bold=bold)}"
    return f"p = {_math_num(p_value, '.3f', bold=bold)}"


def setup_style() -> None:
    from src.supplementary.jbd_figure_typography import FONT_ARIAL, FONT_TIMES, setup_arial_rcparams

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_ARIAL, "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 14.4,
        "axes.titlesize": 17.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 15.8,
        "axes.labelweight": "bold",
        "xtick.labelsize": 14.3,
        "ytick.labelsize": 14.3,
        "legend.fontsize": 13.2,
        "legend.title_fontsize": 13.6,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.linewidth": 0.9,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": TEXT,
        "grid.color": GRID,
        "grid.linewidth": 0.75,
        "grid.alpha": 0.82,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_TIMES,
        "mathtext.it": f"{FONT_TIMES}:italic",
        "mathtext.bf": f"{FONT_TIMES}:bold",
    }
    sns.set_theme(style="whitegrid", context="paper", font=FONT_ARIAL, rc=rc)
    setup_arial_rcparams(rc)


def save_many(fig: plt.Figure, name: str) -> None:
    from src.supplementary.jbd_figure_typography import apply_arial_to_figure, apply_mixed_en_typography

    apply_arial_to_figure(fig)
    apply_mixed_en_typography(fig)
    bases = [FIG_DIR, FIG_ROOT, FIG_MAIN]
    if name.startswith("Figure_mosaic_"):
        bases.append(FINAL_FIG)
    if FIG_SUBMISSION.parent.exists():
        bases.append(FIG_SUBMISSION)
    for base in bases:
        base.mkdir(parents=True, exist_ok=True)
        fig.savefig(base / f"{name}.png", dpi=350, bbox_inches="tight", facecolor="white", pad_inches=0.08)
        fig.savefig(base / f"{name}.pdf", bbox_inches="tight", facecolor="white", pad_inches=0.08)


def write_manuscript_tables(risk: pd.DataFrame, center: pd.DataFrame, bootstrap: dict) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    risk_out = risk.copy()
    risk_out.insert(1, "display_name", risk_out["model_id"].map(LABELS))
    risk_out.to_csv(TABLE_DIR / "T_mosaic_main_comparison.csv", index=False)
    center.to_csv(TABLE_DIR / "T_mosaic_centerwise.csv", index=False)
    pd.DataFrame([bootstrap]).to_csv(TABLE_DIR / "T_mosaic_paired_bootstrap.csv", index=False)
    risk_out.to_csv(TABLE_DIR / "T_kra_semantic_fusion_main_comparison.csv", index=False)
    center.to_csv(TABLE_DIR / "T_kra_semantic_fusion_centerwise.csv", index=False)
    pd.DataFrame([bootstrap]).to_csv(TABLE_DIR / "T_kra_semantic_fusion_paired_bootstrap.csv", index=False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.105,
        1.085,
        label,
        transform=ax.transAxes,
        fontsize=19.0,
        fontweight="bold",
        va="top",
        ha="left",
        fontfamily="Arial",
        color="white",
        bbox={"boxstyle": "round,pad=0.17,rounding_size=0.03", "facecolor": PALETTE["full_lcad_rasa_stablehash"], "edgecolor": "none"},
    )


def _style_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.set_facecolor(PANEL_BG)
    if grid_axis in {"x", "both"}:
        ax.grid(True, axis="x", color=GRID, alpha=0.82, linewidth=0.85)
    else:
        ax.grid(False, axis="x")
    if grid_axis in {"y", "both"}:
        ax.grid(True, axis="y", color=GRID, alpha=0.82, linewidth=0.85)
    else:
        ax.grid(False, axis="y")
    ax.tick_params(axis="both", colors=TEXT, labelsize=14.3)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("Arial")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#C9D2DD")
        ax.spines[side].set_linewidth(0.95)


def plot_metric_dotplot(ax: plt.Axes, risk: pd.DataFrame) -> None:
    metrics = ["auc", "auprc", "f1", "sensitivity", "precision", "balanced_accuracy"]
    metric_view = risk.set_index("model_id")
    ybase = np.arange(len(metrics))[::-1]
    for y, metric in zip(ybase, metrics):
        base = float(metric_view.loc["full_lcad_rasa_stablehash", metric])
        semantic = float(metric_view.loc["semantic_retrieval_positive_ratio", metric])
        full = float(metric_view.loc["kra_semantic_fusion", metric])
        ax.plot([base, full], [y + 0.13, y - 0.13], color=REF, linewidth=2.2, alpha=0.82, solid_capstyle="round", zorder=1)
        ax.scatter(base, y + 0.13, s=86, color=PALETTE["full_lcad_rasa_stablehash"], edgecolor=EDGE, linewidth=0.65, zorder=3)
        ax.scatter(semantic, y, s=78, color=PALETTE["semantic_retrieval_positive_ratio"], marker="D", edgecolor=EDGE, linewidth=0.62, zorder=3)
        ax.scatter(full, y - 0.13, s=94, color=PALETTE["kra_semantic_fusion"], edgecolor=EDGE, linewidth=0.65, zorder=4)
    ax.set_yticks(ybase)
    ax.set_yticklabels([METRIC_LABELS[m] for m in metrics])
    ax.set_xlim(0.35, 0.95)
    ax.set_xlabel("Held-out test metric", labelpad=8)
    ax.set_title("Multi-metric test profile", fontweight="bold", color=TEXT, pad=10)
    ax.text(
        0.02,
        0.965,
        "Panel A marker key:\ncircles = RASA/full; diamonds = retrieval-only.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.4,
        color=TEXT,
        linespacing=1.18,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "white", "edgecolor": GRID, "alpha": 0.90},
        zorder=5,
    )
    handles = [
        plt.Line2D([0], [0], color=REF, linewidth=2.0, alpha=0.82, label="Backbone-to-full link"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["full_lcad_rasa_stablehash"], markeredgecolor=EDGE, markeredgewidth=0.65, markersize=6.8, label="RASA backbone"),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["semantic_retrieval_positive_ratio"], markeredgecolor=EDGE, markeredgewidth=0.62, markersize=6.2, label="Retrieval only"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["kra_semantic_fusion"], markeredgecolor=EDGE, markeredgewidth=0.65, markersize=6.8, label="MOSAIC full"),
    ]
    leg = ax.legend(
        handles=handles,
        title="Panel A markers",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.50, -0.25),
        ncol=2,
        borderpad=0.0,
        labelspacing=0.24,
        handlelength=1.65,
        handletextpad=0.55,
        columnspacing=1.25,
        fontsize=12.8,
        title_fontsize=13.2,
    )
    _style_axis(ax, grid_axis="x")


def plot_roc(ax: plt.Axes, scores: pd.DataFrame) -> None:
    test = scores[scores["split"].eq("test")].copy()
    y = test["y_true"].to_numpy()
    curves = [
        ("full_lcad_rasa_stablehash", test["risk_score"].to_numpy()),
        ("semantic_retrieval_positive_ratio", test["semantic_retrieval_positive_ratio"].to_numpy()),
        ("kra_semantic_fusion", test["semantic_fusion_score"].to_numpy()),
    ]
    for model, s in curves:
        fpr, tpr, auc_val = _roc_curve(y, s)
        ax.plot(fpr, tpr, color=PALETTE[model], linewidth=2.55, label=f"{LABELS[model]} ({_math_num(auc_val)})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=REF, linewidth=1.25)
    ax.set_xlabel("False positive rate", labelpad=8)
    ax.set_ylabel("True positive rate", labelpad=8)
    ax.set_title("Held-out ROC curves", fontweight="bold", color=TEXT, pad=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    leg = ax.legend(frameon=True, loc="lower right", borderpad=0.55, handlelength=2.2)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_alpha(0.88)
    _style_axis(ax, grid_axis="both")


def plot_score_distribution(ax: plt.Axes, scores: pd.DataFrame) -> None:
    test = scores[scores["split"].eq("test")].copy()
    test["Outcome"] = np.where(test["y_true"].eq(1), "CIN2+", "CIN2-")
    rng = np.random.default_rng(42)
    order = ["CIN2-", "CIN2+"]
    colors = {"CIN2-": NEGATIVE_FILL, "CIN2+": ACCENT}
    edges = {"CIN2-": PALETTE["full_lcad_rasa_stablehash"], "CIN2+": ACCENT}
    means: list[float] = []
    for xpos, outcome in enumerate(order):
        vals = test.loc[test["Outcome"].eq(outcome), "semantic_fusion_score"].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            means.append(np.nan)
            continue
        y_grid = np.linspace(max(-0.02, vals.min() - 0.045), min(1.02, vals.max() + 0.045), 220)
        bw = max(0.035, 1.06 * np.std(vals) * (len(vals) ** (-1 / 5)))
        density = np.exp(-0.5 * ((y_grid[:, None] - vals[None, :]) / bw) ** 2).sum(axis=1)
        density = density / max(float(density.max()), 1e-9) * 0.255
        cloud_left = xpos + 0.105
        ax.fill_betweenx(
            y_grid,
            cloud_left,
            cloud_left + density,
            facecolor=colors[outcome],
            edgecolor=edges[outcome],
            linewidth=1.75,
            alpha=0.72 if outcome == "CIN2+" else 0.82,
            zorder=1,
        )
        point_vals = vals
        if vals.size > 90:
            point_vals = rng.choice(vals, size=90, replace=False)
        rain_x = xpos - 0.125 + rng.normal(0, 0.025, size=point_vals.size)
        ax.scatter(
            rain_x,
            point_vals,
            s=22,
            facecolor=edges[outcome],
            edgecolor=EDGE,
            alpha=0.38,
            linewidth=0.35,
            zorder=3,
        )
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        lo, hi = np.percentile(vals, [5, 95])
        box_x = xpos - 0.005
        ax.plot([box_x, box_x], [lo, hi], color=edges[outcome], linewidth=2.0, zorder=4)
        ax.add_patch(
            plt.Rectangle(
                (box_x - 0.045, q1),
                0.090,
                q3 - q1,
                facecolor=colors[outcome],
                edgecolor=edges[outcome],
                linewidth=1.65,
                alpha=0.88,
                zorder=5,
            )
        )
        ax.plot([box_x - 0.047, box_x + 0.047], [med, med], color=EDGE, linewidth=1.45, zorder=6)
        mean_val = float(np.mean(vals))
        means.append(mean_val)
        ax.scatter(
            [box_x],
            [mean_val],
            marker="D",
            s=66,
            facecolor=edges[outcome],
            edgecolor=EDGE,
            linewidth=0.55,
            zorder=7,
        )
    if np.all(np.isfinite(means)):
        ax.plot([0 - 0.005, 1 - 0.005], means, color=TEXT, linewidth=1.65, alpha=0.74, zorder=2)
    ax.axhline(0.50, color=ACCENT, linestyle="--", linewidth=1.35, alpha=0.92)
    ax.text(
        -0.31,
        0.515,
        f"Val threshold {_math_num(0.50, '.2f', bold=True)}",
        color=ACCENT,
        fontsize=13.2,
        fontweight="bold",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 2.2},
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(order)
    ax.set_xlim(-0.48, 1.48)
    ax.set_xlabel("")
    ax.set_ylabel("MOSAIC score", labelpad=8)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Risk-score raincloud by outcome", fontweight="bold", color=TEXT, pad=10)
    _style_axis(ax, grid_axis="y")


def plot_centerwise(ax: plt.Axes, center: pd.DataFrame) -> None:
    view = center.dropna(subset=["auc", "baseline_auc"]).copy()
    view = view.sort_values("auc")
    y = np.arange(len(view))
    for i, row in enumerate(view.itertuples(index=False)):
        ax.plot([row.baseline_auc, row.auc], [i, i], color=REF, linewidth=2.2, alpha=0.82, solid_capstyle="round", zorder=1)
        ax.scatter(row.baseline_auc, i + 0.07, s=88, color=PALETTE["full_lcad_rasa_stablehash"], edgecolor=EDGE, linewidth=0.65, zorder=2)
        ax.scatter(row.auc, i - 0.07, s=96, color=PALETTE["kra_semantic_fusion"], edgecolor=EDGE, linewidth=0.65, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(view["center_id"].str.title())
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Centre-wise AUROC", labelpad=8)
    ax.set_title("Centre-wise AUROC shift", fontweight="bold", color=TEXT, pad=10)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["full_lcad_rasa_stablehash"], markeredgecolor=EDGE, markeredgewidth=0.65, label="MOSAIC-RASA backbone"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["kra_semantic_fusion"], markeredgecolor=EDGE, markeredgewidth=0.65, label="MOSAIC (full)"),
        ],
        frameon=False,
        loc="lower right",
        handletextpad=0.45,
    )
    bootstrap = json.loads((ANALYSIS / "kra_semantic_fusion_vs_full_paired_auc_bootstrap.json").read_text()) if (ANALYSIS / "kra_semantic_fusion_vs_full_paired_auc_bootstrap.json").is_file() else {}
    p_pool = bootstrap.get("paired_bootstrap_p_two_sided")
    if p_pool is not None:
        ax.text(
            0.02,
            0.98,
            f"Pooled paired test:\n{_mixed_pvalue(float(p_pool), bold=True)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13.0,
            fontweight="bold",
            fontfamily="Arial",
            bbox=dict(boxstyle="round,pad=0.28", fc=PSEUDO_FILL, ec=GRID, alpha=0.95),
        )
    _style_axis(ax, grid_axis="x")


def make_summary_figure(risk: pd.DataFrame, center: pd.DataFrame, scores: pd.DataFrame, bootstrap: dict) -> None:
    setup_style()
    fig = plt.figure(figsize=(16.8, 12.8), constrained_layout=True)
    fig._jbd_min_font_size_override = 13.0
    fig._jbd_max_font_size_override = 23.5
    fig._jbd_font_scale_override = 1.0
    fig.set_constrained_layout_pads(w_pad=0.085, h_pad=0.160, wspace=0.120, hspace=0.240)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    plot_metric_dotplot(ax_a, risk)
    plot_roc(ax_b, scores)
    plot_score_distribution(ax_c, scores)
    plot_centerwise(ax_d, center)
    for ax, label in zip([ax_a, ax_b, ax_c, ax_d], list("ABCD")):
        panel_label(ax, label)
    delta = bootstrap["delta_auc"]
    lo = bootstrap["delta_auc_ci_low"]
    hi = bootstrap["delta_auc_ci_high"]
    p = bootstrap["paired_bootstrap_p_two_sided"]
    fig.suptitle(
        "MOSAIC improves held-out risk stratification\n"
        f"(Delta AUROC {_math_num(delta, bold=True)}, {_math_text('95%', bold=True)} CI "
        f"{_math_num(lo, bold=True)} to {_math_num(hi, bold=True)}, p={_math_num(p, bold=True)})",
        y=1.075,
        fontsize=21.0,
        fontweight="bold",
        color=TEXT,
    )
    save_many(fig, "Figure_mosaic_performance_summary")
    save_many(fig, "Figure_kra_semantic_fusion_summary")
    plt.close(fig)


def make_metric_lollipop(risk: pd.DataFrame) -> None:
    """Compact MOSAIC metric comparison as a horizontal lollipop."""
    setup_style()
    metrics = ["auc", "auprc", "f1", "sensitivity", "precision", "balanced_accuracy"]
    matrix = risk.set_index("model_id")[metrics].rename(index=LABELS, columns=METRIC_LABELS)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    fig._jbd_min_font_size_override = 9.2
    fig._jbd_max_font_size_override = 14.2
    y_base = np.arange(len(matrix.columns))[::-1]
    offsets = {"MOSAIC-RASA backbone": 0.22, "Semantic retrieval only": 0.0, "MOSAIC (full)": -0.22}
    for model_id, color in PALETTE.items():
        label = LABELS[model_id]
        if label not in matrix.index:
            continue
        vals = matrix.loc[label].to_numpy(dtype=float)
        y = y_base + offsets.get(label, 0.0)
        ax.hlines(y, 0.35, vals, color=color, linewidth=1.75, alpha=0.58, zorder=1)
        ax.scatter(vals, y, s=72, color=color, marker=MODEL_MARKERS[model_id], edgecolor=EDGE, linewidth=0.6, label=label, zorder=3)
    ax.set_yticks(y_base)
    ax.set_yticklabels(matrix.columns.tolist())
    ax.set_xlim(0.35, 0.95)
    ax.set_xlabel("Held-out test metric", labelpad=8)
    ax.set_title("MOSAIC metric profile", fontweight="bold", color=TEXT, pad=10)
    ax.legend(frameon=False, loc="lower right", fontsize=9.2)
    _style_axis(ax, grid_axis="x")
    save_many(fig, "Figure_mosaic_metrics_heatmap")
    save_many(fig, "Figure_kra_semantic_fusion_metrics_heatmap")
    plt.close(fig)


def main() -> None:
    risk = pd.read_csv(ANALYSIS / "kra_semantic_fusion_risk_comparison.csv")
    center = pd.read_csv(ANALYSIS / "kra_semantic_fusion_centerwise.csv")
    scores = pd.read_csv(ANALYSIS / "kra_semantic_fusion_val_test_scores.csv")
    bootstrap = json.loads((ANALYSIS / "kra_semantic_fusion_vs_full_paired_auc_bootstrap.json").read_text(encoding="utf-8"))
    write_manuscript_tables(risk, center, bootstrap)
    make_summary_figure(risk, center, scores, bootstrap)
    make_metric_lollipop(risk)
    index = [
        "# MOSAIC Figure Index",
        "",
        "- `Figure_mosaic_performance_summary.png/pdf`: four-panel performance, ROC, score distribution, and centre-wise AUROC summary.",
        "- `Figure_mosaic_metrics_heatmap.png/pdf`: compact metric heatmap for manuscript or supplement.",
        "- Source tables: `tables/manuscript/T_mosaic_main_comparison.csv`, `T_mosaic_centerwise.csv`, and `T_mosaic_paired_bootstrap.csv`.",
        "- Legacy aliases retained: `Figure_kra_semantic_fusion_*` and `T_kra_semantic_fusion_*`.",
        "",
    ]
    (FIG_DIR / "MOSAIC_FIGURE_INDEX.md").write_text("\n".join(index), encoding="utf-8")
    (FIG_DIR / "KRA_SEMANTIC_FUSION_FIGURE_INDEX.md").write_text("\n".join(index), encoding="utf-8")
    print(f"Wrote MOSAIC figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
