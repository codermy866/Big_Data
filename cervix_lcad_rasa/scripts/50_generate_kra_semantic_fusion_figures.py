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

ANALYSIS = ROOT / "outputs/publishable/kra_semantic_fusion_analysis"
FIG_DIR = ROOT / "outputs/publishable/figures/jbd_final"
FIG_ROOT = ROOT / "outputs/publishable/figures"
FIG_MAIN = ROOT / "outputs/publishable/figures/main"
FIG_SUBMISSION = ROOT / "outputs/publishable_jbd_submission_v2/figures"
TABLE_DIR = ROOT / "outputs/publishable/tables/manuscript"

PALETTE = {
    "full_lcad_rasa_stablehash": "#1E3A66",
    "semantic_retrieval_positive_ratio": "#ADB093",
    "kra_semantic_fusion": "#E76B6B",
}
LABELS = {
    "full_lcad_rasa_stablehash": "MOSAIC--RASA backbone",
    "semantic_retrieval_positive_ratio": "Semantic retrieval only",
    "kra_semantic_fusion": "MOSAIC (full)",
}
MOSAIC_LABELS = LABELS
METRIC_LABELS = {
    "auc": "AUROC",
    "auprc": "AUPRC",
    "f1": "F1",
    "sensitivity": "Sensitivity",
    "precision": "Precision",
    "balanced_accuracy": "Balanced acc.",
}


def setup_style() -> None:
    from src.supplementary.jbd_figure_typography import setup_arial_rcparams

    setup_arial_rcparams(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": "#3E3425",
            "axes.linewidth": 0.8,
            "grid.color": "#E1CA9E",
            "grid.linewidth": 0.5,
        }
    )
    sns.set_theme(style="whitegrid", context="paper", font="Arial")


def save_many(fig: plt.Figure, name: str) -> None:
    from src.supplementary.jbd_figure_typography import apply_arial_to_figure

    apply_arial_to_figure(fig)
    bases = [FIG_DIR, FIG_ROOT, FIG_MAIN]
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
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left", fontfamily="Arial")


def plot_metric_dotplot(ax: plt.Axes, risk: pd.DataFrame) -> None:
    metrics = ["auc", "auprc", "f1", "sensitivity", "precision", "balanced_accuracy"]
    ybase = np.arange(len(metrics))[::-1]
    offsets = {
        "full_lcad_rasa_stablehash": 0.18,
        "semantic_retrieval_positive_ratio": 0.0,
        "kra_semantic_fusion": -0.18,
    }
    for _, row in risk.iterrows():
        model = row["model_id"]
        values = [float(row[m]) for m in metrics]
        y = ybase + offsets[model]
        ax.scatter(values, y, s=32, color=PALETTE[model], edgecolor="#3E3425", linewidth=0.35, label=LABELS[model], zorder=3)
    ax.set_yticks(ybase)
    ax.set_yticklabels([METRIC_LABELS[m] for m in metrics])
    ax.set_xlim(0.35, 0.95)
    ax.set_xlabel("Held-out test metric")
    ax.set_title("Multi-metric test profile", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.45)
    ax.grid(False, axis="y")
    sns.despine(ax=ax, left=False, bottom=False)


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
        ax.plot(fpr, tpr, color=PALETTE[model], linewidth=1.8, label=f"{LABELS[model]} ({auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#998560", linewidth=0.9)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Held-out ROC curves", fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.02, 0.38))
    sns.despine(ax=ax)


def plot_score_distribution(ax: plt.Axes, scores: pd.DataFrame) -> None:
    test = scores[scores["split"].eq("test")].copy()
    test["Outcome"] = np.where(test["y_true"].eq(1), "CIN2+", "CIN2-")
    sns.violinplot(
        data=test,
        x="Outcome",
        y="semantic_fusion_score",
        hue="Outcome",
        order=["CIN2-", "CIN2+"],
        palette=["#ADB093", "#E76B6B"],
        inner="quartile",
        cut=0,
        linewidth=0.8,
        legend=False,
        ax=ax,
    )
    rng = np.random.default_rng(42)
    sample = test.copy()
    if len(sample) > 180:
        sample = sample.groupby("Outcome", group_keys=False)[sample.columns].apply(lambda x: x.sample(min(len(x), 90), random_state=42))
    xmap = {"CIN2-": 0, "CIN2+": 1}
    ax.scatter(
        [xmap[o] + rng.normal(0, 0.035) for o in sample["Outcome"]],
        sample["semantic_fusion_score"],
        s=9,
        color="#3E3425",
        alpha=0.28,
        linewidth=0,
        zorder=3,
    )
    ax.axhline(0.50, color="#E76B6B", linestyle="--", linewidth=1.1)
    ax.text(
        0.06,
        0.515,
        "Val threshold 0.50",
        color="#E76B6B",
        fontsize=7,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    ax.set_xlabel("")
    ax.set_ylabel("MOSAIC score")
    ax.set_title("Risk-score separation by outcome", fontweight="bold")
    sns.despine(ax=ax)


def plot_centerwise(ax: plt.Axes, center: pd.DataFrame) -> None:
    view = center.dropna(subset=["auc", "baseline_auc"]).copy()
    view = view.sort_values("auc")
    y = np.arange(len(view))
    for i, row in enumerate(view.itertuples(index=False)):
        ax.plot([row.baseline_auc, row.auc], [i, i], color="#998560", linewidth=1.1, zorder=1)
        ax.scatter(row.baseline_auc, i + 0.065, s=34, color=PALETTE["full_lcad_rasa_stablehash"], edgecolor="#3E3425", linewidth=0.35, zorder=2)
        ax.scatter(row.auc, i - 0.065, s=40, color=PALETTE["kra_semantic_fusion"], edgecolor="#3E3425", linewidth=0.35, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(view["center_id"].str.title())
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Centre-wise AUROC")
    ax.set_title("Centre-wise AUROC shift", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.45)
    ax.grid(False, axis="y")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["full_lcad_rasa_stablehash"], markeredgecolor="#3E3425", label="MOSAIC--RASA backbone"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["kra_semantic_fusion"], markeredgecolor="#3E3425", label="MOSAIC (full)"),
        ],
        frameon=False,
        loc="lower right",
    )
    from src.supplementary.jbd_figure_stats import format_pvalue

    bootstrap = json.loads((ANALYSIS / "kra_semantic_fusion_vs_full_paired_auc_bootstrap.json").read_text()) if (ANALYSIS / "kra_semantic_fusion_vs_full_paired_auc_bootstrap.json").is_file() else {}
    p_pool = bootstrap.get("paired_bootstrap_p_two_sided")
    if p_pool is not None:
        ax.text(
            0.02,
            0.98,
            f"Pooled paired test:\n{format_pvalue(float(p_pool))}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            fontfamily="Arial",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F7F8FB", ec="#E1CA9E", alpha=0.9),
        )
    sns.despine(ax=ax, left=False, bottom=False)


def make_summary_figure(risk: pd.DataFrame, center: pd.DataFrame, scores: pd.DataFrame, bootstrap: dict) -> None:
    setup_style()
    fig = plt.figure(figsize=(8.4, 7.1), constrained_layout=True)
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
        f"MOSAIC improves held-out risk stratification (delta AUROC {delta:.3f}, 95% CI {lo:.3f} to {hi:.3f}, p={p:.3f})",
        y=1.015,
        fontsize=10,
        fontweight="bold",
    )
    save_many(fig, "Figure_mosaic_performance_summary")
    save_many(fig, "Figure_kra_semantic_fusion_summary")
    plt.close(fig)


def make_metric_lollipop(risk: pd.DataFrame) -> None:
    """Compact MOSAIC metric comparison — horizontal lollipop (not heatmap)."""
    setup_style()
    metrics = ["auc", "auprc", "f1", "sensitivity", "precision", "balanced_accuracy"]
    matrix = risk.set_index("model_id")[metrics].rename(index=LABELS, columns=METRIC_LABELS)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y_base = np.arange(len(matrix.columns))[::-1]
    offsets = {"MOSAIC--RASA backbone": 0.22, "Semantic retrieval only": 0.0, "MOSAIC (full)": -0.22}
    for model_id, color in PALETTE.items():
        label = LABELS[model_id]
        if label not in matrix.index:
            continue
        vals = matrix.loc[label].to_numpy(dtype=float)
        y = y_base + offsets.get(label, 0.0)
        ax.hlines(y, 0.35, vals, color=color, linewidth=1.4, alpha=0.55, zorder=1)
        ax.scatter(vals, y, s=58, color=color, edgecolor="#3E3425", linewidth=0.45, label=label, zorder=3)
    ax.set_yticks(y_base)
    ax.set_yticklabels(matrix.columns.tolist())
    ax.set_xlim(0.35, 0.95)
    ax.set_xlabel("Held-out test metric")
    ax.set_title("MOSAIC metric profile", fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.grid(True, axis="x", alpha=0.4)
    sns.despine(ax=ax, left=False)
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
