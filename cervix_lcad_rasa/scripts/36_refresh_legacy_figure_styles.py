#!/usr/bin/env python3
"""Replot legacy next-stage figures with Arial + journal palette (no retrain)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supplementary.jbd_figures_seaborn import (
    C0,
    C1,
    C2,
    C4,
    C6,
    EDGE_DARK,
    PALETTE_MAIN,
    TEXT_DARK,
    _cmap_diverging,
    _save,
    _setup_theme,
)


def _tables(project: Path) -> Path:
    return project / "outputs/publishable/tables"


def replot_rasa_pareto(project: Path, fig_dir: Path) -> None:
    p = _tables(project) / "table_rasa_loss_weight_sweep.csv"
    if not p.is_file():
        return
    out = pd.read_csv(p)
    _setup_theme()
    for xcol, stem, title in [
        ("section_completeness", "fig_rasa_pareto_auc_vs_section_alignment", "AUC vs section completeness"),
        ("hallucination_rate", "fig_rasa_pareto_auc_vs_safety", "AUC vs hallucination rate"),
    ]:
        if xcol not in out.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(out[xcol], out["auc"], c=C0, edgecolors=TEXT_DARK, s=90, linewidth=0.8)
        for _, r in out.iterrows():
            ax.annotate(str(r["lambda_align"]), (r[xcol], r["auc"]), fontsize=8, fontfamily="Arial", fontweight="bold", xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(xcol.replace("_", " "))
        ax.set_ylabel("AUROC")
        ax.set_title(title)
        fig.tight_layout()
        _save(fig, fig_dir / stem)


def replot_reference_stratified(project: Path, fig_dir: Path) -> None:
    p = _tables(project) / "table_reference_stratified_evaluation.csv"
    if not p.is_file():
        return
    out = pd.read_csv(p)
    ref = out[out["subset"] == "with_reference"]
    if not len(ref):
        return
    _setup_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    ref = ref.copy()
    ref["label"] = ref["experiment_id"].str.replace("_", " ").str.title()
    sns.barplot(data=ref, x="label", y="auc", palette=PALETTE_MAIN, ax=ax, edgecolor=EDGE_DARK, linewidth=0.9, alpha=0.86, legend=False)
    sns.stripplot(data=ref, x="label", y="auc", color=C4, marker="s", size=8, ax=ax, jitter=False)
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC by model (reference-available subset)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save(fig, fig_dir / "fig_reference_available_metric_comparison")


def replot_threshold_f1(project: Path, fig_dir: Path) -> None:
    p = _tables(project) / "table_threshold_tuned_test_metrics.csv"
    if not p.is_file():
        return
    out = pd.read_csv(p)
    sub = out[out["threshold_type"] == "max_f1"]
    if not len(sub):
        return
    _setup_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = sub.copy()
    sub["label"] = sub["experiment_id"].str.replace("_", " ").str.title()
    sns.barplot(data=sub, x="label", y="f1", palette=PALETTE_MAIN, ax=ax, edgecolor=EDGE_DARK, linewidth=0.9, alpha=0.86, legend=False)
    sns.stripplot(data=sub, x="label", y="f1", color=C4, marker="^", size=8, ax=ax, jitter=False)
    ax.set_title("Test F1 at validation-selected max-F1 threshold")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save(fig, fig_dir / "fig_threshold_f1_curves")


def replot_loco_strict_heatmap(project: Path, fig_dir: Path) -> None:
    p = _tables(project) / "manuscript/S2_loco_strict_retrain.csv"
    if not p.is_file():
        p = _tables(project) / "table_loco_strict_main_results.csv"
    if not p.is_file():
        return
    out = pd.read_csv(p)
    if "auc" not in out.columns:
        return
    col = "held_out_center" if "held_out_center" in out.columns else "center_label"
    piv = out.pivot_table(index="model", columns=col, values="auc", aggfunc="mean")
    _setup_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap=_cmap_diverging(), vmin=0.4, vmax=1.0, ax=ax, linewidths=0.5)
    ax.set_title("Strict LOCO — test AUROC")
    fig.tight_layout()
    _save(fig, fig_dir / "fig_loco_strict_center_heatmap")


def replot_multiseed_boxplot(project: Path, fig_dir: Path) -> None:
    p = _tables(project) / "table_multiseed_raw.csv"
    if not p.is_file():
        p = _tables(project) / "manuscript/S7_multiseed_stability.csv"
    if not p.is_file():
        return
    raw = pd.read_csv(p)
    if "auc" not in raw.columns:
        return
    _setup_theme()
    fig, ax = plt.subplots(figsize=(7, 5))
    model_col = "model" if "model" in raw.columns else "experiment_id"
    sns.boxplot(data=raw, x=model_col, y="auc", palette=[C1, C3] if "C3" in globals() else PALETTE_MAIN, ax=ax, linewidth=0.9, fliersize=0)
    sns.stripplot(data=raw, x=model_col, y="auc", color=C6, size=5.5, alpha=0.7, ax=ax, jitter=0.15)
    ax.set_title("Multi-seed test AUROC")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, fig_dir / "fig_multiseed_auc_boxplot")


def main():
    project = ROOT
    fig_dir = project / "outputs/publishable/figures"
    _setup_theme()
    replot_rasa_pareto(project, fig_dir)
    replot_reference_stratified(project, fig_dir)
    replot_threshold_f1(project, fig_dir)
    replot_loco_strict_heatmap(project, fig_dir)
    replot_multiseed_boxplot(project, fig_dir)
    print(f"Legacy figures refreshed: {fig_dir}")
    print("Font: Arial; palette: #576fa0 #a7b9d7 #e3b87f #fadcb4 #b57979 #dea3a2 #9f9f9f #cfcece")


if __name__ == "__main__":
    main()
