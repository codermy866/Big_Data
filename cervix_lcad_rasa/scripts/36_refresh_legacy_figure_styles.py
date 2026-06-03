#!/usr/bin/env python3
"""Replot legacy next-stage figures with Times New Roman + JBD palette (no retrain)."""

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
    C4,
    C6,
    PALETTE_MAIN,
    _cmap_diverging,
    _save,
    _setup_theme,
)
from src.utils.config import resolve_project_root


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
        ax.scatter(out[xcol], out["auc"], c=C0, edgecolors="#d9d8d8", s=80, linewidth=0.8)
        for _, r in out.iterrows():
            ax.annotate(str(r["lambda_align"]), (r[xcol], r["auc"]), fontsize=8, fontfamily="serif")
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
    sns.barplot(data=ref, x="experiment_id", y="auc", palette=PALETTE_MAIN, ax=ax, legend=False)
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
    sns.barplot(data=sub, x="experiment_id", y="f1", palette=PALETTE_MAIN, ax=ax, legend=False)
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
    sns.boxplot(data=raw, x=model_col, y="auc", palette=PALETTE_MAIN, ax=ax, linewidth=0.8)
    sns.stripplot(data=raw, x=model_col, y="auc", color=C6, size=5, alpha=0.6, ax=ax, jitter=0.15)
    ax.set_title("Multi-seed test AUROC")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, fig_dir / "fig_multiseed_auc_boxplot")


def main():
    project = resolve_project_root()
    fig_dir = project / "outputs/publishable/figures"
    _setup_theme()
    replot_rasa_pareto(project, fig_dir)
    replot_reference_stratified(project, fig_dir)
    replot_threshold_f1(project, fig_dir)
    replot_loco_strict_heatmap(project, fig_dir)
    replot_multiseed_boxplot(project, fig_dir)
    print(f"Legacy figures refreshed: {fig_dir}")
    print("Font: Times New Roman (serif); palette: #8b98b3 ... #d9d8d8")


if __name__ == "__main__":
    main()
