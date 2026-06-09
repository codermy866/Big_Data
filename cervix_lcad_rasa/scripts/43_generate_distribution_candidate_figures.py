#!/usr/bin/env python3
"""Generate candidate box/violin figures from existing publishable results."""

from __future__ import annotations

from pathlib import Path
import sys

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
    C5,
    C7,
    EDGE_DARK,
    TEXT_DARK,
    _save,
    _setup_theme,
)


OUT_DIR = ROOT / "outputs/publishable/figures/distribution_candidates"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _risk_score_violin() -> Path:
    pred_dir = ROOT / "outputs/publishable/predictions/final_per_case"
    files = {
        "Full LCAD-RASA": pred_dir / "full_lcad_rasa_test_predictions.csv",
        "Pseudo-augmented LCAD": pred_dir / "pseudo_augmented_lcad_test_predictions.csv",
        "Real-report only": pred_dir / "real_report_only_decoder_test_predictions.csv",
        "Simple concat fusion": pred_dir / "simple_concat_fusion_test_predictions.csv",
    }
    frames = []
    for model, path in files.items():
        if not path.is_file():
            continue
        d = _read(path)
        d = d[["case_id", "center", "y_true_cin2plus", "risk_score", "threshold_val_selected"]].copy()
        d["Model"] = model
        d["Outcome"] = d["y_true_cin2plus"].map({0: "CIN2- / negative", 1: "CIN2+ / positive"})
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    order = list(files.keys())
    hue_order = ["CIN2- / negative", "CIN2+ / positive"]

    fig, ax = plt.subplots(figsize=(9.8, 5.7))
    sns.violinplot(
        data=data,
        y="Model",
        x="risk_score",
        hue="Outcome",
        order=order,
        hue_order=hue_order,
        split=True,
        inner="quartile",
        cut=0,
        linewidth=0.95,
        density_norm="width",
        palette=[C0, C4],
        saturation=0.92,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        y="Model",
        x="risk_score",
        hue="Outcome",
        order=order,
        hue_order=hue_order,
        dodge=True,
        palette=[C0, C4],
        alpha=0.18,
        size=2.0,
        linewidth=0,
        ax=ax,
        legend=False,
    )
    for y, model in enumerate(order):
        threshold = float(data.loc[data["Model"].eq(model), "threshold_val_selected"].median())
        ax.plot([threshold, threshold], [y - 0.38, y + 0.38], color=TEXT_DARK, linewidth=1.0, linestyle="--", alpha=0.72)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], frameon=False, loc="lower right", title="")
    ax.set_title("Held-out risk-score distributions by outcome", fontsize=14.5, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted risk score")
    ax.set_ylabel("")
    ax.grid(axis="x", color=C7, alpha=0.42)
    ax.grid(axis="y", color=C7, alpha=0.20)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(EDGE_DARK)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.14)
    stem = OUT_DIR / "Figure_candidate_risk_score_violin_by_outcome"
    _save(fig, stem)
    return stem.with_suffix(".png")


def _scarcity_seed_boxplot() -> Path:
    path = ROOT / "outputs/publishable/theme1_alignment/tables/T_theme1_report_supervision_scarcity_curve_raw.csv"
    data = _read(path)
    data["Setup"] = data["setup"].map(
        {
            "real_report_only_surrogate": "Real-report only",
            "lcad_augmented_surrogate": "LCAD-augmented",
        }
    ).fillna(data["setup"])
    data["Report fraction"] = data["real_report_fraction"].map({0.1: "10%", 0.25: "25%", 0.5: "50%", 1.0: "100%"})
    order = ["10%", "25%", "50%", "100%"]
    hue_order = ["Real-report only", "LCAD-augmented"]

    fig, ax = plt.subplots(figsize=(8.5, 5.1))
    sns.boxplot(
        data=data,
        x="Report fraction",
        y="auc",
        hue="Setup",
        order=order,
        hue_order=hue_order,
        palette=[C1, C4],
        width=0.58,
        linewidth=0.95,
        fliersize=0,
        ax=ax,
        boxprops={"alpha": 0.72, "edgecolor": EDGE_DARK},
        medianprops={"color": TEXT_DARK, "linewidth": 1.2},
        whiskerprops={"color": EDGE_DARK, "linewidth": 0.95},
        capprops={"color": EDGE_DARK, "linewidth": 0.95},
    )
    sns.stripplot(
        data=data,
        x="Report fraction",
        y="auc",
        hue="Setup",
        order=order,
        hue_order=hue_order,
        dodge=True,
        palette=[C1, C4],
        marker="o",
        size=4.2,
        alpha=0.86,
        edgecolor=TEXT_DARK,
        linewidth=0.55,
        ax=ax,
        legend=False,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], frameon=False, loc="lower right", title="")
    ax.set_title("Report-scarcity AUROC across random seeds", fontsize=14.5, fontweight="bold", pad=12)
    ax.set_xlabel("Available real-report supervision")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.58, 0.88)
    ax.grid(axis="y", color=C7, alpha=0.42)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.87, bottom=0.15)
    stem = OUT_DIR / "Figure_candidate_scarcity_seed_boxplot"
    _save(fig, stem)
    return stem.with_suffix(".png")


def main() -> None:
    _setup_theme()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = [_risk_score_violin(), _scarcity_seed_boxplot()]
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
