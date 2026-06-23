#!/usr/bin/env python3
"""Regenerate external-baseline AUROC forest plot including MOSAIC (full) and stable-hash backbone."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.supplementary.jbd_figure_typography import FONT_ARIAL, apply_arial_to_figure, setup_arial_rcparams
PROJECT_FIG = ROOT.parent / "figures"
EXTERNAL_TABLE = ROOT / "outputs/publishable/tables/manuscript/T_external_baselines_same_split.csv"
FUSION_SCORES = ROOT / "outputs/publishable/kra_semantic_fusion_analysis/kra_semantic_fusion_val_test_scores.csv"
OUT_DIR = ROOT / "outputs/publishable/external_baselines/figures"
MANUSCRIPT_TABLE = ROOT / "outputs/publishable/tables/manuscript/T_external_baselines_same_split_with_mosaic.csv"

PALETTE = ["#2f5f8f", "#8fb8d8", "#d9a066", "#9e3f3a", "#7f7f7f"]
HIGHLIGHT_FULL = "#9e3f3a"
HIGHLIGHT_BACKBONE = "#2f5f8f"
TEXT = "#343434"
FONT = "Arial"


def setup_theme() -> None:
    setup_arial_rcparams(
        {
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "grid.color": "#d6d6d6",
        }
    )
    sns.set_theme(style="whitegrid", context="talk", font=FONT_ARIAL, palette=PALETTE)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    apply_arial_to_figure(fig)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    if tp[-1] == 0 or fp[-1] == 0:
        return float("nan")
    tpr = tp / tp[-1]
    fpr = fp / fp[-1]
    return float(np.trapz(tpr, fpr))


def f1_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> float:
    pred = (np.asarray(s) >= thr).astype(int)
    y = np.asarray(y, dtype=int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def bootstrap_ci(y: np.ndarray, s: np.ndarray, metric: str = "auc", n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        if metric == "auc":
            vals.append(roc_auc(yb, sb))
        else:
            vals.append(f1_at_threshold(yb, sb, 0.5))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    point = roc_auc(y, s) if metric == "auc" else f1_at_threshold(y, s, 0.5)
    return point, float(lo), float(hi)


def mosaic_rows(test: pd.DataFrame) -> list[dict]:
    rows = []
    specs = [
        ("MOSAIC (full)", "semantic_fusion_score", HIGHLIGHT_FULL, 0.50),
        ("MOSAIC--RASA backbone (stable-hash)", "risk_score", HIGHLIGHT_BACKBONE, 0.39),
    ]
    y = test["y_true"].to_numpy(dtype=int)
    for model, col, color, thr in specs:
        s = test[col].to_numpy(dtype=float)
        auc, auc_lo, auc_hi = bootstrap_ci(y, s, metric="auc")
        f1 = f1_at_threshold(y, s, thr)
        _, f1_lo, f1_hi = bootstrap_ci(y, s, metric="f1")
        rows.append(
            {
                "baseline_id": model.lower().replace(" ", "_").replace("(", "").replace(")", ""),
                "model": model,
                "model_short": model,
                "auc": auc,
                "f1": f1,
                "threshold_val_max_f1": thr,
                "auc_ci_low": auc_lo,
                "auc_ci_high": auc_hi,
                "f1_ci_low": f1_lo,
                "f1_ci_high": f1_hi,
                "highlight_color": color,
                "is_mosaic": True,
            }
        )
    return rows


def build_plot_table() -> pd.DataFrame:
    ext = pd.read_csv(EXTERNAL_TABLE)
    ext = ext[~ext["baseline_id"].eq("full_lcad_rasa_reference")].copy()
    ext["model_short"] = ext["model"]
    ext["highlight_color"] = "#8fb8d8"
    ext["is_mosaic"] = False

    test = pd.read_csv(FUSION_SCORES)
    test = test[test["split"] == "test"].copy()
    mosaic = pd.DataFrame(mosaic_rows(test))

    table = pd.concat([ext, mosaic], ignore_index=True)
    table = table.sort_values("auc", ascending=True).reset_index(drop=True)
    table.to_csv(MANUSCRIPT_TABLE, index=False)
    return table


def plot_forest(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.4))

    ci_max = float(table["auc_ci_high"].max())
    x_left = 0.74
    # Fixed right column for AUROC labels — keeps numbers clear of CI bars and markers.
    label_x = ci_max + 0.040

    for i, (_, r) in enumerate(table.iterrows()):
        color = r["highlight_color"] if bool(r["is_mosaic"]) else PALETTE[i % len(PALETTE)]
        lw = 2.4 if bool(r["is_mosaic"]) else 1.6
        ms = 150 if r["model_short"] == "MOSAIC (full)" else (125 if bool(r["is_mosaic"]) else 95)
        ax.plot(
            [r["auc_ci_low"], r["auc_ci_high"]],
            [i, i],
            color=TEXT,
            lw=lw,
            alpha=0.9,
            zorder=1,
            solid_capstyle="round",
        )
        ax.scatter(
            r["auc"],
            i,
            s=ms,
            color=color,
            edgecolor=TEXT,
            linewidth=0.9,
            zorder=3,
        )
        weight = "bold" if bool(r["is_mosaic"]) else "normal"
        ax.text(
            label_x,
            i,
            f"{float(r['auc']):.3f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight=weight,
            fontfamily=FONT,
            clip_on=False,
            zorder=4,
        )

    ax.set_yticks(range(len(table)))
    labels = []
    label_weights = []
    for _, r in table.iterrows():
        labels.append(str(r["model_short"]))
        label_weights.append("bold" if r["model_short"] == "MOSAIC (full)" else "normal")

    ax.set_yticklabels(labels, fontfamily=FONT, fontsize=10)
    for tick, weight in zip(ax.get_yticklabels(), label_weights):
        tick.set_fontweight(weight)

    ax.set_xlabel("Held-out AUROC with 95% bootstrap CI", fontfamily=FONT, fontsize=11)
    ax.set_title(
        "Same-split AUROC: MOSAIC (full), backbone, and external baselines",
        fontfamily=FONT,
        fontsize=13,
        pad=12,
    )
    ax.set_xlim(x_left, label_x + 0.065)
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    for tick in ax.get_xticklabels():
        tick.set_fontfamily(FONT)
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.36, right=0.82, top=0.90, bottom=0.12)
    out = OUT_DIR / "Figure_external_baselines_auc_forest"
    save_fig(fig, out)

    PROJECT_FIG.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        src = out.with_suffix(suffix)
        dst = PROJECT_FIG / f"Figure_external_baselines_auc_forest{suffix}"
        try:
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
        except shutil.SameFileError:
            pass
    print(f"Wrote {out.with_suffix('.pdf')}")


def main() -> None:
    setup_theme()
    table = build_plot_table()
    plot_forest(table)
    print(table[["model_short", "auc", "auc_ci_low", "auc_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
