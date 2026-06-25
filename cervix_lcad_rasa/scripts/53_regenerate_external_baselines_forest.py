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
from src.supplementary.jbd_figure_typography import (
    FONT_ARIAL,
    FONT_TIMES,
    apply_arial_to_figure,
    apply_mixed_en_typography,
    setup_arial_rcparams,
)
PROJECT_FIG = ROOT.parent / "figures"
FINAL_FIG = ROOT.parent / "final_Fig"
EXTERNAL_TABLE = ROOT / "outputs/publishable/tables/manuscript/T_external_baselines_same_split.csv"
FUSION_SCORES = ROOT / "outputs/publishable/kra_semantic_fusion_analysis/kra_semantic_fusion_val_test_scores.csv"
OUT_DIR = ROOT / "outputs/publishable/external_baselines/figures"
MANUSCRIPT_TABLE = ROOT / "outputs/publishable/tables/manuscript/T_external_baselines_same_split_with_mosaic.csv"

PALETTE = ["#254B6D", "#95A1B2", "#D6DEE8", "#C65A46", "#7D8793"]
HIGHLIGHT_FULL = "#C65A46"
HIGHLIGHT_BACKBONE = "#254B6D"
TEXT = "#17212B"
GRID = "#E2E7EE"
REF = "#95A1B2"
FONT = FONT_ARIAL
NUM_FONT = FONT_TIMES


def setup_theme() -> None:
    setup_arial_rcparams(
        {
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.82,
            "axes.titlesize": 13,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
        }
    )
    sns.set_theme(style="whitegrid", context="talk", font=FONT_ARIAL, palette=PALETTE)


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig._jbd_mixed_en_typography = True
    if not hasattr(fig, "_jbd_min_font_size_override"):
        fig._jbd_min_font_size_override = 9.2
    if not hasattr(fig, "_jbd_max_font_size_override"):
        fig._jbd_max_font_size_override = 15.5
    apply_arial_to_figure(fig)
    apply_mixed_en_typography(fig)
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


def _model_group(row: pd.Series) -> str:
    bid = str(row.get("baseline_id", ""))
    model = str(row.get("model_short", row.get("model", "")))
    if bool(row.get("is_mosaic", False)):
        return "MOSAIC pipeline"
    if bid in {"clinical_lr", "clinical_hist_gradient_boosting"}:
        return "Clinical-only ML"
    if bid in {"oct_only_embedding_mlp", "colposcopy_only_embedding_mlp"}:
        return "Single-modality image"
    if bid in {"late_fusion_mlp", "cross_attention_multimodal_transformer", "contrastive_multimodal_no_report_sections"}:
        return "Multimodal neural baselines"
    if "clinical" in model.lower():
        return "Clinical-only ML"
    return "External baselines"


def _short_label(model: str) -> str:
    mapping = {
        "MOSAIC (full)": "MOSAIC full",
        "MOSAIC--RASA backbone (stable-hash)": "RASA backbone",
        "Clinical-only logistic regression": "Clinical LR",
        "Clinical-only HistGradientBoosting": "Clinical HGB",
        "OCT-only embedding MLP": "OCT-only",
        "Colposcopy-only embedding MLP": "Colposcopy-only",
        "Late-fusion MLP": "Late fusion",
        "Cross-attention multimodal transformer": "Cross-attention",
        "CLIP-style contrastive multimodal baseline": "CLIP-style",
    }
    return mapping.get(model, model)


def _model_color(row: pd.Series) -> str:
    if str(row.get("model_short", "")) == "MOSAIC (full)":
        return HIGHLIGHT_FULL
    if bool(row.get("is_mosaic", False)):
        return HIGHLIGHT_BACKBONE
    group = str(row.get("model_group", ""))
    return {
        "Clinical-only ML": "#7D8793",
        "Single-modality image": "#D6DEE8",
        "Multimodal neural baselines": "#95A1B2",
    }.get(group, "#95A1B2")


def _prepare_plot_table(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    plot = table.copy()
    plot["model_group"] = plot.apply(_model_group, axis=1)
    plot["display_label"] = plot["model_short"].map(_short_label)
    plot["point_color"] = plot.apply(_model_color, axis=1)
    order = [
        "MOSAIC pipeline",
        "Multimodal neural baselines",
        "Clinical-only ML",
        "Single-modality image",
    ]
    order = [g for g in order if g in set(plot["model_group"])]
    plot["model_group"] = pd.Categorical(plot["model_group"], categories=order, ordered=True)
    plot = plot.sort_values(["model_group", "auc"], ascending=[True, True]).reset_index(drop=True)
    return plot, order


def _group_color(group: str) -> str:
    return {
        "MOSAIC pipeline": HIGHLIGHT_FULL,
        "Multimodal neural baselines": "#95A1B2",
        "Clinical-only ML": "#7D8793",
        "Single-modality image": "#254B6D",
    }.get(group, REF)


def _save_named_figure(fig: plt.Figure, name: str) -> None:
    out = OUT_DIR / name
    save_fig(fig, out)
    PROJECT_FIG.mkdir(parents=True, exist_ok=True)
    FINAL_FIG.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        src = out.with_suffix(suffix)
        for dst_dir in (PROJECT_FIG, FINAL_FIG):
            dst = dst_dir / f"{name}{suffix}"
            try:
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
            except shutil.SameFileError:
                pass
    print(f"Wrote {out.with_suffix('.pdf')}")


def plot_forest(table: pd.DataFrame) -> None:
    plot, order = _prepare_plot_table(table)
    y_map = {g: i for i, g in enumerate(order)}
    offsets: dict[str, float] = {}
    for group, sub in plot.groupby("model_group"):
        vals = np.linspace(-0.20, 0.20, len(sub)) if len(sub) > 1 else np.array([0.0])
        for (_, row), off in zip(sub.sort_values("auc").iterrows(), vals):
            offsets[str(row["display_label"])] = float(off)

    fig, ax = plt.subplots(figsize=(12.2, 5.85))
    ax.set_facecolor("white")
    for group in order:
        y = y_map[group]
        sub = plot[plot["model_group"].eq(group)].sort_values("auc")
        mean_auc = float(sub["auc"].mean())
        ax.scatter(
            mean_auc,
            y,
            s=92,
            marker="D",
            color=TEXT,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label="Family mean" if group == order[0] else None,
        )
        for _, r in sub.iterrows():
            label = str(r["display_label"])
            yj = y + offsets[label]
            lw = 2.5 if bool(r["is_mosaic"]) else 1.65
            alpha = 0.90 if bool(r["is_mosaic"]) else 0.58
            ax.plot(
                [r["auc_ci_low"], r["auc_ci_high"]],
                [yj, yj],
                color=REF,
                lw=lw,
                alpha=alpha,
                zorder=1,
                solid_capstyle="round",
            )
            ax.scatter(
                r["auc"],
                yj,
                s=132 if bool(r["is_mosaic"]) else 88,
                color=r["point_color"],
                edgecolor=TEXT,
                linewidth=0.85,
                zorder=4,
                label="MOSAIC models" if bool(r["is_mosaic"]) and label == "MOSAIC full" else None,
            )
            x_text = float(r["auc"]) + 0.006
            label_ha = "left"
            num_x = min(x_text + 0.0032 * min(len(label), 16), 0.958)
            if label == "Cross-attention":
                x_text = float(r["auc"]) - 0.006
                label_ha = "right"
                num_x = float(r["auc"]) + 0.055
            ax.text(
                x_text,
                yj,
                label,
                va="center",
                ha=label_ha,
                fontsize=9.1 if bool(r["is_mosaic"]) else 8.7,
                fontweight="bold" if bool(r["is_mosaic"]) else "normal",
                fontfamily=FONT,
                color=TEXT,
                clip_on=False,
                zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
            )
            ax.text(
                num_x,
                yj,
                f"{float(r['auc']):.3f}",
                va="center",
                ha="left",
                fontsize=9.1 if bool(r["is_mosaic"]) else 8.7,
                fontweight="bold" if bool(r["is_mosaic"]) else "normal",
                fontfamily=NUM_FONT,
                color=TEXT,
                clip_on=False,
                zorder=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.8},
            )

    ax.set_yticks([y_map[g] for g in order])
    ax.set_yticklabels(order, fontfamily=FONT, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Held-out AUROC with 95% bootstrap CI", fontfamily=FONT, fontsize=11.5, fontweight="bold")
    ax.set_title(
        "Same-split AUROC by model family",
        fontfamily=FONT,
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(0.70, 0.985)
    ax.set_ylim(len(order) - 0.45, -0.55)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    for tick in ax.get_xticklabels():
        tick.set_fontfamily(FONT)
    ax.grid(True, axis="x", color=GRID, linewidth=0.9, alpha=0.82)
    ax.grid(False, axis="y")
    ax.text(
        0.985,
        0.035,
        "circle = model AUROC\nline = 95% CI\ndiamond = family mean",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
        fontfamily=FONT,
        color=TEXT,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
    )
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.24, right=0.94, top=0.88, bottom=0.15)
    _save_named_figure(fig, "Figure_external_baselines_auc_forest")


def plot_ecdf_facets(table: pd.DataFrame) -> None:
    plot, order = _prepare_plot_table(table)
    fig, axes = plt.subplots(1, len(order), figsize=(13.4, 5.15), sharex=True, sharey=True)
    fig._jbd_min_font_size_override = 10.0
    fig._jbd_max_font_size_override = 16.0
    if len(order) == 1:
        axes = [axes]

    for ax, group in zip(axes, order):
        sub = plot[plot["model_group"].eq(group)].sort_values("auc").reset_index(drop=True)
        vals = sub["auc"].to_numpy(dtype=float)
        y = np.arange(1, len(vals) + 1, dtype=float) / len(vals)
        color = _group_color(group)
        ax.step(vals, y, where="post", color=color, linewidth=2.8, alpha=0.95)
        ax.scatter(vals, y, s=112, color=sub["point_color"], edgecolor=TEXT, linewidth=0.85, zorder=4)
        for i, row in sub.iterrows():
            x = float(row["auc"])
            x_text = x + 0.0035
            ha = "left"
            if x >= 0.885:
                x_text = x - 0.0038
                ha = "right"
            ax.text(
                x_text,
                y[i],
                str(row["display_label"]),
                ha=ha,
                va="center",
                fontsize=9.8,
                fontweight="bold" if bool(row["is_mosaic"]) else "normal",
                fontfamily=FONT,
                color=TEXT,
                clip_on=False,
            )
        ax.set_title(group, fontsize=13.8, fontweight="bold", fontfamily=FONT, pad=10)
        ax.set_xlabel("AUROC", fontsize=12.2, fontweight="bold", fontfamily=FONT)
        ax.set_xlim(0.775, 0.945)
        ax.set_ylim(0.0, 1.04)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax.grid(True, axis="x", color=GRID, linewidth=0.9, alpha=0.78)
        ax.grid(True, axis="y", color=GRID, linewidth=0.72, alpha=0.48)

    axes[0].set_ylabel("Cumulative model proportion", fontsize=12.2, fontweight="bold", fontfamily=FONT)
    fig.suptitle("Facetted ECDF of held-out AUROC by model family", fontsize=15.6, fontweight="bold", fontfamily=FONT)
    sns.despine(fig=fig)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.18, wspace=0.18)
    _save_named_figure(fig, "Figure_external_baselines_auc_forest_ecdf")


def plot_horizontal_box_observations(table: pd.DataFrame) -> None:
    plot, order = _prepare_plot_table(table)
    fig, ax = plt.subplots(figsize=(12.2, 6.0))
    fig._jbd_min_font_size_override = 10.0
    fig._jbd_max_font_size_override = 16.0
    box_palette = {
        "MOSAIC pipeline": "#E9B5A9",
        "Multimodal neural baselines": "#D4DCE6",
        "Clinical-only ML": "#E2E5EA",
        "Single-modality image": "#EAF0F4",
    }
    sns.boxplot(
        data=plot,
        y="model_group",
        x="auc",
        hue="model_group",
        order=order,
        hue_order=order,
        orient="h",
        ax=ax,
        width=0.48,
        showfliers=False,
        linewidth=1.0,
        palette=box_palette,
        legend=False,
        boxprops={"edgecolor": TEXT, "alpha": 0.88},
        medianprops={"color": TEXT, "linewidth": 1.3},
        whiskerprops={"color": TEXT, "linewidth": 0.9},
        capprops={"color": TEXT, "linewidth": 0.9},
    )

    y_map = {g: i for i, g in enumerate(order)}
    for group in order:
        sub = plot[plot["model_group"].eq(group)].sort_values("auc").reset_index(drop=True)
        offsets = np.linspace(-0.26, 0.26, len(sub)) if len(sub) > 1 else np.array([0.0])
        for offset, (_, row) in zip(offsets, sub.iterrows()):
            y = y_map[group] + float(offset)
            x = float(row["auc"])
            err_low = x - float(row["auc_ci_low"])
            err_high = float(row["auc_ci_high"]) - x
            ax.errorbar(
                x,
                y,
                xerr=np.array([[err_low], [err_high]]),
                fmt="none",
                ecolor=REF,
                elinewidth=2.5 if bool(row["is_mosaic"]) else 1.55,
                capsize=3.0,
                alpha=0.88 if bool(row["is_mosaic"]) else 0.58,
                zorder=3,
            )
            ax.scatter(
                x,
                y,
                s=158 if bool(row["is_mosaic"]) else 92,
                color=row["point_color"],
                edgecolor=TEXT,
                linewidth=0.9,
                zorder=5,
            )
            label_x = min(x + 0.006, 0.936)
            label = str(row["display_label"])
            ax.text(
                label_x,
                y,
                label,
                ha="left",
                va="center",
                fontsize=9.8 if bool(row["is_mosaic"]) else 9.3,
                fontweight="bold" if bool(row["is_mosaic"]) else "normal",
                fontfamily=FONT,
                color=TEXT,
                clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
            )
            num_x = min(label_x + 0.0032 * min(len(label), 16), 0.958)
            ax.text(
                num_x,
                y,
                f"{x:.3f}",
                ha="left",
                va="center",
                fontsize=9.8 if bool(row["is_mosaic"]) else 9.3,
                fontweight="bold" if bool(row["is_mosaic"]) else "normal",
                fontfamily=NUM_FONT,
                color=TEXT,
                clip_on=False,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
            )

    ax.set_xlabel("Held-out AUROC with 95% bootstrap CI", fontsize=12.6, fontweight="bold", fontfamily=FONT)
    ax.set_ylabel("")
    ax.set_title("Horizontal boxplot with model-level observations", fontsize=15.6, fontweight="bold", fontfamily=FONT, pad=13)
    ax.set_xlim(0.70, 0.995)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.grid(True, axis="x", color=GRID, linewidth=0.9, alpha=0.82)
    ax.grid(False, axis="y")
    for label in ax.get_yticklabels():
        label.set_fontfamily(FONT)
        label.set_fontweight("bold")
        label.set_fontsize(12.0)
    for label in ax.get_xticklabels():
        label.set_fontfamily(NUM_FONT)
        label.set_fontsize(11.2)
    ax.text(
        0.985,
        0.055,
        "box = within-family AUROC spread\npoint = individual model\nline = 95% bootstrap CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.4,
        fontfamily=FONT,
        color=TEXT,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
    )
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.22, right=0.935, top=0.86, bottom=0.16)
    _save_named_figure(fig, "Figure_external_baselines_auc_forest_box_observations")


def plot_line_facets(table: pd.DataFrame) -> None:
    plot, order = _prepare_plot_table(table)
    fig, axes = plt.subplots(1, len(order), figsize=(13.4, 5.15), sharey=True)
    fig._jbd_min_font_size_override = 10.0
    fig._jbd_max_font_size_override = 16.0
    if len(order) == 1:
        axes = [axes]

    for ax, group in zip(axes, order):
        sub = plot[plot["model_group"].eq(group)].sort_values("auc").reset_index(drop=True)
        sub["rank"] = np.arange(1, len(sub) + 1)
        color = _group_color(group)
        ax.plot(
            sub["rank"],
            sub["auc"],
            color=color,
            linewidth=2.8,
            marker="o",
            markersize=7.2,
            markeredgecolor=TEXT,
            markeredgewidth=0.8,
        )
        for _, row in sub.iterrows():
            ax.vlines(
                row["rank"],
                row["auc_ci_low"],
                row["auc_ci_high"],
                color=REF,
                linewidth=2.0 if bool(row["is_mosaic"]) else 1.35,
                alpha=0.80 if bool(row["is_mosaic"]) else 0.55,
                zorder=1,
            )
            ax.text(
                float(row["rank"]) + 0.04,
                float(row["auc"]),
                str(row["display_label"]),
                ha="left",
                va="center",
                fontsize=9.4,
                fontweight="bold" if bool(row["is_mosaic"]) else "normal",
                fontfamily=FONT,
                color=TEXT,
            )
        ax.set_title(group, fontsize=13.8, fontweight="bold", fontfamily=FONT, pad=10)
        ax.set_xlabel("Rank within family", fontsize=12.0, fontweight="bold", fontfamily=FONT)
        ax.set_xticks(sub["rank"])
        ax.set_xlim(0.78, max(3.20, len(sub) + 0.38))
        ax.set_ylim(0.74, 0.94)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.grid(True, axis="y", color=GRID, linewidth=0.9, alpha=0.78)
        ax.grid(False, axis="x")
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily(NUM_FONT)
            tick.set_fontsize(11.0)

    axes[0].set_ylabel("Held-out AUROC", fontsize=12.2, fontweight="bold", fontfamily=FONT)
    fig.suptitle("Line plots on multiple facets: AUROC ordered within each family", fontsize=15.6, fontweight="bold", fontfamily=FONT)
    sns.despine(fig=fig)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.18, wspace=0.18)
    _save_named_figure(fig, "Figure_external_baselines_auc_forest_facet_lines")


def plot_wide_lineplot(table: pd.DataFrame) -> None:
    plot, order = _prepare_plot_table(table)
    family_rows = []
    for group in order:
        sub = plot[plot["model_group"].eq(group)].sort_values("auc").reset_index(drop=True)
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            family_rows.append({"rank": rank, "family": group, "auc": float(row["auc"]), "label": row["display_label"]})
    long = pd.DataFrame(family_rows)
    wide = long.pivot(index="rank", columns="family", values="auc")

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    fig._jbd_min_font_size_override = 10.0
    fig._jbd_max_font_size_override = 16.0
    for family in wide.columns:
        ax.plot(
            wide.index,
            wide[family],
            marker="o",
            markersize=7.3,
            linewidth=2.8,
            color=_group_color(family),
            label=family,
        )
        sub = long[long["family"].eq(family)]
        for _, row in sub.iterrows():
            ax.text(
                float(row["rank"]) + 0.04,
                float(row["auc"]),
                str(row["label"]),
                fontsize=9.4,
                fontfamily=FONT,
                color=TEXT,
                va="center",
                ha="left",
            )

    ax.set_title("Wide-form lineplot of rank-ordered AUROC", fontsize=15.6, fontweight="bold", fontfamily=FONT, pad=12)
    ax.set_xlabel("Rank within model family", fontsize=12.4, fontweight="bold", fontfamily=FONT)
    ax.set_ylabel("Held-out AUROC", fontsize=12.4, fontweight="bold", fontfamily=FONT)
    ax.set_xticks(sorted(long["rank"].unique()))
    ax.set_xlim(0.86, float(long["rank"].max()) + 0.55)
    ax.set_ylim(0.76, 0.925)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.legend(frameon=False, loc="lower right", fontsize=10.2, title=None)
    ax.grid(True, color=GRID, linewidth=0.9, alpha=0.82)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily(NUM_FONT)
        tick.set_fontsize(11.0)
    sns.despine(fig=fig, ax=ax)
    fig.subplots_adjust(left=0.11, right=0.86, top=0.86, bottom=0.15)
    _save_named_figure(fig, "Figure_external_baselines_auc_forest_wide_lineplot")


def plot_style_candidates(table: pd.DataFrame) -> None:
    plot_ecdf_facets(table)
    plot_horizontal_box_observations(table)
    plot_line_facets(table)
    plot_wide_lineplot(table)


def main() -> None:
    setup_theme()
    table = build_plot_table()
    plot_forest(table)
    plot_style_candidates(table)
    print(table[["model_short", "auc", "auc_ci_low", "auc_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
