"""Seaborn gallery-style ablation figures (JBD palette)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.supplementary.jbd_figures_seaborn import (
    C0,
    C1,
    C2,
    C4,
    C6,
    C7,
    EDGE_DARK,
    JBD_PALETTE_HEX,
    PALETTE_MAIN,
    TEXT_DARK,
    _cmap_diverging,
    _setup_theme,
)

MANUSCRIPT_REL = "outputs/publishable/tables/manuscript"
OUT_REL = "outputs/publishable/figures/ablation"


def _save(fig: plt.Figure, path: Path) -> None:
    from src.supplementary.jbd_figure_typography import apply_arial_to_figure

    path.parent.mkdir(parents=True, exist_ok=True)
    apply_arial_to_figure(fig)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    try:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.08)
    except Exception:
        pass
    plt.close(fig)


def _read(project: Path, name: str) -> pd.DataFrame | None:
    p = project / MANUSCRIPT_REL / name
    return pd.read_csv(p) if p.is_file() else None


def _pretty_id(s: str) -> str:
    return s.replace("_", " ").replace("no ", "w/o ").title()


def fig_modality_barplot(project: Path, out_dir: Path) -> None:
    df = _read(project, "S3_modality_ablation.csv")
    if df is None:
        return
    _setup_theme()
    df = df.copy()
    df["label"] = df["experiment_id"].map(_pretty_id)
    ref = df[df["experiment_id"] == "full_with_fused"]["auc"].iloc[0]
    df["delta_auc"] = df["auc"] - ref
    df = df.sort_values("auc", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    colors = [C4 if d < -0.03 else (C0 if e == "full_with_fused" else PALETTE_MAIN[i % 8]) for i, (d, e) in enumerate(zip(df["delta_auc"], df["experiment_id"]))]
    sns.barplot(data=df, y="label", x="auc", palette=colors, ax=axes[0], orient="h", edgecolor=EDGE_DARK, linewidth=0.9, alpha=0.86, legend=False)
    sns.stripplot(data=df, y="label", x="auc", color=C2, marker="D", size=7, ax=axes[0], jitter=False)
    axes[0].axvline(ref, color=C6, ls="--", lw=1)
    axes[0].set_xlabel("AUROC (test, threshold-free)")
    axes[0].set_ylabel("")
    axes[0].set_title("Modality ablation — AUROC")
    sns.barplot(data=df, y="label", x="f1", palette=PALETTE_MAIN[: len(df)], ax=axes[1], orient="h", edgecolor=EDGE_DARK, linewidth=0.9, alpha=0.86, legend=False)
    sns.stripplot(data=df, y="label", x="f1", color=C4, marker="^", size=7, ax=axes[1], jitter=False)
    axes[1].set_xlabel("F1 (validation max-F1 threshold on test)")
    axes[1].set_ylabel("")
    axes[1].set_title("Modality ablation — F1")
    fig.suptitle("Supplementary: input modality subsets (n = 288 test)", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "AblationFig_S3_modality_barplot")


def fig_rasa_delta(project: Path, out_dir: Path) -> None:
    df = _read(project, "S5_rasa_component_ablation.csv")
    if df is None:
        return
    _setup_theme()
    df = df.copy()
    ref = df[df["experiment_id"] == "full_lcad_rasa"].iloc[0]
    df["delta_auc"] = df["auc"] - ref["auc"]
    df["delta_f1"] = df["f1"] - ref["f1"]
    df["label"] = df["experiment_id"].map(_pretty_id)
    df = df.sort_values("delta_auc")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = [C4 if v < -0.05 else C0 for v in df["delta_auc"]]
    sns.barplot(data=df, x="delta_auc", y="label", palette=colors, ax=axes[0], orient="h", edgecolor=EDGE_DARK, linewidth=0.9, alpha=0.86, legend=False)
    sns.stripplot(data=df, x="delta_auc", y="label", color=C2, marker="D", size=7, ax=axes[0], jitter=False)
    axes[0].axvline(0, color=C6, lw=1)
    axes[0].set_xlabel("ΔAUROC vs full LCAD-RASA")
    axes[0].set_title("RASA component ablation")
    colors2 = [C4 if v < -0.05 else C0 for v in df["delta_f1"]]
    sns.barplot(data=df, x="delta_f1", y="label", palette=colors2, ax=axes[1], orient="h", edgecolor=EDGE_DARK, linewidth=0.9, alpha=0.86, legend=False)
    sns.stripplot(data=df, x="delta_f1", y="label", color=C4, marker="^", size=7, ax=axes[1], jitter=False)
    axes[1].axvline(0, color=C6, lw=1)
    axes[1].set_xlabel("ΔF1 vs full LCAD-RASA")
    axes[1].set_title("F1 impact (val-selected threshold)")
    fig.tight_layout()
    _save(fig, out_dir / "AblationFig_S5_rasa_delta_bars")


def fig_rasa_auc_f1_scatter(project: Path, out_dir: Path) -> None:
    df = _read(project, "S5_rasa_component_ablation.csv")
    if df is None:
        return
    _setup_theme()
    df = df.copy()
    df["label"] = df["experiment_id"].map(_pretty_id)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=df, x="auc", y="f1", hue="experiment_id", s=180, palette=PALETTE_MAIN, ax=ax, edgecolor=TEXT_DARK, linewidth=0.8)
    for _, r in df.iterrows():
        ax.annotate(r["experiment_id"].split("_")[0][:6], (r["auc"], r["f1"]), fontsize=7, alpha=0.85, xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(0.45, 0.88)
    ax.set_title("RASA ablations: risk–semantic trade-off")
    ax.legend(loc="lower left", fontsize=7, title="Variant", bbox_to_anchor=(1.02, 0))
    fig.tight_layout()
    _save(fig, out_dir / "AblationFig_S5_auc_f1_scatter")


def fig_qc_barplot(project: Path, out_dir: Path) -> None:
    df = _read(project, "S4_lcad_qc_ablation.csv")
    if df is None:
        return
    _setup_theme()
    df = df.copy()
    df["label"] = df["experiment_id"].str.replace("pseudo_", "").str.replace("_", " ").str.title()
    melt = df.melt(id_vars=["label"], value_vars=["auc", "f1"], var_name="metric", value_name="score")
    g = sns.catplot(
        data=melt,
        x="score",
        y="label",
        col="metric",
        kind="bar",
        palette=[C1, C4],
        height=5,
        aspect=0.85,
        sharex=False,
        edgecolor=EDGE_DARK,
        linewidth=0.9,
        alpha=0.86,
    )
    for ax in g.axes.flat:
        sns.stripplot(
            data=melt[melt["metric"].eq(ax.get_title().split(" = ")[-1])] if " = " in ax.get_title() else melt,
            x="score",
            y="label",
            color=C2,
            marker="s",
            size=6,
            jitter=False,
            ax=ax,
        )
    g.set_axis_labels("Score", "QC weighting mode")
    for ax, title in zip(g.axes.flat, ["AUROC", "F1"]):
        ax.set_title(title)
    g.fig.suptitle("LCAD pseudo-report QC / weighting ablation", y=1.03)
    _save(g.fig, out_dir / "AblationFig_S4_qc_catplot")


def fig_combined_heatmap(project: Path, out_dir: Path) -> None:
    frames = []
    for fname, tag in (
        ("S3_modality_ablation.csv", "Modality"),
        ("S5_rasa_component_ablation.csv", "RASA"),
        ("S4_lcad_qc_ablation.csv", "QC"),
    ):
        d = _read(project, fname)
        if d is None:
            continue
        d = d.copy()
        d["block"] = tag
        d["label"] = d["experiment_id"].map(_pretty_id)
        frames.append(d[["block", "label", "auc", "f1"]])
    if not frames:
        return
    _setup_theme()
    all_df = pd.concat(frames, ignore_index=True)
    piv = all_df.set_index(["block", "label"])[["auc", "f1"]]
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(piv) + 2))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap=_cmap_diverging(), center=0.65, vmin=0.4, vmax=0.9, ax=ax, cbar_kws={"label": "Score"})
    ax.set_title("All ablation experiments — AUROC & F1 summary")
    fig.tight_layout()
    _save(fig, out_dir / "AblationFig_combined_heatmap")


def write_ablation_figure_index(out_dir: Path) -> None:
    stems = [
        ("AblationFig_S3_modality_barplot", "barplot", "S3 modality AUROC + F1"),
        ("AblationFig_S5_rasa_delta_bars", "barplot", "S5 ΔAUROC / ΔF1 vs full"),
        ("AblationFig_S5_auc_f1_scatter", "scatterplot", "S5 AUC–F1 trade-off"),
        ("AblationFig_S4_qc_catplot", "catplot", "S4 QC weighting modes"),
        ("AblationFig_combined_heatmap", "heatmap", "Combined ablation matrix"),
    ]
    lines = [
        "# JBD Ablation Figure Index\n",
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n",
        f"Palette: {', '.join(JBD_PALETTE_HEX)}\n",
        "Style reference: https://seaborn.pydata.org/examples/index.html\n",
        "Font: Arial; mathtext: STIX.\n\n",
    ]
    for stem, ptype, desc in stems:
        lines.append(f"## {stem}\n- **Type**: {ptype}\n- {desc}\n- `{stem}.png`, `{stem}.pdf`\n\n")
    (out_dir / "ABLATION_FIGURE_INDEX.md").write_text("".join(lines), encoding="utf-8")


def generate_all_ablation_figures(project: Path) -> Path:
    out_dir = project / OUT_REL
    fig_modality_barplot(project, out_dir)
    fig_rasa_delta(project, out_dir)
    fig_rasa_auc_f1_scatter(project, out_dir)
    fig_qc_barplot(project, out_dir)
    fig_combined_heatmap(project, out_dir)
    write_ablation_figure_index(out_dir)
    return out_dir
