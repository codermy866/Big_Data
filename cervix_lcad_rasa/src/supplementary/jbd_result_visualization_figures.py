"""Result-visualization figures with p-value annotations (journal bar / marginal-scatter style)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from src.supplementary.jbd_figure_stats import (
    add_significance_bracket,
    annotate_p_at_xy,
    format_pvalue,
    load_comparator_pvals,
)
from src.supplementary.jbd_figures_seaborn import (
    C0,
    C1,
    C2,
    C3,
    C4,
    C7,
    EDGE_DARK,
    PALETTE_MAIN,
    TEXT_DARK,
    _read,
    _save,
    _setup_theme,
)

MANUSCRIPT_REL = "outputs/publishable/tables/manuscript"
REVISION_REL = "outputs/publishable/mosaic_revision_audit/tables"
EXTERNAL_REL = "outputs/publishable/tables/manuscript/T_external_baselines_same_split_with_mosaic.csv"

MOSAIC_MODELS = [
    ("kra_semantic_fusion", "MOSAIC (full)", "#E76B6B"),
    ("full_lcad_rasa_stablehash", "MOSAIC--RASA backbone", "#1E3A66"),
    ("semantic_retrieval_positive_ratio", "Semantic retrieval only", "#ADB093"),
]


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left", fontfamily="Arial")


def fig_mosaic_primary_grouped_bars(project: Path, out_dir: Path) -> None:
    """Grouped AUROC/F1 bars for headline MOSAIC variants with paired-bootstrap p brackets."""
    main = _read(project, f"{MANUSCRIPT_REL}/T_mosaic_main_comparison.csv")
    ext = _read(project, EXTERNAL_REL)
    if main is None:
        return
    _setup_theme()
    pvals = load_comparator_pvals(project)
    p_mosaic_vs_backbone = pvals.get("MOSAIC--RASA backbone", pvals.get("MOSAIC (full) vs MOSAIC--RASA backbone"))

    rows = []
    for mid, label, color in MOSAIC_MODELS:
        hit = main[main["model_id"].eq(mid)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        ci_auc = (np.nan, np.nan)
        ci_f1 = (np.nan, np.nan)
        if ext is not None and not ext.empty:
            short = label if label != "MOSAIC--RASA backbone" else "MOSAIC--RASA backbone (stable-hash)"
            eh = ext[ext["model_short"].astype(str).str.contains(label.split("(")[0].strip(), regex=False)]
            if label == "MOSAIC (full)":
                eh = ext[ext["model_short"].eq("MOSAIC (full)")]
            elif label == "MOSAIC--RASA backbone":
                eh = ext[ext["model_short"].str.contains("MOSAIC--RASA backbone", regex=False)]
            if not eh.empty:
                er = eh.iloc[0]
                ci_auc = (float(er.get("auc_ci_low", np.nan)), float(er.get("auc_ci_high", np.nan)))
                ci_f1 = (float(er.get("f1_ci_low", np.nan)), float(er.get("f1_ci_high", np.nan)))
        rows.append(
            {
                "model": label,
                "color": color,
                "auc": float(r["auc"]),
                "f1": float(r["f1"]),
                "auc_ci_low": ci_auc[0],
                "auc_ci_high": ci_auc[1],
                "f1_ci_low": ci_f1[0],
                "f1_ci_high": ci_f1[1],
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    x = np.arange(len(df))
    width = 0.34
    for i, row in df.iterrows():
        ax.bar(
            x[i] - width / 2,
            row["auc"],
            width,
            color=row["color"],
            edgecolor=EDGE_DARK,
            linewidth=0.9,
            alpha=0.88,
            label="AUROC" if i == 0 else "",
        )
        ax.bar(
            x[i] + width / 2,
            row["f1"],
            width,
            color=row["color"],
            edgecolor=EDGE_DARK,
            linewidth=0.9,
            alpha=0.55,
            hatch="///",
            label="F1" if i == 0 else "",
        )
        if not np.isnan(row["auc_ci_low"]):
            ax.errorbar(
                x[i] - width / 2,
                row["auc"],
                yerr=[[row["auc"] - row["auc_ci_low"]], [row["auc_ci_high"] - row["auc"]]],
                fmt="none",
                ecolor=EDGE_DARK,
                elinewidth=1.2,
                capsize=3.5,
                zorder=4,
            )
        if not np.isnan(row["f1_ci_low"]):
            ax.errorbar(
                x[i] + width / 2,
                row["f1"],
                yerr=[[row["f1"] - row["f1_ci_low"]], [row["f1_ci_high"] - row["f1"]]],
                fmt="none",
                ecolor=EDGE_DARK,
                elinewidth=1.2,
                capsize=3.5,
                zorder=4,
            )
        ax.scatter(x[i] - width / 2, row["auc"], s=28, color="white", edgecolor=EDGE_DARK, linewidth=0.7, zorder=5)
        ax.scatter(x[i] + width / 2, row["f1"], marker="^", s=34, color=row["color"], edgecolor=EDGE_DARK, linewidth=0.7, zorder=5)

    ymax = max(float(df["auc"].max()), float(df["f1"].max())) + 0.08
    add_significance_bracket(ax, x[0] - 0.2, x[1] + 0.2, ymax + 0.02, p_mosaic_vs_backbone, h=0.028)
    ax.text(
        (x[0] + x[1]) / 2,
        ymax + 0.09,
        "MOSAIC (full) vs backbone",
        ha="center",
        va="bottom",
        fontsize=8,
        color=TEXT_DARK,
        fontfamily="Arial",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=12, ha="right", fontfamily="Arial")
    ax.set_ylabel("Held-out score (n = 288)", fontfamily="Arial")
    ax.set_ylim(0, ymax + 0.16)
    ax.set_title("Primary MOSAIC comparison with paired-bootstrap significance", fontweight="bold", fontfamily="Arial")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.axhline(0.5, color=C3, ls=":", lw=0.9)
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, out_dir / "Figure_mosaic_primary_results_grouped")


def fig_mosaic_auc_f1_marginal_scatter(project: Path, out_dir: Path) -> None:
    """AUROC vs F1 scatter with marginal boxplots — reference marginal-scatter style."""
    main = _read(project, f"{MANUSCRIPT_REL}/T_mosaic_main_comparison.csv")
    if main is None:
        return
    _setup_theme()
    pvals = load_comparator_pvals(project)

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = GridSpec(4, 4, figure=fig, wspace=0.05, hspace=0.05)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    plot_rows = []
    for mid, label, color in MOSAIC_MODELS:
        hit = main[main["model_id"].eq(mid)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        plot_rows.append({"model": label, "color": color, "auc": float(r["auc"]), "f1": float(r["f1"])})
    df = pd.DataFrame(plot_rows)

    for _, row in df.iterrows():
        ax_main.scatter(row["auc"], row["f1"], s=220, color=row["color"], edgecolor=EDGE_DARK, linewidth=0.9, zorder=3, alpha=0.92)
        ax_main.annotate(
            row["model"],
            (row["auc"], row["f1"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            fontfamily="Arial",
            color=TEXT_DARK,
        )

    if len(df) >= 2:
        r_val = float(np.corrcoef(df["auc"], df["f1"])[0, 1])
        ax_main.text(
            0.04,
            0.96,
            f"r = {r_val:.3f}\nheld-out test (n = 288)",
            transform=ax_main.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontfamily="Arial",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F7F8FB", ec=C3, alpha=0.9),
        )
    p_backbone = pvals.get("MOSAIC--RASA backbone")
    if p_backbone is not None:
        ax_main.text(
            0.04,
            0.78,
            f"MOSAIC (full) vs backbone:\n{format_pvalue(p_backbone)}",
            transform=ax_main.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            fontfamily="Arial",
            color=C4,
        )

    sns.boxplot(data=df, x="auc", y="model", orient="h", palette=df["color"].tolist(), ax=ax_top, width=0.55, fliersize=0, linewidth=0.8)
    sns.boxplot(data=df, x="f1", y="model", orient="v", palette=df["color"].tolist(), ax=ax_right, width=0.55, fliersize=0, linewidth=0.8)
    ax_top.set_xlabel("")
    ax_top.set_ylabel("")
    ax_top.tick_params(labelbottom=False)
    ax_right.set_ylabel("")
    ax_right.set_xlabel("")
    ax_right.tick_params(labelleft=False)

    ax_main.set_xlabel("AUROC", fontfamily="Arial")
    ax_main.set_ylabel("F1 score", fontfamily="Arial")
    ax_main.set_xlim(0.72, 0.94)
    ax_main.set_ylim(0.44, 0.78)
    ax_main.set_title("Risk–discrimination trade-off across MOSAIC variants", fontweight="bold", fontfamily="Arial", pad=10)
    ax_main.grid(True, alpha=0.28)
    sns.despine(ax=ax_main)
    sns.despine(ax=ax_top, left=True, bottom=True)
    sns.despine(ax=ax_right, left=True, bottom=True)
    fig.tight_layout()
    _save(fig, out_dir / "Figure_mosaic_auc_f1_marginal_scatter")


def fig_external_baselines_grouped_bars(project: Path, out_dir: Path) -> None:
    """Top external baselines vs MOSAIC — grouped AUROC bars with p annotations."""
    ext = _read(project, EXTERNAL_REL)
    if ext is None:
        return
    _setup_theme()
    pvals = load_comparator_pvals(project)

    keep = [
        "MOSAIC (full)",
        "CLIP-style contrastive multimodal baseline",
        "Clinical-only HistGradientBoosting",
        "MOSAIC--RASA backbone (stable-hash)",
        "Cross-attention multimodal transformer",
        "Clinical-only logistic regression",
    ]
    plot_df = ext[ext["model_short"].isin(keep)].copy()
    plot_df = plot_df.set_index("model_short").loc[keep].reset_index()

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(plot_df))
    colors = ["#E76B6B" if m == "MOSAIC (full)" else ("#1E3A66" if "MOSAIC--RASA" in m else PALETTE_MAIN[i % len(PALETTE_MAIN)]) for i, m in enumerate(plot_df["model_short"])]
    bars = ax.bar(x, plot_df["auc"], color=colors, edgecolor=EDGE_DARK, linewidth=0.9, alpha=0.88, width=0.62, zorder=1)
    for xi, (_, row) in enumerate(plot_df.iterrows()):
        if not np.isnan(row.get("auc_ci_low", np.nan)):
            ax.errorbar(
                xi,
                row["auc"],
                yerr=[[row["auc"] - row["auc_ci_low"]], [row["auc_ci_high"] - row["auc"]]],
                fmt="none",
                ecolor=EDGE_DARK,
                elinewidth=1.3,
                capsize=4,
                zorder=3,
            )
        ax.scatter(xi, row["auc"], s=42, color="white", edgecolor=EDGE_DARK, linewidth=0.8, zorder=4)
        if row["model_short"] != "MOSAIC (full)":
            p = pvals.get(row["model_short"])
            if row["model_short"] == "CLIP-style contrastive multimodal baseline":
                p_con = _read(project, f"{REVISION_REL}/mosaic_vs_contrastive_paired_bootstrap.csv")
                if p_con is not None and not p_con.empty:
                    p = float(p_con["paired_bootstrap_p_two_sided"].iloc[0])
            annotate_p_at_xy(ax, xi, float(row["auc"]) + 0.045, p, ha="center", va="bottom", fontsize=8)

    mosaic_x = list(plot_df["model_short"]).index("MOSAIC (full)")
    backbone_hit = plot_df[plot_df["model_short"].eq("MOSAIC--RASA backbone (stable-hash)")]
    ref_auc = float(backbone_hit["auc"].iloc[0]) if not backbone_hit.empty else 0.86
    backbone_idx = list(plot_df["model_short"]).index("MOSAIC--RASA backbone (stable-hash)") if not backbone_hit.empty else None
    if backbone_idx is not None:
        y_br = float(plot_df["auc"].max()) + 0.06
        add_significance_bracket(ax, mosaic_x, backbone_idx, y_br, pvals.get("MOSAIC--RASA backbone"), h=0.022)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model_short"], rotation=28, ha="right", fontsize=8.5, fontfamily="Arial")
    ax.set_ylabel("Held-out AUROC (95% bootstrap CI)", fontfamily="Arial")
    ax.set_ylim(0.72, float(plot_df["auc"].max()) + 0.14)
    ax.set_title("External baseline discrimination with paired-bootstrap p-values", fontweight="bold", fontfamily="Arial")
    ax.axhline(ref_auc, color=C3, ls="--", lw=0.8, alpha=0.7)
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, out_dir / "Figure_external_baselines_grouped_bars")
    _ = bars


def fig_centre_auc_grouped_bars(project: Path, out_dir: Path) -> None:
    """Centre-wise AUROC: backbone vs MOSAIC (full) grouped bars."""
    center = _read(project, f"{MANUSCRIPT_REL}/T_mosaic_centerwise.csv")
    if center is None:
        return
    _setup_theme()
    view = center.dropna(subset=["auc", "baseline_auc"]).copy()
    if view.empty:
        return
    view["centre"] = view["center_id"].str.title()
    view = view.sort_values("auc", ascending=False)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(view))
    w = 0.36
    ax.bar(x - w / 2, view["baseline_auc"], w, color=C0, edgecolor=EDGE_DARK, linewidth=0.8, alpha=0.86, label="MOSAIC--RASA backbone")
    ax.bar(x + w / 2, view["auc"], w, color=C4, edgecolor=EDGE_DARK, linewidth=0.8, alpha=0.86, label="MOSAIC (full)")
    for xi, (_, row) in enumerate(view.iterrows()):
        delta = float(row["auc"] - row["baseline_auc"])
        if abs(delta) > 0.02:
            y = max(float(row["auc"]), float(row["baseline_auc"])) + 0.03
            ax.text(xi, y, f"Δ={delta:+.2f}", ha="center", va="bottom", fontsize=8, fontfamily="Arial", color=TEXT_DARK)

    pvals = load_comparator_pvals(project)
    ax.text(
        0.99,
        0.98,
        f"Pooled test:\n{format_pvalue(pvals.get('MOSAIC--RASA backbone'))}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontfamily="Arial",
        bbox=dict(boxstyle="round,pad=0.35", fc="#F7F8FB", ec=C3),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(view["centre"], rotation=15, ha="right", fontfamily="Arial")
    ax.set_ylabel("Centre-wise AUROC", fontfamily="Arial")
    ax.set_ylim(0, 1.08)
    ax.set_title("Centre-wise AUROC shift (held-out cases per centre)", fontweight="bold", fontfamily="Arial")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, out_dir / "Figure_centre_auc_grouped_bars")


def fig_composite_result_panels(project: Path, out_dir: Path) -> None:
    """Three-panel composite (A–C) summarizing headline quantitative results."""
    _setup_theme()
    fig = plt.figure(figsize=(14.2, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.95, 1.25], wspace=0.42)

    # Panel A — reuse grouped logic inline
    ax_a = fig.add_subplot(gs[0, 0])
    main = _read(project, f"{MANUSCRIPT_REL}/T_mosaic_main_comparison.csv")
    pvals = load_comparator_pvals(project)
    if main is not None:
        labels = [m[1] for m in MOSAIC_MODELS if not main[main["model_id"].eq(m[0])].empty]
        aucs = [float(main[main["model_id"].eq(m[0])]["auc"].iloc[0]) for m in MOSAIC_MODELS if not main[main["model_id"].eq(m[0])].empty]
        colors = [m[2] for m in MOSAIC_MODELS if not main[main["model_id"].eq(m[0])].empty]
        x = np.arange(len(labels))
        ax_a.bar(x, aucs, color=colors, edgecolor=EDGE_DARK, linewidth=0.8, alpha=0.9)
        add_significance_bracket(ax_a, 0, 1, max(aucs) + 0.03, pvals.get("MOSAIC--RASA backbone"), h=0.02)
        ax_a.set_xticks(x)
        ax_a.set_xticklabels([l.replace("MOSAIC ", "MOSAIC\n") for l in labels], fontsize=7.5, fontfamily="Arial")
        ax_a.set_ylabel("AUROC", fontfamily="Arial")
        ax_a.set_ylim(0.7, max(aucs) + 0.12)
        ax_a.set_title("Headline MOSAIC variants", fontweight="bold", fontsize=9, fontfamily="Arial")
    _panel_label(ax_a, "A")

    # Panel B — contrastive vs MOSAIC p
    ax_b = fig.add_subplot(gs[0, 1])
    ext = _read(project, EXTERNAL_REL)
    if ext is not None:
        pair = ext[ext["model_short"].isin(["MOSAIC (full)", "CLIP-style contrastive multimodal baseline"])].copy()
        pair = pair.set_index("model_short").loc[["CLIP-style contrastive multimodal baseline", "MOSAIC (full)"]].reset_index()
        x = np.arange(2)
        ax_b.bar(x, pair["auc"], color=["#ADB093", "#E76B6B"], edgecolor=EDGE_DARK, linewidth=0.8, alpha=0.9)
        for xi, (_, row) in enumerate(pair.iterrows()):
            ax_b.errorbar(xi, row["auc"], yerr=[[row["auc"] - row["auc_ci_low"]], [row["auc_ci_high"] - row["auc"]]], fmt="none", ecolor=EDGE_DARK, capsize=3)
        p_con = _read(project, f"{REVISION_REL}/mosaic_vs_contrastive_paired_bootstrap.csv")
        p = float(p_con["paired_bootstrap_p_two_sided"].iloc[0]) if p_con is not None and not p_con.empty else None
        add_significance_bracket(ax_b, 0, 1, float(pair["auc"].max()) + 0.03, p, h=0.02)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(["Contrastive\nbaseline", "MOSAIC\n(full)"], fontsize=8, fontfamily="Arial")
        ax_b.set_ylabel("AUROC", fontfamily="Arial")
        ax_b.set_ylim(0.82, 0.98)
        ax_b.set_title("MOSAIC (full) vs contrastive baseline", fontweight="bold", fontsize=9, fontfamily="Arial")
    _panel_label(ax_b, "B")

    # Panel C — internal ablation p summary (top 4 from recheck)
    ax_c = fig.add_subplot(gs[0, 2])
    recheck = _read(project, f"{MANUSCRIPT_REL}/T_external_baseline_paired_bootstrap_recheck.csv")
    if recheck is not None:
        sub = recheck.head(6).copy()
        sub = sub.sort_values("delta_auc_full_minus_comparator", ascending=True)
        sub["short"] = sub["comparator"].str.replace(" report gen.", " rep.", regex=False).str.replace("Pseudo-augmented (LCAD)", "Pseudo-aug.", regex=False)
        y = np.arange(len(sub))
        for i, (_, row) in enumerate(sub.iterrows()):
            ax_c.plot([row["delta_auc_ci_low"], row["delta_auc_ci_high"]], [i, i], color=EDGE_DARK, lw=1.2)
            ax_c.scatter(row["delta_auc_full_minus_comparator"], i, s=70, color=PALETTE_MAIN[i % len(PALETTE_MAIN)], edgecolor=EDGE_DARK, linewidth=0.7, zorder=3)
            annotate_p_at_xy(ax_c, float(row["delta_auc_ci_high"]) + 0.008, i, float(row["paired_bootstrap_p_two_sided"]), ha="left", fontsize=7.5)
        ax_c.axvline(0, color=C1, ls="--", lw=0.9)
        ax_c.set_yticks(y)
        ax_c.set_yticklabels(sub["short"], fontsize=6.8, fontfamily="Arial")
        ax_c.set_xlabel("ΔAUROC (backbone reference block)", fontfamily="Arial")
        ax_c.set_title("Paired-bootstrap recheck (internal + external)", fontweight="bold", fontsize=9, fontfamily="Arial")
        ax_c.tick_params(axis="y", pad=2)
    _panel_label(ax_c, "C")

    fig.suptitle("Quantitative result summary with paired-bootstrap p-values (held-out n = 288)", fontsize=11, fontweight="bold", fontfamily="Arial", y=1.03)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.18, wspace=0.42)
    _save(fig, out_dir / "Figure_mosaic_result_visualization_panels")


def fig_external_paired_delta_recheck(project: Path, out_dir: Path) -> None:
    """Paired delta AUROC forest with p-value column (no model re-run)."""
    paired = _read(project, f"{MANUSCRIPT_REL}/T_external_baseline_paired_bootstrap_recheck.csv")
    if paired is None:
        return
    _setup_theme()
    p = paired[paired["comparator"].str.contains("Clinical|OCT|Colposcopy|Late|Cross|Contrastive", regex=True)].copy()
    if p.empty:
        return
    p = p.sort_values("delta_auc_full_minus_comparator", ascending=True)
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    ci_max = float(p["delta_auc_ci_high"].max())
    label_x = ci_max + 0.018
    for i, (_, r) in enumerate(p.iterrows()):
        ax.plot([r["delta_auc_ci_low"], r["delta_auc_ci_high"]], [i, i], color=EDGE_DARK, lw=1.5)
        ax.scatter(r["delta_auc_full_minus_comparator"], i, s=120, color=PALETTE_MAIN[i % len(PALETTE_MAIN)], edgecolor=EDGE_DARK, linewidth=0.8, zorder=3)
        annotate_p_at_xy(ax, label_x, i, float(r["paired_bootstrap_p_two_sided"]), ha="left", va="center", fontsize=9)
    ax.axvline(0, ls="--", color=C2, lw=1.2)
    ax.set_yticks(range(len(p)))
    ax.set_yticklabels(p["comparator"].tolist())
    ax.set_xlim(float(p["delta_auc_ci_low"].min()) - 0.03, label_x + 0.10)
    ax.set_xlabel("Paired bootstrap ΔAUROC (Full LCAD-RASA − comparator)")
    ax.set_title("Corrected paired bootstrap recheck with two-sided p-values", fontweight="bold", fontfamily="Arial")
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, out_dir / "Figure_external_baselines_paired_delta_auc")
    import shutil

    for ext in (".png", ".pdf"):
        src = out_dir / f"Figure_external_baselines_paired_delta_auc{ext}"
        if src.is_file():
            dst_dir = project / "outputs/publishable/external_baselines/figures"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if dst.resolve() != src.resolve():
                shutil.copy2(src, dst)


def generate_result_visualization_figures(project: Path) -> list[str]:
    out_dir = project / "outputs/publishable/figures/jbd_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for fn in (
        fig_mosaic_primary_grouped_bars,
        fig_mosaic_auc_f1_marginal_scatter,
        fig_external_baselines_grouped_bars,
        fig_centre_auc_grouped_bars,
        fig_composite_result_panels,
        fig_external_paired_delta_recheck,
    ):
        fn(project, out_dir)
        names.append(fn.__name__)
    return names
