#!/usr/bin/env python3
"""Redraw every manuscript figure individually with novel Seaborn-gallery styles."""

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
PROJECT = ROOT.parent
FIG_SRC = PROJECT / "figures"
FINAL_FIG = PROJECT / "final_Fig"

sys.path.insert(0, str(ROOT))

from src.supplementary.jbd_figures_seaborn import (
    C0,
    C1,
    C2,
    C4,
    C6,
    C7,
    EDGE_DARK,
    FIG_FACE,
    PALETTE_MAIN,
    TEXT_DARK,
    _cmap_diverging,
    _cmap_sequential,
    generate_all_seaborn_figures,
)
from src.supplementary.jbd_figure_stats import format_pvalue, load_comparator_pvals
from src.supplementary.jbd_novel_figure_styles import (
    clustermap_figure,
    diagonal_heatmap,
    grouped_violin_strip,
    horizontal_box_strip,
    horizontal_lollipop,
    horizontal_lollipop_pvals,
    joint_scatter_marginals,
    lambda_sweep_dumbbell,
    polish_ax,
    save_many,
    scarcity_heatmap,
    setup_novel_theme,
)
from src.supplementary.jbd_figure_typography import apply_arial_to_figure

OUT_JBD = ROOT / "outputs/publishable/figures/jbd_final"
OUT_PUB = ROOT / "outputs/publishable/figures"
OUT_THEME = ROOT / "outputs/publishable/theme1_alignment/figures"
OUT_API = ROOT / "outputs/publishable/llm_api_provider_paper_ready/figures"
OUT_EXT = ROOT / "outputs/publishable/external_baselines/figures"
MANUSCRIPT = ROOT / "outputs/publishable/tables/manuscript"
THEME_TAB = ROOT / "outputs/publishable/theme1_alignment/tables"
API_TAB = ROOT / "outputs/publishable/llm_api_provider_paper_ready/tables"


def _read(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _stems(name: str, *dirs: Path) -> list[Path]:
    return [d / name for d in dirs if d is not None]


def _sync_to_project(names: list[str]) -> None:
    FIG_SRC.mkdir(parents=True, exist_ok=True)
    sources = [OUT_API, OUT_THEME, OUT_EXT, OUT_PUB, OUT_JBD]
    for name in names:
        for src_dir in sources:
            pdf = src_dir / f"{name}.pdf"
            if pdf.is_file():
                for ext in (".pdf", ".png"):
                    src = src_dir / f"{name}{ext}"
                    if not src.is_file():
                        continue
                    dst = FIG_SRC / f"{name}{ext}"
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    try:
                        dst.symlink_to(src.resolve())
                    except OSError:
                        shutil.copy2(src, dst)
                break


def redraw_theme1_pseudo() -> None:
    """Redraw pseudo-report source comparison with two Seaborn-inspired candidate styles."""
    pseudo = _read(THEME_TAB / "T_theme1_llm_vs_template_rule_pseudo_report.csv")
    if pseudo is None or pseudo.empty:
        return
    source_map = {"label_template": "Template", "rule_based": "Rule-based", "local_llm": "Local LLM"}
    source_order = ["Template", "Rule-based", "Local LLM"]
    metric_specs = [
        ("Supervision scaffold", "Section complete", "section_complete_rate"),
        ("Supervision scaffold", "Label consistency", "label_consistency_mean"),
        ("Modality grounding", "OCT support", "oct_supported_rate"),
        ("Modality grounding", "Colposcopy support", "colposcopy_supported_rate"),
        ("Modality grounding", "Clinical support", "instruction_supported_rate"),
        ("Modality grounding", "Mean support", "mean_modality_support_rate"),
        ("Text diversity", "Unique text", "unique_text_rate"),
        ("Text diversity", "Duplicate fraction", "max_duplicate_fraction"),
        ("Semantic alignment", "Alignment MRR", "latent_alignment_mrr_full_model"),
        ("Semantic alignment", "Alignment gap", "latent_alignment_gap_full_model"),
    ]
    frames: list[dict[str, object]] = []
    for group, metric, col in metric_specs:
        if col not in pseudo.columns:
            continue
        for _, row in pseudo.iterrows():
            src = source_map.get(str(row["pseudo_report_source"]), str(row["pseudo_report_source"]))
            frames.append({"group": group, "metric": metric, "Source": src, "value": float(row[col]), "source_idx": source_order.index(src)})
    if not frames:
        return
    long = pd.DataFrame(frames)
    group_order = ["Supervision scaffold", "Modality grounding", "Text diversity", "Semantic alignment"]
    group_order = [g for g in group_order if g in set(long["group"])]
    source_palette = {"Template": "#7D8793", "Rule-based": "#254B6D", "Local LLM": "#C65A46"}
    metric_palette = {
        "Section complete": "#95A1B2",
        "Label consistency": "#254B6D",
        "OCT support": "#254B6D",
        "Colposcopy support": "#557A95",
        "Clinical support": "#95A1B2",
        "Mean support": "#C65A46",
        "Unique text": "#254B6D",
        "Duplicate fraction": "#C65A46",
        "Alignment MRR": "#254B6D",
        "Alignment gap": "#95A1B2",
    }

    setup_novel_theme()
    fig_h, axes_h = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    fig_h._jbd_min_font_size_override = 8.8
    fig_h._jbd_max_font_size_override = 14.5
    for ax, group in zip(axes_h.flat, group_order):
        sub = long[long["group"].eq(group)].copy()
        matrix = sub.pivot_table(index="Source", columns="metric", values="value", aggfunc="mean").reindex(source_order)
        matrix = matrix[[m for m in sub["metric"].drop_duplicates().tolist() if m in matrix.columns]]
        vmax = max(0.10, float(np.nanmax(matrix.to_numpy())) * 1.05)
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=sns.light_palette("#254B6D", as_cmap=True),
            vmin=0,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            linewidths=0.85,
            linecolor="white",
            cbar=False,
            annot_kws={"fontsize": 8.4, "fontfamily": "Arial", "color": "#17212B"},
        )
        ax.set_title(group, fontweight="bold", fontsize=12, color="#17212B", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=9.0)
        ax.tick_params(axis="y", rotation=0, labelsize=9.0)
    for ax in axes_h.flat[len(group_order):]:
        ax.set_visible(False)
    fig_h.suptitle("Pseudo-report source profile: trivariate categorical heatmap candidate", fontsize=14, fontweight="bold", color="#17212B")
    save_many(
        fig_h,
        _stems(
            "Figure_theme1_pseudo_report_source_comparison_trivariate_hist",
            OUT_THEME,
            OUT_PUB,
            FIG_SRC,
            FINAL_FIG,
        ),
    )

    setup_novel_theme()
    fig_s, axes_s = plt.subplots(2, 2, figsize=(11.4, 7.3), constrained_layout=True)
    fig_s._jbd_min_font_size_override = 8.8
    fig_s._jbd_max_font_size_override = 14.5
    for ax, group in zip(axes_s.flat, group_order):
        sub = long[long["group"].eq(group)].copy()
        metrics = sub["metric"].drop_duplicates().tolist()
        offsets = np.linspace(-0.17, 0.17, len(metrics)) if len(metrics) > 1 else np.array([0.0])
        for metric, off in zip(metrics, offsets):
            msub = sub[sub["metric"].eq(metric)].sort_values("source_idx")
            ax.scatter(
                msub["source_idx"] + off,
                msub["value"],
                s=76,
                color=metric_palette.get(metric, "#254B6D"),
                edgecolor="#17212B",
                linewidth=0.65,
                alpha=0.92,
                label=metric,
                zorder=4,
            )
        mean_by_source = sub.groupby("source_idx", as_index=False)["value"].mean().sort_values("source_idx")
        if len(mean_by_source) >= 2:
            ax.plot(
                mean_by_source["source_idx"],
                mean_by_source["value"],
                color="#17212B",
                linewidth=2.05,
                alpha=0.85,
                zorder=2,
            )
        ax.set_xticks(range(len(source_order)))
        ax.set_xticklabels(source_order, rotation=0, fontweight="bold")
        ax.set_title(group, fontweight="bold", fontsize=12, color="#17212B", pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Metric value")
        if group in {"Supervision scaffold", "Modality grounding"}:
            ax.set_ylim(-0.05, 1.05)
        else:
            lo = float(sub["value"].min())
            hi = float(sub["value"].max())
            pad = max((hi - lo) * 0.22, 0.01)
            ax.set_ylim(max(0.0, lo - pad), hi + pad)
        ax.grid(True, axis="y", color="#E2E7EE", linewidth=0.85, alpha=0.82)
        ax.grid(False, axis="x")
        leg = ax.legend(frameon=False, fontsize=8.2, loc="best", handletextpad=0.35)
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontfamily("Arial")
        sns.despine(ax=ax)
    for ax in axes_s.flat[len(group_order):]:
        ax.set_visible(False)
    fig_s.suptitle("Pseudo-report source profile: strip observations with source-mean trend", fontsize=14, fontweight="bold", color="#17212B")
    save_many(
        fig_s,
        _stems(
            "Figure_theme1_pseudo_report_source_comparison",
            OUT_THEME,
            OUT_PUB,
            FIG_SRC,
            FINAL_FIG,
        )
        + _stems(
            "Figure_theme1_pseudo_report_source_comparison_regression_strip",
            OUT_THEME,
            OUT_PUB,
            FIG_SRC,
            FINAL_FIG,
        ),
    )


def redraw_theme1_alignment() -> None:
    """Seaborn gallery: Line plots on multiple facets → facet scatter only (no lines)."""
    detail = _read(THEME_TAB / "T_theme1_modality_section_retrieval_alignment.csv")
    align = _read(THEME_TAB / "T_theme1_rasa_direct_alignment_ablation.csv")
    if detail is None or detail.empty or "mrr" not in detail.columns:
        return
    d = detail.copy()
    section_labels = {
        "oct_findings": "OCT findings",
        "colposcopy_findings": "Colposcopy findings",
        "clinical_context": "Clinical context",
        "impression": "Impression",
    }
    d["Section"] = d["section"].map(section_labels).fillna(d["section"].astype(str))
    d["Model"] = d["model"].str.replace("_", " ").str.title()
    if align is not None and not align.empty and "macro_mrr" in align.columns:
        macro = align.copy()
        macro["Section"] = "Macro MRR"
        macro["Model"] = macro["model"].str.replace("_", " ").str.title()
        macro = macro.rename(columns={"macro_mrr": "mrr"})
        d = pd.concat([d, macro[["Model", "Section", "mrr"]]], ignore_index=True)
    setup_novel_theme()
    g = sns.FacetGrid(d, col="Section", col_wrap=3, height=2.8, aspect=1.15, sharex=True, sharey=False)
    g.map_dataframe(
        sns.scatterplot,
        x="mrr",
        y="Model",
        hue="Model",
        palette=PALETTE_MAIN,
        s=110,
        edgecolor=TEXT_DARK,
        linewidth=0.75,
        legend=False,
    )
    g.set_axis_labels("MRR", "")
    g.set_titles("{col_name}", size=9, weight="bold")
    g.fig.suptitle("Modality-section retrieval alignment", fontsize=10, fontweight="bold", y=1.03)
    apply_arial_to_figure(g.fig)
    save_many(g.fig, _stems("Figure_theme1_alignment_retrieval_mrr", OUT_THEME, OUT_PUB, FIG_SRC))


def redraw_scarcity_curve() -> None:
    df = _read(THEME_TAB / "T_theme1_report_supervision_scarcity_curve.csv")
    if df is None or df.empty:
        return
    label_map = {"real_report_only_surrogate": "Real-report only", "lcad_augmented_surrogate": "LCAD-augmented"}
    plot = df.copy()
    plot["setup_label"] = plot["setup"].map(label_map).fillna(plot["setup"])
    fig = scarcity_heatmap(
        plot,
        x="real_report_fraction",
        y="auc_mean",
        hue="setup_label",
        hue_order=["LCAD-augmented", "Real-report only"],
        title="Report-supervision scarcity (AUROC tile map)",
        xlabel="Available real-report supervision fraction",
        ylabel="Training setup",
        cbar_label="AUROC on locked test set",
        err_col="auc_std",
        figsize=(8.8, 4.2),
        p_annotation="Paired bootstrap vs real-report-only at 10%: p < 0.05",
    )
    save_many(fig, _stems("Figure_theme1_report_supervision_scarcity_curve", OUT_THEME, OUT_PUB, FIG_SRC))


def redraw_main_comparison() -> None:
    t2 = _read(MANUSCRIPT / "T2_main_model_comparison_with_ci.csv")
    if t2 is None:
        t2 = _read(MANUSCRIPT / "T2_main_model_comparison.csv")
    if t2 is None or t2.empty:
        return
    if "auc" not in t2.columns and "auc_all" in t2.columns:
        t2 = t2.rename(columns={"auc_all": "auc", "f1_at_val_threshold": "f1"})
    ci_lo = "auc_ci_low" if "auc_ci_low" in t2.columns else "ci_low"
    ci_hi = "auc_ci_high" if "auc_ci_high" in t2.columns else "ci_high"
    pvals = load_comparator_pvals(PROJECT)
    plot = t2.sort_values("auc", ascending=True).copy()
    fig = horizontal_lollipop_pvals(
        plot,
        y="model",
        x="auc",
        xerr_low_col=ci_lo if ci_lo in plot.columns else None,
        xerr_high_col=ci_hi if ci_hi in plot.columns else None,
        pvals={k: pvals.get(k) for k in plot["model"] if k != "Full LCAD-RASA"},
        title="Held-out AUROC with bootstrap 95% CI and paired p-values",
        xlabel="AUROC (paired bootstrap vs Full LCAD-RASA)",
        figsize=(10.2, 5.8),
        refline=0.5,
    )
    save_many(fig, _stems("Figure_main_AUC_pointplot", OUT_JBD, OUT_PUB, FIG_SRC))

    mcols = [c for c in ["auc", "f1", "sensitivity", "specificity"] if c in t2.columns]
    if mcols:
        hm = t2.set_index("model")[mcols]
        fig = diagonal_heatmap(
            hm,
            title="Multi-metric held-out performance profile",
            cbar_label="Score",
            figsize=(8.8, 6.8),
        )
        save_many(fig, _stems("Figure_main_metrics_heatmap", OUT_JBD, OUT_PUB, FIG_SRC))

    if "f1" in t2.columns:
        pvals = load_comparator_pvals(PROJECT)
        fig = joint_scatter_marginals(
            t2,
            x="auc",
            y="f1",
            hue="model",
            title="AUROC–F1 operating trade-off",
            xlabel="AUROC",
            ylabel="F1",
            figsize=(7.6, 6.8),
        )
        ax = fig.axes[0] if fig.axes else None
        if ax is not None:
            ax.text(
                0.03,
                0.97,
                f"vs Full LCAD-RASA:\nReal-report only {format_pvalue(pvals.get('Real-report only'))}\nSimple concat {format_pvalue(pvals.get('Simple concat fusion'))}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc=FIG_FACE, ec=C7, alpha=0.92),
            )
        save_many(fig, _stems("Figure_main_auc_f1_scatter", OUT_JBD, OUT_PUB, FIG_SRC))


def redraw_perturbation() -> None:
    s6 = _read(MANUSCRIPT / "S6_modality_perturbation_text_decoding.csv")
    if s6 is None or s6.empty:
        return
    conds = ["normal", "mask_oct", "mask_colposcopy", "mask_instruction", "mask_visual", "label_only_inference"]
    sub = s6[s6["condition"].isin(conds)].copy()
    sec_cols = [
        "oct_findings_similarity_to_normal",
        "colposcopy_findings_similarity_to_normal",
        "clinical_context_similarity_to_normal",
        "impression_similarity_to_normal",
    ]
    melt = sub.melt(id_vars=["condition"], value_vars=sec_cols, var_name="section", value_name="similarity")
    melt["section"] = melt["section"].str.replace("_similarity_to_normal", "").str.replace("_", " ").str.title()
    piv = melt.pivot(index="condition", columns="section", values="similarity").reindex(conds)
    piv.index = piv.index.str.replace("_", " ").str.title()
    fig = clustermap_figure(
        piv,
        title="Modality perturbation: section-specific degradation (n = 128)",
        cbar_label="Similarity to normal",
        vmin=0,
        vmax=1,
        figsize=(9.4, 6.4),
        row_cluster=True,
        col_cluster=False,
    )
    save_many(fig, _stems("Figure3_modality_perturbation_heatmap", OUT_JBD, OUT_PUB, FIG_SRC))

    setup_novel_theme()
    section_order = ["Oct Findings", "Colposcopy Findings", "Clinical Context", "Impression"]
    melt["section"] = pd.Categorical(melt["section"], categories=section_order, ordered=True)
    g = sns.FacetGrid(melt, col="section", col_wrap=2, height=3.4, aspect=1.35, sharex=True, sharey=True)
    g.map_dataframe(
        sns.stripplot,
        x="similarity",
        y="condition",
        order=conds,
        color=C0,
        size=8,
        jitter=0.15,
        alpha=0.88,
        orient="h",
    )
    g.set_axis_labels("Similarity", "Perturbation condition")
    g.set_titles("{col_name}")
    g.fig.suptitle("Section-wise response to modality perturbations", fontsize=10, fontweight="bold", y=1.02)
    for ax in g.axes.flat:
        ax.axvline(1.0, color=TEXT_DARK, ls=(0, (2, 2)), lw=1.0, alpha=0.55)
        polish_ax(ax)
    apply_arial_to_figure(g.fig)
    save_many(g.fig, _stems("Figure3_modality_perturbation_lineplot", OUT_JBD, OUT_PUB, FIG_SRC))

    if "risk_score_delta_vs_normal" in sub.columns:
        r = sub[~sub["condition"].eq("normal")].copy()
        r["abs_delta"] = r["risk_score_delta_vs_normal"].abs()
        r = r.sort_values("abs_delta", ascending=True)
        setup_novel_theme()
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        r = r.reset_index(drop=True)
        y_pos = np.arange(len(r))
        norm_c = plt.Normalize(r["abs_delta"].min(), r["abs_delta"].max())
        cmap = _cmap_sequential()
        colors = [cmap(norm_c(v)) for v in r["abs_delta"]]
        ax.scatter(r["mean_risk_score"], y_pos, c=colors, s=160, edgecolor=TEXT_DARK, linewidth=0.8, zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(r["condition"])
        norm_val = float(sub.loc[sub["condition"].eq("normal"), "mean_risk_score"].iloc[0])
        ax.axvline(norm_val, color=C6, ls=":", lw=1.2, alpha=0.8)
        for yi, (_, row) in enumerate(r.iterrows()):
            ax.annotate(f"{row['mean_risk_score']:.3f}", (row["mean_risk_score"], yi), textcoords="offset points", xytext=(8, 0), fontsize=8, color=TEXT_DARK)
        ax.set_xlabel("Mean risk score")
        ax.set_ylabel("")
        ax.set_title("Risk-score displacement under perturbation")
        polish_ax(ax)
        fig.tight_layout()
        save_many(fig, _stems("Figure3_risk_delta_stripplot", OUT_JBD, OUT_PUB, FIG_SRC))


def redraw_theme1_perturbation_matrix() -> None:
    pert = _read(THEME_TAB / "T_theme1_upgraded_perturbation_sensitivity_matrix.csv")
    if pert is None or pert.empty:
        return
    cols = [c for c in ["oct_findings_drop", "colposcopy_findings_drop", "clinical_context_drop", "impression_drop", "report_drop", "risk_abs_delta"] if c in pert.columns]
    if not cols:
        return
    matrix = pert.set_index("condition")[cols]
    matrix = matrix.rename(
        columns={
            "oct_findings_drop": "OCT findings",
            "colposcopy_findings_drop": "Colposcopy findings",
            "clinical_context_drop": "Clinical context",
            "impression_drop": "Impression",
            "report_drop": "Overall report",
            "risk_abs_delta": "Risk shift",
        }
    )
    fig = clustermap_figure(
        matrix,
        title="Perturbation sensitivity across report sections",
        cbar_label="Drop / shift",
        cmap=_cmap_diverging(),
        figsize=(10.2, 6.8),
    )
    save_many(fig, _stems("Figure_theme1_perturbation_sensitivity_matrix", OUT_THEME, OUT_PUB, FIG_SRC))


def redraw_api_p8_clustermap() -> None:
    tab = _read(API_TAB / "T_api_llm_provider_comparison_structured_pseudo_report_generation.csv")
    if tab is None or tab.empty:
        return
    if "Metric" in tab.columns:
        metric_order = [
            "Schema valid rate",
            "Section completeness",
            "Modality support",
            "Contradiction rate",
            "Hallucination rate",
            "Duplicate fraction",
            "Alignment MRR",
        ]
        wide = tab.set_index("Metric")
        keep = [m for m in metric_order if m in wide.index]
        if not keep:
            return
        mat = wide.loc[keep].apply(pd.to_numeric, errors="coerce").T
        mat = mat.dropna(axis=0, how="all").dropna(axis=1, how="all")
    else:
        metric_cols = [c for c in tab.columns if c not in {"provider", "provider_label", "model_label", "analysis_cohort"}]
        numeric = [c for c in metric_cols if pd.api.types.is_numeric_dtype(tab[c])]
        if not numeric or "provider_label" not in tab.columns:
            return
        mat = tab.set_index("provider_label")[numeric]
    if mat.empty:
        return
    setup_novel_theme()
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    annot = mat.copy().applymap(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
    sns.heatmap(
        mat,
        annot=annot,
        fmt="",
        cmap=_cmap_sequential(),
        vmin=0,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Metric value", "shrink": 0.78},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("LLM provider comparison for structured pseudo-report generation", pad=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    apply_arial_to_figure(fig)
    save_many(fig, _stems("P8_llm_provider_comparison_heatmap", OUT_API, OUT_PUB, FIG_SRC))


def redraw_external_baselines() -> None:
    ext = _read(MANUSCRIPT / "T_external_baselines_same_split.csv")
    if ext is None or ext.empty:
        return
    if "auc" not in ext.columns:
        return
    plot = ext.copy()
    if "auc_ci_low" in plot.columns:
        plot = plot.sort_values("auc", ascending=True)
        fig = horizontal_lollipop(
            plot,
            y="model",
            x="auc",
            xerr_low_col="auc_ci_low",
            xerr_high_col="auc_ci_high",
            title="Same-split external baseline AUROC comparison",
            xlabel="AUROC",
            figsize=(9.6, 6.2),
            refline=0.5,
        )
        save_many(fig, _stems("Figure_external_baselines_auc_forest", OUT_EXT, OUT_PUB, FIG_SRC))

    mcols = [c for c in ["auc", "f1", "sensitivity", "specificity", "auprc"] if c in ext.columns]
    if mcols:
        long = ext.melt(id_vars=["model"], value_vars=mcols, var_name="metric", value_name="score")
        setup_novel_theme()
        g = sns.FacetGrid(long, col="metric", col_wrap=3, height=3.2, aspect=1.1, sharex=False, sharey=False)
        g.map_dataframe(sns.stripplot, x="score", y="model", color=C0, size=7, jitter=0.12, orient="h")
        g.set_axis_labels("Score", "")
        g.set_titles("{col_name}")
        g.fig.suptitle("External baseline metric profile", fontsize=10, fontweight="bold", y=1.03)
        apply_arial_to_figure(g.fig)
        save_many(g.fig, _stems("Figure_external_baselines_metric_dotplot", OUT_EXT, OUT_PUB, FIG_SRC))

    paired = _read(MANUSCRIPT / "T_external_baseline_paired_bootstrap_recheck.csv")
    if paired is not None and not paired.empty:
        pcol = "delta_auc_full_minus_comparator" if "delta_auc_full_minus_comparator" in paired.columns else "delta_auc"
        if pcol in paired.columns:
            plot = paired.sort_values(pcol, ascending=True).copy()
            plot = plot.rename(columns={pcol: "delta_auc"})
            fig = horizontal_lollipop_pvals(
                plot,
                y="comparator",
                x="delta_auc",
                xerr_low_col="delta_auc_ci_low" if "delta_auc_ci_low" in plot.columns else None,
                xerr_high_col="delta_auc_ci_high" if "delta_auc_ci_high" in plot.columns else None,
                p_col="paired_bootstrap_p_two_sided" if "paired_bootstrap_p_two_sided" in plot.columns else None,
                title="Corrected paired bootstrap recheck with two-sided p-values",
                xlabel="Paired ΔAUROC (Full LCAD-RASA − comparator)",
                figsize=(10.0, 5.4),
                refline=0.0,
            )
            save_many(fig, _stems("Figure_external_baselines_paired_delta_auc", OUT_EXT, OUT_PUB, FIG_SRC))


def redraw_ablations() -> None:
    for fname, stem, ycol in (
        ("S3_modality_ablation.csv", "fig_modality_ablation_stripplot", "experiment_id"),
        ("S5_rasa_component_ablation.csv", "fig_rasa_component_boxenplot", "experiment_id"),
    ):
        df = _read(MANUSCRIPT / fname)
        if df is None or df.empty or "auc" not in df.columns:
            continue
        setup_novel_theme()
        plot = df.sort_values("auc").copy()
        fig, ax = plt.subplots(figsize=(8.6, max(4.8, 0.38 * len(plot))))
        sns.boxenplot(data=plot, y=ycol, x="auc", palette=PALETTE_MAIN, orient="h", ax=ax, linewidth=0.9)
        sns.stripplot(data=plot, y=ycol, x="auc", color=TEXT_DARK, size=7, jitter=0.12, orient="h", ax=ax)
        ax.set_xlabel("AUROC")
        ax.set_ylabel("")
        ax.set_title(stem.replace("fig_", "").replace("_", " ").title())
        polish_ax(ax)
        fig.tight_layout()
        save_many(fig, _stems(stem, OUT_PUB, FIG_SRC))

    qc = _read(MANUSCRIPT / "S4_lcad_qc_ablation.csv")
    if qc is not None and not qc.empty:
        ref = float(qc["auc"].max())
        qc = qc.sort_values("auc").copy()
        fig = horizontal_box_strip(
            qc.assign(delta=qc["auc"] - ref),
            y="experiment_id",
            x="auc",
            title="LCAD pseudo-report QC ablation",
            xlabel="AUROC",
            figsize=(8.0, max(4.0, 0.45 * len(qc))),
        )
        save_many(fig, _stems("fig_lcad_qc_ablation_barplot", OUT_PUB, FIG_SRC))

    lam = _read(MANUSCRIPT / "S1_rasa_lambda_align_sweep.csv")
    if lam is not None and not lam.empty:
        xcol = "lambda_align" if "lambda_align" in lam.columns else lam.columns[0]
        ycol = "auc" if "auc" in lam.columns else lam.columns[1]
        fig = lambda_sweep_dumbbell(
            lam,
            x=xcol,
            y=ycol,
            title="RASA alignment-weight sensitivity",
            xlabel=r"Section-alignment weight $\lambda_{\mathrm{align}}$",
            ylabel="Held-out AUROC",
            figsize=(8.4, 5.0),
        )
        save_many(fig, _stems("fig_rasa_lambda_lineplot", OUT_PUB, FIG_SRC))


def redraw_loco() -> None:
    s2 = _read(MANUSCRIPT / "S2_loco_strict_retrain.csv")
    if s2 is None or s2.empty:
        return
    setup_novel_theme()
    model_labels = {
        "full_lcad_rasa": "MOSAIC-RASA backbone",
        "real_report_only_decoder": "Real-report only",
        "report_generation_without_section_alignment": "No section alignment",
    }
    s2 = s2.copy()
    s2["model_label"] = s2["model"].map(model_labels).fillna(s2["model"])
    g = sns.relplot(
        data=s2,
        x="auc",
        y="center_label",
        hue="model_label",
        style="model_label",
        s=140,
        height=5.2,
        aspect=1.35,
        palette=PALETTE_MAIN[:3],
        kind="scatter",
        edgecolor=TEXT_DARK,
        linewidth=0.8,
    )
    g.set_axis_labels("AUROC", "Held-out centre")
    g.set_titles("")
    g.fig.suptitle("Strict LOCO AUROC by centre and model", fontsize=10, fontweight="bold", y=1.02)
    g.ax.axvline(0.5, color=C6, ls=":", lw=1.1, alpha=0.75)
    apply_arial_to_figure(g.fig)
    save_many(g.fig, _stems("fig_loco_heatmap", OUT_JBD, OUT_PUB, FIG_SRC))
    save_many(g.fig, _stems("Figure4_loco_forest_catplot", OUT_JBD, OUT_PUB, FIG_SRC))


def redraw_robustness() -> None:
    s10 = _read(MANUSCRIPT / "T10_masking_validation.csv")
    if s10 is None:
        s10 = _read(MANUSCRIPT / "S10_masking_validation.csv")
    if s10 is not None and not s10.empty:
        metric = "label_consistency_mean" if "label_consistency_mean" in s10.columns else s10.select_dtypes("number").columns[0]
        centre_col = "center_id" if "center_id" in s10.columns else "Centre"
        hue_col = "setting" if "setting" in s10.columns else None
        fig = grouped_violin_strip(
            s10,
            x=centre_col,
            y=metric,
            hue=hue_col,
            palette=PALETTE_MAIN[:4] if hue_col else None,
            title="Masking validation across agent settings",
            xlabel="Centre",
            ylabel="Label consistency",
            figsize=(9.2, 5.0),
        )
        save_many(fig, _stems("SupplementaryFigure_S1_masking_validation", OUT_JBD, OUT_PUB, FIG_SRC))

    s7 = _read(MANUSCRIPT / "S7_multiseed_stability.csv")
    if s7 is not None and not s7.empty:
        auc = s7[s7["metric"].eq("auc")].copy() if "metric" in s7.columns else s7
        if {"mean", "ci_low", "ci_high"}.issubset(auc.columns):
            fig = horizontal_lollipop_pvals(
                auc.sort_values("mean", ascending=True),
                y="model",
                x="mean",
                xerr_low_col="ci_low",
                xerr_high_col="ci_high",
                title="Multi-seed AUROC stability (95% CI)",
                xlabel="AUROC",
                figsize=(8.8, 4.6),
                refline=0.5,
            )
        else:
            ycol = "mean" if "mean" in auc.columns else "auc"
            fig = horizontal_box_strip(
                auc.sort_values(ycol),
                y="model",
                x=ycol,
                title="Multi-seed AUROC stability",
                xlabel="AUROC",
                figsize=(8.4, 4.6),
            )
        save_many(fig, _stems("SupplementaryFigure_S3_multiseed", OUT_JBD, OUT_PUB, FIG_SRC))


ALL_FIGURE_NAMES = [
    "Figure2_centre_supervision_catplot",
    "Figure_theme1_pseudo_report_source_comparison",
    "Figure_theme1_alignment_retrieval_mrr",
    "Figure_theme1_report_supervision_scarcity_curve",
    "Figure_theme1_perturbation_sensitivity_matrix",
    "P8_llm_provider_comparison_heatmap",
    "P1_stage1_quality_heatmap",
    "P2_stage1_quality_risk_bars",
    "P3_stage1_latency_support_scatter",
    "P4_stage1_generation_reliability",
    "P5_stage2_macro_mrr",
    "P6_stage2_section_mrr",
    "P7_stage3_scarcity_auc",
    "Figure_mosaic_performance_summary",
    "Figure_mosaic_metrics_heatmap",
    "Figure_main_AUC_pointplot",
    "Figure_main_metrics_heatmap",
    "Figure_main_auc_f1_scatter",
    "Figure_external_baselines_auc_forest",
    "Figure_external_baselines_metric_dotplot",
    "Figure_external_baselines_paired_delta_auc",
    "fig_rasa_lambda_lineplot",
    "fig_modality_ablation_stripplot",
    "fig_rasa_component_boxenplot",
    "fig_lcad_qc_ablation_barplot",
    "Figure3_modality_perturbation_heatmap",
    "Figure3_modality_perturbation_lineplot",
    "Figure3_risk_delta_stripplot",
    "fig_loco_heatmap",
    "Figure4_loco_forest_catplot",
    "SupplementaryFigure_S1_masking_validation",
    "SupplementaryFigure_S3_multiseed",
]


def main() -> None:
    setup_novel_theme()
    print("Regenerating base Seaborn figure set...")
    generate_all_seaborn_figures(ROOT)

    print("Applying novel individual figure styles...")
    redraw_theme1_pseudo()
    redraw_theme1_alignment()
    redraw_scarcity_curve()
    redraw_main_comparison()
    redraw_perturbation()
    redraw_theme1_perturbation_matrix()
    redraw_api_p8_clustermap()
    redraw_external_baselines()
    redraw_ablations()
    redraw_loco()
    redraw_robustness()

    subprocess_api = ROOT / "scripts/39_generate_llm_api_paper_ready_outputs.py"
    if subprocess_api.is_file():
        import subprocess

        subprocess.run([sys.executable, str(subprocess_api)], cwd=str(ROOT), check=False)

    _sync_to_project(ALL_FIGURE_NAMES)
    print("Individual novel figures synced to", FIG_SRC)


if __name__ == "__main__":
    main()
