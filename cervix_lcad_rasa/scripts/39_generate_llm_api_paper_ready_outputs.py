#!/usr/bin/env python3
"""Generate paper-ready tables and seaborn figures for staged LLM API experiments."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supplementary.jbd_figures_seaborn import (
    C0,
    C1,
    C2,
    C4,
    C6,
    EDGE_DARK,
    FIG_FACE,
    JBD_PALETTE_HEX,
    NATURE_HEATMAP_DIV,
    NATURE_HEATMAP_SEQ,
    TEXT_DARK,
    _cmap_diverging,
    _cmap_sequential,
)
from src.supplementary.jbd_figure_typography import apply_arial_to_figure, setup_arial_rcparams
from src.supplementary.jbd_novel_figure_styles import proportion_heatmap, scarcity_heatmap

PILOT = ROOT / "outputs/publishable/llm_api_provider_comparison_aihubmix_pilot10"
FREE_EXT = ROOT / "outputs/publishable/llm_api_provider_comparison_aihubmix_stage123_100"
PAID_GPT_EXT = ROOT / "outputs/publishable/llm_api_provider_comparison_aihubmix_gpt_paid_100"
EXT = PAID_GPT_EXT if (PAID_GPT_EXT / "tables/T_api_provider_quality_comparison.csv").is_file() else FREE_EXT
OUT = ROOT / "outputs/publishable/llm_api_provider_paper_ready"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
MANUSCRIPT = ROOT / "outputs/publishable/tables/manuscript"

API_PROVIDERS = {"aihubmix_gpt", "aihubmix_qwen", "aihubmix_glm", "aihubmix_gemini"}
PROVIDER_LABELS = {
    "label_template": "Template",
    "rule_based": "Rule-based",
    "local_llm": "Local embedding LLM",
    "aihubmix_gpt": "GPT-5.5",
    "aihubmix_qwen": "Qwen-Plus",
    "aihubmix_glm": "GLM-4.7-Flash",
    "aihubmix_gemini": "Gemini-3.1-Pro",
}
MORANDI_HEX = JBD_PALETTE_HEX
MORANDI = sns.color_palette(MORANDI_HEX)
MORANDI_SEQ = _cmap_sequential()
MORANDI_DIV = _cmap_diverging()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _write(df: pd.DataFrame, name: str, manuscript: bool = True) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    p = TABLES / name
    df.to_csv(p, index=False)
    if manuscript:
        MANUSCRIPT.mkdir(parents=True, exist_ok=True)
        df.to_csv(MANUSCRIPT / name, index=False)
    return p


def _cohort(provider: str) -> str:
    if provider == "aihubmix_gpt":
        if EXT == PAID_GPT_EXT:
            return "paid_gpt_100_requested"
        return "extended_100_requested_69_valid"
    if provider in {"aihubmix_qwen", "aihubmix_glm"}:
        return "pilot10_valid"
    if provider == "aihubmix_gemini":
        return "pilot10_9_valid"
    return "offline_100_baseline" if provider in {"label_template", "rule_based", "local_llm"} else "unknown"


def _provider_label(provider: str) -> str:
    return PROVIDER_LABELS.get(provider, provider)


def _combine_quality() -> pd.DataFrame:
    ext = _read(EXT / "tables/T_api_provider_quality_comparison.csv")
    pilot = _read(PILOT / "tables/T_api_provider_quality_comparison.csv")
    rows = []
    for provider in ["label_template", "rule_based", "local_llm", "aihubmix_gpt"]:
        hit = ext[ext["provider"].eq(provider)] if not ext.empty else pd.DataFrame()
        if not hit.empty:
            rows.append(hit.iloc[0].to_dict())
    for provider in ["aihubmix_qwen", "aihubmix_glm", "aihubmix_gemini"]:
        hit = pilot[pilot["provider"].eq(provider)] if not pilot.empty else pd.DataFrame()
        if not hit.empty:
            rows.append(hit.iloc[0].to_dict())
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.insert(1, "provider_label", out["provider"].map(_provider_label))
    out.insert(2, "model_label", out["provider"].map(_provider_label))
    out.insert(3, "analysis_cohort", out["provider"].map(_cohort))
    preferred = [
        "provider",
        "provider_label",
        "model_label",
        "analysis_cohort",
        "n_cases",
        "schema_valid_rate",
        "section_completeness",
        "oct_supported_rate",
        "colposcopy_supported_rate",
        "instruction_supported_rate",
        "mean_modality_support_rate",
        "label_consistency_mean",
        "contradiction_rate",
        "hallucination_rate",
        "qc_pass_rate",
        "qc_score_mean",
        "unique_text_rate",
        "max_duplicate_fraction",
        "mean_latency_seconds",
        "mean_estimated_cost_usd",
        "cost_per_1000_cases_usd",
        "mean_prompt_tokens",
        "mean_completion_tokens",
    ]
    return out[[c for c in preferred if c in out.columns]].sort_values(
        ["provider"].map if False else "provider"
    )


def _error_class(text: str) -> str:
    t = str(text).lower()
    if "free model quota" in t or "429" in t or "rate limit" in t:
        return "rate_or_free_quota_limit"
    if "unknown model" in t or "not supported" in t:
        return "model_unavailable"
    if "decommissioned" in t:
        return "model_decommissioned"
    if not t or t == "nan":
        return ""
    return "other_api_error"


def _generation_reliability() -> pd.DataFrame:
    status = _read(EXT / "tables/T_api_provider_generation_status.csv")
    if status.empty:
        return status
    if "error" not in status.columns:
        status["error"] = ""
    status["error_class"] = status["error"].map(_error_class)
    piv = (
        status.groupby(["provider", "status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["cached", "ok", "parse_warning", "error"]:
        if col not in piv.columns:
            piv[col] = 0
    quality = _read(EXT / "tables/T_api_provider_quality_comparison.csv")
    n_valid = quality[["provider", "n_cases"]] if not quality.empty else pd.DataFrame(columns=["provider", "n_cases"])
    out = piv.merge(n_valid.rename(columns={"n_cases": "n_valid_structured_reports"}), on="provider", how="left")
    out["n_requested_or_cached"] = out[["cached", "ok", "parse_warning", "error"]].sum(axis=1)
    out["provider_label"] = out["provider"].map(_provider_label)
    out["model_label"] = out["provider"].map(_provider_label)
    out["analysis_cohort"] = out["provider"].map(_cohort)
    err = (
        status[status["status"].eq("error")]
        .groupby("provider")["error_class"]
        .agg(lambda s: s.value_counts().index[0] if len(s) else "")
        .reset_index(name="dominant_error_class")
    )
    out = out.merge(err, on="provider", how="left")
    if "dominant_error_class" not in out.columns:
        out["dominant_error_class"] = ""
    out["dominant_error_class"] = out["dominant_error_class"].fillna("")
    out["structured_report_yield"] = out["n_valid_structured_reports"] / out["n_requested_or_cached"].replace(0, np.nan)
    cols = [
        "provider",
        "provider_label",
        "model_label",
        "analysis_cohort",
        "n_requested_or_cached",
        "cached",
        "ok",
        "parse_warning",
        "error",
        "n_valid_structured_reports",
        "structured_report_yield",
        "dominant_error_class",
    ]
    return out[[c for c in cols if c in out.columns]]


def _combine_alignment() -> tuple[pd.DataFrame, pd.DataFrame]:
    ext = _read(EXT / "tables/T_api_provider_alignment_comparison.csv")
    pilot = _read(PILOT / "tables/T_api_provider_alignment_comparison.csv")
    rows = []
    for provider in ["label_template", "rule_based", "local_llm", "aihubmix_gpt"]:
        hit = ext[ext["provider"].eq(provider)] if not ext.empty else pd.DataFrame()
        if not hit.empty:
            rows.append(hit.iloc[0].to_dict())
    for provider in ["aihubmix_qwen", "aihubmix_glm", "aihubmix_gemini"]:
        hit = pilot[pilot["provider"].eq(provider)] if not pilot.empty else pd.DataFrame()
        if not hit.empty:
            rows.append(hit.iloc[0].to_dict())
    macro = pd.DataFrame(rows)
    if not macro.empty:
        macro.insert(1, "provider_label", macro["provider"].map(_provider_label))
        macro.insert(2, "model_label", macro["provider"].map(_provider_label))
        macro.insert(3, "analysis_cohort", macro["provider"].map(_cohort))

    ext_sec = _read(EXT / "tables/T_api_provider_alignment_by_section.csv")
    pilot_sec = _read(PILOT / "tables/T_api_provider_alignment_by_section.csv")
    sec_rows = []
    for provider in ["label_template", "rule_based", "local_llm", "aihubmix_gpt"]:
        hit = ext_sec[ext_sec["provider"].eq(provider)] if not ext_sec.empty else pd.DataFrame()
        if not hit.empty:
            sec_rows.append(hit)
    for provider in ["aihubmix_qwen", "aihubmix_glm", "aihubmix_gemini"]:
        hit = pilot_sec[pilot_sec["provider"].eq(provider)] if not pilot_sec.empty else pd.DataFrame()
        if not hit.empty:
            sec_rows.append(hit)
    sec = pd.concat(sec_rows, ignore_index=True) if sec_rows else pd.DataFrame()
    if not sec.empty:
        sec.insert(1, "provider_label", sec["provider"].map(_provider_label))
        sec.insert(2, "model_label", sec["provider"].map(_provider_label))
        sec.insert(3, "analysis_cohort", sec["provider"].map(_cohort))
    return macro, sec


def _combine_scarcity() -> pd.DataFrame:
    ext = _read(EXT / "tables/T_api_provider_downstream_scarcity_surrogate.csv")
    pilot = _read(PILOT / "tables/T_api_provider_downstream_scarcity_surrogate.csv")
    rows = []
    if not ext.empty and "provider" in ext.columns:
        rows.append(ext[ext["provider"].eq("aihubmix_gpt")])
    if not pilot.empty and "provider" in pilot.columns:
        rows.append(pilot[pilot["provider"].isin(["aihubmix_gemini"])])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out.insert(1, "provider_label", out["provider"].map(_provider_label))
        out.insert(2, "model_label", out["provider"].map(_provider_label))
        out.insert(3, "analysis_cohort", out["provider"].map(_cohort))
    return out


def _setup_style() -> None:
    sns.set_theme(style="ticks", context="talk", font="Arial", font_scale=1.05, palette=MORANDI)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.8,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 13,
            "patch.edgecolor": "#3a3a3a",
            "lines.linewidth": 2.2,
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _polish_axes(ax: plt.Axes, *, rotate_x: int = 0, legend: bool = False) -> None:
    ax.title.set_fontweight("bold")
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(9)
    ax.yaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(9)
    for label in ax.get_xticklabels():
        label.set_fontsize(8)
        label.set_rotation(rotate_x)
        if rotate_x:
            label.set_ha("right")
    for label in ax.get_yticklabels():
        label.set_fontsize(8)
    if legend and ax.get_legend() is not None:
        for text in ax.get_legend().get_texts():
            text.set_fontsize(8)
        title = ax.get_legend().get_title()
        if title is not None:
            title.set_fontsize(8)
            title.set_fontweight("bold")


def _plot_quality(quality: pd.DataFrame) -> None:
    plot = quality.copy()
    metrics = [
        "schema_valid_rate",
        "section_completeness",
        "mean_modality_support_rate",
        "qc_pass_rate",
        "unique_text_rate",
    ]
    heat = plot.set_index("provider_label")[metrics].astype(float).rename(
        columns={
            "schema_valid_rate": "Schema validity",
            "section_completeness": "Section completeness",
            "mean_modality_support_rate": "Modality support",
            "qc_pass_rate": "QC pass",
            "unique_text_rate": "Unique text",
        }
    )
    setup_arial_rcparams()
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap=MORANDI_SEQ,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Rate", "shrink": 0.78},
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Structured pseudo-report generation quality")
    ax.set_xlabel("")
    ax.set_ylabel("")
    _polish_axes(ax)
    apply_arial_to_figure(fig)
    _save(fig, "P1_stage1_quality_heatmap")

    long = plot.melt(
        id_vars=["provider_label", "analysis_cohort"],
        value_vars=["contradiction_rate", "max_duplicate_fraction"],
        var_name="risk_metric",
        value_name="rate",
    )
    long["risk_metric"] = long["risk_metric"].map(
        {
            "contradiction_rate": "Contradiction",
            "max_duplicate_fraction": "Duplicate fraction",
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    sns.boxenplot(
        data=long,
        y="provider_label",
        x="rate",
        hue="risk_metric",
        palette=[MORANDI_HEX[4], MORANDI_HEX[2]],
        linewidth=0.8,
        orient="h",
        ax=ax,
    )
    sns.stripplot(
        data=long,
        y="provider_label",
        x="rate",
        hue="risk_metric",
        dodge=True,
        marker="s",
        size=7,
        edgecolor=EDGE_DARK,
        linewidth=0.8,
        palette=[MORANDI_HEX[4], MORANDI_HEX[2]],
        orient="h",
        ax=ax,
        legend=False,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Rate")
    ax.set_ylabel("")
    ax.set_title("Clinical consistency and repetition risk")
    ax.legend(frameon=False, title="Metric", loc="lower right")
    _polish_axes(ax, legend=True)
    apply_arial_to_figure(fig)
    _save(fig, "P2_stage1_quality_risk_bars")

    api = plot[plot["provider"].isin(API_PROVIDERS)].dropna(subset=["mean_latency_seconds"])
    if not api.empty:
        api = api.sort_values("mean_latency_seconds").copy()
        tokens = api["mean_completion_tokens"].astype(float)
        token_min = float(tokens.min())
        token_max = float(tokens.max())

        def _bubble_size(value: float) -> float:
            if not np.isfinite(value) or token_max <= token_min:
                return 360.0
            return 180.0 + 520.0 * ((value - token_min) / (token_max - token_min))

        provider_colors = {
            "aihubmix_gpt": MORANDI_HEX[4],
            "aihubmix_qwen": MORANDI_HEX[0],
            "aihubmix_glm": MORANDI_HEX[2],
            "aihubmix_gemini": MORANDI_HEX[1],
        }
        label_offsets = {
            "GPT-5.5": (12, -4),
            "Qwen-Plus": (12, -18),
            "GLM-4.7-Flash": (-122, -18),
            "Gemini-3.1-Pro": (-132, 12),
        }

        fig, ax = plt.subplots(figsize=(9.8, 5.8))
        for _, row in api.iterrows():
            x = float(row["mean_latency_seconds"])
            y = float(row["mean_modality_support_rate"])
            model = str(row["provider_label"])
            n_cases = int(row["n_cases"]) if pd.notna(row.get("n_cases", np.nan)) else 0
            color = provider_colors.get(str(row["provider"]), MORANDI_HEX[6])
            size = _bubble_size(float(row["mean_completion_tokens"]))
            ax.scatter(
                x,
                y,
                s=size,
                color=color,
                edgecolor="#343434",
                linewidth=0.9,
                alpha=0.92,
                zorder=3,
            )
            dx, dy = label_offsets.get(model, (10, 10))
            ax.annotate(
                f"{model}\n(n={n_cases})",
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="center",
                fontsize=11.5,
                fontweight="bold",
                color="#2f2f2f",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.86),
                arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8, alpha=0.75),
                zorder=4,
            )

        if token_max > token_min:
            legend_values = np.unique(np.round(np.linspace(token_min, token_max, 3) / 100) * 100).astype(int)
            handles = [
                ax.scatter(
                    [],
                    [],
                    s=_bubble_size(float(v)),
                    color=MORANDI_HEX[6],
                    edgecolor="#343434",
                    linewidth=0.8,
                    alpha=0.82,
                )
                for v in legend_values
            ]
            ax.legend(
                handles,
                [f"{v:,}" for v in legend_values],
                title="Completion tokens",
                frameon=False,
                loc="lower right",
            )
        ax.set_xlabel("Mean latency per case (s)")
        ax.set_ylabel("Modality support rate")
        x_pad = max(0.8, (api["mean_latency_seconds"].max() - api["mean_latency_seconds"].min()) * 0.08)
        ax.set_xlim(api["mean_latency_seconds"].min() - x_pad, api["mean_latency_seconds"].max() + x_pad)
        ax.set_ylim(0.76, 1.035)
        ax.axhline(1.0, color="#666666", lw=1.0, ls=":", alpha=0.65, zorder=1)
        ax.set_title("Latency and modality support of API-generated pseudo reports")
        _polish_axes(ax, legend=True)
        _save(fig, "P3_stage1_latency_support_scatter")


def _plot_reliability(reliability: pd.DataFrame) -> None:
    if reliability.empty:
        return
    plot = reliability[reliability["provider"].isin(["aihubmix_gpt", "label_template", "local_llm", "rule_based"])].copy()
    if plot.empty:
        return
    status_cols = ["cached", "ok", "parse_warning", "error"]
    status_labels = {
        "cached": "Cached",
        "ok": "Valid",
        "parse_warning": "Parse warning",
        "error": "Error",
    }
    matrix = plot.set_index("provider_label")[status_cols].rename(columns=status_labels).astype(float)
    props = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    fig = proportion_heatmap(
        props,
        title="Generation reliability in the 100-case cohort",
        cbar_label="Outcome fraction",
        figsize=(9.4, 4.8),
    )
    apply_arial_to_figure(fig)
    _save(fig, "P4_stage1_generation_reliability")


def _plot_alignment(macro: pd.DataFrame, section: pd.DataFrame) -> None:
    if macro.empty:
        return
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    order = macro.sort_values("macro_mrr", ascending=False)["provider_label"]
    plot = macro.sort_values("macro_mrr", ascending=True).copy()
    y_pos = np.arange(len(plot))
    ax.hlines(y_pos, 0, plot["macro_mrr"], color=MORANDI_HEX[7], linewidth=6, alpha=0.9)
    sns.scatterplot(
        data=plot,
        x="macro_mrr",
        y="provider_label",
        hue="provider_label",
        palette=MORANDI_HEX[: len(plot)],
        s=230,
        edgecolor="#3a3a3a",
        linewidth=0.8,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Macro MRR")
    ax.set_ylabel("")
    ax.set_title("Modality-section retrieval alignment")
    _polish_axes(ax)
    _save(fig, "P5_stage2_macro_mrr")

    if not section.empty:
        sec = section.copy()
        sec["section"] = (
            sec["section"]
            .str.replace("_", " ", regex=False)
            .str.replace("oct", "OCT", regex=False)
            .str.title()
            .str.replace("Oct", "OCT", regex=False)
        )
        g = sns.FacetGrid(
            sec,
            col="section",
            col_wrap=2,
            height=3.2,
            aspect=1.45,
            sharex=True,
            sharey=True,
        )
        section_palette = {
            provider: MORANDI_HEX[i % len(MORANDI_HEX)]
            for i, provider in enumerate(sec["provider_label"].drop_duplicates())
        }
        g.map_dataframe(
            sns.scatterplot,
            x="mrr",
            y="provider_label",
            hue="provider_label",
            palette=section_palette,
            s=145,
            edgecolor="#3a3a3a",
            linewidth=0.7,
            legend=False,
        )
        g.set_axis_labels("MRR", "")
        g.set_titles("{col_name}")
        g.fig.suptitle("Section-specific semantic alignment", y=1.03, fontweight="bold", fontsize=10)
        for ax in g.axes.flat:
            ax.set_xlim(0, max(0.45, float(sec["mrr"].max()) * 1.12))
            ax.grid(True, axis="x", alpha=0.35)
            _polish_axes(ax)
        _save(g.fig, "P6_stage2_section_mrr")


def _plot_scarcity(scarcity: pd.DataFrame) -> None:
    if scarcity.empty or "auc_mean" not in scarcity.columns:
        return
    plot = scarcity.copy()
    plot["provider_label"] = plot["provider_label"].astype(str)
    fig = scarcity_heatmap(
        plot,
        x="real_report_fraction",
        y="auc_mean",
        hue="provider_label",
        hue_order=sorted(plot["provider_label"].unique()),
        title="Downstream performance under report-supervision scarcity",
        xlabel="Available real-report supervision fraction",
        ylabel="Provider / model",
        cbar_label="AUROC",
        err_col="auc_std" if "auc_std" in plot.columns else None,
        figsize=(9.0, max(4.2, 0.42 * plot["provider_label"].nunique())),
    )
    apply_arial_to_figure(fig)
    _save(fig, "P7_stage3_scarcity_auc")


def _summary(
    quality: pd.DataFrame,
    reliability: pd.DataFrame,
    alignment: pd.DataFrame,
    scarcity: pd.DataFrame,
) -> None:
    lines = [
        "# Paper-Ready LLM API Experiment Summary\n\n",
        "This folder consolidates staged LLM API pseudo-report experiments into manuscript-ready tables and seaborn-style figures.\n\n",
        "## Stage 1: Quality\n\n",
    ]
    if not quality.empty:
        gpt = quality[quality["provider"].eq("aihubmix_gpt")]
        if not gpt.empty:
            r = gpt.iloc[0]
            cohort_label = "paid extended cohort" if str(r.get("analysis_cohort", "")).startswith("paid") else "extended cohort"
            cost = r.get("cost_per_1000_cases_usd", np.nan)
            cost_text = (
                f", cost per 1000 cases ${cost:.2f}"
                if pd.notna(cost)
                else ""
            )
            lines.append(
        f"- GPT-5.5 {cohort_label}: {int(r['n_cases'])} valid structured reports, "
                f"schema-valid rate {r['schema_valid_rate']:.3f}, section completeness {r['section_completeness']:.3f}, "
                f"modality support {r['mean_modality_support_rate']:.3f}, contradiction rate {r['contradiction_rate']:.3f}, "
                f"latency {r['mean_latency_seconds']:.2f} s/case{cost_text}.\n"
            )
        api = quality[quality["provider"].isin(API_PROVIDERS)]
        if not api.empty:
            best = api.sort_values(["contradiction_rate", "mean_latency_seconds"], ascending=[True, True]).iloc[0]
            lines.append(f"- Best available API trade-off by low contradiction and latency: {best['provider_label']}.\n")
    lines.append("\n## Stage 2: Alignment\n\n")
    if not alignment.empty:
        top_api = alignment[alignment["provider"].isin(API_PROVIDERS)].sort_values("macro_mrr", ascending=False)
        if not top_api.empty:
            row = top_api.iloc[0]
            lines.append(f"- Best API macro MRR in available cohorts: {row['provider_label']} (MRR={row['macro_mrr']:.3f}).\n")
    lines.append("\n## Stage 3: Downstream Surrogate\n\n")
    if not scarcity.empty:
        low = scarcity[scarcity["real_report_fraction"].eq(scarcity["real_report_fraction"].min())]
        for _, row in low.iterrows():
            lines.append(
                f"- {row['provider_label']} at {row['real_report_fraction']:.0%} real reports: "
                f"AUROC={row['auc_mean']:.3f}, F1={row['f1_mean']:.3f}.\n"
            )
    lines.append("\n## Claim Limits\n\n")
    lines.append("- Cohorts are intentionally staged: GPT-5.5 has an extended cohort; Qwen-Plus/GLM-4.7-Flash/Gemini-3.1-Pro remain pilot cohorts.\n")
    lines.append("- API failures and rate limits are reported as part of operational reliability, not hidden.\n")
    lines.append("- Stage 3 is a lightweight surrogate and does not replace full LCAD-RASA retraining.\n")
    lines.append("- External APIs receive only de-identified structured evidence, not raw images, paths, patient names, hospital identifiers, or internal patient IDs.\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PAPER_READY_LLM_API_EXPERIMENT_SUMMARY.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    _setup_style()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    quality = _combine_quality()
    reliability = _generation_reliability()
    alignment, section = _combine_alignment()
    scarcity = _combine_scarcity()

    _write(quality, "T_api_stage1_quality_for_manuscript.csv")
    _write(reliability, "T_api_stage1_generation_reliability.csv")
    _write(alignment, "T_api_stage2_alignment_for_manuscript.csv")
    _write(section, "T_api_stage2_alignment_by_section_for_manuscript.csv")
    _write(scarcity, "T_api_stage3_downstream_scarcity_surrogate.csv")

    _plot_quality(quality)
    _plot_reliability(reliability)
    _plot_alignment(alignment, section)
    _plot_scarcity(scarcity)

    _summary(quality, reliability, alignment, scarcity)
    print(f"Wrote paper-ready outputs to {OUT}")


if __name__ == "__main__":
    main()
