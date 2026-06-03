#!/usr/bin/env python3
"""Build the manuscript LLM provider comparison table requested for JBD Theme 1."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
READY = ROOT / "outputs/publishable/llm_api_provider_paper_ready"
TABLES = READY / "tables"
FIGS = READY / "figures"
MANUSCRIPT = ROOT / "outputs/publishable/tables/manuscript"

PROVIDER_ORDER = [
    ("label_template", "Template"),
    ("rule_based", "Rule-based"),
    ("local_llm", "Local embedding LLM"),
    ("aihubmix_qwen", "Qwen-Plus"),
    ("aihubmix_glm", "GLM-4.7-Flash"),
    ("aihubmix_gemini", "Gemini-3.1-Pro"),
    ("aihubmix_gpt", "GPT-5.5"),
]
MORANDI_HEX = [
    "#8b98b3",
    "#abb8cc",
    "#dbb98c",
    "#edd6b8",
    "#b57979",
    "#dea3a2",
    "#b3b0b0",
    "#d9d8d8",
]
MORANDI = sns.color_palette(MORANDI_HEX)
MORANDI_DIV = LinearSegmentedColormap.from_list(
    "morandi_div",
    ["#8b98b3", "#d9d8d8", "#b57979"],
    N=256,
)

METRICS = [
    ("schema_valid_rate", "Schema valid rate", ".3f"),
    ("section_completeness", "Section completeness", ".3f"),
    ("mean_modality_support_rate", "Modality support", ".3f"),
    ("contradiction_rate", "Contradiction rate", ".3f"),
    ("hallucination_rate", "Hallucination rate", ".3f"),
    ("max_duplicate_fraction", "Duplicate fraction", ".3f"),
    ("macro_mrr", "Alignment MRR", ".3f"),
    ("cost_per_1000_cases_usd", "Cost per 1000 cases", "cost"),
    ("mean_latency_seconds", "Latency per case (s)", ".2f"),
]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _fmt(value, fmt: str) -> str:
    if fmt == "cost":
        if pd.isna(value) or str(value).strip().lower() in {"", "nan", "na", "none", "<na>"}:
            return "not_configured"
        try:
            return f"{float(value):.2f}"
        except Exception:
            return str(value)
    if pd.isna(value):
        return "not_configured" if not fmt else "NA"
    if not fmt:
        return str(value)
    try:
        return format(float(value), fmt)
    except Exception:
        return str(value)


def _setup_style() -> None:
    sns.set_theme(style="ticks", context="talk", font="Arial", font_scale=1.05, palette=MORANDI)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.8,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "patch.edgecolor": "#3a3a3a",
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        }
    )


def main() -> None:
    _setup_style()
    quality = _read(TABLES / "T_api_stage1_quality_for_manuscript.csv")
    alignment = _read(TABLES / "T_api_stage2_alignment_for_manuscript.csv")
    reliability = _read(TABLES / "T_api_stage1_generation_reliability.csv")

    merged = quality.merge(
        alignment[["provider", "macro_mrr"]] if not alignment.empty else pd.DataFrame(columns=["provider", "macro_mrr"]),
        on="provider",
        how="left",
    )
    if "cost_per_1000_cases_usd" not in merged.columns:
        merged["cost_per_1000_cases_usd"] = pd.NA
    merged["cost_per_1000_cases_usd"] = merged["cost_per_1000_cases_usd"].astype("object")
    merged.loc[merged["provider"].isin(["label_template", "rule_based", "local_llm"]), "cost_per_1000_cases_usd"] = "0 API cost"
    if not reliability.empty:
        merged = merged.merge(
            reliability[["provider", "n_requested_or_cached", "n_valid_structured_reports", "structured_report_yield", "dominant_error_class"]],
            on="provider",
            how="left",
        )

    wide_rows = []
    for key, label, fmt in METRICS:
        row = {"Metric": label}
        for provider, p_label in PROVIDER_ORDER:
            hit = merged[merged["provider"].eq(provider)]
            row[p_label] = "NA" if hit.empty else _fmt(hit.iloc[0].get(key, pd.NA), fmt)
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)

    cohort = {"Metric": "Analysis cohort"}
    valid_n = {"Metric": "Valid reports / requested"}
    reliability_note = {"Metric": "Operational note"}
    for provider, p_label in PROVIDER_ORDER:
        hit = merged[merged["provider"].eq(provider)]
        if hit.empty:
            cohort[p_label] = "NA"
            valid_n[p_label] = "NA"
            reliability_note[p_label] = "NA"
            continue
        r = hit.iloc[0]
        cohort[p_label] = str(r.get("analysis_cohort", ""))
        requested = r.get("n_requested_or_cached", r.get("n_cases", pd.NA))
        valid = r.get("n_valid_structured_reports", r.get("n_cases", pd.NA))
        if pd.isna(requested):
            requested = r.get("n_cases", pd.NA)
        if pd.isna(valid) and not pd.isna(requested):
            valid = requested
        if pd.isna(requested) or pd.isna(valid):
            valid_n[p_label] = str(int(r.get("n_cases", 0)))
        else:
            valid_n[p_label] = f"{int(valid)}/{int(requested)}"
        note = r.get("dominant_error_class", "")
        if pd.isna(note) or not str(note).strip():
            note = "completed"
        reliability_note[p_label] = str(note)
    wide = pd.concat([pd.DataFrame([cohort, valid_n, reliability_note]), wide], ignore_index=True)

    TABLES.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)
    out_name = "T_api_llm_provider_comparison_structured_pseudo_report_generation.csv"
    wide.to_csv(TABLES / out_name, index=False)
    wide.to_csv(MANUSCRIPT / out_name, index=False)

    long_metrics = ["schema_valid_rate", "section_completeness", "mean_modality_support_rate", "contradiction_rate", "hallucination_rate", "max_duplicate_fraction", "macro_mrr"]
    plot = merged[merged["provider"].isin([p for p, _ in PROVIDER_ORDER])].copy()
    label_map = dict(PROVIDER_ORDER)
    plot["Provider"] = plot["provider"].map(label_map)
    heat = plot.set_index("Provider")[long_metrics].astype(float)
    rename = {
        "schema_valid_rate": "Schema",
        "section_completeness": "Completeness",
        "mean_modality_support_rate": "Modality support",
        "contradiction_rate": "Contradiction",
        "hallucination_rate": "Hallucination",
        "max_duplicate_fraction": "Duplication",
        "macro_mrr": "Alignment MRR",
    }
    heat = heat.rename(columns=rename)

    fig, ax = plt.subplots(figsize=(12.6, 6.2))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap=MORANDI_DIV,
        center=0.5,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Metric value"},
        annot_kws={"size": 12, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Structured pseudo-report generation: model comparison")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.title.set_fontweight("bold")
    ax.title.set_fontsize(18)
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
        label.set_fontweight("bold")
        label.set_rotation(25)
        label.set_ha("right")
    for label in ax.get_yticklabels():
        label.set_fontsize(12)
    FIGS.mkdir(parents=True, exist_ok=True)
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(FIGS / "P8_llm_provider_comparison_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / "P8_llm_provider_comparison_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    print(TABLES / out_name)


if __name__ == "__main__":
    main()
