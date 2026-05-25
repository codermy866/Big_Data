"""
Rich Seaborn-style figures for JBD LCAD-RASA results.
Inspired by: https://seaborn.pydata.org/examples/index.html
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

MANUSCRIPT_REL = "outputs/publishable/tables/manuscript"
PRED_REL = "outputs/publishable/predictions/final_per_case"
TABLES_REL = "outputs/publishable/tables"

# Seaborn gallery–style palette (husl + muted accents)
PALETTE_MAIN = sns.color_palette("husl", 8)
PALETTE_MODEL = {
    "Full LCAD-RASA": "#2A9D8F",
    "LCAD w/o section alignment": "#E9C46A",
    "Real-report only": "#E76F51",
    "Simple concat fusion": "#8D99AE",
    "Image-only report gen.": "#457B9D",
    "Instruction-only report gen.": "#9B5DE5",
    "Fusion w/o report anchor": "#F15BB5",
}


def _setup_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.85,
        rc={
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.35,
        },
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    except Exception:
        pass
    plt.close(fig)


def _read(project: Path, rel: str) -> pd.DataFrame | None:
    p = project / rel
    return pd.read_csv(p) if p.is_file() else None


def _model_palette(df: pd.DataFrame, col: str = "model") -> dict:
    keys = df[col].astype(str).unique().tolist() if col in df.columns else []
    out = {}
    for i, k in enumerate(keys):
        out[k] = PALETTE_MODEL.get(k, PALETTE_MAIN[i % len(PALETTE_MAIN)])
    return out


# ---------------------------------------------------------------------------
# Main-text figures (jbd_final/)
# ---------------------------------------------------------------------------


def fig01_pipeline_schematic(out_dir: Path) -> None:
    """Conceptual flow — matplotlib + seaborn colors (no bar chart)."""
    _setup_theme()
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        "Multicentre cohort\nn = 1,897",
        "ResNet50\nembeddings",
        "LCAD pseudo-reports\n(local structured)",
        "QC &\nweighting",
        "RASA section\nalignment",
        "Risk +\nstructured report",
    ]
    colors = sns.color_palette("crest", len(stages))
    xs = np.linspace(0.06, 0.94, len(stages))
    for i, (x, s, c) in enumerate(zip(xs, stages, colors)):
        ax.add_patch(FancyBboxPatch((x - 0.075, 0.28), 0.15, 0.44, boxstyle="round,pad=0.02", fc=c, ec="#264653", lw=1.2, alpha=0.92))
        ax.text(x, 0.5, s, ha="center", va="center", fontsize=9, color="#1d3557", fontweight="medium")
        if i < len(stages) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.085, 0.5), xytext=(x + 0.085, 0.5), arrowprops=dict(arrowstyle="-|>", color="#264653", lw=2))
    ax.set_title("LCAD-RASA: case-level report supervision under big-data cervical screening", fontsize=12, pad=12)
    _save(fig, out_dir / "Figure1_pipeline_schematic")


def fig02_centre_supervision(project: Path, out_dir: Path) -> None:
    """Catplot + proportional line — gallery: catplot / relplot."""
    t1b = _read(project, f"{MANUSCRIPT_REL}/T1b_centre_scale_and_supervision.csv")
    if t1b is None:
        return
    _setup_theme()
    long = t1b.melt(
        id_vars=["Centre", "Cases"],
        value_vars=["Real reports", "Pseudo-report candidates"],
        var_name="Supervision",
        value_name="Count",
    )
    long["Supervision"] = long["Supervision"].str.replace("Pseudo-report candidates", "Pseudo-report candidate")

    g = sns.catplot(
        data=long,
        x="Centre",
        y="Count",
        hue="Supervision",
        kind="bar",
        palette=["#2A9D8F", "#E76F51"],
        height=5,
        aspect=1.35,
        edgecolor="#333",
        linewidth=0.6,
        legend_out=True,
    )
    g.set_axis_labels("Centre", "Cases")
    g.fig.suptitle("Centre-level report supervision imbalance", y=1.02, fontsize=13)
    g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=18, ha="right")
    _save(g.fig, out_dir / "Figure2_centre_supervision_catplot")

    # Proportion line (relplot style)
    prop = long.copy()
    prop["fraction"] = prop.groupby("Centre")["Count"].transform(lambda x: x / x.sum())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.lineplot(data=prop, x="Centre", y="fraction", hue="Supervision", marker="o", linewidth=2.5, ax=ax, palette=["#2A9D8F", "#E76F51"])
    ax.set_ylabel("Fraction of cases")
    ax.set_title("Report supervision mix by centre")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Supervision", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=18, ha="right")
    fig.tight_layout()
    _save(fig, out_dir / "Figure2_centre_supervision_fraction_line")
    t1b.to_csv(out_dir / "Figure2_centre_supervision_source.csv", index=False)


def fig03_perturbation(project: Path, out_dir: Path) -> None:
    """Heatmap + lineplot — gallery: heatmap / lineplot."""
    s6 = _read(project, f"{MANUSCRIPT_REL}/S6_modality_perturbation_text_decoding.csv")
    if s6 is None:
        return
    _setup_theme()
    conds = [
        "normal",
        "mask_oct",
        "mask_colposcopy",
        "mask_instruction",
        "mask_visual",
        "label_only_inference",
    ]
    sub = s6[s6["condition"].isin(conds)].copy()
    sec_cols = [
        "oct_findings_similarity_to_normal",
        "colposcopy_findings_similarity_to_normal",
        "clinical_context_similarity_to_normal",
        "impression_similarity_to_normal",
    ]
    melt = sub.melt(id_vars=["condition"], value_vars=sec_cols, var_name="section", value_name="similarity")
    melt["section"] = melt["section"].str.replace("_similarity_to_normal", "").str.replace("_", " ")

    # Heatmap (clustermap-style without cluster)
    piv = melt.pivot(index="condition", columns="section", values="similarity").reindex(conds)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(
        piv,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Similarity to normal"},
        ax=ax,
    )
    ax.set_title("Modality perturbation: section-specific degradation (n = 128)")
    ax.set_xlabel("")
    fig.tight_layout()
    _save(fig, out_dir / "Figure3_modality_perturbation_heatmap")

    # Lineplot with markers
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=melt, x="condition", y="similarity", hue="section", marker="o", linewidth=2, ax=ax, palette="Set2")
    ax.set_ylim(0, 1.08)
    ax.set_title("Perturbation response by report section")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Section", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir / "Figure3_modality_perturbation_lineplot")

    # Risk delta strip
    if "risk_score_delta_vs_normal" in sub.columns:
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.stripplot(
            data=sub,
            x="condition",
            y="risk_score_delta_vs_normal",
            size=10,
            jitter=0.25,
            palette="mako",
            ax=ax,
            alpha=0.85,
        )
        sns.pointplot(data=sub, x="condition", y="risk_score_delta_vs_normal", color="#264653", markers="D", linestyles="-", ax=ax, errorbar=None)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_title("Risk score shift vs normal input")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        _save(fig, out_dir / "Figure3_risk_delta_stripplot")
    sub.to_csv(out_dir / "Figure3_modality_perturbation_source.csv", index=False)


def fig_main_model_comparison(project: Path, out_dir: Path) -> None:
    """Pointplot + metrics heatmap — gallery: pointplot / heatmap."""
    t2 = _read(project, f"{MANUSCRIPT_REL}/T2_main_model_comparison_with_ci.csv")
    if t2 is None:
        t2 = _read(project, f"{MANUSCRIPT_REL}/T2_main_model_comparison.csv")
    if t2 is None:
        return
    _setup_theme()
    if "auc" not in t2.columns and "auc_all" in t2.columns:
        t2 = t2.rename(columns={"auc_all": "auc", "f1_at_val_threshold": "f1"})

    # Pointplot with CI
    plot_df = t2.copy()
    if "ci_low" in plot_df.columns:
        plot_df["order"] = plot_df["auc"].rank(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        pal = _model_palette(plot_df)
        for i, row in plot_df.iterrows():
            yi = list(plot_df.index).index(i)
            c = pal.get(row["model"], "#333")
            ax.errorbar(
                row["auc"],
                yi,
                xerr=[[row["auc"] - row["ci_low"]], [row["ci_high"] - row["auc"]]],
                fmt="o",
                ecolor="#555",
                elinewidth=2,
                capsize=4,
                markersize=10,
                color=c,
                linestyle="none",
            )
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df["model"])
        ax.set_xlabel("AUROC (95% bootstrap CI)")
        ax.set_title("Main model comparison (test n = 288)")
        ax.set_xlim(0.35, 0.95)
        ax.axvline(0.5, color="gray", ls=":", lw=1, alpha=0.7)
        fig.tight_layout()
        _save(fig, out_dir / "Figure_main_AUC_pointplot")

    # Metrics heatmap
    mcols = [c for c in ["auc", "f1", "sensitivity", "specificity"] if c in t2.columns]
    if mcols:
        hm = t2.set_index("model")[mcols]
        fig, ax = plt.subplots(figsize=(7, 4 + 0.35 * len(hm)))
        sns.heatmap(hm, annot=True, fmt=".3f", cmap="vlag", center=0.5, vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Score"})
        ax.set_title("Multi-metric profile (validation-selected threshold)")
        fig.tight_layout()
        _save(fig, out_dir / "Figure_main_metrics_heatmap")

    # F1 vs AUC scatter (joint-style)
    if "f1" in t2.columns and "auc" in t2.columns:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        sns.scatterplot(data=t2, x="auc", y="f1", hue="model", s=180, palette=_model_palette(t2), ax=ax, edgecolor="white", linewidth=0.8)
        for _, r in t2.iterrows():
            ax.annotate(r["model"].split()[0], (r["auc"], r["f1"]), fontsize=7, alpha=0.8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlim(0.3, 0.9)
        ax.set_title("Risk–semantic trade-off (AUC vs F1)")
        ax.legend(loc="lower right", fontsize=7, title="Model")
        fig.tight_layout()
        _save(fig, out_dir / "Figure_main_auc_f1_scatter")
    t2.to_csv(out_dir / "Figure_main_AUC_comparison_source.csv", index=False)


def fig_per_case_distributions(project: Path, out_dir: Path) -> None:
    """Violin + kde + joint — gallery: violinplot / kdeplot / jointplot."""
    pred_dir = project / PRED_REL
    if not pred_dir.is_dir():
        return
    frames = []
    for p in sorted(pred_dir.glob("*_test_predictions.csv")):
        model = p.name.replace("_test_predictions.csv", "")
        d = pd.read_csv(p)
        d["model"] = model.replace("_", " ").replace("full lcad rasa", "Full LCAD-RASA").title()
        if "full_lcad_rasa" in p.name:
            d["model"] = "Full LCAD-RASA"
        elif "real_report" in p.name:
            d["model"] = "Real-report only"
        elif "simple_concat" in p.name:
            d["model"] = "Simple concat"
        elif "report_generation" in p.name:
            d["model"] = "LCAD w/o section"
        frames.append(d)
    if not frames:
        return
    _setup_theme()
    allp = pd.concat(frames, ignore_index=True)
    allp["CIN2+ label"] = allp["y_true_cin2plus"].map({0: "Negative", 1: "Positive"})
    core = allp[allp["model"].isin(["Full LCAD-RASA", "Real-report only", "Simple concat", "LCAD w/o section"])]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.violinplot(data=core, x="model", y="risk_score", hue="CIN2+ label", split=True, inner="quart", palette=["#457B9D", "#E76F51"], ax=ax, linewidth=0.8)
    sns.swarmplot(data=core.sample(min(400, len(core)), random_state=42), x="model", y="risk_score", hue="CIN2+ label", dodge=True, size=2, alpha=0.35, ax=ax, legend=False)
    ax.set_title("Predicted risk distributions by model and true label (test set)")
    ax.set_ylabel("Risk score")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _save(fig, out_dir / "Figure_risk_violin_swarm")

    # KDE for full model only
    full = allp[allp["model"] == "Full LCAD-RASA"]
    if len(full):
        g = sns.displot(data=full, x="risk_score", hue="CIN2+ label", kind="kde", fill=True, height=4.5, aspect=1.4, palette=["#457B9D", "#E76F51"], linewidth=2)
        g.fig.suptitle("Full LCAD-RASA: risk score density", y=1.02)
        _save(g.fig, out_dir / "Figure_full_model_kdeplot")

        g2 = sns.jointplot(data=full, x="risk_score", y="correct", hue="CIN2+ label", kind="scatter", height=5, palette=["#457B9D", "#E76F51"], marginal_kws=dict(fill=True))
        g2.ax_joint.set_ylabel("Prediction correct (0/1)")
        g2.ax_joint.set_xlabel("Risk score")
        g2.fig.suptitle("Joint: risk vs correctness (Full LCAD-RASA)", y=1.02)
        _save(g2.fig, out_dir / "Figure_full_model_jointplot")


# ---------------------------------------------------------------------------
# Supplementary / legacy figure names (outputs/publishable/figures/)
# ---------------------------------------------------------------------------


def supp_masking(project: Path, out_dir: Path) -> None:
    s10 = _read(project, f"{MANUSCRIPT_REL}/S10_masking_validation.csv")
    if s10 is None:
        return
    _setup_theme()
    centres = [c for c in s10["center_id"].unique() if "pooled" not in str(c).lower()]
    sub = s10[s10["center_id"].isin(centres) | s10["center_id"].astype(str).str.contains("pooled")]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.pointplot(data=sub, x="setting", y="label_consistency_mean", hue="center_id", markers="o", linestyles="-", ax=ax, palette="Set1", errorbar=None)
    ax.set_ylim(0.45, 0.82)
    ax.set_title("LCAD masking validation: label-consistency proxy")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, out_dir / "fig_masking_pointplot")
    _save(fig, out_dir / "SupplementaryFigure_S1_masking_validation")


def supp_loco(project: Path, out_dir: Path) -> None:
    s2 = _read(project, f"{MANUSCRIPT_REL}/S2_loco_strict_retrain.csv")
    if s2 is None:
        s2 = _read(project, f"{TABLES_REL}/table_loco_main_results.csv")
    if s2 is None:
        return
    _setup_theme()
    s2 = s2.copy()
    s2["model_short"] = s2["model"].astype(str).str.replace("_", " ")
    piv = s2.pivot_table(index="center_label", columns="model_short", values="auc", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="icefire", center=0.65, vmin=0.3, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Strict LOCO: AUROC by held-out centre and model")
    fig.tight_layout()
    _save(fig, out_dir / "fig_loco_heatmap")

    g = sns.catplot(
        data=s2,
        x="auc",
        y="center_label",
        hue="model_short",
        kind="point",
        dodge=0.45,
        join=False,
        height=5,
        aspect=1.3,
        palette="tab10",
        markers="d",
        linestyles="",
    )
    g.set(xlabel="AUROC", title="Cross-centre generalisation (LOCO)")
    g.ax.axvline(0.5, ls=":", c="gray")
    _save(g.fig, out_dir / "Figure4_loco_forest_catplot")
    _save(g.fig, out_dir / "SupplementaryFigure_S2_loco_catplot")


def supp_lambda_sweep(project: Path, out_dir: Path) -> None:
    s1 = _read(project, f"{MANUSCRIPT_REL}/S1_rasa_lambda_align_sweep.csv")
    if s1 is None:
        return
    _setup_theme()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=s1, x="lambda_align", y="auc", marker="o", linewidth=2.5, color="#2A9D8F", ax=ax)
    ax.fill_between(s1["lambda_align"], s1["auc"] - 0.02, s1["auc"] + 0.02, alpha=0.2, color="#2A9D8F")
    ax.set_xlabel("RASA section-alignment weight λ_align")
    ax.set_ylabel("AUROC")
    ax.set_title("Hyperparameter sweep: λ_align vs discrimination")
    fig.tight_layout()
    _save(fig, out_dir / "fig_rasa_lambda_lineplot")


def supp_modality_ablation(project: Path, out_dir: Path) -> None:
    s3 = _read(project, f"{MANUSCRIPT_REL}/S3_modality_ablation.csv")
    if s3 is None:
        return
    _setup_theme()
    s3 = s3.copy()
    s3["modality_set"] = s3["experiment_id"].str.replace("_", " + ")
    s3 = s3.sort_values("auc", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=s3, y="modality_set", x="auc", palette=sns.color_palette("flare", len(s3)), ax=ax, orient="h")
    ax.set_xlim(0.55, 0.85)
    ax.set_title("Modality ablation: AUROC by input combination")
    fig.tight_layout()
    _save(fig, out_dir / "fig_modality_ablation_barplot")

    # Dot + strip composite
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.stripplot(data=s3, x="auc", y="modality_set", size=12, palette="viridis", ax=ax, jitter=False)
    sns.pointplot(data=s3, x="auc", y="modality_set", color="#264653", markers="X", linestyles="", ax=ax)
    ax.set_title("Modality subsets (strip + point)")
    fig.tight_layout()
    _save(fig, out_dir / "fig_modality_ablation_stripplot")
    _save(fig, out_dir / "fig_modality_ablation_section_heatmap")  # alias for old name


def supp_rasa_components(project: Path, out_dir: Path) -> None:
    s5 = _read(project, f"{MANUSCRIPT_REL}/S5_rasa_component_ablation.csv")
    if s5 is None:
        return
    _setup_theme()
    ref = s5[s5["experiment_id"] == "full_lcad_rasa"]
    ref_auc = float(ref["auc"].iloc[0]) if len(ref) else 0.8
    s5 = s5.copy()
    s5["delta_auc"] = s5["auc"] - ref_auc
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in s5["delta_auc"]]
    sns.barplot(data=s5, x="experiment_id", y="delta_auc", palette=colors, ax=ax, legend=False)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("RASA component ablation: ΔAUROC vs full model")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    _save(fig, out_dir / "fig_rasa_component_ablation")

    # Boxen-style via boxenplot
    if "f1" in s5.columns:
        melt = s5.melt(id_vars=["experiment_id"], value_vars=["auc", "f1"], var_name="metric", value_name="value")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxenplot(data=melt, x="experiment_id", y="value", hue="metric", palette="pastel", ax=ax)
        ax.set_title("Component ablation metrics (boxen)")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        _save(fig, out_dir / "fig_rasa_component_boxenplot")


def supp_multiseed(project: Path, out_dir: Path) -> None:
    s7 = _read(project, f"{MANUSCRIPT_REL}/S7_multiseed_stability.csv")
    if s7 is None:
        return
    _setup_theme()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sub = s7[s7["metric"] == "auc"]
    sns.pointplot(
        data=sub,
        x="model",
        y="mean",
        hue="model",
        palette=_model_palette(sub, "model"),
        ax=ax,
        errorbar=("ci", 95),
        markers="o",
        linestyles="",
        legend=False,
    )
    ax.set_ylabel("AUC (multi-seed mean ± CI)")
    ax.set_title("Random-seed stability")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, out_dir / "SupplementaryFigure_S3_multiseed")

    # Facet by metric
    g = sns.FacetGrid(s7, col="metric", height=3.5, aspect=1.1, sharey=False)
    g.map_dataframe(sns.pointplot, x="model", y="mean", errorbar=None, palette="husl")
    g.set_xticklabels(rotation=25)
    g.fig.suptitle("Stability across metrics and seeds", y=1.05)
    _save(g.fig, out_dir / "fig_multiseed_facetgrid")


def supp_qc_and_scalability(project: Path, out_dir: Path) -> None:
    s4 = _read(project, f"{MANUSCRIPT_REL}/S4_lcad_qc_ablation.csv")
    if s4 is not None:
        _setup_theme()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=s4, x="experiment_id", y="auc", palette="Blues", ax=ax, linewidth=0.8)
        sns.swarmplot(data=s4, x="experiment_id", y="auc", color="#264653", size=7, ax=ax)
        ax.set_title("LCAD QC ablation (box + swarm)")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        _save(fig, out_dir / "fig_lcad_qc_ablation_barplot")

    s11 = _read(project, f"{MANUSCRIPT_REL}/S11_scalability_and_runtime.csv")
    if s11 is not None:
        _setup_theme()
        pipe = s11[s11["section"] == "pipeline_scale"] if "section" in s11.columns else s11
        key = pipe[pipe["metric"].isin(["total_cases", "total_images", "real_report_cases", "pseudo_report_cases"])]
        if len(key):
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=key, x="value", y="metric", palette="crest", orient="h", ax=ax)
            ax.set_xscale("log")
            ax.set_xlabel("Count (log scale)")
            ax.set_title("Pipeline scale (S11)")
            fig.tight_layout()
            _save(fig, out_dir / "fig_pipeline_runtime_breakdown")
            _save(fig, out_dir / "SupplementaryFigure_S4_scalability")

    centre = _read(project, f"{TABLES_REL}/table_loco_center_characteristics.csv")
    if centre is None:
        centre = _read(project, f"{MANUSCRIPT_REL}/T1b_centre_scale_and_supervision.csv")
    if centre is not None:
        _setup_theme()
        c = centre.copy()
        if "center" in c.columns:
            c = c.rename(columns={"center": "Centre", "cases": "Cases"})
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if "OCT images" in c.columns:
            melt = c.melt(id_vars=["Centre"], value_vars=["OCT images", "Colposcopy images"], var_name="Modality", value_name="Images")
            sns.barplot(data=melt, x="Centre", y="Images", hue="Modality", palette="muted", ax=ax)
            ax.set_yscale("log")
            ax.set_title("Imaging volume by centre (log scale)")
        elif "Cases" in c.columns:
            sns.barplot(data=c, x="Centre", y="Cases", hue="Centre", palette="mako", ax=ax, legend=False)
            ax.set_title("Cases per centre")
        elif "cases" in c.columns:
            sns.barplot(data=c, x="center", y="cases", hue="center", palette="mako", ax=ax, legend=False)
            ax.set_title("Cases per centre")
        ax.tick_params(axis="x", rotation=18)
        fig.tight_layout()
        _save(fig, out_dir / "fig_centerwise_data_scale")


def supp_perturbation_extended(project: Path, out_dir: Path) -> None:
    pert = _read(project, f"{MANUSCRIPT_REL}/S6b_modality_perturbation_extended.csv")
    if pert is None:
        pert = _read(project, f"{TABLES_REL}/table_modality_perturbation_extended.csv")
    if pert is None:
        return
    _setup_theme()
    sim_cols = [c for c in pert.columns if "similarity" in c.lower() or c.endswith("_to_normal")]
    if not sim_cols:
        return
    sub = pert.head(20)
    melt = sub.melt(id_vars=["condition"] if "condition" in sub.columns else [], value_vars=sim_cols[:6], var_name="metric", value_name="value")
    if "condition" not in melt.columns:
        return
    piv = melt.pivot_table(index="condition", columns="metric", values="value", aggfunc="mean")
    cg = sns.clustermap(piv.fillna(0), cmap="mako_r", figsize=(10, 8), linewidths=0.3, annot=True, fmt=".2f", dendrogram_ratio=0.12)
    cg.fig.suptitle("Extended perturbation: clustered similarity", y=1.02)
    _save(cg.fig, out_dir / "fig_perturbation_section_dependency_heatmap")
    _save(cg.fig, out_dir / "fig_perturbation_clustermap")


def supp_pairwise_tests(project: Path, out_dir: Path) -> None:
    pw = _read(project, f"{MANUSCRIPT_REL}/T2_pairwise_statistical_tests.csv")
    if pw is None:
        return
    _setup_theme()
    pw = pw.copy()
    pw["neg_log_p"] = -np.log10(pw["bootstrap_p_auc"].clip(1e-6, 1))
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.scatterplot(data=pw, x="delta_auc", y="neg_log_p", hue="comparator", s=120, palette="Set2", ax=ax, edgecolor="w")
    ax.axvline(0, color="gray", ls="--")
    ax.set_xlabel("ΔAUROC (comparator − Full LCAD-RASA)")
    ax.set_ylabel("−log10(bootstrap p)")
    ax.set_title("Paired comparisons vs full model")
    ax.legend(loc="upper left", fontsize=7, title="")
    fig.tight_layout()
    _save(fig, out_dir / "fig_pairwise_comparison_scatter")


def write_figure_index(out_dir: Path, entries: list[tuple[str, str, str]]) -> None:
    lines = [
        "# JBD Figure Index (Seaborn gallery style)\n",
        f"Regenerated: {datetime.now(timezone.utc).isoformat()}\n",
        "Reference: https://seaborn.pydata.org/examples/index.html\n\n",
    ]
    for stem, plot_type, desc in entries:
        lines.append(f"## {stem}\n- **Plot type**: {plot_type}\n- {desc}\n- Files: `{stem}.png`, `{stem}.pdf`\n\n")
    (out_dir / "JBD_FINAL_FIGURE_INDEX.md").write_text("".join(lines), encoding="utf-8")


def generate_all_seaborn_figures(project: Path) -> list[str]:
    """Regenerate jbd_final + publishable/figures with varied Seaborn plot types."""
    _setup_theme()
    jbd_final = project / "outputs/publishable/figures/jbd_final"
    pub_fig = project / "outputs/publishable/figures"
    jbd_final.mkdir(parents=True, exist_ok=True)
    pub_fig.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, str, str]] = []

    fig01_pipeline_schematic(jbd_final)
    entries.append(("Figure1_pipeline_schematic", "flow schematic", "Five-stage pipeline"))

    fig02_centre_supervision(project, jbd_final)
    entries.append(("Figure2_centre_supervision_catplot", "catplot + lineplot", "Centre supervision"))
    entries.append(("Figure2_centre_supervision_fraction_line", "lineplot", "Supervision fractions"))

    fig03_perturbation(project, jbd_final)
    entries.append(("Figure3_modality_perturbation_heatmap", "heatmap", "Section similarity matrix"))
    entries.append(("Figure3_modality_perturbation_lineplot", "lineplot", "Perturbation by section"))
    entries.append(("Figure3_risk_delta_stripplot", "stripplot + pointplot", "Risk shift under perturbation"))

    fig_main_model_comparison(project, jbd_final)
    entries.append(("Figure_main_AUC_pointplot", "pointplot + errorbar", "Main AUROC with CI"))
    entries.append(("Figure_main_metrics_heatmap", "heatmap", "Multi-metric profile"))
    entries.append(("Figure_main_auc_f1_scatter", "scatterplot", "AUC–F1 trade-off"))

    fig_per_case_distributions(project, jbd_final)
    entries.append(("Figure_risk_violin_swarm", "violin + swarm", "Risk by model and label"))
    entries.append(("Figure_full_model_kdeplot", "kdeplot", "Full model risk density"))
    entries.append(("Figure_full_model_jointplot", "jointplot", "Risk vs correctness"))

    # Mirror key figures to pub_fig with legacy names
    for name in (
        "Figure_main_AUC_pointplot",
        "Figure3_modality_perturbation_heatmap",
        "Figure2_centre_supervision_catplot",
    ):
        src = jbd_final / f"{name}.png"
        if src.is_file():
            shutil.copy2(src, pub_fig / f"{name}.png")

    supp_masking(project, jbd_final)
    supp_masking(project, pub_fig)
    entries.append(("SupplementaryFigure_S1_masking_validation", "pointplot", "Masking validation"))

    supp_loco(project, jbd_final)
    supp_loco(project, pub_fig)
    entries.append(("fig_loco_heatmap", "heatmap", "LOCO AUROC matrix"))
    entries.append(("Figure4_loco_forest_catplot", "catplot strip/point", "LOCO forest"))

    supp_lambda_sweep(project, pub_fig)
    entries.append(("fig_rasa_lambda_lineplot", "lineplot", "λ_align sweep"))

    supp_modality_ablation(project, pub_fig)
    entries.append(("fig_modality_ablation_stripplot", "strip + point", "Modality ablation"))

    supp_rasa_components(project, pub_fig)
    entries.append(("fig_rasa_component_boxenplot", "boxenplot", "RASA components"))

    supp_multiseed(project, jbd_final)
    supp_multiseed(project, pub_fig)

    supp_qc_and_scalability(project, pub_fig)
    entries.append(("fig_lcad_qc_ablation_barplot", "box + swarm", "QC ablation"))

    supp_perturbation_extended(project, pub_fig)
    entries.append(("fig_perturbation_clustermap", "clustermap", "Extended perturbation"))

    supp_pairwise_tests(project, jbd_final)
    entries.append(("fig_pairwise_comparison_scatter", "scatterplot", "Paired statistical tests"))

    write_figure_index(jbd_final, entries)
    write_figure_index(pub_fig, entries)

    return [e[0] for e in entries]
