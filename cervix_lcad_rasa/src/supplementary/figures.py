"""Generate supplementary figures (matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    if df.empty or y not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x].astype(str), df[y].astype(float))
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_heatmap(matrix: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_all_figures(tables_dir: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    qc = tables_dir / "table_lcad_qc_ablation.csv"
    if qc.is_file():
        df = pd.read_csv(qc)
        _save_bar(df, "experiment_id", "rouge_l", figures_dir / "fig_lcad_qc_ablation_barplot.png", "LCAD QC ablation ROUGE-L")

    if qc.is_file() and "qc_score" in pd.read_csv(tables_dir.parent / "manifests" / "full_manifest_publishable_with_llm_pseudo.csv", nrows=5).columns:
        mpath = tables_dir.parent / "manifests" / "full_manifest_publishable_with_llm_pseudo.csv"
        if mpath.is_file():
            mdf = pd.read_csv(mpath)
            pseudo = mdf[mdf["training_report_type"] == "pseudo"]
            if len(pseudo) and "qc_score" in pseudo.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(pseudo["qc_score"].dropna(), bins=20)
                ax.set_title("Pseudo-report QC score distribution")
                fig.savefig(figures_dir / "fig_lcad_qc_score_distribution.png", dpi=150)
                plt.close(fig)

    loco = tables_dir / "table_loco_main_results.csv"
    if loco.is_file():
        df = pd.read_csv(loco)
        if "held_out_center" in df.columns and "auc" in df.columns:
            piv = df.pivot_table(index="model", columns="held_out_center", values="auc", aggfunc="mean")
            _save_heatmap(piv.fillna(0).values, list(piv.columns), figures_dir / "fig_loco_heatmap.png", "LOCO AUC by center")

    pert = tables_dir / "table_modality_perturbation_extended.csv"
    if pert.is_file():
        df = pd.read_csv(pert)
        if "condition" in df.columns and "oct_findings_similarity" in df.columns:
            cols = [c for c in df.columns if c.endswith("_similarity") or c == "eds_oct_findings"]
            if cols:
                mat = df[cols].fillna(0).values[: min(12, len(df))]
                labs = df["condition"].astype(str).tolist()[: mat.shape[0]]
                _save_heatmap(mat, cols, figures_dir / "fig_perturbation_section_dependency_heatmap.png", "Perturbation section dependency")

    mod = tables_dir / "table_modality_ablation.csv"
    if mod.is_file():
        df = pd.read_csv(mod)
        if "experiment_id" in df.columns and "rouge_l" in df.columns:
            _save_bar(df, "experiment_id", "rouge_l", figures_dir / "fig_modality_ablation_section_heatmap.png", "Modality ablation")

    rasa = tables_dir / "table_rasa_component_ablation.csv"
    if rasa.is_file():
        df = pd.read_csv(rasa)
        _save_bar(df, "experiment_id", "rouge_l", figures_dir / "fig_rasa_component_ablation.png", "RASA component ablation")

    scale = tables_dir / "table_scalability_pipeline_statistics.csv"
    if scale.is_file():
        df = pd.read_csv(scale)
        _save_bar(df, "metric", "value", figures_dir / "fig_pipeline_runtime_breakdown.png", "Pipeline scale metrics")

    center = tables_dir / "table_loco_center_characteristics.csv"
    if center.is_file():
        df = pd.read_csv(center)
        _save_bar(df, "center", "total_cases", figures_dir / "fig_centerwise_data_scale.png", "Cases per center")
