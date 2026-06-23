#!/usr/bin/env python3
"""Redraw all manuscript figures into final_Fig/ with compact Arial typography."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
FINAL_FIG = PROJECT / "final_Fig"

sys.path.insert(0, str(ROOT))


def _run() -> None:
    import subprocess

    FINAL_FIG.mkdir(parents=True, exist_ok=True)

    # Regenerate canonical figures with Arial theme via existing pipelines.
    subprocess.run([sys.executable, str(ROOT / "scripts/41_restyle_all_experiment_figures.py")], cwd=str(ROOT), check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/56_redraw_individual_novel_figures.py")], cwd=str(ROOT), check=True)

    names = [
        "Figure1_mosaic_overview",
        "Figure2_centre_supervision_catplot",
        "Figure_theme1_pseudo_report_source_comparison",
        "P8_llm_provider_comparison_heatmap",
        "P1_stage1_quality_heatmap",
        "P2_stage1_quality_risk_bars",
        "P3_stage1_latency_support_scatter",
        "P4_stage1_generation_reliability",
        "Figure_theme1_alignment_retrieval_mrr",
        "P5_stage2_macro_mrr",
        "P6_stage2_section_mrr",
        "Figure_theme1_report_supervision_scarcity_curve",
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
        "Figure_theme1_perturbation_sensitivity_matrix",
        "fig_loco_heatmap",
        "Figure4_loco_forest_catplot",
        "SupplementaryFigure_S1_masking_validation",
        "SupplementaryFigure_S3_multiseed",
    ]

    sources = [
        PROJECT / "figures",
        ROOT / "outputs/publishable/figures/jbd_final",
        ROOT / "outputs/publishable/figures",
        ROOT / "outputs/publishable/theme1_alignment/figures",
        ROOT / "outputs/publishable/llm_api_provider_paper_ready/figures",
        ROOT / "outputs/publishable/external_baselines/figures",
    ]

    copied: list[str] = []
    missing: list[str] = []
    for name in names:
        ok = False
        for src_dir in sources:
            for ext in (".pdf", ".png"):
                src = src_dir / f"{name}{ext}"
                if src.is_file() or src.is_symlink():
                    real = src.resolve()
                    if real.is_file():
                        shutil.copy2(real, FINAL_FIG / f"{name}{ext}")
                        copied.append(f"{name}{ext}")
                        ok = True
        if not ok:
            missing.append(name)

    manifest = FINAL_FIG / "FIGURE_MANIFEST.txt"
    manifest.write_text(
        "Arial-redrawn manuscript figures (compact typography)\n"
        f"Copied: {len(copied)} files\n"
        f"Missing stems: {len(missing)}\n\n"
        + "\n".join(f"- {n}" for n in sorted(set(copied)))
        + ("\n\nMissing:\n" + "\n".join(f"- {m}" for m in missing) if missing else "")
    )
    print(f"Exported {len(set(copied))} file entries to {FINAL_FIG}")
    if missing:
        print("Missing:", ", ".join(missing))


if __name__ == "__main__":
    _run()
