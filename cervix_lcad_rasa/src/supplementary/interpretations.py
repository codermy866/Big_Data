"""Auto-generate interpretation markdown from result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _df_summary(path: Path, title: str) -> str:
    if not path.is_file():
        return f"## {title}\n\n*not available in the current run*\n\n"
    df = pd.read_csv(path)
    lines = [f"## {title}\n", f"Source: `{path.name}` ({len(df)} rows)\n"]
    if len(df):
        try:
            lines.append(df.head(12).to_markdown(index=False))
        except ImportError:
            lines.append("```\n" + df.head(12).to_string(index=False) + "\n```")
        lines.append("\n")
    return "\n".join(lines) + "\n"


def write_interpretations(tables_dir: Path) -> None:
    mapping = {
        "BASELINE_COMPARISON_INTERPRETATION.md": ("table_baseline_comparison.csv", "Baseline comparison"),
        "LCAD_QC_ABLATION_INTERPRETATION.md": ("table_lcad_qc_ablation.csv", "LCAD QC ablation"),
        "LOCO_INTERPRETATION_FOR_JBD.md": ("table_loco_main_results.csv", "LOCO generalization"),
        "MODALITY_ABLATION_INTERPRETATION.md": ("table_modality_ablation.csv", "Modality ablation"),
        "RASA_COMPONENT_INTERPRETATION.md": ("table_rasa_component_ablation.csv", "RASA component ablation"),
        "MODALITY_PERTURBATION_INTERPRETATION.md": ("table_modality_perturbation_extended.csv", "Modality perturbation"),
        "REPORT_SAFETY_INTERPRETATION.md": ("table_report_safety_metrics.csv", "Report safety"),
        "SCALABILITY_INTERPRETATION_FOR_JBD.md": ("table_scalability_pipeline_statistics.csv", "Scalability"),
        "MAIN_RESULTS_INTERPRETATION_FOR_MANUSCRIPT.md": ("table_main_results_for_manuscript.csv", "Main results"),
    }
    for fname, (csv_name, title) in mapping.items():
        body = _df_summary(tables_dir / csv_name, title)
        extra = {
            "BASELINE_COMPARISON_INTERPRETATION.md": (
                "\n### Key questions\n1. LCAD-RASA vs real-report-only: compare `full_lcad_rasa` vs `real_report_only_decoder` on ROUGE-L and label consistency.\n"
                "2. vs simple concat: compare against `simple_concat_fusion`.\n"
                "3. Section alignment: compare `report_generation_without_section_alignment` vs full model.\n"
            ),
            "LOCO_INTERPRETATION_FOR_JBD.md": (
                "\n### JBD narrative\nCross-centre LOCO shows whether report-anchored alignment generalizes under heterogeneous report supervision.\n"
            ),
            "SCALABILITY_INTERPRETATION_FOR_JBD.md": (
                "\n### JBD narrative\n137k+ images processed via cached embeddings; pseudo-report QC scales linearly with report-missing cases.\n"
            ),
        }.get(fname, "")
        (tables_dir / fname).write_text(body + extra, encoding="utf-8")
