"""Generate Results_JBD_DRAFT.md from actual CSV tables (Prompt 12)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SECTIONS = [
    ("1. Multicentre multimodal data scale and report-supervision imbalance", "table_scalability_pipeline_statistics.csv"),
    ("2. LCAD calibration and pseudo-report quality control", "table_lcad_qc_ablation.csv"),
    ("3. RASA improves report-anchored multimodal semantic alignment", "table_baseline_comparison.csv"),
    ("4. Modality-specific evidence dependency", "table_modality_ablation.csv"),
    ("5. Cross-centre generalization under heterogeneous supervision", "table_loco_main_results.csv"),
    ("6. Safety, hallucination and clinical plausibility", "table_report_safety_metrics.csv"),
    ("7. Scalability and computational efficiency", "table_runtime_efficiency.csv"),
]


def build_results_draft(tables_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Results — Journal of Big Data (draft)\n", "*Auto-generated from `outputs/publishable/tables/*.csv`. No fabricated values.*\n"]
    for title, csv_name in SECTIONS:
        p = tables_dir / csv_name
        lines.append(f"## {title}\n")
        if not p.is_file():
            lines.append("not available in the current run.\n")
            continue
        df = pd.read_csv(p)
        if df.empty:
            lines.append("Table empty.\n")
            continue
        # lead sentence from first row metrics
        row = df.iloc[0].to_dict()
        nums = {k: v for k, v in row.items() if isinstance(v, (int, float)) and not pd.isna(v)}
        if nums:
            top = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(nums.items())[:5])
            lines.append(f"Primary metrics ({row.get('experiment_id', csv_name)}): {top}.\n")
        try:
            lines.append(df.head(8).to_markdown(index=False))
        except ImportError:
            lines.append("```\n" + df.head(8).to_string(index=False) + "\n```")
        lines.append("\n")

    lines.append("## Recommended main-text tables\n")
    lines.append("- Main Table 1: `table_scalability_pipeline_statistics.csv`\n")
    lines.append("- Main Table 2: `table_baseline_comparison.csv` + `table_rasa_component_ablation.csv`\n")
    lines.append("- Supplementary: LOCO, modality ablation, perturbation extended, safety\n")

    (out_dir / "RESULTS_JBD_DRAFT.md").write_text("\n".join(lines), encoding="utf-8")
    calls = ["# Figure and table call-outs\n", "- Fig.1: pipeline scale (`fig_centerwise_data_scale.png`)\n", "- Fig.2: baseline + RASA (`fig_rasa_component_ablation.png`)\n", "- Fig.3: LOCO heatmap + perturbation heatmap\n"]
    (out_dir / "FIGURE_TABLE_CALL_OUTS.md").write_text("\n".join(calls), encoding="utf-8")
