#!/usr/bin/env python3
"""Step 13: Export experiment summary and main tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import resolve_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _copy_table(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_dir", default="outputs")
    args = p.parse_args()
    out = Path(args.outputs_dir)
    if not out.is_absolute():
        out = resolve_project_root() / out
    tables = out / "tables"

    mappings = {
        "main_table_data_profile.csv": tables / "centre_modality_summary.csv",
        "main_table_masking_validation.csv": tables / "report_rich_masking_validation_metrics.csv",
        "main_table_pseudo_report_qc.csv": tables / "pseudo_report_quality_summary.csv",
        "main_table_model_performance.csv": tables / "full_lcad_rasa/eval_report_metrics.csv",
        "main_table_agent_ablation.csv": tables / "agent_setting_ablation.csv",
        "main_table_rasa_ablation.csv": tables / "rasa_component_ablation.csv",
        "main_table_modality_ablation.csv": tables / "modality_ablation_summary.csv",
        "main_table_modality_perturbation.csv": tables / "modality_perturbation_summary.csv",
        "main_table_center_generalization.csv": tables / "leave_one_center_out_summary.csv",
    }
    for dst_name, src in mappings.items():
        _copy_table(src, tables / dst_name)

    lines = ["# LCAD-RASA Experiment Summary", ""]
    for name in mappings:
        pth = tables / name
        lines.append(f"## {name}")
        if pth.is_file():
            df = pd.read_csv(pth)
            lines.append("```\n" + df.head(20).to_string(index=False) + "\n```")
        else:
            lines.append("_pending_")
        lines.append("")

    exp_summary = out / "experiment_summary.md"
    exp_summary.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", exp_summary)


if __name__ == "__main__":
    main()
