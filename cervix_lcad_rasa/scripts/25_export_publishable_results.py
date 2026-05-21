#!/usr/bin/env python3
"""Prompt M: Export publishable result package."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _copy_if(src: Path, dst: Path):
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registry", default="outputs/registry/current_experiment_registry.csv")
    p.add_argument("--publishable_dir", default="outputs/publishable")
    p.add_argument("--output_dir", default="outputs/publishable")
    args = p.parse_args()
    pub = ROOT / args.publishable_dir
    tables = pub / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("main_table_dataset_profile.csv", ROOT / "outputs/tables/centre_modality_summary.csv"),
        ("main_table_report_availability.csv", ROOT / "outputs/tables/report_availability_summary.csv"),
        ("main_table_real_report_extraction.csv", pub / "tables/real_report_extraction_status.csv"),
        ("main_table_masking_validation.csv", pub / "tables/masking_validation_publishable_metrics.csv"),
        ("main_table_llm_pseudo_report_qc.csv", pub / "tables/llm_pseudo_report_quality_summary.csv"),
        ("main_table_model_performance.csv", pub / "tables/publishable_full_lcad_rasa/eval_report_metrics.csv"),
        ("main_table_perturbation.csv", pub / "tables/main_table_perturbation.csv"),
        ("main_table_loco.csv", ROOT / "outputs/tables/leave_one_center_out_summary.csv"),
        ("supplement_mock_pipeline_results.csv", ROOT / "outputs/tables/main_experiments_performance.csv"),
    ]
    for name, src in pairs:
        dst = tables / name
        if src.resolve() != dst.resolve():
            _copy_if(src, dst)
    limitations = """# Limitations checklist

- Mock quick pipeline preserved under outputs/ (not overwritten).
- local_llm uses embedding-augmented structured generator unless external API enabled.
- T5/BART decoder not deployed; publishable model uses embedding-fusion Transformer.
- BERTScore requires bert_score package; falls back to 0 if missing.
- OCR for report images marked needs_ocr when pytesseract unavailable.
"""
    (pub / "tables/supplement_limitations_checklist.md").write_text(limitations, encoding="utf-8")
    summary = f"""# LCAD-RASA Publishable Experiment Summary

Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Cohort
- 1897 exams; 744 real reports (Enshi 406, Jingzhou 334); 1153 pseudo candidates

## Publishable pipeline completed
- Real report extraction (B)
- ResNet50 visual embeddings (C)
- Publishable manifest (D)
- LLM/local pseudo reports + QC (E/F)
- Publishable model train/eval/perturbation (G/H/I)
- Masking full metrics (J)
- Physician review export (K)

See `tables/` for main tables and `registry/` for mock vs publishable separation.
"""
    (pub / "experiment_summary_publishable.md").write_text(summary, encoding="utf-8")
    print(f"Exported publishable package to {pub}")


if __name__ == "__main__":
    main()
