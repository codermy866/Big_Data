#!/usr/bin/env python3
"""Run full revised dual-centre LCAD-RASA experiment battery (method §11 + execution prompt)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.modality_ablation import run_modality_ablations
from src.pipeline.revised_experiments import (
    load_experiment_registry,
    run_loco_summary,
    run_masking_per_centre,
    run_qc_ablation,
    run_stages_00_05,
    run_training_experiments,
)
from src.utils.config import load_config, resolve_project_root
from src.utils.io import write_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAIN_EXPERIMENTS = [
    "enshi_real_only",
    "jingzhou_real_only",
    "dual_real_only",
    "lcad_augmented",
    "simple_fusion",
    "fusion_plus_risk",
    "fusion_plus_section_alignment",
    "full_lcad_rasa",
]

RASA_ABLATIONS = [
    "simple_fusion",
    "fusion_plus_risk",
    "fusion_plus_section_alignment",
    "full_lcad_rasa",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-prep", action="store_true", help="Skip steps 00-05 if manifest+QC exist")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-loco", action="store_true")
    p.add_argument("--quick", action="store_true", help="3 epochs, 80 steps for all training")
    return p.parse_args()


def main():
    args = parse_args()
    project = resolve_project_root()
    train_cfg = load_config("configs/train.yaml", project)
    eval_cfg = load_config("configs/eval.yaml", project)
    dcfg = load_config("configs/data.yaml", project)
    eval_cfg["_data"] = dcfg
    eval_cfg["_train"] = train_cfg
    eval_cfg.setdefault("outputs", {})["generated_reports"] = dcfg["outputs"]["generated_reports"]
    eval_cfg.setdefault("outputs", {})["tables"] = dcfg["outputs"]["tables"]

    if args.quick:
        train_cfg["training"]["num_epochs"] = 3
        train_cfg["training"]["max_steps_per_epoch"] = 80

    manifest = project / "configs/experiments.yaml"
    reg = load_experiment_registry(manifest)
    with_pseudo = project / reg["manifest"]["with_pseudo"]

    python = sys.executable
    if not args.skip_prep:
        logger.info("=== Stages 00-05: audit, manifest, evidence, masking, pseudo, QC ===")
        run_stages_00_05(project, python)
        tables = project / "outputs/tables"
        run_masking_per_centre(
            project / reg["manifest"]["full"],
            project / dcfg["outputs"]["modality_evidence"],
            tables,
        )

    train_cfg["manifest"]["path"] = str(with_pseudo)
    train_cfg["manifest"]["pseudo_reports_dir"] = dcfg["outputs"]["pseudo_reports"]

    tables = project / "outputs/tables"

    if not args.skip_train:
        logger.info("=== §11.1–11.3 Training experiments ===")
        perf = run_training_experiments(project, train_cfg, eval_cfg, with_pseudo, MAIN_EXPERIMENTS)
        write_csv(perf, tables / "main_experiments_performance.csv")

        logger.info("=== §11.5 QC ablation ===")
        qc_df = run_qc_ablation(project, train_cfg, eval_cfg, with_pseudo)
        write_csv(qc_df, tables / "pseudo_report_qc_ablation.csv")

    logger.info("=== §11.7 Modality ablation ===")
    run_modality_ablations(with_pseudo, project / dcfg["outputs"]["modality_evidence"], tables)

    if not args.skip_loco:
        logger.info("=== §11.8 LOCO ===")
        loco = run_loco_summary(project, train_cfg, eval_cfg, with_pseudo)
        write_csv(loco, tables / "leave_one_center_out_summary.csv")

    logger.info("=== Export checklist ===")
    import subprocess

    subprocess.run([python, "scripts/12_export_implementation_checklist.py"], cwd=project, check=True)
    subprocess.run(
        [python, "scripts/10_finalize_summary.py", "--out", "outputs/tables/FINAL_EXPERIMENT_SUMMARY.md"],
        cwd=project,
        check=True,
    )
    subprocess.run([python, "scripts/11_export_results.py"], cwd=project, check=True)
    logger.info("=== Revised experiments complete ===")


if __name__ == "__main__":
    main()
