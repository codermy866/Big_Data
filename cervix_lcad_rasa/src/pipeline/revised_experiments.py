"""Orchestrate revised dual-centre LCAD-RASA experiments (method §11)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.distillation.masking_validation import run_masking_validation
from src.pipeline.modality_ablation import run_modality_ablations
from src.pipeline.train_eval import run_eval, run_train
from src.training.experiment_modes import apply_train_filter, load_experiment_registry
from src.utils.config import ensure_dir, load_config
from src.utils.io import write_csv


def _merge_cfg(base: dict, exp_name: str, registry: dict) -> dict:
    cfg = json.loads(json.dumps(base))
    spec = registry["experiments"].get(exp_name, {})
    cfg["training"]["experiment_name"] = exp_name
    cfg["experiment_spec"] = spec
    td = registry.get("training_defaults", {})
    cfg["training"]["num_epochs"] = int(td.get("num_epochs", cfg["training"].get("num_epochs", 5)))
    cfg["training"]["max_steps_per_epoch"] = int(
        td.get("max_steps_per_epoch", cfg["training"].get("max_steps_per_epoch", 200))
    )
    return cfg


def run_stages_00_05(project: Path, python: str) -> None:
    import subprocess

    steps = [
        [python, "scripts/00_audit_data.py"],
        [python, "scripts/01_build_manifest.py"],
        [python, "scripts/02_extract_modality_evidence.py"],
        [python, "scripts/03_report_rich_masking_validation.py", "--client", "mock"],
    ]
    for cmd in steps:
        subprocess.run(cmd, cwd=project, check=True)

    pseudo = project / "outputs/pseudo_reports"
    if pseudo.is_dir():
        shutil.rmtree(pseudo)
    pseudo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [python, "scripts/04_generate_pseudo_reports.py", "--client", "mock", "--setting", "modality_plus_label"],
        cwd=project,
        check=True,
    )
    subprocess.run([python, "scripts/05_qc_pseudo_reports.py"], cwd=project, check=True)


def run_masking_per_centre(manifest: Path, evidence_dir: Path, tables: Path) -> None:
    df = pd.read_csv(manifest)
    rows = []
    for cid in ["enshi", "jingzhou", "xiangyang"]:
        sub = df[(df["center_id"] == cid) & (df["has_real_report"] == 1)]
        if sub.empty:
            continue
        mdf = run_masking_validation(sub, evidence_dir, tables.parent / "masking_validation", max_cases=0)
        for _, r in mdf.iterrows():
            rows.append({"center_id": cid, **r.to_dict()})
    write_csv(pd.DataFrame(rows), tables / "masking_validation_by_centre_detailed.csv")


def run_training_experiments(
    project: Path,
    train_cfg: dict,
    eval_cfg: dict,
    manifest: Path,
    experiment_names: list[str],
) -> pd.DataFrame:
    registry = load_experiment_registry(project / "configs/experiments.yaml")
    rows = []
    for name in experiment_names:
        cfg = _merge_cfg(train_cfg, name, registry)
        cfg["manifest"]["path"] = str(manifest)
        ckpt = run_train(name, cfg)
        ecfg = dict(eval_cfg)
        ecfg["evaluation"]["experiment"] = name
        m = run_eval(name, ecfg, manifest, ckpt)
        rows.append({"experiment": name, "checkpoint": str(ckpt), **m})
    return pd.DataFrame(rows)


def run_qc_ablation(
    project: Path,
    train_cfg: dict,
    eval_cfg: dict,
    manifest_path: Path,
) -> pd.DataFrame:
    registry = load_experiment_registry(project / "configs/experiments.yaml")
    df = pd.read_csv(manifest_path)
    rows = []
    for ab in registry.get("qc_ablations", []):
        mdf = df.copy()
        if ab["require_qc_pass"]:
            mdf.loc[mdf["needs_pseudo_report"] == 1, "pseudo_report_pass_qc"] = mdf.loc[
                mdf["needs_pseudo_report"] == 1, "pseudo_report_pass_qc"
            ].fillna(0).astype(int)
            mdf = mdf[(mdf["has_real_report"] == 1) | (mdf["pseudo_report_pass_qc"] == 1)]
        if not ab.get("use_confidence_weight", True):
            mdf.loc[mdf["needs_pseudo_report"] == 1, "pseudo_training_weight"] = mdf.loc[
                mdf["needs_pseudo_report"] == 1, "pseudo_report_pass_qc"
            ].astype(float)

        tmp = project / "outputs/manifests" / f"qc_ablation_{ab['name']}.csv"
        mdf.to_csv(tmp, index=False)
        cfg = _merge_cfg(train_cfg, "lcad_augmented", registry)
        cfg["training"]["experiment_name"] = f"qc_ablation_{ab['name']}"
        cfg["manifest"]["path"] = str(tmp)
        ckpt = run_train(cfg["training"]["experiment_name"], cfg)
        m = run_eval(cfg["training"]["experiment_name"], eval_cfg, tmp, ckpt)
        rows.append({"qc_ablation": ab["name"], **m})
    return pd.DataFrame(rows)


def run_loco_summary(
    project: Path,
    train_cfg: dict,
    eval_cfg: dict,
    manifest: Path,
) -> pd.DataFrame:
    """Leave-one-centre-out: train on 4 centres, test on held-out (5-fold summary)."""
    df = pd.read_csv(manifest)
    centers = sorted(df["center_id"].unique())
    rows = []
    registry = load_experiment_registry(project / "configs/experiments.yaml")
    for held in centers:
        train_df = df[(df["center_id"] != held) & (df["split"] == "train")]
        test_df = df[(df["center_id"] == held) & (df["split"] == "test")]
        if len(train_df) < 50 or len(test_df) < 10:
            rows.append({"held_out_center": held, "status": "skipped_insufficient_n", "n_train": len(train_df), "n_test": len(test_df)})
            continue
        tmp_manifest = project / "outputs/manifests" / f"loco_holdout_{held}.csv"
        part = df.copy()
        part.loc[part["center_id"] == held, "split"] = "test"
        part.loc[part["center_id"] != held, "split"] = "train"
        part.to_csv(tmp_manifest, index=False)

        cfg = _merge_cfg(train_cfg, "lcad_augmented", registry)
        exp = f"loco_holdout_{held}"
        cfg["training"]["experiment_name"] = exp
        cfg["manifest"]["path"] = str(tmp_manifest)
        cfg["training"]["num_epochs"] = 3
        cfg["training"]["max_steps_per_epoch"] = 80
        try:
            ckpt = run_train(exp, cfg)
            m = run_eval(exp, eval_cfg, tmp_manifest, ckpt)
            n_real_test = int((test_df["has_real_report"] == 1).sum())
            rows.append(
                {
                    "held_out_center": held,
                    "status": "completed",
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "n_test_real_report": n_real_test,
                    **m,
                }
            )
        except Exception as exc:
            rows.append({"held_out_center": held, "status": f"failed:{exc}", "n_test": len(test_df)})
    return pd.DataFrame(rows)
