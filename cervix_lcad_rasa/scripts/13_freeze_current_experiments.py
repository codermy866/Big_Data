#!/usr/bin/env python3
"""Prompt A: Freeze current mock/quick experiment outputs into registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def scan_outputs(outputs_dir: Path) -> list[dict]:
    rows = []
    stages = [
        ("00_audit", "scripts/00_audit_data.py", ["data_audit_report.md"], "outputs/data_audit_report.md"),
        ("01_manifest", "scripts/01_build_manifest.py", ["manifests/full_manifest.csv"], "outputs/manifests/full_manifest.csv"),
        ("02_evidence", "scripts/02_extract_modality_evidence.py", ["tables/modality_evidence_status.csv"], "outputs/modality_evidence"),
        ("03_masking", "scripts/03_report_rich_masking_validation.py", ["tables/report_rich_masking_validation_metrics.csv"], "outputs/masking_validation"),
        ("04_pseudo", "scripts/04_generate_pseudo_reports.py", [], "outputs/pseudo_reports"),
        ("05_qc", "scripts/05_qc_pseudo_reports.py", ["manifests/full_manifest_with_pseudo_reports.csv"], "outputs/manifests/full_manifest_with_pseudo_reports.csv"),
    ]
    for eid, script, files, out in stages:
        status = "completed" if (outputs_dir / out.split("/", 1)[-1]).exists() or (outputs_dir / Path(out).name).exists() else "missing"
        rows.append(
            {
                "experiment_id": eid,
                "stage": eid.split("_")[0],
                "script_or_module": script,
                "output_files": out,
                "checkpoint_path": "",
                "status": status,
                "notes": "mock/quick pipeline",
            }
        )
    ckpt_root = outputs_dir / "checkpoints"
    if ckpt_root.is_dir():
        for d in sorted(ckpt_root.iterdir()):
            if d.is_dir() and (d / "best.ckpt").is_file():
                rows.append(
                    {
                        "experiment_id": d.name,
                        "stage": "train",
                        "script_or_module": "scripts/06_train_lcad_rasa.py",
                        "output_files": str(d / "best.ckpt"),
                        "checkpoint_path": str(d / "best.ckpt"),
                        "status": "completed",
                        "notes": "quick training checkpoint",
                    }
                )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs_dir", default="outputs")
    p.add_argument("--output_dir", default="outputs/registry")
    args = p.parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = ROOT / args.outputs_dir
    df = pd.DataFrame(scan_outputs(outputs_dir))
    df.to_csv(out_dir / "current_experiment_registry.csv", index=False)
    md = ["# Current Experiment Registry", "", f"Scanned: `{outputs_dir}`", "", df.to_string(index=False)]
    (out_dir / "current_experiment_registry.md").write_text("\n".join(md), encoding="utf-8")
    missing = """# Missing Publishable Items

1. Visual encoder (ResNet/ViT) embeddings — in progress under outputs/publishable/
2. T5/BART text decoder — publishable model uses embedding-fusion decoder
3. Real LLM Agent — local_llm client (embedding-augmented)
4. OCR/real report text — scripts/14
5. BLEU/METEOR/BERTScore — scripts/20, 22
6. Full masking metrics — script 22
7. Perturbation degradation — script 21
8. Physician review publishable — script 23
"""
    (out_dir / "missing_publishable_items.md").write_text(missing, encoding="utf-8")
    print(f"Wrote registry ({len(df)} rows) to {out_dir}")


if __name__ == "__main__":
    main()
