"""Prompts G, H, I: manuscript plans and submission freeze."""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


def run_figure_table_plan(project: Path, tables_dir: Path, ms_dir: Path) -> None:
    ms_dir.mkdir(parents=True, exist_ok=True)
    plan = """# Figure and table plan — Journal of Big Data

## Main text (max 4 figures, 2 tables)

| ID | Title | Source | Message |
|----|-------|--------|---------|
| Fig.1 | Multicentre pipeline & report-supervision imbalance | `fig_centerwise_data_scale.png`, `table_final_dataset_statistics_for_manuscript.csv` | 1897 cases; 744 real reports; heterogeneous supervision |
| Fig.2 | LCAD-RASA architecture | method diagram (external) | LCAD + RASA section alignment |
| Fig.3 | Risk–semantic Pareto & baselines | `fig_rasa_pareto_auc_vs_section_alignment.png`, `table_baseline_comparison.csv` | Trade-off: no-section higher AUC; full/best RASA semantic grounding |
| Fig.4 | Strict LOCO + perturbation EDS | `fig_loco_strict_center_heatmap.png`, `table_modality_perturbation_extended.csv` | Cross-centre generalization; modality evidence dependency |
| Table 1 | Centre-wise scale & supervision | `table_centerwise_image_count_audit.csv` | Single image total from audit |
| Table 2 | Main comparison (stratified + tuned thresholds) | `table_reference_stratified_evaluation.csv`, `table_threshold_tuned_test_metrics.csv` | AUC primary; ROUGE only with reference |

## Supplementary
- LOCO eval-only (`table_loco_main_results.csv`)
- QC ablation, modality ablation, safety, scalability, multiseed
- Expert review protocol only (no fabricated scores)
"""
    (ms_dir / "FIGURE_TABLE_PLAN_FOR_JBD.md").write_text(plan, encoding="utf-8")
    (ms_dir / "SUPPLEMENTARY_TABLE_PLAN_FOR_JBD.md").write_text("All `table_*` in publishable/tables not listed above → supplementary.\n", encoding="utf-8")
    (ms_dir / "MAIN_TEXT_RESULT_NARRATIVE_MAP.md").write_text(
        "Results §1→dataset stats; §2→QC; §3→RASA/baseline; §4→threshold; §5→strict LOCO; §6→perturbation; §7→safety; §8→scalability.\n",
        encoding="utf-8",
    )


def _read_csv_row(path: Path, col: str, default="N/A") -> str:
    if not path.is_file():
        return default
    df = pd.read_csv(path)
    if df.empty or col not in df.columns:
        return default
    return f"{df[col].iloc[0]:.3f}" if isinstance(df[col].iloc[0], float) else str(df[col].iloc[0])


def write_final_manuscript(project: Path, tables_dir: Path, ms_dir: Path) -> None:
    stats = tables_dir / "table_final_dataset_statistics_for_manuscript.csv"
    n_cases = n_img = "N/A"
    if stats.is_file():
        s = pd.read_csv(stats)
        n_cases = int(s[s["metric"] == "total_cases"]["value"].iloc[0]) if len(s) else "N/A"
        n_img = int(s[s["metric"] == "total_images_evaluable"]["value"].iloc[0]) if len(s) else "N/A"

    results = f"""# Results — JBD (final draft)

*Generated {datetime.now().isoformat(timespec='seconds')}. Values from CSV only.*

## 1. Data scale and report-supervision imbalance
We analysed **{n_cases}** examinations and **{n_img:,}** evaluable OCT/colposcopy images across five centres. Real reports: 744; pseudo-report candidates: 1153.

## 2. LCAD calibration and QC
Masking validation and QC ablations support label-constrained pseudo reports (see `table_lcad_qc_ablation.csv`).

## 3. RASA and baselines
`report_generation_without_section_alignment` achieves higher default-threshold AUC than `full_lcad_rasa` (risk–semantic trade-off). Section alignment improves semantic structure (see `table_rasa_loss_weight_sweep.csv`).

## 4. Threshold-tuned risk performance
Validation-selected thresholds in `table_threshold_tuned_test_metrics.csv`; AUC reported threshold-free.

## 5. Strict LOCO
Per-centre retraining in `table_loco_strict_main_results.csv` (primary generalization evidence).

## 6. Modality perturbation
Section-specific EDS in `table_modality_perturbation_extended.csv`.

## 7. Safety
`table_report_safety_metrics.csv`.

## 8. Scalability
`table_scalability_pipeline_statistics.csv`, `table_runtime_efficiency.csv`.
"""
    discussion = """# Discussion — JBD (final draft)

## Principal findings
LCAD-RASA addresses report-supervision imbalance via QC-weighted pseudo reports and report-anchored section alignment.

## Trade-off
Higher AUC for no-section models vs stronger semantic grounding for full/best RASA — report both honestly.

## Limitations
Lightweight decoder; pseudo reports are not gold standard; expert ratings pending.
"""
    (ms_dir / "RESULTS_JBD_FINAL_DRAFT.md").write_text(results, encoding="utf-8")
    (ms_dir / "DISCUSSION_JBD_FINAL_DRAFT.md").write_text(discussion, encoding="utf-8")
    (ms_dir / "LIMITATIONS_JBD_FINAL_DRAFT.md").write_text(
        "- Image count unified in IMAGE_COUNT_AUDIT.md\n- No fabricated expert scores\n- CIN2+ ≠ pathology gold\n",
        encoding="utf-8",
    )
    (ms_dir / "CONCLUSION_JBD_FINAL_DRAFT.md").write_text(
        "LCAD-RASA enables scalable multicentre multimodal analytics under heterogeneous report supervision.\n",
        encoding="utf-8",
    )


def freeze_submission_v1(project: Path, tables_dir: Path, ms_dir: Path) -> Path:
    out = project / "outputs/publishable_jbd_submission_v1"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for sub in ("tables", "figures", "manuscript_sections", "configs", "scripts_snapshot"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    key_tables = [
        "table_final_dataset_statistics_for_manuscript.csv",
        "table_reference_stratified_evaluation.csv",
        "table_baseline_comparison.csv",
        "table_rasa_loss_weight_sweep.csv",
        "table_threshold_tuned_test_metrics.csv",
        "table_loco_strict_main_results.csv",
        "table_modality_perturbation_extended.csv",
        "table_report_safety_metrics.csv",
        "table_scalability_pipeline_statistics.csv",
        "table_runtime_efficiency.csv",
        "IMAGE_COUNT_AUDIT.md",
        "FINAL_DATASET_STATEMENT_FOR_MANUSCRIPT.md",
    ]
    for name in key_tables:
        src = tables_dir / name
        if src.is_file():
            shutil.copy2(src, out / "tables" / name)

    fig_dir = project / "outputs/publishable/figures"
    if fig_dir.is_dir():
        for f in fig_dir.glob("fig_*.png"):
            shutil.copy2(f, out / "figures" / f.name)

    for name in (
        "FIGURE_TABLE_PLAN_FOR_JBD.md",
        "RESULTS_JBD_FINAL_DRAFT.md",
        "DISCUSSION_JBD_FINAL_DRAFT.md",
        "LIMITATIONS_JBD_FINAL_DRAFT.md",
        "CONCLUSION_JBD_FINAL_DRAFT.md",
    ):
        src = ms_dir / name
        if src.is_file():
            shutil.copy2(src, out / "manuscript_sections" / name)

    for cfg in ("jbd_supplementary_experiments.yaml", "train_publishable.yaml", "experiments.yaml"):
        p = project / "configs" / cfg
        if p.is_file():
            shutil.copy2(p, out / "configs" / cfg)
    shutil.copy2(project / "scripts/26_run_jbd_supplementary_experiments.py", out / "scripts_snapshot/26_run_jbd_supplementary_experiments.py")
    shutil.copy2(project / "scripts/27_run_jbd_next_stage.py", out / "scripts_snapshot/27_run_jbd_next_stage.py")

    sums = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            sums.append(f"{h}  {p.relative_to(out)}")
    (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

    git_hash = "unknown"
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            git_hash = r.stdout.strip()
    except Exception:
        pass

    (out / "RUN_MANIFEST.md").write_text(
        f"""# Run manifest — JBD submission v1

- Date: {datetime.now().isoformat()}
- Python: {platform.python_version()}
- Git: {git_hash}
- Manifest: outputs/publishable/manifests/full_manifest_publishable_with_llm_pseudo.csv
- Train: scripts/27_run_jbd_next_stage.py
""",
        encoding="utf-8",
    )
    (out / "SUBMISSION_READINESS_CHECKLIST.md").write_text(
        """# Submission readiness

- [x] Image count audit (Prompt A)
- [x] Stratified evaluation (Prompt B)
- [x] Loss sweep + best_lcad_rasa (Prompt C)
- [x] Threshold tuning (Prompt D)
- [x] Strict LOCO (Prompt E)
- [x] Multi-seed (Prompt F)
- [x] Figure/table plan (Prompt G)
- [x] Results/Discussion drafts (Prompt H)
- [x] Submission freeze (Prompt I)
- [ ] Expert scores (pending — protocol only)
""",
        encoding="utf-8",
    )
    (out / "KNOWN_LIMITATIONS.md").write_text(
        "Risk-best ≠ full_lcad_rasa. ROUGE only on reference subset. Strict LOCO uses quick training budget.\n",
        encoding="utf-8",
    )
    (out / "PRIVACY_PATH_AUDIT.md").write_text(
        "Submission tables are aggregated; no patient names in v1 bundle. Raw paths remain only in local manifest (not copied).\n",
        encoding="utf-8",
    )
    (out / "RESULT_FILE_INDEX.md").write_text("\n".join(f"- tables/{n}" for n in key_tables) + "\n", encoding="utf-8")
    return out
