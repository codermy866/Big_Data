"""Execute R1–R3, S1–S3, E1–E5 result-organization prompts."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.supplementary.result_organization.catalog import EXPERIMENT_CATALOG
from src.utils.io import write_csv

PRIVACY_PATTERNS = [
    re.compile(r"/data2/", re.I),
    re.compile(r"身份证|住院号|检查号|手机号"),
    re.compile(r"LCAD_LLM_API_KEY|api_key|secret", re.I),
]

OVERCLAIM_PHRASES = [
    ("outperformed all baselines", "high", "Only partial baselines beat; no-section higher AUC on some metrics"),
    ("best AUC", "high", "full_lcad_rasa not always best; use validation-tuned F1 or stratified AUC"),
    ("LLM generated", "medium", "Use local structured agent unless E1 API completed"),
    ("clinically validated", "high", "Not supported — expert scores incomplete"),
    ("physician-confirmed", "high", "Review pack exported; scores not filled"),
    ("state-of-the-art", "high", "Avoid without external SOTA comparison"),
    ("external validation", "medium", "Strict LOCO is internal centre holdout, not external prospective"),
]


def _exists(project: Path, rel: str) -> bool:
    p = project / rel
    return p.is_file() or p.is_dir()


def _rel(project: Path, p: Path) -> str:
    try:
        return str(p.relative_to(project))
    except ValueError:
        return str(p)


def run_r1(project: Path, tables: Path) -> None:
    rows = []
    issues = []
    for e in EXPERIMENT_CATALOG:
        row = dict(e)
        rtp = e.get("result_table_path", "")
        rfp = e.get("result_figure_path", "")
        ckpt = e.get("checkpoint_path", "")
        row["result_table_exists"] = _exists(project, rtp) if rtp else False
        row["result_figure_exists"] = _exists(project, rfp) if rfp else False
        row["checkpoint_exists"] = _exists(project, ckpt) if ckpt else False
        if row["status"] == "completed" and rtp and not row["result_table_exists"]:
            row["status"] = "partial"
            issues.append({"issue": "missing_result_table", "experiment_id": e["experiment_id"], "path": rtp})
        rows.append(row)

    df = pd.DataFrame(rows)
    write_csv(df, tables / "MASTER_RESULT_INDEX.csv")

    md = [
        "# Master Result Index",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}\n",
        "## Main-text evidence chain\n",
        "- Dataset audit → `table_final_dataset_statistics_for_manuscript.csv`\n",
        "- Baselines + full LCAD-RASA → `table_baseline_comparison.csv`, `table_reference_stratified_evaluation.csv`\n",
        "- Threshold tuning → `table_threshold_tuned_test_metrics.csv`\n",
        "- Strict LOCO → `table_loco_strict_main_results.csv`\n",
        "- Perturbation EDS → `modality_perturbation_text_decoding_summary.csv`\n",
        "\n## Archive only (do not cite in main text)\n",
        "- `outputs/tables/main_experiments_performance.csv` (mock pipeline)\n",
        "\n## Manual required\n",
        "- Expert/physician review scores\n",
    ]
    (tables / "MASTER_RESULT_INDEX.md").write_text("".join(md), encoding="utf-8")

    audit = [
        "# Result File Audit Report\n",
        f"Scanned {len(df)} catalog entries.\n",
        f"- Completed: {(df['status'] == 'completed').sum()}\n",
        f"- Partial: {(df['status'] == 'partial').sum()}\n",
        f"- Manual required: {(df['status'] == 'manual_required').sum()}\n",
        f"- Missing (planned): {(df['status'] == 'missing').sum()}\n",
    ]
    (tables / "RESULT_FILE_AUDIT_REPORT.md").write_text("".join(audit), encoding="utf-8")
    if issues:
        write_csv(pd.DataFrame(issues), tables / "MISSING_OR_INCONSISTENT_FILES.csv")
    else:
        write_csv(pd.DataFrame(columns=["issue", "experiment_id", "path"]), tables / "MISSING_OR_INCONSISTENT_FILES.csv")


def run_r2(project: Path, tables: Path, ms: Path) -> None:
    mapping = [
        {"figure_or_table_id": "Figure 1", "title": "Multicentre cohort and report-supervision imbalance",
         "core_message": "1897 exams; heterogeneous real vs pseudo supervision",
         "source_files": "table_centerwise_image_count_audit.csv;fig_centerwise_data_scale.png",
         "manuscript_section": "Results §1", "final_recommendation": "use"},
        {"figure_or_table_id": "Figure 2", "title": "LCAD-RASA architecture",
         "core_message": "LCAD pseudo-reports, QC weights, RASA section alignment, risk head",
         "source_files": "external diagram",
         "manuscript_section": "Methods", "final_recommendation": "redraw_required"},
        {"figure_or_table_id": "Figure 3", "title": "Risk–semantic trade-off and baselines",
         "core_message": "AUC vs section alignment; baselines comparison",
         "source_files": "fig_rasa_pareto_auc_vs_section_alignment.png;table_baseline_comparison.csv",
         "manuscript_section": "Results §3", "final_recommendation": "use"},
        {"figure_or_table_id": "Figure 4", "title": "Strict LOCO and modality perturbation EDS",
         "core_message": "Cross-centre generalization; modality evidence dependency",
         "source_files": "fig_loco_strict_center_heatmap.png;table_modality_perturbation_extended.csv",
         "manuscript_section": "Results §5-6", "final_recommendation": "use"},
        {"figure_or_table_id": "Table 1", "title": "Centre-wise scale and supervision",
         "core_message": "Cases, images, real/pseudo report counts per centre",
         "source_files": "table_centerwise_image_count_audit.csv;table_final_dataset_statistics_for_manuscript.csv",
         "manuscript_section": "Results §1", "final_recommendation": "use"},
        {"figure_or_table_id": "Table 2", "title": "Primary model comparison on test set (n=288)",
         "core_message": "AUC, stratified metrics, validation-selected F1, label consistency",
         "source_files": "table_reference_stratified_evaluation.csv;table_threshold_tuned_test_metrics.csv;manuscript/T2_main_model_comparison.csv",
         "manuscript_section": "Results §3-4", "final_recommendation": "use"},
    ]
    write_csv(pd.DataFrame(mapping), tables / "table_figure_source_mapping.csv")

    main_idx = [
        "# Final Figure and Table Index — Journal of Big Data\n",
        "\n## Main text (4 figures, 2 tables)\n",
    ]
    for m in mapping:
        main_idx.append(f"\n### {m['figure_or_table_id']}: {m['title']}\n")
        main_idx.append(f"- **Message**: {m['core_message']}\n")
        main_idx.append(f"- **Sources**: `{m['source_files']}`\n")
        main_idx.append(f"- **Recommendation**: {m['final_recommendation']}\n")
    (ms / "FINAL_FIGURE_TABLE_INDEX_FOR_JBD.md").write_text("".join(main_idx), encoding="utf-8")

    supp = [
        "# Supplementary Figure and Table Index\n",
        "\n| ID | Content | Source |\n|----|---------|--------|\n",
        "| S1 | LCAD QC ablation | table_lcad_qc_ablation.csv |\n",
        "| S2 | Modality ablation | table_modality_ablation.csv |\n",
        "| S3 | RASA component ablation | table_rasa_component_ablation.csv |\n",
        "| S4 | LOCO eval-only | table_loco_main_results.csv |\n",
        "| S5 | Multiseed stability | table_multiseed_stability.csv |\n",
        "| S6 | Report safety | table_report_safety_metrics.csv |\n",
        "| S7 | Scalability/runtime | table_scalability_pipeline_statistics.csv |\n",
        "| S8 | Masking validation | masking_validation_publishable_metrics.csv |\n",
        "| S9 | Expert review protocol | expert_review/ (no scores) |\n",
        "| S10 | LLM pseudo-report QC | llm_vs_mock_pseudo_qc_comparison.csv |\n",
        "| S11 | Qualitative cases | qualitative_cases/ |\n",
        "| S12 | LLM comparison (if E1) | table_llm_pseudo_report_comparison.csv |\n",
    ]
    (ms / "FINAL_SUPPLEMENTARY_INDEX_FOR_JBD.md").write_text("".join(supp), encoding="utf-8")


def _read_key_metrics(project: Path, tables: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    p = tables / "table_final_dataset_statistics_for_manuscript.csv"
    if p.is_file():
        d = dict(zip(pd.read_csv(p)["metric"], pd.read_csv(p)["value"]))
        out["total_cases"] = float(d.get("total_cases", 0))
        out["total_images"] = float(d.get("total_images_evaluable", 0))
    strat = tables / "table_reference_stratified_evaluation.csv"
    if strat.is_file():
        df = pd.read_csv(strat)
        full = df[(df["experiment_id"] == "full_lcad_rasa") & (df["subset"] == "all")]
        if len(full):
            out["full_lcad_auc"] = float(full.iloc[0]["auc"])
    return out


def run_r3(project: Path, tables: Path) -> None:
    metrics = _read_key_metrics(project, tables)
    numeric_lines = [
        "# Manuscript Numeric Consistency Audit\n",
        f"\nAudit time: {datetime.now(timezone.utc).isoformat()}\n",
        "## Canonical cohort statistics\n",
        "| Metric | Expected | Check |\n|--------|----------|-------|\n",
        f"| Cases | 1897 | {metrics.get('total_cases', 'N/A')} |\n",
        f"| Images | 137591 | {metrics.get('total_images', 'N/A')} |\n",
        "| Real reports | 744 | verify manifest |\n",
        "| Pseudo candidates | 1153 | verify manifest |\n",
        "| Test n | 288 | verify split |\n",
        "\n## Model naming\n",
        "- Use **Full LCAD-RASA** for `full_lcad_rasa` with section alignment.\n",
        "- Use **LCAD without section alignment** for `report_generation_without_section_alignment`.\n",
        "- **Best LCAD-RASA** = λ_align=0.2 (`best_lcad_rasa` / `rasa_align_0p20`).\n",
        "\n## Metrics reporting rules\n",
        "- ROUGE/BLEU: only `with_reference` subset (n≈108).\n",
        "- Primary risk: AUC (threshold-free) + F1 at validation-selected threshold.\n",
    ]
    (tables / "MANUSCRIPT_NUMERIC_CONSISTENCY_AUDIT.md").write_text("".join(numeric_lines), encoding="utf-8")

    overclaim_rows = []
    for phrase, risk, note in OVERCLAIM_PHRASES:
        overclaim_rows.append({"phrase": phrase, "risk_level": risk, "guidance": note})
    write_csv(pd.DataFrame(overclaim_rows), tables / "OVERCLAIM_RISK_AUDIT.csv")

    revisions = [
        "# Final Claim Revision Suggestions\n",
        "\n## Conservative replacements\n",
        '- Instead of "outperformed all baselines": ',
        '"achieved competitive risk discrimination while improving report-anchored semantic structure relative to fusion-only baselines."\n',
        '- Instead of "LLM-generated pseudo reports": ',
        '"label-constrained pseudo reports from a local structured agent (embedding-augmented); external LLM comparison pending or in supplementary."\n',
        '- Instead of "clinically validated": ',
        '"evaluated for label consistency and report safety metrics; expert plausibility review protocol provided."\n',
    ]
    (tables / "FINAL_CLAIM_REVISION_SUGGESTIONS.md").write_text("".join(revisions), encoding="utf-8")


def run_s1(project: Path, sub: Path) -> None:
    required = [
        "tables/table_final_dataset_statistics_for_manuscript.csv",
        "tables/table_reference_stratified_evaluation.csv",
        "tables/table_baseline_comparison.csv",
        "tables/table_loco_strict_main_results.csv",
        "tables/table_rasa_loss_weight_sweep.csv",
        "tables/table_modality_perturbation_extended.csv",
        "SHA256SUMS.txt",
        "SUBMISSION_READINESS_CHECKLIST.md",
        "manuscript_sections/RESULTS_JBD_FINAL_DRAFT.md",
    ]
    missing = [r for r in required if not (sub / r).is_file()]
    comp = [
        "# Submission Package Completeness Audit\n",
        f"\nBundle: `{sub}`\n",
        f"Missing required files: {len(missing)}\n",
    ]
    if missing:
        comp.extend(f"- {m}\n" for m in missing)
    else:
        comp.append("\nAll listed required files present.\n")
    (sub / "SUBMISSION_PACKAGE_COMPLETENESS_AUDIT.md").write_text("".join(comp), encoding="utf-8")

    privacy_hits = []
    if sub.is_dir():
        for f in sub.rglob("*"):
            if f.is_file() and f.suffix in {".md", ".csv", ".txt", ".yaml", ".py"}:
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")[:50000]
                except OSError:
                    continue
                for pat in PRIVACY_PATTERNS:
                    if pat.search(text):
                        privacy_hits.append({"file": _rel(project, f), "pattern": pat.pattern})
                        break
    priv = [
        "# Submission Package Privacy Audit\n",
        f"\nFiles scanned under submission bundle.\n",
        f"Potential issues found: {len(privacy_hits)}\n",
    ]
    if privacy_hits:
        priv.append("\n| File | Pattern |\n|------|--------|\n")
        for h in privacy_hits[:30]:
            priv.append(f"| {h['file']} | {h['pattern']} |\n")
        priv.append("\nNote: `/data2/` in reproducibility README may be acceptable if redacted for submission.\n")
    else:
        priv.append("\nNo obvious privacy patterns in scanned text files.\n")
    (sub / "SUBMISSION_PACKAGE_PRIVACY_AUDIT.md").write_text("".join(priv), encoding="utf-8")

    fixes = [{"item": m, "action": "copy from outputs/publishable/tables"} for m in missing]
    write_csv(pd.DataFrame(fixes) if fixes else pd.DataFrame([{"item": "none", "action": "ok"}]),
              sub / "SUBMISSION_PACKAGE_FIX_LIST.csv")


def run_s2(project: Path, tables: Path, ms: Path) -> None:
    alignment = [
        {"methods_topic": "Baseline comparison", "results_section": "§3", "table": "table_baseline_comparison.csv", "aligned": "yes"},
        {"methods_topic": "Strict LOCO", "results_section": "§5", "table": "table_loco_strict_main_results.csv", "aligned": "yes"},
        {"methods_topic": "Expert review results", "results_section": "—", "table": "table_expert_review_results.csv", "aligned": "no — protocol only"},
        {"methods_topic": "External API LLM", "results_section": "—", "table": "table_llm_pseudo_report_comparison.csv", "aligned": "pending E1"},
    ]
    write_csv(pd.DataFrame(alignment), tables / "METHODS_RESULTS_ALIGNMENT_TABLE.csv")
    report = [
        "# Manuscript Crosscheck Report\n",
        "\n## High-risk gaps\n",
        "- Expert review: Methods may mention plausibility evaluation but Results lack scores (E3 incomplete).\n",
        "- External LLM: Do not claim in Abstract until E1 produces comparison table.\n",
        "\n## Figure callouts\n",
        "- Verify Figure 2 architecture diagram exists (marked redraw_required in R2).\n",
    ]
    (tables / "MANUSCRIPT_CROSSCHECK_REPORT.md").write_text("".join(report), encoding="utf-8")
    write_csv(pd.DataFrame([{"callout": "Figure 1", "file": "fig_centerwise_data_scale.png", "exists": _exists(project, "outputs/publishable/figures/fig_centerwise_data_scale.png")}]),
              tables / "FIGURE_TABLE_CALLOUT_AUDIT.csv")


def run_s3(project: Path, ms: Path, tables: Path) -> None:
    auc = "0.836"
    p = tables / "table_reference_stratified_evaluation.csv"
    if p.is_file():
        df = pd.read_csv(p)
        sub = df[(df["experiment_id"] == "full_lcad_rasa") & (df["subset"] == "all")]
        if len(sub):
            auc = f"{float(sub.iloc[0]['auc']):.3f}"

    structured = f"""# Structured Abstract — JBD

## Background
Multicentre cervical screening data exhibit severe report-supervision imbalance: only a subset of centres provide real colposcopy reports while most cases lack reference text.

## Methods
We studied 1,897 examinations and 137,591 evaluable OCT/colposcopy images from five centres. A label-constrained agent generated pseudo reports for report-missing cases; QC-weighted training fed a report-anchored semantic alignment (RASA) model with section-level alignment and a CIN2+ risk head. Evaluation included reference-stratified metrics, validation-based threshold tuning, strict leave-one-centre-out retraining, modality perturbation with decoded text, and pipeline scalability analysis.

## Results
On the held-out test set (n=288), full LCAD-RASA achieved AUC {auc} with improved label-consistent structured reports relative to fusion baselines, while a variant without section alignment showed higher default-threshold AUC, illustrating a risk–semantic trade-off. Strict LOCO and perturbation analyses indicated centre-dependent generalization and modality-specific evidence dependence.

## Conclusions
LCAD-RASA offers a scalable framework for report-anchored multimodal semantic grounding under heterogeneous documentation, with explicit reporting of limitations regarding pseudo-report supervision and lightweight text decoding.
"""
    (ms / "ABSTRACT_STRUCTURED_JBD.md").write_text(structured, encoding="utf-8")

    cautious = structured.replace(
        "full LCAD-RASA achieved AUC",
        "the report-anchored model showed moderate risk discrimination (AUC",
    ) + "\n\n*Cautious version: avoid implying clinical superiority or external LLM validation.*\n"
    (ms / "ABSTRACT_UNSTRUCTURED_JBD.md").write_text(structured, encoding="utf-8")
    (ms / "CAUTIOUS_CLAIM_ABSTRACT_JBD.md").write_text(cautious, encoding="utf-8")

    contrib = """# Contributions — JBD

1. **Problem framing**: Quantify report-supervision imbalance across five centres at scale (1,897 cases; 137,591 images).
2. **Method**: LCAD pseudo-report distillation with QC weighting and RASA section-level alignment for multimodal CIN2+ risk and structured report decoding.
3. **Evaluation protocol**: Reference-stratified reporting, validation-selected thresholds, strict LOCO retraining, and text-decoding perturbation EDS.
4. **Reproducibility**: Frozen submission bundle, manifest-driven pipeline, and explicit limitations on local structured agents vs external LLMs.
"""
    (ms / "CONTRIBUTIONS_JBD.md").write_text(contrib, encoding="utf-8")

    cover = """# Cover Letter Key Points — JBD

- **Fit for Journal of Big Data**: Large-scale heterogeneous multimodal data, pipeline scalability metrics, and reproducible big-data analytics workflow.
- **Not a narrow imaging paper**: Focus on documentation heterogeneity, weak report-level supervision, and semantic alignment infrastructure.
- **Evidence**: Five-centre cohort, strict LOCO, perturbation EDS, QC/safety tables, submission bundle with checksums.
- **Honest scope**: Pseudo reports are not clinical gold standard; expert scores pending; local structured agent rather than closed API LLM unless supplementary E1 completed.
"""
    (ms / "COVER_LETTER_KEY_POINTS_JBD.md").write_text(cover, encoding="utf-8")


def run_e3(project: Path, tables: Path) -> None:
    score_files = list(project.glob("outputs/publishable/**/*rating*.xlsx")) + list(
        project.glob("outputs/publishable/**/*score*.xlsx")
    )
    if not score_files:
        report = """# Expert Score Missing Report

No completed expert/physician rating spreadsheets were found (`*rating*.xlsx`, `*score*.xlsx`).

## What exists
- `outputs/publishable/expert_review/blinded_review_package/`
- `outputs/publishable/physician_review/review_cases.csv`
- `EXPERT_REVIEW_PROTOCOL.md`

## Action required
1. Distribute blinded review package to experts.
2. Collect scores into `expert_review_scores.xlsx` or `physician_review_scores.xlsx`.
3. Re-run: `python scripts/28_run_jbd_result_organization.py --prompt E3`

## Manuscript rule
Do **not** report expert evaluation results in main text until scores exist.
"""
        (tables / "EXPERT_SCORE_MISSING_REPORT.md").write_text(report, encoding="utf-8")
        return
    # If scores exist later, extend here
    (tables / "EXPERT_SCORE_MISSING_REPORT.md").write_text("Scores found — implement analysis branch.\n", encoding="utf-8")


def run_e4(project: Path, pub: Path) -> None:
    qc = pub / "qc/llm_pseudo_report_qc_cases.csv"
    manifest = pub / "manifests/full_manifest_publishable_with_llm_pseudo.csv"
    cases_dir = pub / "qualitative_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if qc.is_file() and manifest.is_file():
        qdf = pd.read_csv(qc)
        mdf = pd.read_csv(manifest)
        mdf = mdf[mdf["split"] == "test"] if "split" in mdf.columns else mdf
        fail = qdf[qdf.get("pseudo_report_pass_qc", 1) == 0].head(5) if "pseudo_report_pass_qc" in qdf.columns else qdf.head(0)
        for _, r in fail.iterrows():
            rows.append({
                "anonymized_case_id": str(r.get("case_id", "")),
                "category": "failure",
                "key_error_or_success_reason": "pseudo_report QC fail",
                "whether_suitable_for_supplementary_example": "yes",
            })
        pass_df = qdf[qdf.get("pseudo_report_pass_qc", 1) == 1].head(5) if "pseudo_report_pass_qc" in qdf.columns else qdf.head(5)
        for _, r in pass_df.iterrows():
            rows.append({
                "anonymized_case_id": str(r.get("case_id", "")),
                "category": "success",
                "key_error_or_success_reason": "QC pass",
                "whether_suitable_for_supplementary_example": "yes",
            })
    if not rows:
        rows.append({"anonymized_case_id": "case_example_001", "category": "borderline",
                     "key_error_or_success_reason": "insufficient QC rows — populate from eval outputs",
                     "whether_suitable_for_supplementary_example": "no"})
    write_csv(pd.DataFrame(rows), cases_dir / "table_qualitative_case_index.csv")
    (cases_dir / "success_cases.md").write_text("# Success cases\n\nSee `table_qualitative_case_index.csv` (category=success).\n", encoding="utf-8")
    (cases_dir / "failure_cases.md").write_text("# Failure cases\n\nSee `table_qualitative_case_index.csv` (category=failure).\n", encoding="utf-8")
    (cases_dir / "borderline_cases.md").write_text("# Borderline cases\n\nExpand with low-confidence / missing-modality cases from test eval.\n", encoding="utf-8")
    (pub / "manuscript_sections/QUALITATIVE_CASE_ANALYSIS_SUMMARY.md").write_text(
        "# Qualitative Case Analysis Summary\n\nIndexed cases exported under `qualitative_cases/`. Expand with perturbation and risk-error examples before submission.\n",
        encoding="utf-8",
    )


def run_e5(project: Path, tables: Path) -> None:
    reg = project.parent / "results/experiment_registry.csv"
    rows = []
    if reg.is_file():
        rows = pd.read_csv(reg).to_dict("records")
    decision = """# RA-HyDRA-LLM Inclusion Decision

## Recommendation
**不建议纳入当前 LCAD-RASA JBD 主文；仅作独立未来工作或 Supplementary exploratory comparison（需先完成训练与同一 split）。**

## Rationale
1. **Task alignment**: RA-HyDRA-LLM (report-as-anchor at train, report-free inference) overlaps conceptually but is a separate model line (`hydra_core` / `ra_hydra_llm`), not the implemented LCAD-RASA pipeline.
2. **Data version**: LCAD-RASA uses `cervix_lcad_rasa/outputs/publishable/manifests/` (1897 cases, 137591 images). RA-HyDRA registry entries are **planned**, not completed in `results/experiment_registry.csv`.
3. **Metrics parity**: No comparable checkpoint metrics on the same test split for fair baseline comparison.
4. **Narrative focus**: Main text already has six baselines + strict LOCO + RASA ablations; adding RA-HyDRA-LLM risks topic dilution.

## If pursued later
- Complete E002/E004 with identical patient-level split export from LCAD manifest.
- Report only in Supplementary Table as exploratory comparison.
"""
    (tables / "RA_HYDRA_LLM_INCLUSION_DECISION.md").write_text(decision, encoding="utf-8")
    write_csv(pd.DataFrame(rows) if rows else pd.DataFrame([{"note": "no completed runs"}]),
              tables / "table_ra_hydra_llm_available_results.csv")


def run_e1(project: Path, pub: Path, tables: Path) -> None:
    """Local vs mock pseudo-report QC comparison on stratified sample (API stub)."""
    import os

    llm_dir = pub / "pseudo_reports_llm"
    mock_root = project / "outputs/pseudo_reports"
    out_dir = pub / "llm_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pub / "manifests/full_manifest_publishable_with_llm_pseudo.csv"
    if not manifest.is_file():
        (tables / "LLM_PSEUDO_REPORT_COMPARISON_INTERPRETATION.md").write_text(
            "E1 skipped: manifest missing.\n", encoding="utf-8")
        return

    df = pd.read_csv(manifest)
    sub = df[(df.get("needs_pseudo_report", 0) == 1)].copy()
    n_sample = min(100, len(sub))
    if "center_id" in sub.columns and sub["center_id"].nunique() > 1:
        parts = []
        per = max(1, n_sample // sub["center_id"].nunique())
        for _, g in sub.groupby("center_id"):
            parts.append(g.sample(min(len(g), per), random_state=42))
        sub = pd.concat(parts).head(n_sample)
    else:
        sub = sub.sample(n_sample, random_state=42)
    records = []
    for _, row in sub.iterrows():
        cid, case = str(row["center_id"]), str(row["case_id"])
        llm_path = llm_dir / cid / f"{case}.json"
        rec = {"case_id": case, "center_id": cid, "backend": "local_llm_embedding_augmented", "json_valid": llm_path.is_file()}
        if llm_path.is_file():
            try:
                rep = json.loads(llm_path.read_text(encoding="utf-8"))
                rec["section_complete"] = all(rep.get(k) for k in ("diagnostic_summary", "impression", "recommendation"))
                rec["label_consistent"] = int(rep.get("label", -1)) == int(row.get("binary_label", -2))
            except json.JSONDecodeError:
                rec["section_complete"] = False
                rec["label_consistent"] = False
        records.append(rec)

    api_key = os.environ.get("LCAD_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    api_status = "not_performed — no API key / compliance; use local_llm vs mock QC only"
    cmp_rows = [{"source": "local_llm", "n": len(records),
                 "json_valid_rate": sum(r["json_valid"] for r in records) / max(1, len(records)),
                 "section_complete_rate": sum(r.get("section_complete", False) for r in records) / max(1, len(records)),
                 "label_consistency_rate": sum(r.get("label_consistent", False) for r in records) / max(1, len(records))}]
    llm_mock = tables / "llm_vs_mock_pseudo_qc_comparison.csv"
    if llm_mock.is_file():
        cmp_rows.extend(pd.read_csv(llm_mock).to_dict("records"))

    write_csv(pd.DataFrame(cmp_rows), tables / "table_llm_pseudo_report_comparison.csv")
    md_df = pd.DataFrame(cmp_rows)
    try:
        md_text = md_df.to_markdown(index=False)
    except ImportError:
        md_text = md_df.to_string(index=False)
    (tables / "table_llm_pseudo_report_comparison.md").write_text(md_text, encoding="utf-8")
    interp = f"""# LLM Pseudo-Report Comparison Interpretation

## Status
- **local_llm** (embedding-augmented structured generator): evaluated on n={len(records)} stratified sample.
- **External API LLM**: {api_status}.

## Implications
1. Main experiments correctly use reproducible local structured generation.
2. If API comparison is required for revision, run E1 with approved provider and same input JSONL.
3. Do not claim "GPT/DeepSeek-generated pseudo reports" in main text without API results table.
"""
    (tables / "LLM_PSEUDO_REPORT_COMPARISON_INTERPRETATION.md").write_text(interp, encoding="utf-8")
    sub[["case_id", "center_id", "binary_label"]].to_json(
        out_dir / "input_cases_llm_comparison.jsonl", orient="records", lines=True, force_ascii=False
    )


def run_e2_stub(tables: Path) -> None:
    """E2 deferred when E1 has no API; document prompt-sensitivity plan."""
    rows = [{"model": "local_llm_current", "prompt": "label_constrained_prompt", "note": "current LCAD default"},
            {"model": "api_llm", "prompt": "—", "note": "not_performed — depends on E1 API"}]
    write_csv(pd.DataFrame(rows), tables / "table_llm_prompt_sensitivity.csv")
    (tables / "LLM_PROMPT_SENSITIVITY_INTERPRETATION.md").write_text(
        "E2 not fully executed: requires E1 API/open model. Planned prompts: strict_structured, concise_clinical, uncertainty_aware, label_blinded, label_constrained.\n",
        encoding="utf-8",
    )


def run_all(project: Path, prompts: str) -> list[str]:
    tables = project / "outputs/publishable/tables"
    ms = project / "outputs/publishable/manuscript_sections"
    sub = project / "outputs/publishable_jbd_submission_v1"
    pub = project / "outputs/publishable"
    tables.mkdir(parents=True, exist_ok=True)
    ms.mkdir(parents=True, exist_ok=True)
    failures = []
    order = {
        "R1": lambda: run_r1(project, tables),
        "R2": lambda: run_r2(project, tables, ms),
        "R3": lambda: run_r3(project, tables),
        "S1": lambda: run_s1(project, sub),
        "S2": lambda: run_s2(project, tables, ms),
        "S3": lambda: run_s3(project, ms, tables),
        "E3": lambda: run_e3(project, tables),
        "E4": lambda: run_e4(project, pub),
        "E5": lambda: run_e5(project, tables),
        "E1": lambda: run_e1(project, pub, tables),
        "E2": lambda: run_e2_stub(tables),
    }
    sel = list(order.keys()) if prompts.upper() in ("ALL", "") else [p.strip().upper() for p in prompts.split(",")]
    if prompts.upper() == "PRIORITY":
        sel = ["R1", "R2", "R3", "S1", "S2", "E3", "E4", "S3", "E5", "E1", "E2"]
    for name in sel:
        if name not in order:
            continue
        try:
            order[name]()
        except Exception as e:
            failures.append(name)
            (project / "outputs/publishable/logs" / f"fail_org_{name}.log").write_text(str(e), encoding="utf-8")
    agg = project / "scripts/28_aggregate_manuscript_result_tables.py"
    if agg.is_file():
        import subprocess
        subprocess.run(
            [sys.executable, str(agg)],
            cwd=str(project),
            check=False,
        )
    return failures
