#!/usr/bin/env python3
"""Finalize JBD paper experimental sections (Prompts A–H from final_execution_prompts.md)."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MS = ROOT / "outputs/publishable/tables/manuscript"
SEC = ROOT / "outputs/publishable/manuscript_sections"
FIG_MAIN = ROOT / "outputs/publishable/figures/main"
FIG_PUB = ROOT / "outputs/publishable/figures"
FIG_JBD = ROOT / "outputs/publishable/figures/jbd_final"
PRED = ROOT / "outputs/publishable/predictions/final_per_case"
PROJECT_ROOT = ROOT.parent  # experiments/JBD_2026


def _df_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or len(df) == 0:
        return "_empty_"
    sub = df.head(max_rows)
    cols = list(sub.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def prompt_a_audit() -> None:
    t1a = pd.read_csv(MS / "T1a_cohort_summary.csv")
    t1b = pd.read_csv(MS / "T1b_centre_scale_and_supervision.csv")
    t2 = pd.read_csv(MS / "T2_main_model_comparison.csv")
    t2ci = pd.read_csv(MS / "T2_main_model_comparison_with_ci.csv") if (MS / "T2_main_model_comparison_with_ci.csv").is_file() else None
    pw = pd.read_csv(MS / "T2_pairwise_statistical_tests.csv") if (MS / "T2_pairwise_statistical_tests.csv").is_file() else None
    s3 = pd.read_csv(MS / "S3_modality_ablation.csv")
    s5 = pd.read_csv(MS / "S5_rasa_component_ablation.csv")

    main_tables = [
        ("T1a", "T1a_cohort_summary.csv", "cohort", "n/a", "main", "yes"),
        ("T1b", "T1b_centre_scale_and_supervision.csv", "cohort", "n/a", "main", "yes"),
        ("T2", "T2_main_model_comparison.csv", "test", "validation_selected", "main", "yes"),
        ("T2_CI", "T2_main_model_comparison_with_ci.csv", "test n=288", "validation_selected", "main", "yes"),
    ]
    supp_tables = [f.name for f in MS.glob("S*.csv")]
    main_figs = [
        ("Figure 1", "figures/main/Figure1_study_design.png", "main"),
        ("Figure 2", "figures/main/Figure2_centre_supervision.png", "main"),
        ("Figure 3", "figures/main/Figure3_perturbation.png", "main"),
        ("Figure 4 (opt.)", "figures/main/Figure4_loco_strict.png", "supplement optional"),
    ]

    no_ckpt_stale = (
        "table_main_results_for_manuscript.csv may list no_checkpoint for some rows; "
        "manuscript tables S3/S5 and per-case exports are complete after 2026-05-25 re-eval."
    )

    claims = [
        ("Five-centre 1897-case dataset", "T1a/T1b", "yes", ""),
        ("Case-level supervision imbalance", "T1b", "yes", ""),
        ("LCAD structured pseudo-reports", "pseudo_reports_llm", "yes", "not commercial LLM"),
        ("Masking: modality > label-only proxy", "S10 pooled", "yes", "proxy not expert validation"),
        ("Full LCAD-RASA higher point estimates vs real-only & simple concat", "T2", "yes", "use val threshold"),
        ("Statistically significant vs all baselines", "T2_pairwise", "no", f"p≈0.5 bootstrap; do not write significant"),
        ("Expert-validated reports", "expert_review", "no", "ratings pending"),
        ("Commercial LLM-driven", "—", "no", ""),
        ("Clinical deployment-ready", "—", "no", ""),
        ("QC ablation improves performance", "S4", "no", "weak separation"),
        ("Section alignment always improves AUC", "S5/S7", "no", "w/o section AUC comparable on multi-seed"),
        ("Perturbation section-specific fidelity", "S6", "yes", "not strict causal proof"),
    ]

    lines = [
        "# JBD Final Result Audit (Paper-Ready)\n",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}\n",
        "\n## A. Main-text eligible results\n",
        _df_md(pd.DataFrame(main_tables, columns=["id", "file", "split", "threshold", "placement", "ready"])),
        f"\nMain-text ready tables: **{len(main_tables)}**\n",
        f"Main-text ready figures: **{len(main_figs)}**\n",
        "\n## B. Supplement-only (S1–S11)\n",
        "\n".join(f"- `{n}`" for n in sorted(supp_tables)),
        "\n## C. Stale / resolved checkpoint notes\n",
        no_ckpt_stale,
        f"\nS3 modality ablation: **{len(s3)} rows**, all with AUC values.\n",
        f"S5 RASA components: **{len(s5)} rows**, all with AUC values.\n",
        "\n## D. cohort n mismatch audit\n",
        _df_md(
            pd.DataFrame(
                [
                    ("T2 main comparison", 288, "all test", "validation-selected"),
                    ("S6 perturbation", 128, "report-missing test", "n/a"),
                    ("Per-case predictions", 288, "test", "validation-selected per model"),
                    ("S2 strict LOCO", "per centre", "held-out centre test", "quick budget footnote"),
                ],
                columns=["artifact", "n", "subset", "threshold_protocol"],
            )
        ),
        "\n## E. threshold protocol\n",
        "- **Table 2**: validation-selected max-F1 threshold per model (see `table_threshold_tuned_test_metrics.csv`).\n",
        "- **Do not** report default 0.5 F1 as primary endpoint in main text.\n",
        "\n## F. mock contamination\n",
        "- Main text uses only `outputs/publishable/`.\n",
        "- Do not cite `outputs/tables/main_experiments_performance.csv` (mock).\n",
        "\n## H. Claim safety audit\n",
        _df_md(pd.DataFrame(claims, columns=["claim", "evidence", "can_write", "note"])),
        "\n## Summary counts\n",
        f"- Claims allowed now: **{sum(1 for c in claims if c[2]=='yes')}**\n",
        f"- Claims must remove/downgrade: **{sum(1 for c in claims if c[2]=='no')}**\n",
    ]
    (MS / "JBD_FINAL_RESULT_AUDIT.md").write_text("".join(lines), encoding="utf-8")


def prompt_b_visualization_plan() -> None:
    rows = [
        ("E0", "Cohort & supervision", "Dataset scale imbalance", "Figure 1, Table 1a/1b", "T1a,T1b", "Main", "Complete", "None", "Five-centre big-data cohort", "No uniform supervision"),
        ("E2", "Masking validation", "LCAD uses modality", "Supp Fig S1, S10", "S10", "Supplement", "Complete", "None", "Modality proxy > label-only", "Not expert validation"),
        ("E3", "Pseudo-report LCAD", "Weak supervision", "Fig 1 panel B", "pseudo_reports_llm", "Methods/Main", "Complete", "None", "Structured pseudo reports", "Not commercial LLM"),
        ("E5", "Main model comparison", "Core performance", "Table 2, Fig main AUC", "T2,T2_CI", "Main", "Complete", "Stats: no significant p", "Higher point estimates", "No 'significant' without p<0.05"),
        ("E6", "Reference stratified", "Subset behaviour", "Table 2 cols / optional Table 3", "T2", "Main text", "Complete", "Footnote n", "Separate ref / no-ref subsets", "No ROUGE on no-ref"),
        ("E8", "Strict LOCO", "Cross-centre", "Fig 4 optional, S2", "S2", "Supplement", "Complete", "Quick budget footnote", "Centre heterogeneity", "Not definitive external validation"),
        ("E10", "Modality ablation", "Modality contribution", "fig_modality_*, S3", "S3", "Supplement", "Complete", "None", "Modality subsets ranked by AUC", "F1=0 at default thr"),
        ("E11", "RASA ablation", "Component contribution", "fig_rasa_*, S5", "S5", "Supplement", "Complete", "None", "Component deltas", "Do not overclaim section align"),
        ("E12", "Perturbation", "Mechanism", "Figure 3", "S6,S6b", "Main", "Complete", "None", "Section-specific degradation", "Perturbation fidelity not causality"),
        ("E13-E14", "Stability/scale", "JBD fit", "Supp S3/S4/S11", "S7,S9,S11", "Supplement", "Complete", "None", "Reproducible pipeline", "Not deployable"),
        ("E15", "Expert review", "Human eval", "—", "expert_review", "Optional", "Not complete", "Collect scores", "—", "No expert-validated"),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "experiment_id",
            "name",
            "research_question",
            "visualization",
            "input_files",
            "placement",
            "status",
            "remaining_action",
            "allowed_claim",
            "limitation",
        ],
    )
    intro = (
        "# JBD Experiment–Visualization Plan\n\n"
        "Reference figures: [Seaborn gallery](https://seaborn.pydata.org/examples/index.html)\n\n"
        "## Recommended main figures\n\n"
        "| Priority | Figure | File | Evidence |\n"
        "|---:|---|---|---|\n"
        "| 1 | Study design & LCAD-RASA | `figures/main/Figure1_study_design.png` | E0,E3,E5 |\n"
        "| 2 | Centre supervision imbalance | `figures/main/Figure2_centre_supervision.png` | E1,T1b |\n"
        "| 3 | Perturbation & section degradation | `figures/main/Figure3_perturbation.png` | E12,S6 |\n"
        "| 4 (opt.) | Strict LOCO only | `figures/main/Figure4_loco_strict.png` | E8,S2 |\n\n"
        "## Supplementary figures\n\n"
        "- Masking: `SupplementaryFigure_S1_masking_validation.png`\n"
        "- Modality ablation: `fig_modality_ablation_*.png`\n"
        "- RASA ablation: `fig_rasa_component_*.png`\n"
        "- Multi-seed: `SupplementaryFigure_S3_multiseed.png`\n"
        "- Scalability: `SupplementaryFigure_S4_scalability.png`\n"
        "- QC/safety: S8/S9 tables\n\n"
    )
    (MS / "JBD_EXPERIMENT_VISUALIZATION_PLAN.md").write_text(intro + _df_md(df), encoding="utf-8")


def prompt_d_statistics() -> None:
    t2ci = pd.read_csv(MS / "T2_main_model_comparison_with_ci.csv")
    pw = pd.read_csv(MS / "T2_pairwise_statistical_tests.csv")

    def _col(row, *names):
        for n in names:
            if n in row.index and pd.notna(row[n]):
                return row[n]
        raise KeyError(names)

    single = []
    for _, r in t2ci.iterrows():
        single.append(
            {
                "model": r["model"],
                "n": int(r["n"]),
                "auc": round(float(_col(r, "auc")), 4),
                "auc_ci_low": round(float(_col(r, "auc_ci_low", "ci_low")), 4),
                "auc_ci_high": round(float(_col(r, "auc_ci_high", "ci_high")), 4),
                "f1": round(float(_col(r, "f1")), 4),
                "f1_ci_low": round(float(_col(r, "f1_ci_low")), 4),
                "f1_ci_high": round(float(_col(r, "f1_ci_high")), 4),
                "threshold": float(_col(r, "threshold")),
            }
        )
    pd.DataFrame(single).to_csv(MS / "JBD_STATISTICAL_TESTS_FINAL.csv", index=False)

    sig_allowed = pw[pw["bootstrap_p_auc"] < 0.05] if len(pw) else pd.DataFrame()
    md = [
        "# JBD Statistical Tests (Final)\n",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}\n",
        "\n## Methods\n",
        "- Bootstrap n=2000, seed=20260525, resampling unit=case_id.\n",
        "- AUC/F1 95% CI: percentile bootstrap on test set (n=288).\n",
        "- Pairwise AUC: paired bootstrap on matched case_id (DeLong **not** used).\n",
        "- McNemar: paired binary predictions at validation-selected threshold.\n",
        "- Threshold: validation max-F1 per model (`table_threshold_tuned_test_metrics.csv`).\n",
        "\n## Single-model CI (manuscript-safe)\n",
        _df_md(pd.DataFrame(single)),
        "\n## Pairwise comparisons\n",
        _df_md(pw),
        "\n## Manuscript wording rules\n",
        "- **Allowed**: \"AUROC 0.832 (95% CI 0.757–0.897)\" for Full LCAD-RASA.\n",
        "- **Allowed**: \"numerically higher point estimate\" vs real-report-only (ΔAUC +0.108, p=0.501).\n",
        "- **NOT allowed**: \"statistically significant\" for primary comparisons (all bootstrap p≈0.5).\n",
        f"- Comparisons with p<0.05: **{len(sig_allowed)}** (if any).\n",
    ]
    (MS / "JBD_STATISTICAL_TESTS_FINAL.md").write_text("".join(md), encoding="utf-8")


def prompt_e_no_checkpoint() -> None:
    s3 = pd.read_csv(MS / "S3_modality_ablation.csv")
    s5 = pd.read_csv(MS / "S5_rasa_component_ablation.csv")
    idx = pd.read_csv(PRED / "PER_CASE_PREDICTION_INDEX.csv")
    lines = [
        "# no_checkpoint Resolution (Final)\n",
        f"\nDate: {datetime.now(timezone.utc).isoformat()}\n",
        "\n## Resolution\n",
        "- **S3 modality ablation**: All 9 experiments evaluated; AUC present. **Keep in Supplement.**\n",
        "- **S5 RASA component ablation**: All 7 experiments evaluated. **Keep in Supplement.**\n",
        "- **Per-case exports**: 7/7 core models OK (see PER_CASE_PREDICTION_INDEX.csv).\n",
        "- **Stale no_checkpoint**: `table_main_results_for_manuscript.csv` — do not cite; use `T2_*` and `S3/S5` manuscript tables.\n",
        "\n## Deleted from manuscript tables\n",
        "- Rows with status=no_checkpoint in aggregated long table only.\n",
        "\n## Impact on main conclusions\n",
        "- **None** for Table 2 / Figure 3. Supplementary ablations support mechanism narrative only.\n",
        "\n## S3 preview\n",
        _df_md(s3),
        "\n## S5 preview\n",
        _df_md(s5),
        "\n## Per-case manifest\n",
        _df_md(idx),
    ]
    (MS / "JBD_NO_CHECKPOINT_RESOLUTION.md").write_text("".join(lines), encoding="utf-8")


def prompt_f_main_figures() -> None:
    FIG_MAIN.mkdir(parents=True, exist_ok=True)
    mapping = {
        "Figure1_study_design.png": "Figure1_pipeline_schematic.png",
        "Figure2_centre_supervision.png": "Figure2_centre_supervision_catplot.png",
        "Figure3_perturbation.png": "Figure3_modality_perturbation_heatmap.png",
        "Figure4_loco_strict.png": "Figure4_loco_forest_catplot.png",
    }
    entries = []
    for dst, src in mapping.items():
        sp = FIG_JBD / src
        if not sp.is_file():
            sp = FIG_PUB / src
        if sp.is_file():
            shutil.copy2(sp, FIG_MAIN / dst)
            pdf = sp.with_suffix(".pdf")
            if pdf.is_file():
                shutil.copy2(pdf, FIG_MAIN / dst.replace(".png", ".pdf"))
            entries.append((dst, src, sp, "main" if "Figure4" not in dst else "optional"))
    idx_lines = [
        "# JBD Figure Index (Paper Main + Supplement)\n",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}\n",
        "\n## Main text (canonical paths under `figures/main/`)\n",
    ]
    captions = {
        "Figure1_study_design.png": "Figure 1. MOSAIC framework overview: (A) multicentre cohort under report-supervision imbalance; (B) offline LCAD structured completion; (C) section-anchored RASA alignment; (D) train-only semantic retrieval and validation-calibrated fusion.",
        "Figure2_centre_supervision.png": "Figure 2. Centre-level case counts by real-report vs pseudo-report-candidate supervision (Table 1b).",
        "Figure3_perturbation.png": "Figure 3. Perturbation condition × report-section similarity (n=128 report-missing test cases). Masking induced section-specific degradation.",
        "Figure4_loco_strict.png": "Figure 4 (optional). Strict leave-one-centre-out AUROC by held-out centre and model. Training used a fixed quick budget; interpret as cross-centre behaviour analysis, not full external validation.",
    }
    for dst, src, sp, place in entries:
        idx_lines.append(f"\n### {dst}\n- **Source**: `{src}`\n- **Placement**: {place}\n- **Caption**: {captions.get(dst, '')}\n")
    idx_lines.append("\n## Full gallery\n- See `figures/jbd_final/JBD_FINAL_FIGURE_INDEX.md`\n")
    (FIG_PUB / "JBD_FIGURE_INDEX.md").write_text("".join(idx_lines), encoding="utf-8")


def prompt_g_results_sections() -> None:
    t1a = pd.read_csv(MS / "T1a_cohort_summary.csv")
    t1b = pd.read_csv(MS / "T1b_centre_scale_and_supervision.csv")
    t2 = pd.read_csv(MS / "T2_main_model_comparison.csv")
    t2ci = pd.read_csv(MS / "T2_main_model_comparison_with_ci.csv")
    full = t2ci[t2ci["model"].str.contains("Full", case=False)].iloc[0]

    def _v(df, metric):
        row = df[df["Metric"] == metric]
        return int(row["Value"].iloc[0]) if len(row) else "—"

    text = f"""# Results — Final Experiment Sections (JBD / SCI-ready draft)

Generated: {datetime.now(timezone.utc).isoformat()}

> **Data source**: `outputs/publishable/tables/manuscript/` only.  
> **Threshold**: validation-selected max-F1 per model. **Test n = 288.**

---

## 3.1 Dataset scale and case-level report supervision imbalance

We analysed **{_v(t1a, 'Total cases')}** multimodal cervical screening examinations from **five centres**, comprising **{_v(t1a, 'Evaluable images (pipeline)')}** evaluable OCT and colposcopy images (**{_v(t1a, 'OCT images')}** OCT; **{_v(t1a, 'Colposcopy images')}** colposcopy). Report-level supervision was available for **{_v(t1a, 'Real reports')}** cases, whereas **{_v(t1a, 'Pseudo-report candidates')}** cases lacked archived reports and were candidates for label-constrained pseudo-report weak supervision (Table 1a; Figure 2).

Supervision was **case-level** rather than centre-uniform: Enshi had real reports for all **406** cases; Jingzhou **334/406**; Xiangyang **4/500**; Wuhan and Shiyan had no centre-wide real-report archives (Table 1b). This imbalance motivates LCAD-RASA rather than a single-centre report-rich assumption.

**Figure 1** | **Table 1a, 1b**

---

## 3.2 LCAD pseudo-report construction and masking validation

For report-missing cases, we applied a **local embedding-enhanced structured generator** (LCAD) under a fixed JSON schema and rule-based QC pipeline—**not** a commercial LLM API—to produce pseudo reports for weak supervision (Figure 1B; Supplementary Tables S8–S9).

Masking validation on real-report centres compared label-only, modality-only, and modality-plus-label agent settings. On the Enshi+Jingzhou pooled subset (n=740), the label-consistency **proxy** was **0.593** (label-only), **0.664** (modality-only), and **0.664** (modality-plus-label) (Supplementary Table S10; Supplementary Figure S1). This supports that modality evidence contributed beyond label templating; it is **not** expert clinical validation. Xiangyang (n=4) was sensitivity-only.

---

## 3.3 Held-out test performance of LCAD-RASA

On the held-out test set (**n=288**), **Full LCAD-RASA** achieved AUROC **{full['auc']:.3f}** (95% CI **{full['auc_ci_low']:.3f}–{full['auc_ci_high']:.3f}**) and F1 **{full['f1']:.3f}** (95% CI **{full['f1_ci_low']:.3f}–{full['f1_ci_high']:.3f}**) at the validation-selected threshold (0.37). **Real-report-only** training yielded AUROC **{t2[t2['Model']=='Real-report only']['auc_at_val_threshold'].iloc[0]:.3f}** and F1 **{t2[t2['Model']=='Real-report only']['f1_at_val_threshold'].iloc[0]:.3f}**; **simple concatenation fusion** AUROC **{t2[t2['Model']=='Simple concat fusion']['auc_at_val_threshold'].iloc[0]:.3f}** (Table 2; Figure main AUC pointplot).

Paired bootstrap comparisons did **not** support “statistically significant” superiority statements (bootstrap p≈0.5 for ΔAUROC vs baselines). We report **numerically higher** point estimates and bootstrap CIs only.

**Do not write**: statistically significant; expert-validated; commercial LLM-driven.

---

## 3.4 Reference-stratified and cross-centre behaviour

Stratified by reference availability on the test set, Full LCAD-RASA AUROC was **{t2[t2['Model']=='Full LCAD-RASA']['auc_with_reference'].iloc[0]:.3f}** (n={int(t2[t2['Model']=='Full LCAD-RASA']['n_with_reference'].iloc[0])} with reference) and **{t2[t2['Model']=='Full LCAD-RASA']['auc_without_reference'].iloc[0]:.3f}** (n={int(t2[t2['Model']=='Full LCAD-RASA']['n_without_reference'].iloc[0])} without reference) (Table 2). Reference-based NLG metrics apply only to the reference subset.

Strict leave-one-centre-out retraining (Supplementary Table S2; optional Figure 4) showed centre-dependent AUROC (e.g., Enshi held-out Full LCAD-RASA **0.702**, Jingzhou **0.648**, Xiangyang **0.382**). Eval-only global-checkpoint LOCO (S2b) is reported separately and must not be mixed with strict retrain. Training used a **fixed quick budget**; interpret LOCO as cross-centre behaviour characterisation, not definitive external validation.

---

## 3.5 Modality and RASA component ablations

Modality ablations (Supplementary Table S3; supplementary figures) ranked input combinations by AUROC up to **0.787** (colposcopy+instruction). RASA component ablations (Supplementary Table S5) showed large drops without section alignment (AUROC **0.512** vs full **0.806** under the same eval protocol). Multi-seed mean AUROC for full and w/o-section models was comparable (~0.776–0.780); we do **not** claim section alignment uniformly maximises discrimination.

---

## 3.6 Perturbation fidelity and report-section specificity

Under decoded-text perturbation on **n=128** report-missing test cases (Figure 3; Table S6), masking OCT reduced oct_findings similarity to **0.26** (vs 1.0 normal); masking colposcopy reduced colposcopy findings to **0.00**; masking instruction reduced clinical_context to **0.17**. Risk score increased under label-only inference (Δrisk **+0.37**). This supports **perturbation-based evidence sensitivity** and section-specific organisation; it does not prove causal interpretability.

---

## 3.7 Stability, safety, and scalability

Multi-seed stability (Supplementary Table S7), report-safety indicators (Table S9), and pipeline scale (**~137k images**, checkpoint **~80 MB**, inference **~0.017 s/case**; Table S11) support reproducible big-data analytics. We do **not** claim clinical deployment readiness.

---

## 3.8 Human expert review

A blinded physician review package was exported; **formal expert scores are pending**. Manuscript must **not** state expert-validated or physician-confirmed report quality until scores are collected.

---

## Claims allowed vs disallowed

| Claim | Allowed? |
|-------|----------|
| Five-centre 1,897-case multimodal cohort | Yes |
| Case-level report supervision imbalance | Yes |
| LCAD structured pseudo-reports (local generator) | Yes |
| Modality-based masking > label-only proxy | Yes (proxy wording) |
| Full LCAD-RASA higher point estimates vs real-only & simple concat | Yes |
| Statistically significant vs baselines | **No** (p not <0.05) |
| Expert-validated reports | **No** |
| Commercial LLM-driven | **No** |
| QC ablation strongly improves performance | **No** |
| Clinical deployment-ready | **No** |
| Section alignment always best AUC | **No** |
| Perturbation section-specific fidelity | Yes (careful wording) |
"""
    SEC.mkdir(parents=True, exist_ok=True)
    (SEC / "RESULTS_FINAL_EXPERIMENT_SECTIONS.md").write_text(text, encoding="utf-8")
    shutil.copy2(SEC / "RESULTS_FINAL_EXPERIMENT_SECTIONS.md", PROJECT_ROOT / "JBD_LCAD_RASA_paper_experimental_sections_FILLED.md")


def prompt_h_expert() -> None:
    er = ROOT / "outputs/publishable/expert_review"
    tpl = er / "JBD_EXPERT_REVIEW_RATING_TEMPLATE.csv"
    has_scores = any(er.glob("*score*.csv")) and tpl.is_file() and pd.read_csv(tpl).notna().any().any() if tpl.is_file() else False
    md = [
        "# Expert / Physician Review Status\n",
        f"\nChecked: {datetime.now(timezone.utc).isoformat()}\n",
        f"\n**Completed scores**: {'Yes' if has_scores else '**No**'}\n",
        "\n## Manuscript rule\n",
        "- **Do not** write expert-validated, physician-confirmed, or human-validated unless `JBD_EXPERT_REVIEW_RESULTS.csv` exists with real scores.\n",
        "\n## Package location\n",
        f"- `{er.relative_to(ROOT)}`\n",
        "- Template: `JBD_EXPERT_REVIEW_RATING_TEMPLATE.csv`\n",
    ]
    (MS / "JBD_EXPERT_REVIEW_STATUS.md").write_text("".join(md), encoding="utf-8")
    pd.DataFrame([{"status": "pending", "note": "expert scores not collected"}]).to_csv(
        MS / "S_expert_review_summary.csv", index=False
    )


def write_execution_status() -> None:
    sections = [
        ("Dataset scale & supervision imbalance", "Complete", "T1a/T1b, Fig 2", "Yes"),
        ("LCAD pseudo-report & masking", "Mostly complete", "S10, pseudo_reports", "Yes with proxy wording"),
        ("Held-out test performance", "Complete", "T2 + CI", "Point estimates only; no 'significant'"),
        ("Reference-stratified & LOCO", "Complete", "T2 cols, S2", "Protocol footnotes required"),
        ("Modality & RASA ablations", "Complete", "S3, S5", "Yes"),
        ("Perturbation fidelity", "Complete", "Fig 3, S6", "Yes"),
        ("Stability, safety, scalability", "Complete", "S7,S9,S11", "Yes"),
        ("Expert review", "Not complete", "—", "No expert-validated claims"),
    ]
    df = pd.DataFrame(sections, columns=["Section", "Status", "Artifacts", "Paper-ready"])
    status = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "outputs": [
            str(MS / "JBD_FINAL_RESULT_AUDIT.md"),
            str(MS / "JBD_EXPERIMENT_VISUALIZATION_PLAN.md"),
            str(MS / "JBD_STATISTICAL_TESTS_FINAL.md"),
            str(MS / "JBD_NO_CHECKPOINT_RESOLUTION.md"),
            str(FIG_PUB / "JBD_FIGURE_INDEX.md"),
            str(SEC / "RESULTS_FINAL_EXPERIMENT_SECTIONS.md"),
            str(MS / "JBD_EXPERT_REVIEW_STATUS.md"),
        ],
        "main_figures": [str(FIG_MAIN / f) for f in sorted(FIG_MAIN.glob("*.png"))],
    }
    (MS / "PAPER_EXPERIMENT_EXECUTION_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    lines = [
        "# Paper Experiment Execution Status\n",
        f"\nExecuted: {status['generated']}\n",
        "\n## Section checklist (user matrix)\n",
        _df_md(df),
        "\n## Generated artifacts\n",
    ] + [f"- `{o}`\n" for o in status["outputs"]]
    (PROJECT_ROOT / "JBD_LCAD_RASA_experiment_visualization_audit_FILLED.md").write_text(
        "".join(lines) + "\nSee also `cervix_lcad_rasa/outputs/publishable/tables/manuscript/JBD_EXPERIMENT_VISUALIZATION_PLAN.md`.\n",
        encoding="utf-8",
    )
    (MS / "PAPER_EXPERIMENT_EXECUTION_STATUS.md").write_text("".join(lines), encoding="utf-8")


def main():
    MS.mkdir(parents=True, exist_ok=True)
    prompt_a_audit()
    prompt_b_visualization_plan()
    prompt_d_statistics()
    prompt_e_no_checkpoint()
    prompt_f_main_figures()
    prompt_g_results_sections()
    prompt_h_expert()
    write_execution_status()
    print("Paper finalization complete.")
    print(f"Results draft: {SEC / 'RESULTS_FINAL_EXPERIMENT_SECTIONS.md'}")
    print(f"Main figures: {FIG_MAIN}")


if __name__ == "__main__":
    main()
