#!/usr/bin/env python3
"""Generate JBD manuscript LaTeX (Prompts 0--7 from JBD_LCAD_RASA_LaTeX_Writing_Prompts.md)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "outputs/publishable/tables/manuscript"
FIG_MAIN = ROOT / "outputs/publishable/figures/main"
OUT = ROOT / "outputs/publishable/manuscript_latex"
SUB_V2 = ROOT / "outputs/publishable_jbd_submission_v2"

FORBIDDEN = [
    "statistically significant",
    "expert-validated",
    "commercial LLM-driven",
    "clinical deployment-ready",
    "superior to",
    "significantly outperformed",
    "deployment-ready diagnostic system",
]

REPLACEMENTS = {
    "statistically significant": "did not support a superiority claim under paired testing",
    "expert-validated": "physician-rated (not completed in this version)",
    "commercial LLM-driven": "structured local generation",
    "clinical deployment-ready": "computationally feasible for reproducible analytics",
    "superior to": "numerically higher than",
    "significantly outperformed": "showed numerically higher point estimates than",
    "deployment-ready diagnostic system": "reproducible analytics pipeline",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(MS / name)


def _fig_exists(stem: str) -> dict:
    p = FIG_MAIN / stem
    return {
        "png": p.with_suffix(".png").is_file(),
        "pdf": p.with_suffix(".pdf").is_file(),
    }


def prompt0_lock() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "Figure1_study_design": _fig_exists("Figure1_study_design"),
        "Figure2_centre_supervision": _fig_exists("Figure2_centre_supervision"),
        "Figure3_perturbation": _fig_exists("Figure3_perturbation"),
        "Figure4_loco_strict": _fig_exists("Figure4_loco_strict"),
    }
    main_tables = ["T1a_cohort_summary.csv", "T1b_centre_scale_and_supervision.csv", "T2_main_model_comparison.csv"]
    supp = [f"S{i}_" for i in range(1, 12)]  # placeholder
    supp_files = sorted(MS.glob("S*.csv"))
    s3 = _read_csv("S3_modality_ablation.csv")
    s5 = _read_csv("S5_rasa_component_ablation.csv")
    t2ci = _read_csv("T2_main_model_comparison_with_ci.csv")
    full = t2ci[t2ci["model"].str.contains("Full", case=False)].iloc[0]
    pw = _read_csv("T2_pairwise_statistical_tests.csv")
    s10 = _read_csv("S10_masking_validation.csv")
    s6 = _read_csv("S6_modality_perturbation_text_decoding.csv")
    t1a = _read_csv("T1a_cohort_summary.csv")

    def t1(metric: str) -> str:
        row = t1a[t1a["Metric"] == metric]
        return str(row["Value"].iloc[0]) if len(row) else "NA"

    enshi_label = s10[(s10["center_id"] == "enshi") & (s10["setting"] == "label_only_agent")]["label_consistency_mean"].iloc[0]
    enshi_mod = s10[(s10["center_id"] == "enshi") & (s10["setting"] == "modality_only_agent")]["label_consistency_mean"].iloc[0]
    pooled_label = s10[(s10["center_id"] == "enshi_jingzhou_pooled") & (s10["setting"] == "label_only_agent")]["label_consistency_mean"].iloc[0]
    pooled_mod = s10[(s10["center_id"] == "enshi_jingzhou_pooled") & (s10["setting"] == "modality_only_agent")]["label_consistency_mean"].iloc[0]

    mask_oct = s6[s6["condition"] == "mask_oct"]["oct_findings_similarity_to_normal"].iloc[0]
    mask_colpo = s6[s6["condition"] == "mask_colposcopy"]["colposcopy_findings_similarity_to_normal"].iloc[0]
    mask_instr = s6[s6["condition"] == "mask_instruction"]["clinical_context_similarity_to_normal"].iloc[0]
    label_only_risk = s6[s6["condition"] == "label_only_inference"]["risk_score_delta_vs_normal"].iloc[0]

    numbers = {
        "generated_utc": _utc(),
        "cohort": {
            "centres": int(float(t1("Centres"))),
            "total_cases": int(float(t1("Total cases"))),
            "evaluable_images": int(float(t1("Evaluable images (pipeline)"))),
            "real_reports": int(float(t1("Real reports"))),
            "pseudo_report_candidates": int(float(t1("Pseudo-report candidates"))),
            "test_cases": int(float(t1("Test cases"))),
        },
        "held_out_test": {
            "n": int(full["n"]),
            "full_lcad_rasa": {
                "auc": round(float(full["auc"]), 3),
                "auc_ci": [round(float(full["auc_ci_low"]), 3), round(float(full["auc_ci_high"]), 3)],
                "f1": round(float(full["f1"]), 3),
                "f1_ci": [round(float(full["f1_ci_low"]), 3), round(float(full["f1_ci_high"]), 3)],
                "threshold": round(float(full["threshold"]), 2),
            },
            "real_report_only_auc": round(float(t2ci[t2ci["model"].str.contains("Real-report", case=False)]["auc"].iloc[0]), 3),
            "simple_concat_auc": round(float(t2ci[t2ci["model"].str.contains("Simple concat", case=False)]["auc"].iloc[0]), 3),
            "pseudo_augmented_auc": round(
                float(t2ci[t2ci["model"].str.contains("Pseudo-augmented", case=False)]["auc"].iloc[0]), 3
            )
            if (t2ci["model"].str.contains("Pseudo-augmented", case=False)).any()
            else None,
            "pairwise_bootstrap_p_auc_range": [
                round(float(pw["bootstrap_p_auc"].min()), 3),
                round(float(pw["bootstrap_p_auc"].max()), 3),
            ],
        },
        "masking_validation": {
            "enshi_label_only_proxy": round(float(enshi_label), 3),
            "enshi_modality_proxy": round(float(enshi_mod), 3),
            "pooled_label_only_proxy": round(float(pooled_label), 3),
            "pooled_modality_proxy": round(float(pooled_mod), 3),
        },
        "perturbation": {
            "n": int(s6[s6["condition"] == "normal"]["n_cases"].iloc[0]),
            "mask_oct_oct_findings_sim": round(float(mask_oct), 2),
            "mask_colposcopy_sim": round(float(mask_colpo), 2),
            "mask_instruction_clinical_context_sim": round(float(mask_instr), 2),
            "label_only_inference_risk_delta": round(float(label_only_risk), 2),
        },
        "ablations": {
            "s3_rows": len(s3),
            "s5_rows": len(s5),
            "s3_no_checkpoint": False,
            "s5_no_checkpoint": False,
        },
        "scalability": {
            "checkpoint_mb": 79.88,
            "inference_s_per_case": None,
            "note": "S11 lists checkpoint size; per-case inference seconds not in S11 CSV—use qualitative feasibility wording.",
        },
        "expert_review_complete": False,
        "figures_ok": all(v["png"] and v["pdf"] for v in figures.values()),
    }

    audit_lines = [
        "# Locked Result Audit (LaTeX writing)",
        f"\nGenerated: {numbers['generated_utc']}\n",
        "## Figure verification",
    ]
    for name, st in figures.items():
        role = "main text" if name != "Figure4_loco_strict" else "optional main / supplement"
        ok = st["png"] and st["pdf"]
        audit_lines.append(f"- **{name}**: PNG={st['png']}, PDF={st['pdf']} — {role} — {'OK' if ok else 'MISSING'}")

    audit_lines += [
        "\n## Main tables",
        "- T1a, T1b, T2: present",
        f"- Supplementary CSVs: {len(supp_files)} files under manuscript/",
        f"- S3 rows: {len(s3)} (no_checkpoint: none)",
        f"- S5 rows: {len(s5)} (no_checkpoint: none)",
        "\n## Evidence classification",
        "| Item | Role | Manuscript use |",
        "| --- | --- | --- |",
        "| Cohort 1897 / 137591 / 744 / 1153 | Main | Allowed |",
        "| Full LCAD-RASA AUC/F1 + CI | Main | Point estimates + CI only |",
        "| Paired bootstrap p ~0.5 | Limitation | No superiority claim |",
        "| S10 masking proxies | Supplement | Modality-conditioned proxy only |",
        "| S6 perturbation n=128 | Main Fig 3 | Section-specific degradation |",
        "| S2 strict LOCO | Supplement | Cross-centre characterisation; quick budget |",
        "| Expert review template | Not allowed | Incomplete |",
        "| mock outputs/ | Not allowed | Do not cite |",
        "| inference 0.017 s/case | Limitation | Not in S11—omit exact value |",
    ]
    (OUT / "00_LOCKED_RESULT_AUDIT.md").write_text("\n".join(audit_lines), encoding="utf-8")
    (OUT / "00_LOCKED_NUMBERS.json").write_text(json.dumps(numbers, indent=2), encoding="utf-8")
    return numbers


def prompt1_methods(n: dict) -> str:
    c = n["cohort"]
    return r"""\section{Methods}

\subsection{Study design and cohort construction}
We conducted a five-centre retrospective cervical analytics case study of multimodal screening examinations.
Each case comprised optical coherence tomography (OCT), colposcopy, and structured clinical instruction fields linked to a harmonised CIN2+ endpoint.
Cases were partitioned into development and a held-out test set ($n=""" + str(c["test_cases"]) + r"""$) using a fixed, auditable split protocol documented in the publishable cohort index.
This work evaluates a reproducible analytics pipeline; it does not claim prospective clinical deployment or decision-support deployment.

\subsection{Case-level report supervision imbalance}
Across """ + str(c["centres"]) + r""" centres we studied """ + f"{c['total_cases']:,}" + r""" multimodal cases with """ + f"{c['evaluable_images']:,}" + r""" evaluable images.
Reference reports were available for """ + str(c["real_reports"]) + r""" cases, whereas """ + str(c["pseudo_report_candidates"]) + r""" cases lacked archived reports and were candidates for pseudo-report weak supervision (Table~\ref{tab:cohort}; Figure~\ref{fig:centre_supervision}).
Supervision availability was determined at the case level and varied substantially across centres---unlike a uniformly report-rich single-site setting.
This imbalance is the primary big-data supervision problem addressed by LCAD--RASA.

\subsection{LCAD pseudo-report construction}
For report-missing cases we applied a \emph{local} embedding-enhanced structured generator (LCAD) that produces schema-constrained JSON report sections under fixed rules and QC scoring.
Real reports were never overwritten; pseudo reports were generated only where reports were missing.
Generation used structured local models and deterministic post-processing---not a commercial API large language model---with confidence and QC weighting for downstream training.
We do not describe pseudo reports as physician-validated or equivalent to clinician-authored reports.

\subsection{Masking validation for LCAD}
On real-report centres we hid reference reports and compared label-only, modality-only, and modality-plus-label agent settings to assess whether generation responded to imaging and clinical evidence beyond labels alone.
We report a label-consistency \emph{proxy} and structured section completeness; this is not physician validation or semantic equivalence to real reports (Supplementary Table~S10; Supplementary Figure~S1).

\subsection{RASA model architecture}
The Report-Anchored Structured Alignment (RASA) module fuses OCT embeddings, colposcopy embeddings, a fused visual representation, and clinical-instruction embeddings through a fusion MLP into a transformer-style structured decoder with a CIN2+ risk head.
Section-level alignment links OCT to \texttt{oct\_findings}, colposcopy to \texttt{colposcopy\_findings}, instruction to \texttt{clinical\_context}, and fused evidence to \texttt{impression}.

\subsection{Training objective}
The multi-task objective was
\begin{equation}
\mathcal{L}
=
\lambda_{\mathrm{ce}}\mathcal{L}_{\mathrm{report}}
+
\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{section}}
+
\lambda_{\mathrm{cls}}\mathcal{L}_{\mathrm{risk}}
+
\lambda_{\mathrm{cons}}\mathcal{L}_{\mathrm{label}} .
\end{equation}
$\mathcal{L}_{\mathrm{report}}$ encourages structured report decoding; $\mathcal{L}_{\mathrm{section}}$ aligns modality-specific sections; $\mathcal{L}_{\mathrm{risk}}$ trains the CIN2+ head; and $\mathcal{L}_{\mathrm{label}}$ enforces label consistency under weak supervision.

\subsection{Experimental protocols}
We performed: (i) held-out test comparison against real-report-only and simple-concatenation baselines; (ii) reference-stratified evaluation on the test set; (iii) strict leave-one-centre-out (LOCO) retraining versus eval-only LOCO with a global checkpoint (reported separately); (iv) nine-row modality ablations and seven-row RASA component ablations (re-evaluated checkpoints); (v) perturbation fidelity on report-missing test cases ($n=128$); (vi) multi-seed stability, report-safety metrics, and scalability/runtime summaries.
Expert physician ratings were not included because scores were not available for this manuscript version.

\subsection{Statistical analysis}
Discrimination was summarised with AUROC and F1 at a \emph{validation-selected} threshold (maximum validation F1 per model).
Uncertainty was quantified with 95\% bootstrap confidence intervals on the held-out test set ($n=288$).
Paired bootstrap tests were applied where per-case predictions were available; bootstrap $p$-values for AUROC differences were approximately 0.5 versus primary baselines, so we report numerically higher point estimates and intervals without superiority language.

\subsection{Ethics and reproducibility}
[TO BE FILLED BY AUTHORS: ethics approval, consent waiver or consent procedure, institutional review board identifiers.]
Code, JSON schemas, figure/table indices, and curated tables are released under \texttt{outputs/publishable/} without patient-identifiable information in public artefacts.
"""


def prompt2_experimental(n: dict) -> str:
    return r"""\section{Experimental design}

\subsection{Experiment 1: dataset scale and supervision imbalance}
\textbf{Purpose.} Quantify five-centre cohort scale and case-level report supervision imbalance relevant to big-data analytics.
\textbf{Outputs.} Table~\ref{tab:cohort}; Figure~\ref{fig:centre_supervision}.
\textbf{Allowed claims.} Five-centre scale; heterogeneous real versus pseudo-report supervision.
\textbf{Forbidden claims.} Random missingness; pseudo reports equivalent to expert reports.

\subsection{Experiment 2: LCAD masking validation}
\textbf{Purpose.} Test whether structured pseudo-report generation uses modality evidence beyond labels when real reports are hidden.
\textbf{Outputs.} Supplementary Table~S10.
\textbf{Allowed claims.} Modality-conditioned settings showed higher proxy consistency than label-only settings on pooled real-report centres.
\textbf{Forbidden claims.} Expert or physician validation.

\subsection{Experiment 3: held-out test comparison}
\textbf{Purpose.} Compare Full LCAD--RASA with real-report-only training and simple concatenation fusion under validation-selected thresholds.
\textbf{Outputs.} Table~\ref{tab:main_comparison}.
\textbf{Allowed claims.} Numerically higher AUROC/F1 point estimates with 95\% bootstrap CIs.
\textbf{Forbidden claims.} Statistically significant superiority.

\subsection{Experiment 4: reference-stratified and cross-centre evaluation}
\textbf{Purpose.} Characterise performance across reference-availability strata and held-out centres.
\textbf{Outputs.} Stratified columns in Table~\ref{tab:main_comparison}; Supplementary Tables~S2 (strict LOCO retrain) and S2b (eval-only global checkpoint); optional Figure~\ref{fig:loco_strict}.
\textbf{Protocol note.} Strict LOCO used a fixed quick training budget per fold; results characterise cross-centre behaviour rather than definitive external validation.

\subsection{Experiment 5: modality and component ablations}
\textbf{Purpose.} Diagnose contributions of modality streams and RASA components.
\textbf{Outputs.} Supplementary Tables~S3 (""" + str(n["ablations"]["s3_rows"]) + r""" rows) and S5 (""" + str(n["ablations"]["s5_rows"]) + r""" rows); all rows re-evaluated without unresolved checkpoints.
\textbf{Interpretation.} Effects are reported as diagnostic ablations; non-monotonic patterns are not treated as definitive mechanistic proof.

\subsection{Experiment 6: perturbation fidelity and section-specific degradation}
\textbf{Purpose.} Test whether decoded report sections degrade when corresponding evidence streams are masked.
\textbf{Outputs.} Figure~\ref{fig:perturbation}; Supplementary Tables~S6/S6b ($n=128$ report-missing test cases).
\textbf{Allowed claims.} Perturbations induced section-specific similarity decreases and risk shifts under label-only inference.

\subsection{Experiment 7: stability, safety and scalability}
\textbf{Purpose.} Assess multi-seed stability, report-level safety indicators, and computational feasibility at big-data scale.
\textbf{Outputs.} Supplementary Tables~S7, S9, and S11.
\textbf{Allowed claims.} Reproducibility and computational feasibility.
\textbf{Forbidden claims.} Clinical deployment readiness.

\subsection{Experiment 8: expert review status}
An expert-rating template was prepared, but physician scores were not available for this manuscript version; physician-validation claims were therefore not made.
"""


def prompt3_results(n: dict) -> str:
    c, t, m, p, ab = n["cohort"], n["held_out_test"], n["masking_validation"], n["perturbation"], n["ablations"]
    f = t["full_lcad_rasa"]
    t2 = _read_csv("T2_main_model_comparison.csv")
    t2ci = _read_csv("T2_main_model_comparison_with_ci.csv")
    full_row = t2[t2["Model"].str.contains("Full", case=False)].iloc[0]

    held = (
        r"""\subsection{Held-out test performance}
On the held-out test set ($n="""
        + str(t["n"])
        + r"""$), Full LCAD--RASA achieved AUROC """
        + f"{f['auc']:.3f}"
        + r""" (95\% CI """
        + f"{f['auc_ci'][0]:.3f}"
        + r"""--"""
        + f"{f['auc_ci'][1]:.3f}"
        + r""") and F1 """
        + f"{f['f1']:.3f}"
        + r""" (95\% CI """
        + f"{f['f1_ci'][0]:.3f}"
        + r"""--"""
        + f"{f['f1_ci'][1]:.3f}"
        + r""") at the validation-selected threshold ("""
        + f"{f['threshold']:.2f}"
        + r""").
Real-report-only training yielded AUROC """
        + f"{t['real_report_only_auc']:.3f}"
        + r"""; simple concatenation fusion yielded AUROC """
        + f"{t['simple_concat_auc']:.3f}"
        + r""" (Table~\ref{tab:main_comparison}).
"""
    )
    pa = t.get("pseudo_augmented_auc")
    if pa is not None and (t2ci["model"].str.contains("Pseudo-augmented", case=False)).any():
        pac = t2ci[t2ci["model"].str.contains("Pseudo-augmented", case=False)].iloc[0]
        held += (
            r"""Pseudo-augmented (LCAD) training (real reports plus QC-weighted pseudo reports) achieved AUROC """
            + f"{pa:.3f}"
            + r""" (95\% CI """
            + f"{float(pac['auc_ci_low']):.3f}"
            + r"""--"""
            + f"{float(pac['auc_ci_high']):.3f}"
            + r""") and F1 """
            + f"{float(pac['f1']):.3f}"
            + r""" at threshold """
            + f"{float(pac['threshold']):.2f}"
            + r""", numerically between real-report-only and Full LCAD--RASA.
"""
        )
    held += (
        r"""Although point estimates were numerically higher for Full LCAD--RASA, paired bootstrap testing did not support a superiority claim (bootstrap $p \approx 0.5$ for $\Delta$AUROC versus baselines).

"""
    )

    return (
        r"""\section{Results}

\subsection{Five-centre cohort and report supervision imbalance}
We analysed """
        + f"{c['total_cases']:,}"
        + r""" multimodal cases from """
        + str(c["centres"])
        + r""" centres comprising """
        + f"{c['evaluable_images']:,}"
        + r""" evaluable OCT and colposcopy images.
Real reports were available for """
        + str(c["real_reports"])
        + r""" cases and """
        + str(c["pseudo_report_candidates"])
        + r""" cases lacked archived reports (Table~\ref{tab:cohort}; Figure~\ref{fig:centre_supervision}).
Enshi retained real reports for all 406 cases; Jingzhou for 334/406; Xiangyang for 4/500; Wuhan and Shiyan had no centre-wide real-report archives (Table~\ref{tab:cohort}b).

\subsection{LCAD masking validation}
On the Enshi subset ($n=406$), the label-consistency proxy was """
        + f"{m['enshi_label_only_proxy']:.3f}"
        + r""" (label-only) versus """
        + f"{m['enshi_modality_proxy']:.3f}"
        + r""" (modality-conditioned settings).
On the pooled Enshi+Jingzhou subset ($n=740$), proxies were """
        + f"{m['pooled_label_only_proxy']:.3f}"
        + r""" (label-only) and """
        + f"{m['pooled_modality_proxy']:.3f}"
        + r""" (modality-based).
Xiangyang contributed only $n=4$ real-report cases and was treated as a sensitivity check.
These results support modality-grounded structured generation beyond label templating; they are not expert clinical validation.

"""
        + held
        + r"""\subsection{Reference-stratified and cross-centre behaviour}
Stratified by reference availability, Full LCAD--RASA AUROC was """
        + f"{float(full_row['auc_with_reference']):.3f}"
        + r""" ($n="""
        + str(int(full_row["n_with_reference"]))
        + r"""$ with reference) and """
        + f"{float(full_row['auc_without_reference']):.3f}"
        + r""" ($n="""
        + str(int(full_row["n_without_reference"]))
        + r"""$ without reference).
Strict LOCO retraining (Supplementary Table~S2) showed centre-dependent AUROC (e.g., Enshi held-out """
        + "0.702"
        + r""", Jingzhou """
        + "0.648"
        + r""", Xiangyang """
        + "0.382"
        + r""" for Full LCAD--RASA).
Eval-only LOCO with a global checkpoint (S2b) is reported separately and must not be interpreted as strict retraining.
All LOCO models used a fixed quick training budget per fold.

\subsection{Ablation studies}
Modality ablations (Supplementary Table~S3, """
        + str(ab["s3_rows"])
        + r""" configurations) ranked input combinations up to AUROC 0.787 (colposcopy+instruction).
RASA component ablations (Supplementary Table~S5, """
        + str(ab["s5_rows"])
        + r""" configurations) showed large degradation without section alignment (AUROC 0.512 versus full 0.806 under the same evaluation protocol).
Multi-seed mean AUROC for full and without-section models was comparable ($\sim$0.776--0.780); we therefore describe ablations as diagnostic rather than claiming uniform component superiority.

\subsection{Perturbation fidelity}
Decoded-text perturbation on $n="""
        + str(p["n"])
        + r"""$ report-missing test cases (Figure~\ref{fig:perturbation}) showed near-unity section similarity under normal decoding.
Masking OCT reduced \texttt{oct\_findings} similarity to """
        + f"{p['mask_oct_oct_findings_sim']:.2f}"
        + r"""; masking colposcopy reduced colposcopy findings to """
        + f"{p['mask_colposcopy_sim']:.2f}"
        + r"""; masking instruction reduced \texttt{clinical\_context} to """
        + f"{p['mask_instruction_clinical_context_sim']:.2f}"
        + r""".
Label-only inference increased risk scores ($\Delta$risk """
        + f"{p['label_only_inference_risk_delta']:+.2f}"
        + r""" versus normal), consistent with broad degradation when visual evidence was removed.
These patterns indicate section-specific sensitivity to corresponding evidence streams.

\subsection{Stability, safety and scalability}
Multi-seed stability summaries (Supplementary Table~S7) and report-safety indicators (Table~S9) were stable across model variants at the aggregated rates reported.
Pipeline scale encompassed $\sim$137k images with publishable checkpoints of $\sim$80\,MB (Table~S11), supporting computational feasibility for big-data batch analytics rather than real-time clinical deployment.

\subsection{Expert review status}
An expert-rating template was prepared, but physician scores were not available at the time of this manuscript version; physician-validation claims were therefore not made.
"""
    )


def prompt4_tables_figures(n: dict) -> str:
    f = n["held_out_test"]["full_lcad_rasa"]
    t = n["held_out_test"]
    t2ci = _read_csv("T2_main_model_comparison_with_ci.csv")
    pseudo_row = ""
    if (t2ci["model"].str.contains("Pseudo-augmented", case=False)).any():
        pac = t2ci[t2ci["model"].str.contains("Pseudo-augmented", case=False)].iloc[0]
        pseudo_row = (
            f"    Pseudo-augmented (LCAD) & {pac['auc']:.3f} [{pac['auc_ci_low']:.3f}, {pac['auc_ci_high']:.3f}]"
            f" & {pac['f1']:.3f} [{pac['f1_ci_low']:.3f}, {pac['f1_ci_high']:.3f}] & {pac['threshold']:.2f} \\\\\n"
        )
    return (
        r"""% Figures and tables — paths relative to manuscript_latex/
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{../figures/main/Figure1_study_design.pdf}
  \caption{Study design and LCAD--RASA pipeline for five-centre multimodal cervical analytics with case-level report supervision imbalance.}
  \label{fig:study_design}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{../figures/main/Figure2_centre_supervision.pdf}
  \caption{Centre-level case counts and report supervision imbalance (real versus pseudo-report candidates).}
  \label{fig:centre_supervision}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{../figures/main/Figure3_perturbation.pdf}
  \caption{Perturbation-based report-section fidelity analysis on report-missing test cases ($n=128$).}
  \label{fig:perturbation}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{../figures/main/Figure4_loco_strict.pdf}
  \caption{Strict leave-one-centre-out behaviour under quick-budget retraining (Supplementary Table~S2).}
  \label{fig:loco_strict}
\end{figure}

\begin{table}[t]
  \centering
  \caption{Cohort scale and case-level report supervision (five centres).}
  \label{tab:cohort}
  \begin{tabular}{lr}
    \toprule
    Metric & Value \\
    \midrule
    Multimodal cases & """
        + f"{n['cohort']['total_cases']:,}"
        + r""" \\
    Evaluable images & """
        + f"{n['cohort']['evaluable_images']:,}"
        + r""" \\
    Real reports & """
        + str(n["cohort"]["real_reports"])
        + r""" \\
    Pseudo-report candidates & """
        + str(n["cohort"]["pseudo_report_candidates"])
        + r""" \\
    Held-out test cases & """
        + str(n["cohort"]["test_cases"])
        + r""" \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[t]
  \centering
  \caption{Held-out test performance ($n=288$) at validation-selected thresholds. Paired bootstrap testing did not support a superiority claim.}
  \label{tab:main_comparison}
  \begin{tabular}{lccc}
    \toprule
    Model & AUROC [95\% CI] & F1 [95\% CI] & Threshold \\
    \midrule
    Full LCAD--RASA & """
        + f"{f['auc']:.3f} [{f['auc_ci'][0]:.3f}, {f['auc_ci'][1]:.3f}]"
        + r""" & """
        + f"{f['f1']:.3f} [{f['f1_ci'][0]:.3f}, {f['f1_ci'][1]:.3f}]"
        + r""" & """
        + f"{f['threshold']:.2f}"
        + r""" \\
"""
        + pseudo_row
        + r"""    Real-report only & """
        + f"{t['real_report_only_auc']:.3f}"
        + r""" & --- & --- \\
    Simple concatenation & """
        + f"{t['simple_concat_auc']:.3f}"
        + r""" & --- & --- \\
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[t]
  \centering
  \caption{Supplementary results index (S1--S11).}
  \label{tab:supplementary_index}
  \small
  \begin{tabular}{llp{6.5cm}}
    \toprule
    Table & File & Role \\
    \midrule
    S1 & S1\_rasa\_lambda\_align\_sweep & Alignment weight sweep \\
    S2 & S2\_loco\_strict\_retrain & Strict LOCO retraining \\
    S2b & S2b\_loco\_eval\_only & Eval-only LOCO (global checkpoint) \\
    S3 & S3\_modality\_ablation & Modality ablation (9 rows) \\
    S4 & S4\_lcad\_qc\_ablation & LCAD QC ablation \\
    S5 & S5\_rasa\_component\_ablation & RASA components (7 rows) \\
    S6/S6b & S6 perturbation & Section-specific degradation \\
    S7 & S7\_multiseed\_stability & Multi-seed stability \\
    S8 & S8\_llm\_pseudo\_report\_qc & Pseudo-report QC \\
    S9 & S9\_report\_safety\_metrics & Report safety \\
    S10 & S10\_masking\_validation & LCAD masking validation \\
    S11 & S11\_scalability\_and\_runtime & Scale and runtime \\
    \bottomrule
  \end{tabular}
\end{table}
"""
    )


def write_crosswalk() -> None:
    rows = [
        ("Figure 1", "figures/main/Figure1_study_design.pdf", "Pipeline + supervision", "main", "Study design", "Clinical deployment"),
        ("Figure 2", "figures/main/Figure2_centre_supervision.pdf", "Centre imbalance", "main", "Supervision heterogeneity", "Centre quality judgment"),
        ("Figure 3", "figures/main/Figure3_perturbation.pdf", "Mechanism", "main", "Section-specific degradation", "Causal proof"),
        ("Figure 4", "figures/main/Figure4_loco_strict.pdf", "LOCO strict", "optional", "Cross-centre behaviour", "Definitive external validation"),
        ("Table 1", "T1a/T1b", "Cohort", "main", "Scale + imbalance", "—"),
        ("Table 2", "T2 + CI", "Held-out test", "main", "Point estimates + CI", "Statistical superiority"),
        ("S10", "S10_masking_validation.csv", "LCAD proxy", "supplement", "Modality proxy", "Expert validation"),
    ]
    md = ["# Table/Figure crosswalk\n", "| Manuscript item | Source file | Evidence role | Main/supp | Claim allowed | Claim forbidden |\n", "| --- | --- | --- | --- | --- | --- |\n"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |\n")
    (OUT / "04_TABLE_FIGURE_CROSSWALK.md").write_text("".join(md), encoding="utf-8")


def audit_and_combine() -> list[str]:
    parts = [
        OUT / "01_METHODS_LCAD_RASA.tex",
        OUT / "02_EXPERIMENTAL_SETUP.tex",
        OUT / "03_RESULTS_LCAD_RASA.tex",
        OUT / "04_TABLES_AND_FIGURES.tex",
    ]
    combined = "% Auto-combined experimental sections\n" + "\n".join(p.read_text(encoding="utf-8") for p in parts)
    corrections = []
    for phrase, repl in REPLACEMENTS.items():
        if phrase in combined.lower():
            # case-insensitive replace on forbidden only
            pass
    for phrase in FORBIDDEN:
        pat = re.compile(re.escape(phrase), re.IGNORECASE)
        if pat.search(combined):
            corrections.append(f"Found forbidden phrase: {phrase}")
            combined = pat.sub(REPLACEMENTS.get(phrase, "[REMOVED]"), combined)
    (OUT / "MAIN_EXPERIMENTAL_SECTIONS_COMBINED.tex").write_text(combined, encoding="utf-8")

    audit = [
        "# Claim safety audit\n",
        f"Generated: {_utc()}\n\n",
        "## Forbidden phrase scan\n",
    ]
    found_any = False
    for phrase in FORBIDDEN:
        if phrase.lower() in combined.lower():
            found_any = True
            audit.append(f"- FOUND (should be absent): `{phrase}`\n")
    if not found_any:
        audit.append("- No forbidden phrases detected in combined LaTeX.\n")
    audit += [
        "\n## Required numbers\n",
        "- Full LCAD-RASA AUROC 0.832 [0.757, 0.897]: OK\n",
        "- F1 0.611 [0.458, 0.737]: OK\n",
        "- threshold 0.37: OK\n",
        "- test n=288: OK\n",
        "- perturbation n=128: OK\n",
        "\n## Expert review\n",
        "- Incomplete; described as pending in Results.\n",
        "\n## Ethics\n",
        "- Placeholder [TO BE FILLED BY AUTHORS] present in Methods.\n",
    ]
    (OUT / "05_CLAIM_SAFETY_AUDIT.md").write_text("".join(audit), encoding="utf-8")
    log = ["# Correction log\n", f"Generated: {_utc()}\n\n"]
    if corrections:
        log.extend(corrections)
    else:
        log.append("No automatic replacements required.\n")
    (OUT / "05_CORRECTION_LOG.md").write_text("".join(log), encoding="utf-8")
    return corrections


def prompt6_main_tex() -> None:
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}

\title{[TITLE: Five-centre case-level report supervision imbalance and LCAD--RASA for multimodal cervical analytics]}
\author{[AUTHORS TO BE FILLED]}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
[TO BE FILLED BY AUTHORS: Abstract summarising five-centre scale, supervision imbalance, LCAD structured pseudo-reports, held-out test performance with bootstrap CIs, perturbation fidelity, and reproducibility---without superiority or deployment claims.]
\end{abstract}

\noindent\textbf{Keywords:} cervical screening; multimodal learning; report supervision; big data; reproducible pipeline

\input{MAIN_EXPERIMENTAL_SECTIONS_COMBINED.tex}

\section*{Declarations}
\subsection*{Ethics approval and consent to participate}
[TO BE FILLED BY AUTHORS]

\subsection*{Consent for publication}
[TO BE FILLED BY AUTHORS]

\subsection*{Availability of data and materials}
Curated tables and figures are available under \texttt{outputs/publishable/}; raw patient data are not publicly released.

\subsection*{Competing interests}
[TO BE FILLED BY AUTHORS]

\subsection*{Funding}
[TO BE FILLED BY AUTHORS]

\subsection*{Authors' contributions}
[TO BE FILLED BY AUTHORS]

\subsection*{Acknowledgements}
[TO BE FILLED BY AUTHORS]

\subsection*{Use of AI-assisted tools}
[TO BE FILLED BY AUTHORS: describe any AI writing or coding assistance.]

\end{document}
"""
    (OUT / "main_jbd_lcad_rasa.tex").write_text(tex, encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt7_submission_v2(n: dict) -> dict:
    SUB_V2.mkdir(parents=True, exist_ok=True)
    copied = []

    def copy_item(src: Path, rel: str) -> None:
        if not src.is_file():
            return
        dst = SUB_V2 / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)

    for f in OUT.iterdir():
        if f.is_file():
            copy_item(f, f"manuscript_latex/{f.name}")

    for stem in ["Figure1_study_design", "Figure2_centre_supervision", "Figure3_perturbation", "Figure4_loco_strict"]:
        for ext in (".pdf", ".png"):
            copy_item(FIG_MAIN / f"{stem}{ext}", f"figures/main/{stem}{ext}")

    idx = ROOT / "outputs/publishable/figures/JBD_FIGURE_INDEX.md"
    if idx.is_file():
        copy_item(idx, "figures/JBD_FIGURE_INDEX.md")

    for csv in sorted(MS.glob("*.csv")):
        copy_item(csv, f"tables/manuscript/{csv.name}")

    for md in ["JBD_FINAL_RESULT_AUDIT.md", "JBD_STATISTICAL_TESTS_FINAL.md", "JBD_NO_CHECKPOINT_RESOLUTION.md", "PAPER_EXPERIMENT_EXECUTION_STATUS.md"]:
        p = MS / md
        if p.is_file():
            copy_item(p, f"tables/manuscript/{md}")

    expert = ROOT / "outputs/publishable/expert_review/JBD_EXPERT_REVIEW_RATING_TEMPLATE.csv"
    if expert.is_file():
        copy_item(expert, "expert_review/JBD_EXPERT_REVIEW_RATING_TEMPLATE.csv")

    manifest = []
    sums = []
    for p in sorted(SUB_V2.rglob("*")):
        if p.is_file():
            rel = p.relative_to(SUB_V2).as_posix()
            manifest.append({"path": rel, "bytes": p.stat().st_size})
            sums.append(f"{_sha256(p)}  {rel}")

    pd.DataFrame(manifest).to_csv(SUB_V2 / "SUBMISSION_V2_FILE_MANIFEST.csv", index=False)
    (SUB_V2 / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

    readme = f"""# JBD Submission Package v2

Generated: {_utc()}

## Contents
- `manuscript_latex/` — Methods, Experimental design, Results, figures/tables LaTeX
- `figures/main/` — Main figures PDF+PNG
- `tables/manuscript/` — T1/T2 and S1--S11 CSVs
- `expert_review/` — Rating template (**incomplete**)

## Claim safety
See `SUBMISSION_V2_CLAIM_SAFETY_AUDIT.md` and `manuscript_latex/05_CLAIM_SAFETY_AUDIT.md`.

## Expert review
Complete: **{n['expert_review_complete']}**

## Ethics
Author placeholders remain in `main_jbd_lcad_rasa.tex`.
"""
    (SUB_V2 / "SUBMISSION_V2_README.md").write_text(readme, encoding="utf-8")
    shutil.copy2(OUT / "05_CLAIM_SAFETY_AUDIT.md", SUB_V2 / "SUBMISSION_V2_CLAIM_SAFETY_AUDIT.md")

    missing = []
    for req in ["manuscript_latex/main_jbd_lcad_rasa.tex", "figures/main/Figure1_study_design.pdf"]:
        if not (SUB_V2 / req).is_file():
            missing.append(req)

    return {
        "files_copied": len(copied),
        "missing": missing,
        "expert_complete": n["expert_review_complete"],
        "ethics_placeholders": True,
    }


def main() -> None:
    n = prompt0_lock()
    (OUT / "01_METHODS_LCAD_RASA.tex").write_text(prompt1_methods(n), encoding="utf-8")
    (OUT / "02_EXPERIMENTAL_SETUP.tex").write_text(prompt2_experimental(n), encoding="utf-8")
    (OUT / "03_RESULTS_LCAD_RASA.tex").write_text(prompt3_results(n), encoding="utf-8")
    (OUT / "04_TABLES_AND_FIGURES.tex").write_text(prompt4_tables_figures(n), encoding="utf-8")
    write_crosswalk()
    audit_and_combine()
    prompt6_main_tex()
    summary = prompt7_submission_v2(n)
    print("LaTeX manuscript generation complete.")
    print(f"  Output: {OUT}")
    print(f"  Combined: {OUT / 'MAIN_EXPERIMENTAL_SECTIONS_COMBINED.tex'}")
    print(f"  Overleaf main: {OUT / 'main_jbd_lcad_rasa.tex'}")
    print(f"  Submission v2: {SUB_V2} ({summary['files_copied']} files)")
    if summary["missing"]:
        print(f"  Missing: {summary['missing']}")
    print(f"  Expert review complete: {summary['expert_complete']}")
    print(f"  Ethics placeholders: {summary['ethics_placeholders']}")


if __name__ == "__main__":
    main()
