"""Aggregate perturbation experiment metrics with text decoding."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation_publishable.clinical_consistency import clinical_metrics
from src.evaluation_publishable.hallucination import hallucination_flags, hallucination_rates
from src.evaluation_publishable.report_metrics import compute_reference_metrics, rouge_l
from src.evaluation_publishable.section_consistency import SECTION_KEYS, section_completeness, section_supported_scores


def full_report_text(sections: dict[str, str]) -> str:
    return " ".join(str(sections.get(k, "")) for k in SECTION_KEYS)


def section_similarity(a: str, b: str) -> float:
    return rouge_l(a or "", b or "")


def aggregate_condition(
    condition: str,
    case_records: list[dict[str, Any]],
    normal_by_case: dict[str, dict[str, str]],
    refs: dict[str, str] | None = None,
) -> dict[str, Any]:
    refs = refs or {}
    n = len(case_records)
    ref_n = sum(1 for c in case_records if refs.get(c["case_id"], ""))
    rouges, bleus, meteors, berts = [], [], [], []
    lc_list, contra, sect_comp, risk_scores = [], [], [], []
    all_flags = []
    sim_report, sim_oct, sim_col, sim_ctx, sim_imp = [], [], [], [], []

    for rec in case_records:
        cid = rec["case_id"]
        sec = rec["generated_sections"]
        text = full_report_text(sec)
        label = int(rec.get("binary_label", 0))
        cm = clinical_metrics(text, label)
        lc_list.append(cm["label_consistency"])
        contra.append(cm["contradiction_rate"])
        sc = section_completeness(sec)
        sect_comp.append(sc["overall_section_completeness"])
        risk_scores.append(float(rec.get("risk_score", 0.5)))
        flags = hallucination_flags(sec, condition)
        all_flags.append(flags)
        rec["hallucination_flags"] = flags

        ref = refs.get(cid, "")
        if ref and len(ref) > 20:
            rm = compute_reference_metrics(text, ref)
            rouges.append(rm["rouge_l"])
            bleus.append(rm["bleu"])
            meteors.append(rm["meteor"])
            berts.append(rm["bertscore_f1"])

        if cid in normal_by_case:
            ns = normal_by_case[cid]
            sim_report.append(section_similarity(text, full_report_text(ns)))
            sim_oct.append(section_similarity(sec.get("oct_findings", ""), ns.get("oct_findings", "")))
            sim_col.append(section_similarity(sec.get("colposcopy_findings", ""), ns.get("colposcopy_findings", "")))
            sim_ctx.append(section_similarity(sec.get("clinical_context", ""), ns.get("clinical_context", "")))
            sim_imp.append(section_similarity(sec.get("impression", ""), ns.get("impression", "")))

    hal = hallucination_rates(all_flags)
    row = {
        "condition": condition,
        "n_cases": n,
        "reference_available_n": ref_n,
        "rouge_l": float(np.mean(rouges)) if rouges else np.nan,
        "bleu": float(np.mean(bleus)) if bleus else np.nan,
        "meteor": float(np.mean(meteors)) if meteors else np.nan,
        "bertscore": float(np.mean(berts)) if berts else np.nan,
        "label_consistency": float(np.mean(lc_list)),
        "contradiction_rate": float(np.mean(contra)),
        "section_completeness": float(np.mean(sect_comp)),
        "mean_risk_score": float(np.mean(risk_scores)),
        "risk_score_std": float(np.std(risk_scores)),
        "report_similarity_to_normal": float(np.mean(sim_report)) if sim_report else np.nan,
        "oct_findings_similarity_to_normal": float(np.mean(sim_oct)) if sim_oct else np.nan,
        "colposcopy_findings_similarity_to_normal": float(np.mean(sim_col)) if sim_col else np.nan,
        "clinical_context_similarity_to_normal": float(np.mean(sim_ctx)) if sim_ctx else np.nan,
        "impression_similarity_to_normal": float(np.mean(sim_imp)) if sim_imp else np.nan,
        **hal,
    }
    return row


def by_section_rows(condition: str, case_records: list[dict], normal_by_case: dict) -> list[dict]:
    rows = []
    for sec in SECTION_KEYS:
        present, sims, hall = [], [], []
        for rec in case_records:
            s = rec["generated_sections"].get(sec, "")
            present.append(len(s.strip()) >= 15)
            cid = rec["case_id"]
            if cid in normal_by_case:
                sims.append(section_similarity(s, normal_by_case[cid].get(sec, "")))
            hall.append(
                1.0
                if any(
                    f in rec.get("hallucination_flags", [])
                    for f in (
                        "oct_missing_hallucination",
                        "colposcopy_missing_hallucination",
                        "instruction_missing_hallucination",
                    )
                )
                else 0.0
            )
        rows.append(
            {
                "condition": condition,
                "section": sec,
                "section_present_rate": float(np.mean(present)),
                "section_similarity_to_normal": float(np.mean(sims)) if sims else np.nan,
                "missing_modality_hallucination_rate": float(np.mean(hall)),
            }
        )
    return rows
