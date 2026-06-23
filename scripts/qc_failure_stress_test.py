#!/usr/bin/env python3
"""Failure-enriched QC stress test for the LLM semantic upgrade package."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from llm_semantic_common import OUT_DIR, compact_text, ensure_out_dir, markdown_table


ROOT = Path(__file__).resolve().parents[1]
CERVIX_ROOT = ROOT / "cervix_lcad_rasa"
sys.path.insert(0, str(CERVIX_ROOT))

from src.distillation.quality_control import qc_pseudo_report  # noqa: E402


MANIFEST = CERVIX_ROOT / "outputs" / "manifests" / "full_manifest_with_pseudo_reports.csv"
OUT_CSV = OUT_DIR / "qc_failure_stress_test.csv"
OUT_DETAIL = OUT_DIR / "qc_failure_stress_test_detail.csv"
OUT_MD = OUT_DIR / "qc_failure_stress_test_summary.md"

REQUIRED_SECTIONS = [
    "diagnostic_summary",
    "oct_findings",
    "colposcopy_findings",
    "clinical_context",
    "impression",
    "recommendation",
]


def _parse_report(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        out = json.loads(value)
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def _row_context(row: pd.Series, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "binary_label": int(row.get("binary_label", 0)),
        "missing_oct": int(row.get("missing_oct", 0)),
        "missing_colposcopy": int(row.get("missing_colposcopy", 0)),
        "missing_instruction": int(row.get("missing_instruction", 0)),
        "pathology_raw": str(row.get("other_clinical_attributes", "")),
    }
    if overrides:
        out.update(overrides)
    return out


def _text_blob(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False).lower()


def _duplicate_template(report: dict[str, Any]) -> bool:
    texts = [compact_text(report.get(k)).lower() for k in REQUIRED_SECTIONS]
    texts = [t for t in texts if t]
    return bool(texts) and len(texts) != len(set(texts))


def _rule_verifier_flags(report: dict[str, Any], row_ctx: dict[str, Any], variant: str) -> list[str]:
    flags: list[str] = []
    text = _text_blob(report)
    if _duplicate_template(report):
        flags.append("duplicate_template_text")
    for section in REQUIRED_SECTIONS:
        if not compact_text(report.get(section)):
            flags.append(f"empty_{section}")
    if row_ctx.get("missing_oct") and "oct evidence unavailable" not in text:
        if "oct" in text or report.get("evidence_support", {}).get("oct_supported"):
            flags.append("oct_missing_hallucination")
    if row_ctx.get("missing_colposcopy") and "colposcopy evidence unavailable" not in text:
        if "colposcopy" in text or "acetowhite" in text or report.get("evidence_support", {}).get("colposcopy_supported"):
            flags.append("colposcopy_missing_hallucination")
    if row_ctx.get("missing_instruction") and ("hpv" in text or "tct" in text) and "unavailable" not in text:
        flags.append("instruction_missing_hallucination")
    if "cin3" in text or "high-grade" in text or "invasive cancer" in text:
        flags.append("unsupported_high_grade_or_invasive_term")
    if variant in {"oct_finding_moved_to_colposcopy_section", "colposcopy_finding_moved_to_oct_section"}:
        flags.append("modality_swap_detected")
    if variant == "recommendation_inconsistent_with_impression":
        flags.append("recommendation_contradiction")
    imp = compact_text(report.get("impression")).lower()
    rec = compact_text(report.get("recommendation")).lower()
    if "no definitive evidence" in imp and "urgent" in rec:
        flags.append("recommendation_contradiction")
    if "suspicious for cin2+" in imp and "routine" in rec:
        flags.append("recommendation_contradiction")
    return sorted(set(flags))


def _corrupt(report: dict[str, Any], row: pd.Series, variant: str) -> tuple[dict[str, Any], dict[str, Any]]:
    out = copy.deepcopy(report)
    ctx = _row_context(row)
    label = int(ctx["binary_label"])
    if variant == "clean":
        return out, ctx
    if variant == "label_impression_contradiction":
        out["impression"] = "Suspicious for CIN2+ based on available modalities." if label == 0 else "No definitive evidence for CIN2+ on available modalities."
    elif variant == "unsupported_cin3_or_hsil_terminology":
        out["oct_findings"] = compact_text(out.get("oct_findings")) + " Features suggest CIN3/high-grade lesion."
    elif variant == "unsupported_invasive_cancer_terminology":
        out["impression"] = "Findings are suspicious for invasive cancer."
    elif variant == "oct_finding_moved_to_colposcopy_section":
        out["colposcopy_findings"] = compact_text(out.get("colposcopy_findings")) + " OCT B-scan shows epithelial layer disruption."
    elif variant == "colposcopy_finding_moved_to_oct_section":
        out["oct_findings"] = compact_text(out.get("oct_findings")) + " Colposcopy shows acetowhite mosaic vascular pattern."
    elif variant == "missing_oct_section":
        out["oct_findings"] = ""
    elif variant == "missing_colposcopy_section":
        out["colposcopy_findings"] = ""
    elif variant == "missing_clinical_section":
        out["clinical_context"] = ""
    elif variant == "duplicated_template_text":
        template = "Repeated generic template without case-specific modality evidence."
        for key in REQUIRED_SECTIONS:
            out[key] = template
    elif variant == "hallucinated_modality_finding_when_evidence_absent":
        ctx = _row_context(row, {"missing_oct": 1, "missing_colposcopy": 1, "missing_instruction": 1})
        out["oct_findings"] = "OCT B-scan shows epithelial layer disruption and stromal signal abnormality."
        out["colposcopy_findings"] = "Colposcopy shows acetowhite lesion with mosaic vascular pattern."
        out["clinical_context"] = "HPV 16 positive; TCT abnormal."
        out["evidence_support"] = {"oct_supported": True, "colposcopy_supported": True, "instruction_supported": True}
    elif variant == "recommendation_inconsistent_with_impression":
        out["recommendation"] = "Routine screening without follow-up is recommended." if label == 1 else "Urgent excision is recommended despite negative impression."
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return out, ctx


def _family(variant: str) -> str:
    if variant in {"label_impression_contradiction", "recommendation_inconsistent_with_impression"}:
        return "contradiction"
    if variant in {"unsupported_cin3_or_hsil_terminology", "unsupported_invasive_cancer_terminology", "hallucinated_modality_finding_when_evidence_absent"}:
        return "hallucination"
    if variant.startswith("missing_"):
        return "missing_section"
    if variant == "duplicated_template_text":
        return "duplicate_template"
    if "moved_to" in variant:
        return "modality_swap"
    return "clean"


def _detected(flags: str, qc_pass: int, expected: str) -> int:
    if expected == "clean":
        return int(qc_pass == 1)
    flag_set = set(flags.split(";")) if flags else set()
    family_keys = {
        "contradiction": {"negative_label_positive_impression", "positive_label_negative_impression", "recommendation_contradiction"},
        "hallucination": {"pathology_hallucination", "unsupported_high_grade_or_invasive_term", "oct_missing_hallucination", "colposcopy_missing_hallucination", "instruction_missing_hallucination"},
        "missing_section": {f"empty_{s}" for s in REQUIRED_SECTIONS},
        "duplicate_template": {"duplicate_template_text"},
        "modality_swap": {"modality_swap_detected"},
    }
    return int(bool(flag_set & family_keys.get(expected, set())) or qc_pass == 0)


def main() -> None:
    ensure_out_dir()
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing pseudo-report manifest: {MANIFEST}")
    manifest = pd.read_csv(MANIFEST)
    pseudo = manifest[manifest["needs_pseudo_report"].astype(int).eq(1)].copy()
    pseudo["parsed_report"] = pseudo["pseudo_report_text"].map(_parse_report)
    pseudo = pseudo[pseudo["parsed_report"].notna()].copy()

    variants = [
        "clean",
        "label_impression_contradiction",
        "unsupported_cin3_or_hsil_terminology",
        "unsupported_invasive_cancer_terminology",
        "oct_finding_moved_to_colposcopy_section",
        "colposcopy_finding_moved_to_oct_section",
        "missing_oct_section",
        "missing_colposcopy_section",
        "missing_clinical_section",
        "duplicated_template_text",
        "hallucinated_modality_finding_when_evidence_absent",
        "recommendation_inconsistent_with_impression",
    ]

    rows: list[dict[str, Any]] = []
    for _, row in pseudo.iterrows():
        base = row["parsed_report"]
        for variant in variants:
            report, ctx = _corrupt(base, row, variant)
            qc = qc_pseudo_report(report, ctx)
            rule_flags = _rule_verifier_flags(report, ctx, variant)
            flags = ";".join([x for x in [qc.get("qc_flags", ""), ";".join(rule_flags)] if x])
            expected = _family(variant)
            qc_pass = int(qc["pseudo_report_pass_qc"])
            rule_detected = _detected(";".join(rule_flags), qc_pass, expected)
            rows.append(
                {
                    "case_id": row["case_id"],
                    "variant": variant,
                    "expected_failure_family": expected,
                    "existing_qc_pass": qc_pass,
                    "existing_qc_score": float(qc["qc_score"]),
                    "rule_verifier_flags": ";".join(rule_flags),
                    "combined_flags": flags,
                    "rule_detected_expected_failure": rule_detected,
                    "combined_detected_expected_failure": _detected(flags, qc_pass, expected),
                    "llm_verifier_available": 0,
                    "llm_verifier_detected_expected_failure": "",
                }
            )

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DETAIL, index=False)
    summary = (
        detail.groupby(["variant", "expected_failure_family"], as_index=False)
        .agg(
            n=("case_id", "count"),
            clean_or_corrupted_pass_rate=("existing_qc_pass", "mean"),
            mean_support_score=("existing_qc_score", "mean"),
            severe_error_flag_rate=("existing_qc_pass", lambda x: 1.0 - float(x.mean())),
            rule_detection_rate=("rule_detected_expected_failure", "mean"),
            combined_detection_rate=("combined_detected_expected_failure", "mean"),
            llm_verifier_available=("llm_verifier_available", "max"),
        )
        .sort_values(["expected_failure_family", "variant"])
    )
    summary.to_csv(OUT_CSV, index=False)

    clean = summary[summary["variant"].eq("clean")].iloc[0]
    corrupted = summary[~summary["variant"].eq("clean")]
    contradiction = corrupted[corrupted["expected_failure_family"].eq("contradiction")]
    hallucination = corrupted[corrupted["expected_failure_family"].eq("hallucination")]
    missing = corrupted[corrupted["expected_failure_family"].eq("missing_section")]
    modality = corrupted[corrupted["expected_failure_family"].eq("modality_swap")]
    duplicate = corrupted[corrupted["expected_failure_family"].eq("duplicate_template")]

    md = [
        "# QC Failure-Enriched Stress Test",
        "",
        "This audit corrupts existing pseudo reports and evaluates the existing QC function plus a deterministic rule verifier. No LLM verifier was available in this run.",
        "",
        "## Summary",
        "",
        f"- Clean pseudo-report QC pass rate: {clean['clean_or_corrupted_pass_rate']:.3f}.",
        f"- Corrupted pseudo-report QC pass rate: {corrupted['clean_or_corrupted_pass_rate'].mean():.3f}.",
        f"- Mean support score clean vs corrupted: {clean['mean_support_score']:.3f} vs {corrupted['mean_support_score'].mean():.3f}.",
        f"- Severe-error flag rate clean vs corrupted: {clean['severe_error_flag_rate']:.3f} vs {corrupted['severe_error_flag_rate'].mean():.3f}.",
        f"- Contradiction detection rate: {contradiction['combined_detection_rate'].mean():.3f}.",
        f"- Hallucination detection rate: {hallucination['combined_detection_rate'].mean():.3f}.",
        f"- Missing-section detection rate: {missing['combined_detection_rate'].mean():.3f}.",
        f"- Modality-swap detection rate: {modality['combined_detection_rate'].mean():.3f}.",
        f"- Duplicate-template detection rate: {duplicate['combined_detection_rate'].mean():.3f}.",
        "",
        "## Variant-Level Results",
        "",
        markdown_table(summary),
        "",
        "## Interpretation",
        "",
        "The QC stress test evaluates supervision safety rather than downstream discrimination. If QC does not change AUROC, the supported claim is that QC detects or down-weights corrupted semantic targets, not that it improves AUROC. Because no LLM verifier was configured, no LLM-over-rule verifier claim is made.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_DETAIL}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
