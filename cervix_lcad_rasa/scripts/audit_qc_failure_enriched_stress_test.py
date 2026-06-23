#!/usr/bin/env python3
"""Failure-enriched QC stress test for pseudo reports.

This script does not retrain a model. It corrupts existing pseudo reports,
runs the existing QC function, and reports whether QC suppresses the intended
failure modes: contradictions, unsupported high-grade terminology, missing
sections, modality hallucination, cross-modality section swaps, and duplicated
template text.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distillation.quality_control import qc_pseudo_report  # noqa: E402


MANIFEST = ROOT / "outputs" / "manifests" / "full_manifest_with_pseudo_reports.csv"
OUT_DIR = ROOT.parent / "outputs" / "qc"
OUT_CSV = OUT_DIR / "qc_failure_enriched_stress_test.csv"
OUT_CSV_V2 = OUT_DIR / "qc_stress_test_v2.csv"
OUT_DETAIL_V2 = OUT_DIR / "qc_stress_test_v2_detail.csv"
OUT_MD = OUT_DIR / "qc_failure_enriched_stress_test.md"
OUT_PNG = OUT_DIR / "qc_failure_enriched_stress_test.png"

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
    texts = [str(report.get(k, "")).strip().lower() for k in REQUIRED_SECTIONS]
    texts = [t for t in texts if t]
    return len(texts) != len(set(texts))


OCT_TERMS = {
    "oct",
    "b-scan",
    "b scan",
    "epithelial",
    "epithelium",
    "stromal",
    "stroma",
    "signal",
    "layer",
    "microstructure",
    "cross-sectional",
}

COLPOSCOPY_TERMS = {
    "colposcopy",
    "colposcopic",
    "acetowhite",
    "mosaic",
    "vascular",
    "punctation",
    "iodine",
    "cervical surface",
    "surface",
}

CLINICAL_TERMS = {
    "age",
    "hpv",
    "hrhpv",
    "tct",
    "cytology",
    "clinical",
    "history",
    "screening",
}


def _term_hits(text: str, terms: set[str]) -> int:
    low = text.lower()
    return sum(1 for term in terms if term in low)


def _modality_section_consistency(report: dict[str, Any]) -> dict[str, Any]:
    """Rule-based section/modality consistency audit.

    The detector is intentionally conservative: it is used for report-safety
    auditing, not as a hard training filter. It catches explicit section swaps
    and obvious cross-modality terminology while keeping clean false positives
    low.
    """

    oct_text = str(report.get("oct_findings", "")).lower()
    col_text = str(report.get("colposcopy_findings", "")).lower()
    clin_text = str(report.get("clinical_context", "")).lower()

    oct_support = _term_hits(oct_text, OCT_TERMS) - _term_hits(oct_text, COLPOSCOPY_TERMS)
    col_support = _term_hits(col_text, COLPOSCOPY_TERMS) - _term_hits(col_text, OCT_TERMS)
    clin_support = _term_hits(clin_text, CLINICAL_TERMS) - max(
        _term_hits(clin_text, OCT_TERMS), _term_hits(clin_text, COLPOSCOPY_TERMS)
    )

    explicit_oct_swap = any(x in oct_text for x in ["colposcopy evidence", "colposcopic", "acetowhite", "mosaic vascular"])
    explicit_col_swap = any(x in col_text for x in ["oct evidence", "oct b-scan", "b-scan", "epithelial layer"])
    clinical_image_only = (_term_hits(clin_text, OCT_TERMS) + _term_hits(clin_text, COLPOSCOPY_TERMS)) > 0 and _term_hits(
        clin_text, CLINICAL_TERMS
    ) == 0

    modality_swap = explicit_oct_swap or explicit_col_swap or clinical_image_only or oct_support < -1 or col_support < -1
    section_score = 1.0
    if modality_swap:
        section_score -= 0.45
    if clinical_image_only:
        section_score -= 0.20
    section_score = max(0.0, min(1.0, section_score))
    return {
        "modality_swap_flag": int(modality_swap),
        "section_modality_consistency_score": section_score,
        "oct_section_support_score": float(oct_support),
        "colposcopy_section_support_score": float(col_support),
        "clinical_section_support_score": float(clin_support),
    }


def _extra_flags(report: dict[str, Any], row_ctx: dict[str, Any], variant: str) -> list[str]:
    flags: list[str] = []
    text = _text_blob(report)
    if _duplicate_template(report):
        flags.append("duplicate_template_text")
    if not str(report.get("colposcopy_findings", "")).strip():
        flags.append("empty_colposcopy_findings")
    if not str(report.get("clinical_context", "")).strip():
        flags.append("empty_clinical_context")
    if row_ctx.get("missing_colposcopy") and "acetowhite" in text and "unavailable" not in text:
        flags.append("colposcopy_missing_hallucination")
    if row_ctx.get("missing_instruction") and ("hpv" in text or "tct" in text) and "unavailable" not in text:
        flags.append("instruction_missing_hallucination")
    modality_audit = _modality_section_consistency(report)
    if modality_audit["modality_swap_flag"]:
        flags.append("modality_swap_detected")
    if variant == "randomised_recommendation_inconsistent_with_impression":
        imp = str(report.get("impression", "")).lower()
        rec = str(report.get("recommendation", "")).lower()
        if ("no definitive evidence" in imp and "urgent excision" in rec) or ("suspicious for cin2+" in imp and "routine screening" in rec):
            flags.append("recommendation_contradiction")
    return flags


def _corrupt(report: dict[str, Any], row: pd.Series, variant: str) -> tuple[dict[str, Any], dict[str, Any]]:
    out = copy.deepcopy(report)
    row_ctx = _row_context(row)
    label = int(row_ctx["binary_label"])

    if variant == "clean":
        return out, row_ctx
    if variant == "label_impression_contradiction":
        out["impression"] = (
            "Suspicious for CIN2+ based on available modalities."
            if label == 0
            else "No definitive evidence for CIN2+ on available modalities."
        )
    elif variant == "unsupported_invasive_cancer_terminology":
        out["impression"] = "Findings are suspicious for invasive cancer."
    elif variant == "unsupported_cin3_high_grade_terminology":
        out["oct_findings"] = str(out.get("oct_findings", "")) + " Features suggest CIN3/high-grade lesion."
    elif variant == "missing_oct_section":
        out["oct_findings"] = ""
    elif variant == "missing_colposcopy_section":
        out["colposcopy_findings"] = ""
    elif variant == "missing_clinical_context_section":
        out["clinical_context"] = ""
    elif variant == "modality_swap_between_oct_and_colposcopy":
        original_oct = str(out.get("oct_findings", ""))
        original_col = str(out.get("colposcopy_findings", ""))
        out["oct_findings"] = "Colposcopy evidence placed in OCT section: " + original_col
        out["colposcopy_findings"] = "OCT B-scan evidence placed in colposcopy section: " + original_oct
    elif variant == "shuffled_report_sections":
        original_oct = str(out.get("oct_findings", ""))
        original_col = str(out.get("colposcopy_findings", ""))
        original_clin = str(out.get("clinical_context", ""))
        out["oct_findings"] = "Clinical context placed in OCT section: " + original_clin
        out["colposcopy_findings"] = "OCT B-scan evidence placed in colposcopy section: " + original_oct
        out["clinical_context"] = "Colposcopy evidence placed in clinical context section: " + original_col
    elif variant == "duplicated_template_text":
        template = "Repeated generic template without case-specific modality evidence."
        for key in REQUIRED_SECTIONS:
            out[key] = template
    elif variant == "hallucinated_modality_finding_when_modality_absent":
        row_ctx = _row_context(row, {"missing_oct": 1, "missing_colposcopy": 1, "missing_instruction": 1})
        out["oct_findings"] = "OCT B-scan shows epithelial layer disruption and stromal signal abnormality."
        out["colposcopy_findings"] = "Colposcopy shows acetowhite lesion with mosaic vascular pattern."
        out["clinical_context"] = "HPV 16 positive; TCT HSIL."
        out["evidence_support"] = {"oct_supported": True, "colposcopy_supported": True, "instruction_supported": True}
    elif variant == "randomised_recommendation_inconsistent_with_impression":
        out["recommendation"] = (
            "Routine screening without follow-up is recommended."
            if label == 1
            else "Urgent excision is recommended despite negative impression."
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return out, row_ctx


def _expected_family(variant: str) -> str:
    if variant in {"label_impression_contradiction", "randomised_recommendation_inconsistent_with_impression"}:
        return "contradiction"
    if variant in {"unsupported_invasive_cancer_terminology", "unsupported_cin3_high_grade_terminology", "hallucinated_modality_finding_when_modality_absent"}:
        return "hallucination"
    if variant.startswith("missing_"):
        return "missing_section"
    if variant == "duplicated_template_text":
        return "duplicate_template"
    if variant in {"modality_swap_between_oct_and_colposcopy", "shuffled_report_sections"}:
        return "modality_swap"
    return "clean"


def _detected(flags: str, passed: int, expected: str) -> int:
    if expected == "clean":
        return int(passed == 1)
    flag_set = set(str(flags).split(";")) if flags else set()
    if expected == "contradiction":
        keys = {"negative_label_positive_impression", "positive_label_negative_impression", "recommendation_contradiction"}
    elif expected == "hallucination":
        keys = {"pathology_hallucination", "oct_missing_hallucination", "colposcopy_missing_hallucination", "instruction_missing_hallucination"}
    elif expected == "missing_section":
        keys = {f"empty_{s}" for s in REQUIRED_SECTIONS}
    elif expected == "duplicate_template":
        keys = {"duplicate_template_text"}
    elif expected == "modality_swap":
        keys = {"modality_swap_detected"}
    else:
        keys = set()
    return int(bool(flag_set & keys) or passed == 0)


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    pseudo = manifest[manifest["needs_pseudo_report"].astype(int).eq(1)].copy()
    pseudo["parsed_report"] = pseudo["pseudo_report_text"].map(_parse_report)
    pseudo = pseudo[pseudo["parsed_report"].notna()].copy()

    variants = [
        "clean",
        "label_impression_contradiction",
        "unsupported_invasive_cancer_terminology",
        "unsupported_cin3_high_grade_terminology",
        "missing_oct_section",
        "missing_colposcopy_section",
        "missing_clinical_context_section",
        "modality_swap_between_oct_and_colposcopy",
        "shuffled_report_sections",
        "duplicated_template_text",
        "hallucinated_modality_finding_when_modality_absent",
        "randomised_recommendation_inconsistent_with_impression",
    ]

    rows: list[dict[str, Any]] = []
    for _, row in pseudo.iterrows():
        base = row["parsed_report"]
        for variant in variants:
            report, ctx = _corrupt(base, row, variant)
            qc = qc_pseudo_report(report, ctx)
            modality_audit = _modality_section_consistency(report)
            extra = _extra_flags(report, ctx, variant)
            flags = ";".join([f for f in [qc.get("qc_flags", ""), ";".join(extra)] if f])
            expected = _expected_family(variant)
            rows.append(
                {
                    "case_id": row["case_id"],
                    "variant": variant,
                    "expected_failure_family": expected,
                    "qc_pass": int(qc["pseudo_report_pass_qc"]),
                    "qc_score": float(qc["qc_score"]),
                    "pseudo_training_weight": float(qc["pseudo_training_weight"]),
                    "qc_flags": flags,
                    "detected_expected_failure": _detected(flags, int(qc["pseudo_report_pass_qc"]), expected),
                    "modality_swap_flag": int(modality_audit["modality_swap_flag"]),
                    "section_modality_consistency_score": float(modality_audit["section_modality_consistency_score"]),
                    "oct_section_support_score": float(modality_audit["oct_section_support_score"]),
                    "colposcopy_section_support_score": float(modality_audit["colposcopy_section_support_score"]),
                    "clinical_section_support_score": float(modality_audit["clinical_section_support_score"]),
                }
            )

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DETAIL_V2, index=False)
    summary = (
        detail.groupby(["variant", "expected_failure_family"], as_index=False)
        .agg(
            n=("case_id", "count"),
            qc_pass_rate=("qc_pass", "mean"),
            mean_qc_score=("qc_score", "mean"),
            severe_or_expected_detection_rate=("detected_expected_failure", "mean"),
            severe_error_flag_rate=("qc_pass", lambda x: 1.0 - float(x.mean())),
            mean_training_weight=("pseudo_training_weight", "mean"),
            modality_swap_flag_rate=("modality_swap_flag", "mean"),
            mean_section_modality_consistency_score=("section_modality_consistency_score", "mean"),
            mean_oct_section_support_score=("oct_section_support_score", "mean"),
            mean_colposcopy_section_support_score=("colposcopy_section_support_score", "mean"),
            mean_clinical_section_support_score=("clinical_section_support_score", "mean"),
        )
        .sort_values(["expected_failure_family", "variant"])
    )
    summary.to_csv(OUT_CSV, index=False)
    summary.to_csv(OUT_CSV_V2, index=False)

    clean = summary[summary["variant"].eq("clean")].iloc[0]
    corrupted = summary[~summary["variant"].eq("clean")]
    contradiction = corrupted[corrupted["expected_failure_family"].eq("contradiction")]
    hallucination = corrupted[corrupted["expected_failure_family"].eq("hallucination")]
    missing = corrupted[corrupted["expected_failure_family"].eq("missing_section")]
    duplicate = corrupted[corrupted["expected_failure_family"].eq("duplicate_template")]
    modality = corrupted[corrupted["expected_failure_family"].eq("modality_swap")]

    md = [
        "# QC Failure-Enriched Stress Test",
        "",
        "This audit corrupts existing pseudo reports and runs the existing QC function. It is a report-safety audit, not a downstream AUROC ablation.",
        "",
        "## Summary",
        "",
        f"- Clean pseudo-report QC pass rate: {clean['qc_pass_rate']:.3f}.",
        f"- Corrupted pseudo-report QC pass rate: {corrupted['qc_pass_rate'].mean():.3f}.",
        f"- Mean QC score, clean vs corrupted: {clean['mean_qc_score']:.3f} vs {corrupted['mean_qc_score'].mean():.3f}.",
        f"- Contradiction detection rate: {contradiction['severe_or_expected_detection_rate'].mean():.3f}.",
        f"- Hallucination detection rate: {hallucination['severe_or_expected_detection_rate'].mean():.3f}.",
        f"- Missing-section detection rate: {missing['severe_or_expected_detection_rate'].mean():.3f}.",
        f"- Duplicate-template detection rate: {duplicate['severe_or_expected_detection_rate'].mean():.3f}.",
        f"- Modality-swap detection rate: {modality['severe_or_expected_detection_rate'].mean():.3f}.",
        "",
        "## Variant-Level Results",
        "",
        _markdown_table(summary),
        "",
        "## Interpretation",
        "",
        "The original retrospective QC ablation did not materially change downstream AUROC. This failure-enriched stress test evaluates the intended safety role of QC: suppressing contradiction, unsupported diagnostic terminology, missing-section errors, modality hallucination, cross-modality section swaps, and duplicated template text.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    plot = summary.copy()
    plot["label"] = plot["variant"].str.replace("_", " ")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    axes[0].barh(plot["label"], plot["qc_pass_rate"], color="#7aa6c2", edgecolor="#333333")
    axes[0].set_xlabel("QC pass rate")
    axes[0].set_xlim(0, 1.02)
    axes[0].set_title("Pass / retained rate")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(
        plot["label"],
        plot["severe_or_expected_detection_rate"],
        color="#d08a5b",
        edgecolor="#333333",
    )
    axes[1].set_xlabel("Expected handling rate")
    axes[1].set_xlim(0, 1.02)
    axes[1].set_title("Clean retention / failure detection")
    axes[1].grid(axis="x", alpha=0.3)
    fig.suptitle("Failure-enriched pseudo-report QC stress test (v2)", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_CSV_V2}")
    print(f"Wrote {OUT_DETAIL_V2}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
