#!/usr/bin/env python3
"""Build leakage-controlled inputs for LLM/rule semantic tagging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from llm_semantic_common import (
    MANIFEST_CANDIDATES,
    OUT_DIR,
    RASA_SCORE_CANDIDATES,
    compact_text,
    contains_forbidden_outcome,
    contains_identifier_like_text,
    ensure_out_dir,
    first_existing,
    markdown_table,
    parse_json_dict,
    redact_outcome_and_ids,
    read_first_existing,
)


SECTION_KEYS = [
    "diagnostic_summary",
    "oct_findings",
    "colposcopy_findings",
    "clinical_context",
    "impression",
    "recommendation",
]


def _boolish(value: Any) -> int:
    text = compact_text(value).lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n", ""}:
        return 0
    try:
        return int(float(text))
    except Exception:
        return 0


def _row_text(row: pd.Series) -> tuple[str, str, str]:
    split = compact_text(row.get("split")).lower()
    parts: list[str] = []
    policy: list[str] = []

    clinical_parts = []
    for col in ["age", "hpv", "tct", "other_clinical_attributes", "reference_clinical_context"]:
        value = compact_text(row.get(col))
        if value:
            clinical_parts.append(f"{col}: {value}")
    if clinical_parts:
        parts.append("clinical_evidence: " + " ".join(clinical_parts))
        policy.append("structured_clinical_fields")

    if split == "train":
        pseudo = parse_json_dict(row.get("pseudo_report_text"))
        if pseudo:
            section_text = []
            for key in SECTION_KEYS:
                value = compact_text(pseudo.get(key))
                if value:
                    section_text.append(f"{key}: {value}")
            if section_text:
                parts.append("training_pseudo_sections: " + " ".join(section_text))
                policy.append("train_pseudo_report_sections")
        real_text = compact_text(row.get("real_report_text")) or compact_text(row.get("training_report_text"))
        if real_text:
            parts.append("training_real_or_normalized_report: " + real_text)
            policy.append("train_report_text")
    else:
        # Validation/test inputs exclude pseudo reports because many pseudo
        # sections were generated under weak-label constraints. Use only
        # structured evidence and archived/normalized text after redaction.
        normalized = compact_text(row.get("training_report_text"))
        reference = compact_text(row.get("reference_report_text"))
        real_text = compact_text(row.get("real_report_text")) if _boolish(row.get("has_real_report")) else ""
        for label, value in [
            ("normalized_archived_or_structured_text", normalized),
            ("reference_report_text", reference),
            ("archived_report_text", real_text),
        ]:
            if value:
                parts.append(f"{label}: {value}")
                policy.append(f"eval_{label}_redacted")

    availability = []
    for col in ["missing_oct", "missing_colposcopy", "missing_instruction", "missing_report"]:
        availability.append(f"{col}: {_boolish(row.get(col))}")
    parts.append("availability: " + " ".join(availability))
    policy.append("modality_availability_flags")

    raw_text = " ".join(parts)
    safe_text = redact_outcome_and_ids(raw_text)
    if len(safe_text) > 6000:
        safe_text = safe_text[:6000].rsplit(" ", 1)[0]
    return safe_text, ";".join(policy), raw_text


def _merge_scores(df: pd.DataFrame) -> pd.DataFrame:
    score_path = first_existing(RASA_SCORE_CANDIDATES)
    if score_path is None:
        return df
    scores = pd.read_csv(score_path)
    keep = [c for c in ["case_id", "risk_score", "semantic_retrieval_score", "semantic_fusion_score"] if c in scores.columns]
    if "case_id" not in keep:
        return df
    scores = scores[keep].drop_duplicates("case_id")
    scores = scores.rename(
        columns={
            "risk_score": "existing_rasa_score",
            "semantic_retrieval_score": "existing_semantic_retrieval_score",
            "semantic_fusion_score": "existing_mosaic_score",
        }
    )
    return df.merge(scores, on="case_id", how="left")


def main() -> None:
    out_dir = ensure_out_dir()
    manifest_path = first_existing(MANIFEST_CANDIDATES)
    manifest = read_first_existing(MANIFEST_CANDIDATES, required_name="full case manifest")

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        safe_text, policy, raw_text = _row_text(row)
        case_id = compact_text(row.get("case_id"))
        patient_id = compact_text(row.get("patient_id"))
        split = compact_text(row.get("split")).lower()
        report_source = "real" if _boolish(row.get("has_real_report")) else "pseudo_candidate"
        if not _boolish(row.get("has_real_report")) and not _boolish(row.get("has_pseudo_report")):
            report_source = "none"

        has_case_id = bool(case_id and case_id in safe_text)
        has_patient_id = bool(patient_id and patient_id in safe_text)
        has_identifier = contains_identifier_like_text(safe_text)
        has_forbidden = contains_forbidden_outcome(safe_text)
        leakage_flag = int((split in {"val", "test"} and (has_forbidden or has_identifier or has_case_id or has_patient_id)))

        rows.append(
            {
                "case_id": case_id,
                "split": split,
                "center_id": compact_text(row.get("center_id")),
                "report_source": report_source,
                "has_real_report": _boolish(row.get("has_real_report")),
                "has_pseudo_report": _boolish(row.get("has_pseudo_report")),
                "safe_text": safe_text,
                "safe_text_char_count": len(safe_text),
                "safe_text_policy": policy,
                "y_true": int(float(row.get("binary_label", 0))),
                "pseudo_report_qc_score": row.get("qc_score", ""),
                "pseudo_report_confidence": row.get("pseudo_report_confidence", ""),
            }
        )
        audit_rows.append(
            {
                "case_id": case_id,
                "split": split,
                "safe_text_policy": policy,
                "has_case_id_in_safe_text": int(has_case_id),
                "has_patient_id_in_safe_text": int(has_patient_id),
                "has_path_or_identifier_like_text": int(has_identifier),
                "has_forbidden_diagnostic_terms": int(has_forbidden),
                "leakage_flag": leakage_flag,
                "raw_text_used_before_redaction": int(bool(compact_text(raw_text))),
            }
        )

    out = pd.DataFrame(rows)
    out = _merge_scores(out)
    audit = pd.DataFrame(audit_rows)

    out.to_csv(out_dir / "semantic_tagging_input.csv", index=False)
    audit.to_csv(out_dir / "semantic_tagging_leakage_safety_audit.csv", index=False)

    split_summary = out.groupby("split", as_index=False).agg(
        n=("case_id", "count"),
        mean_safe_text_chars=("safe_text_char_count", "mean"),
        real_reports=("has_real_report", "sum"),
        pseudo_report_cases=("has_pseudo_report", "sum"),
    )
    leakage_summary = audit.groupby("split", as_index=False).agg(
        n=("case_id", "count"),
        leakage_flags=("leakage_flag", "sum"),
        forbidden_term_flags=("has_forbidden_diagnostic_terms", "sum"),
        identifier_flags=("has_path_or_identifier_like_text", "sum"),
    )

    md = [
        "# Semantic Tagging Input and Leakage Safety Audit",
        "",
        "The semantic-tag input table was built from the locked manifest with case-level train/validation/test splits. Outcome and pathology-like terms were redacted from `safe_text`, and validation/test pseudo-report label phrases were excluded.",
        "",
        f"- Manifest source: `{manifest_path}`" if manifest_path else "- Manifest source: missing",
        f"- Cases written: {len(out)}",
        "- Validation/test labels are retained only in the `y_true` column for final evaluation; they are not embedded in `safe_text`.",
        "- The retrieval bank is constructed downstream from training cases only.",
        "",
        "## Split Summary",
        "",
        markdown_table(split_summary),
        "",
        "## Leakage Audit Summary",
        "",
        markdown_table(leakage_summary),
        "",
        "## Safety Rule",
        "",
        "A nonzero validation/test leakage flag means a case should not be used for LLM/rule tag extraction until the offending text source is removed or redacted further.",
    ]
    (out_dir / "semantic_tagging_leakage_safety_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'semantic_tagging_input.csv'}")
    print(f"Wrote {out_dir / 'semantic_tagging_leakage_safety_audit.csv'}")
    print(f"Wrote {out_dir / 'semantic_tagging_leakage_safety_audit.md'}")


if __name__ == "__main__":
    main()
