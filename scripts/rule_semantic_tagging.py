#!/usr/bin/env python3
"""Deterministic rule-based semantic tagging fallback."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from llm_semantic_common import OUT_DIR, TAG_COLUMNS, compact_text, ensure_out_dir, join_tags


INPUT = OUT_DIR / "semantic_tagging_input.csv"
OUTPUT = OUT_DIR / "rule_semantic_tags.csv"


def _has(text: str, *patterns: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _tags_for_row(row: pd.Series) -> dict[str, Any]:
    text = compact_text(row.get("safe_text")).lower()

    oct_tags = []
    col_tags = []
    clinical_tags = []
    impression_tags = []
    severity_tags = []
    evidence_tags = []
    missing_flags = []

    if _has(text, r"\boct\b", r"b[- ]?scan", r"epithelial", r"stroma", r"stromal", r"layer", r"oct_emb"):
        oct_tags.append("oct_evidence_present")
        evidence_tags.append("oct")
    if _has(text, r"oct evidence unavailable", r"missing_oct:\s*1"):
        oct_tags.append("oct_unavailable")
        missing_flags.append("missing_oct")
    if _has(text, r"epithelial", r"layer disruption", r"layer_disruption"):
        oct_tags.append("epithelial_layer_signal")
    if _has(text, r"stroma", r"stromal"):
        oct_tags.append("stromal_signal")
    if _has(text, r"oct_read", r"oct_emb_norm", r"embedding"):
        oct_tags.append("oct_embedding_signature")

    if _has(text, r"colposcop", r"acetowhite", r"iodine", r"mosaic", r"punctation", r"vascular", r"transformation zone", r"colposcopy_emb"):
        col_tags.append("colposcopy_evidence_present")
        evidence_tags.append("colposcopy")
    if _has(text, r"colposcopy evidence unavailable", r"missing_colposcopy:\s*1"):
        col_tags.append("colposcopy_unavailable")
        missing_flags.append("missing_colposcopy")
    if _has(text, r"acetowhite", r"aceto"):
        col_tags.append("acetowhite_pattern")
    if _has(text, r"mosaic"):
        col_tags.append("mosaic_pattern")
    if _has(text, r"punctation"):
        col_tags.append("punctation_pattern")
    if _has(text, r"vascular", r"vessel"):
        col_tags.append("vascular_pattern")
    if _has(text, r"iodine", r"schiller"):
        col_tags.append("iodine_staining_context")
    if _has(text, r"col_read", r"colposcopy_emb_norm"):
        col_tags.append("colposcopy_embedding_signature")

    if _has(text, r"\bage\b", r"\bhpv\b", r"\btct\b", r"cytology", r"ascus", r"asc-us", r"nilm", r"clinical_context"):
        clinical_tags.append("clinical_context_present")
        evidence_tags.append("clinical")
    if _has(text, r"missing_instruction:\s*1"):
        missing_flags.append("missing_clinical_instruction")
    if _has(text, r"\bhpv\s*[:= ]\s*(16|18|31|33|35|39|45|51|52|53|56|58|59|66|68|73|81|82)\b"):
        clinical_tags.append("hrhpv_genotype_present")
    elif _has(text, r"\bhpv\b"):
        clinical_tags.append("hpv_field_present")
    if _has(text, r"\btct\b", r"cytology", r"ascus", r"asc-us", r"nilm"):
        clinical_tags.append("cytology_field_present")
    if _has(text, r"asc-us", r"ascus"):
        clinical_tags.append("cytology_ascus")
    if _has(text, r"nilm"):
        clinical_tags.append("cytology_nilm")

    if _has(text, r"routine screening", r"routine follow", r"no definitive evidence"):
        impression_tags.append("low_suspicion_language")
    if _has(text, r"follow[- ]?up", r"correlate", r"review", r"recommendation"):
        impression_tags.append("management_context")
    if _has(text, r"suspicious", r"abnormal", r"urgent"):
        impression_tags.append("abnormal_or_suspicious_language")
    if _has(text, r"negative impression", r"positive impression", r"inconsistent", r"contradiction"):
        impression_tags.append("explicit_inconsistency_language")

    if _has(text, r"abnormal", r"suspicious", r"urgent"):
        severity_tags.append("elevated_attention_language")
    if _has(text, r"routine", r"no definitive", r"negative"):
        severity_tags.append("low_attention_language")

    contradiction = int(
        ("elevated_attention_language" in severity_tags and "low_attention_language" in severity_tags)
        or "explicit_inconsistency_language" in impression_tags
    )

    section_count = sum(
        bool(x)
        for x in [
            set(oct_tags) - {"oct_unavailable"},
            set(col_tags) - {"colposcopy_unavailable"},
            set(clinical_tags),
            set(impression_tags),
        ]
    )
    support_score = min(1.0, section_count / 4.0 + 0.05 * len(set(evidence_tags)))
    if contradiction:
        support_score = max(0.0, support_score - 0.25)

    raw = {
        "oct_tags": oct_tags,
        "colposcopy_tags": col_tags,
        "clinical_tags": clinical_tags,
        "impression_tags": impression_tags,
        "severity_tags": severity_tags,
        "modality_evidence": sorted(set(evidence_tags)),
        "missing_section_flags": sorted(set(missing_flags)),
        "contradiction_flag": contradiction,
        "support_score": round(float(support_score), 4),
    }
    tag_text = join_tags(
        oct_tags
        + col_tags
        + clinical_tags
        + impression_tags
        + severity_tags
        + evidence_tags
        + missing_flags
    )
    return {
        "case_id": row["case_id"],
        "split": row["split"],
        "center_id": row["center_id"],
        "source": "rule",
        "valid_json": 1,
        "parse_error": 0,
        "oct_tags": join_tags(oct_tags),
        "colposcopy_tags": join_tags(col_tags),
        "clinical_tags": join_tags(clinical_tags),
        "impression_tags": join_tags(impression_tags),
        "severity_tags": join_tags(severity_tags),
        "modality_evidence": join_tags(sorted(set(evidence_tags))),
        "missing_section_flags": join_tags(sorted(set(missing_flags))),
        "contradiction_flag": contradiction,
        "support_score": round(float(support_score), 4),
        "tag_text": tag_text,
        "raw_json": json.dumps(raw, ensure_ascii=False, sort_keys=True),
    }


def main() -> None:
    ensure_out_dir()
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing semantic input: {INPUT}")
    df = pd.read_csv(INPUT)
    rows = [_tags_for_row(row) for _, row in df.iterrows()]
    out = pd.DataFrame(rows, columns=TAG_COLUMNS)
    out.to_csv(OUTPUT, index=False)
    summary = out.groupby("split", as_index=False).agg(
        n=("case_id", "count"),
        contradiction_flag_rate=("contradiction_flag", "mean"),
        mean_support_score=("support_score", "mean"),
    )
    summary.to_csv(OUT_DIR / "rule_semantic_tag_summary.csv", index=False)
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {OUT_DIR / 'rule_semantic_tag_summary.csv'}")


if __name__ == "__main__":
    main()
