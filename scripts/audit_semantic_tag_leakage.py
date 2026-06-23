#!/usr/bin/env python3
"""Audit semantic-tag inputs and outputs for target or identifier leakage."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEM_DIR = ROOT / "outputs" / "llm_semantic"
REV_DIR = ROOT / "outputs" / "revision"

INPUTS = {
    "semantic_tagging_input": SEM_DIR / "semantic_tagging_input.csv",
    "rule_semantic_tags": SEM_DIR / "rule_semantic_tags.csv",
    "rule_tag_retrieval_predictions": SEM_DIR / "rule_tag_retrieval_predictions.csv",
}

TEXT_COLUMNS = [
    "safe_text",
    "tag_text",
    "normalised_semantic_summary",
    "normalized_semantic_summary",
    "oct_tags",
    "colposcopy_tags",
    "clinical_tags",
    "impression_tags",
    "severity_tags",
    "modality_evidence",
    "missing_section_flags",
    "raw_json",
]

FORBIDDEN_PATTERNS = {
    "target_or_label_token": r"\b(y_true|label|target|outcome)\b",
    "pathology_terms": r"\b(pathology|histology|biopsy result|final diagnosis)\b",
    "cin_terms": r"\b(CIN\s*2|CIN\s*3|CIN2\+|CIN3\+|CIN\s*2\s*\+|CIN\s*3\s*\+)\b",
    "diagnostic_terms": r"\b(HSIL|carcinoma|invasive cancer|cancer confirmed|malignant|high-grade lesion|low-grade lesion|AIS|SCC|adenocarcinoma)\b",
    "training_label_phrase": r"\b(positive training label|negative training label)\b",
}

PATH_PATTERNS = {
    "slash_path": r"(^|\s)(/[^ \t\r\n]+|[A-Za-z]:\\[^ \t\r\n]+)",
    "image_or_record_ext": r"\.(jpg|jpeg|png|dcm|nii|xml|ini|pdf)\b",
    "backslash_path": r"\\[^ \t\r\n]+\\",
}

ID_PATTERNS = {
    "case_id_like": r"\bM\d{3,8}_\d{4}_P\d{5,10}\b",
    "patient_id_like": r"\b(patient_id|case_id|subject_id|身份证|住院号|病历号)\b",
    "hospital_identifier": r"\b(Renmin Hospital|Enshi Central Hospital|Xiangyang Central Hospital|Shiyan People's Hospital|Jingzhou First People's Hospital)\b",
}


def compact(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def redact(text: str) -> str:
    out = text
    for patterns in (FORBIDDEN_PATTERNS, PATH_PATTERNS, ID_PATTERNS):
        for pat in patterns.values():
            out = re.sub(pat, "[REDACTED]", out, flags=re.IGNORECASE)
    return out[:220]


def scan_row(row: pd.Series) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    for col in TEXT_COLUMNS:
        if col not in row.index:
            continue
        text = compact(row.get(col))
        if not text:
            continue
        for name, pat in FORBIDDEN_PATTERNS.items():
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append({"issue": "forbidden_term", "pattern": name, "column": col, "snippet": redact(text)})
        for name, pat in PATH_PATTERNS.items():
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append({"issue": "path_like_token", "pattern": name, "column": col, "snippet": redact(text)})
        for name, pat in ID_PATTERNS.items():
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append({"issue": "identifier_like_token", "pattern": name, "column": col, "snippet": redact(text)})
    return hits


def load_inputs() -> pd.DataFrame:
    if not INPUTS["semantic_tagging_input"].exists():
        raise FileNotFoundError(INPUTS["semantic_tagging_input"])
    base = pd.read_csv(INPUTS["semantic_tagging_input"])
    base_for_cleaning = base.copy()
    keep = ["case_id", "split", "center_id", "safe_text"]
    merged = base[[c for c in keep if c in base.columns]].copy()
    if INPUTS["rule_semantic_tags"].exists():
        tags = pd.read_csv(INPUTS["rule_semantic_tags"])
        tag_cols = ["case_id", "split", "center_id"] + [c for c in TEXT_COLUMNS if c in tags.columns]
        merged = merged.merge(tags[tag_cols], on=["case_id", "split", "center_id"], how="left", suffixes=("", "_rule"))
    return merged, base_for_cleaning


def main() -> None:
    REV_DIR.mkdir(parents=True, exist_ok=True)
    df, base_for_cleaning = load_inputs()
    examples: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    cleaned = df.copy()

    for split, sub in df.groupby("split", dropna=False):
        split_hits: dict[str, set[str]] = {
            "forbidden_term": set(),
            "path_like_token": set(),
            "identifier_like_token": set(),
        }
        for _, row in sub.iterrows():
            row_hits = scan_row(row)
            case_id = str(row.get("case_id", ""))
            for hit in row_hits:
                split_hits[str(hit["issue"])].add(case_id)
                if len(examples) < 20:
                    examples.append({"case_id": case_id, "split": split, **hit})
        summary_rows.append(
            {
                "split": split,
                "cases_scanned": int(sub["case_id"].nunique()),
                "cases_with_forbidden_terms": len(split_hits["forbidden_term"]),
                "cases_with_path_like_tokens": len(split_hits["path_like_token"]),
                "cases_with_identifiers": len(split_hits["identifier_like_token"]),
                "passes_leakage_audit": int(not any(split_hits.values())),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("split")
    summary.to_csv(REV_DIR / "semantic_tag_leakage_audit.csv", index=False)
    examples_df = pd.DataFrame(examples)
    if not examples_df.empty:
        examples_df.to_csv(REV_DIR / "semantic_tag_leakage_examples_redacted.csv", index=False)

    passed = bool(summary["passes_leakage_audit"].all())
    lines = [
        "# Semantic-Tag Leakage Audit",
        "",
        "Text fields scanned: `safe_text`, `tag_text`, normalized summary columns when present, all tag columns, `modality_evidence`, `missing_section_flags`, and `raw_json`.",
        "",
        "| split | cases scanned | forbidden-term cases | path-like cases | identifier cases | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['split']} | {row['cases_scanned']} | {row['cases_with_forbidden_terms']} | "
            f"{row['cases_with_path_like_tokens']} | {row['cases_with_identifiers']} | {row['passes_leakage_audit']} |"
        )
    lines.extend(["", f"Overall pass: {'yes' if passed else 'no'}."])
    if examples:
        lines.extend(["", "## Redacted Examples", ""])
        for ex in examples[:10]:
            lines.append(
                f"- `{ex['split']}` `{ex['case_id']}` `{ex['issue']}` in `{ex['column']}` "
                f"({ex['pattern']}): {ex['snippet']}"
            )
    else:
        lines.extend(["", "No offending rows were detected."])
    (REV_DIR / "semantic_tag_leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not passed:
        cleaned_input = base_for_cleaning.copy()
        if "safe_text" in cleaned_input.columns:
            cleaned_input["safe_text"] = cleaned_input["safe_text"].map(lambda x: redact(compact(x)))
            cleaned_input["safe_text_char_count"] = cleaned_input["safe_text"].map(len)
            cleaned_input["safe_text_policy"] = cleaned_input.get("safe_text_policy", "").astype(str) + ";strict_revision_redaction"
        cleaned_input.to_csv(REV_DIR / "semantic_tagging_input_cleaned_required.csv", index=False)
        (REV_DIR / "INVALIDATED_RESULTS.md").write_text(
            "Semantic-tag leakage was detected. Existing semantic-tag retrieval and fusion results must be regenerated from cleaned inputs before being used as evidence.\n",
            encoding="utf-8",
        )
    print(f"Wrote {REV_DIR / 'semantic_tag_leakage_audit.csv'}")
    print(f"Wrote {REV_DIR / 'semantic_tag_leakage_audit.md'}")


if __name__ == "__main__":
    main()
