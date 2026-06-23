#!/usr/bin/env python3
"""LLM semantic tag extraction with strict JSON output.

If no LLM provider is configured, this script writes explicit unavailable
artifacts and exits successfully. It never fabricates LLM tags from the rule
fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

import pandas as pd

from llm_semantic_common import OUT_DIR, TAG_COLUMNS, compact_text, empty_csv, ensure_out_dir, join_tags


INPUT = OUT_DIR / "semantic_tagging_input.csv"
CSV_OUT = OUT_DIR / "llm_semantic_tags.csv"
JSONL_OUT = OUT_DIR / "llm_semantic_tags.jsonl"

SCHEMA_KEYS = {
    "oct_tags",
    "colposcopy_tags",
    "clinical_tags",
    "impression_tags",
    "severity_tags",
    "modality_evidence",
    "missing_section_flags",
    "contradiction_flag",
    "support_score",
}


SYSTEM_PROMPT = """You extract evidence-grounded cervical screening semantic tags.
Return strict JSON only. Do not diagnose. Do not infer labels. Do not use CIN,
HSIL, LSIL, invasive cancer, pathology, or histology terms as tags. Use only
the supplied de-identified safe_text evidence. Required keys:
oct_tags, colposcopy_tags, clinical_tags, impression_tags, severity_tags,
modality_evidence, missing_section_flags, contradiction_flag, support_score.
All tag fields are arrays of short lowercase snake_case strings.
contradiction_flag is 0 or 1. support_score is a number in [0, 1].
"""


def _provider_available(provider: str) -> tuple[bool, str]:
    if provider == "none":
        return False, "provider set to none"
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY absent"
        return True, "openai"
    if provider == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return True, "openai"
        return False, "no supported LLM API key found"
    return False, f"unsupported provider {provider}"


def _extract_json(text: str) -> dict[str, Any]:
    raw = compact_text(text)
    try:
        obj = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM response was not a JSON object")
    missing = SCHEMA_KEYS - set(obj)
    if missing:
        raise ValueError(f"Missing keys: {sorted(missing)}")
    return obj


def _call_openai(safe_text: str, *, model: str, timeout: int = 60) -> str:
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "safe_text:\n" + safe_text[:6000]},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=data,
        headers={
            "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        parsed = json.loads(resp.read().decode("utf-8"))
    return parsed["choices"][0]["message"]["content"]


def _normalise_row(row: pd.Series, obj: dict[str, Any], raw_text: str, parse_error: int = 0) -> dict[str, Any]:
    def list_field(key: str) -> str:
        value = obj.get(key, [])
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            value = []
        return join_tags(value)

    contradiction = int(bool(obj.get("contradiction_flag", 0)))
    try:
        support = max(0.0, min(1.0, float(obj.get("support_score", 0.0))))
    except Exception:
        support = 0.0
    tag_text = join_tags(
        []
        + list_field("oct_tags").split("|")
        + list_field("colposcopy_tags").split("|")
        + list_field("clinical_tags").split("|")
        + list_field("impression_tags").split("|")
        + list_field("severity_tags").split("|")
        + list_field("modality_evidence").split("|")
        + list_field("missing_section_flags").split("|")
    )
    return {
        "case_id": row["case_id"],
        "split": row["split"],
        "center_id": row["center_id"],
        "source": "llm",
        "valid_json": int(parse_error == 0),
        "parse_error": parse_error,
        "oct_tags": list_field("oct_tags"),
        "colposcopy_tags": list_field("colposcopy_tags"),
        "clinical_tags": list_field("clinical_tags"),
        "impression_tags": list_field("impression_tags"),
        "severity_tags": list_field("severity_tags"),
        "modality_evidence": list_field("modality_evidence"),
        "missing_section_flags": list_field("missing_section_flags"),
        "contradiction_flag": contradiction,
        "support_score": round(support, 4),
        "tag_text": tag_text,
        "raw_json": raw_text,
    }


def _write_unavailable(reason: str) -> None:
    empty_csv(CSV_OUT, TAG_COLUMNS)
    JSONL_OUT.write_text("", encoding="utf-8")
    lines = [
        "# LLM Semantic Tags Not Available",
        "",
        f"- Reason: {reason}.",
        "- No LLM semantic tags were fabricated from deterministic rule outputs.",
        "- Downstream LLM-specific retrieval and fusion files are therefore written as empty schema-only artifacts unless a valid LLM tag table is later generated.",
    ]
    (OUT_DIR / "LLM_NOT_AVAILABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"LLM unavailable: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=os.environ.get("LLM_SEMANTIC_PROVIDER", "auto"))
    parser.add_argument("--model", default=os.environ.get("LLM_SEMANTIC_MODEL", "gpt-4o-mini"))
    parser.add_argument("--limit", type=int, default=0, help="0 means all cases")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    ensure_out_dir()
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing semantic input: {INPUT}")
    available, status = _provider_available(args.provider)
    if not available:
        _write_unavailable(status)
        return
    provider = status
    df = pd.read_csv(INPUT)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    rows: list[dict[str, Any]] = []
    with JSONL_OUT.open("w", encoding="utf-8") as jf:
        for _, row in df.iterrows():
            raw_response = ""
            obj: dict[str, Any] | None = None
            parse_error = 1
            last_error = ""
            for attempt in range(args.max_retries + 1):
                try:
                    if provider == "openai":
                        raw_response = _call_openai(compact_text(row["safe_text"]), model=args.model)
                    else:
                        raise RuntimeError(f"Unsupported provider {provider}")
                    obj = _extract_json(raw_response)
                    parse_error = 0
                    break
                except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError, KeyError) as exc:
                    last_error = repr(exc)
                    time.sleep(min(2.0, 0.5 * (attempt + 1)))
            if obj is None:
                obj = {
                    "oct_tags": [],
                    "colposcopy_tags": [],
                    "clinical_tags": [],
                    "impression_tags": [],
                    "severity_tags": [],
                    "modality_evidence": [],
                    "missing_section_flags": ["llm_parse_or_call_failed"],
                    "contradiction_flag": 0,
                    "support_score": 0,
                }
                raw_response = json.dumps({"error": last_error}, ensure_ascii=False)
            record = _normalise_row(row, obj, raw_response, parse_error=parse_error)
            rows.append(record)
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.sleep:
                time.sleep(args.sleep)

    out = pd.DataFrame(rows, columns=TAG_COLUMNS)
    out.to_csv(CSV_OUT, index=False)
    valid = int(out["valid_json"].sum()) if not out.empty else 0
    parse_errors = int(out["parse_error"].sum()) if not out.empty else 0
    (OUT_DIR / "LLM_TAGGING_RUN_SUMMARY.md").write_text(
        "\n".join(
            [
                "# LLM Semantic Tagging Run Summary",
                "",
                f"- Provider: {provider}",
                f"- Model: {args.model}",
                f"- Cases attempted: {len(out)}",
                f"- Valid JSON rows: {valid}",
                f"- Parse/call error rows: {parse_errors}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {CSV_OUT}")
    print(f"Wrote {JSONL_OUT}")


if __name__ == "__main__":
    main()
