#!/usr/bin/env python3
"""Shared helpers for the LLM semantic-tag upgrade audit.

The helpers intentionally avoid scikit-learn so the pipeline can run in the
current lightweight environment.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "llm_semantic"

MANIFEST_CANDIDATES = [
    ROOT / "cervix_lcad_rasa" / "outputs" / "publishable" / "manifests" / "full_manifest_publishable_with_llm_pseudo.csv",
    ROOT / "cervix_lcad_rasa" / "outputs" / "publishable" / "manifests" / "full_manifest_publishable.csv",
    ROOT / "cervix_lcad_rasa" / "outputs" / "manifests" / "full_manifest_with_pseudo_reports.csv",
    ROOT / "cervix_lcad_rasa" / "outputs" / "manifests" / "full_manifest.csv",
]

RASA_SCORE_CANDIDATES = [
    ROOT
    / "cervix_lcad_rasa"
    / "outputs"
    / "publishable"
    / "kra_semantic_fusion_analysis"
    / "kra_semantic_fusion_val_test_scores.csv",
    ROOT
    / "cervix_lcad_rasa"
    / "outputs"
    / "publishable"
    / "kra_rasa_stablehash_analysis"
    / "full_lcad_rasa_val_test_scores.csv",
]

EXTERNAL_BASELINE_TABLE = (
    ROOT
    / "cervix_lcad_rasa"
    / "outputs"
    / "publishable"
    / "external_baselines"
    / "tables"
    / "T_external_baselines_same_split.csv"
)

FULL_MOSAIC_BOOTSTRAP = ROOT / "outputs" / "statistics" / "full_mosaic_vs_external_baselines_paired_bootstrap.csv"

TAG_COLUMNS = [
    "case_id",
    "split",
    "center_id",
    "source",
    "valid_json",
    "parse_error",
    "oct_tags",
    "colposcopy_tags",
    "clinical_tags",
    "impression_tags",
    "severity_tags",
    "modality_evidence",
    "missing_section_flags",
    "contradiction_flag",
    "support_score",
    "tag_text",
    "raw_json",
]

RETRIEVAL_COLUMNS = [
    "case_id",
    "split",
    "center_id",
    "source",
    "topk",
    "retrieval_prior",
    "mean_similarity",
    "max_similarity",
    "retrieved_positive_fraction",
    "retrieval_support_weight",
    "support_score",
    "contradiction_flag",
    "y_true",
    "top_train_case_ids",
]

ABLATION_COLUMNS = [
    "row_id",
    "semantic_source",
    "uses_llm",
    "train_only_bank",
    "validation_calibrated_fusion",
    "available",
    "auroc",
    "auprc",
    "f1",
    "sensitivity",
    "specificity",
    "precision",
    "balanced_accuracy",
    "alpha",
    "threshold",
    "contradiction_flag_rate",
    "mean_support_score",
    "note",
]


FORBIDDEN_PATTERNS = [
    r"\by_true\b",
    r"\bbinary_label\b",
    r"\blabel_phrase\b",
    r"\bweak supervision\b",
    r"\blabel[-_ ]?constrained\b",
    r"\bcin\s*0\s*[-+/ ]?\s*1\b",
    r"\bcin\s*1\b",
    r"\bcin\s*2\b",
    r"\bcin\s*3\b",
    r"\bcin\s*2\s*\+\b",
    r"\bcin\s*3\s*\+\b",
    r"\bcin2\+\b",
    r"\bcin3\+\b",
    r"\bhsil\b",
    r"\blsil\b",
    r"\bais\b",
    r"\bscc\b",
    r"\binvasive cancer\b",
    r"\bcervical cancer\b",
    r"\bcarcinoma\b",
    r"\bhistopathology\b",
    r"\bhistology\b",
    r"\bpathology\b",
    r"\bpathologic\b",
    r"\bpositive label\b",
    r"\bnegative label\b",
    r"\bdiagnostic label\b",
    # Chinese diagnostic/pathology terms, kept as escapes to preserve ASCII source.
    r"\u75c5\u7406",
    r"\u6d78\u6da6\u764c",
    r"\u9cde\u764c",
    r"\u817a\u764c",
    r"\u539f\u4f4d\u764c",
    r"\u9ad8\u7ea7",
    r"\u4f4e\u7ea7",
    r"\u764c",
]

ID_LIKE_PATTERNS = [
    r"M\d{3,8}_\d{4}_P\d{5,10}",
    r"\bP\d{5,}\b",
    r"/[A-Za-z0-9_.\-/]+",
    r"[A-Za-z]:\\[A-Za-z0-9_.\\-]+",
    r"\.(?:jpg|jpeg|png|bmp|pdf|xml|ini|dcm)\b",
]


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def read_first_existing(paths: Iterable[Path], *, required_name: str) -> pd.DataFrame:
    path = first_existing(paths)
    if path is None:
        write_missing_inputs(
            [
                {
                    "name": required_name,
                    "expected_schema": "case_id, split, center_id, binary_label, safe evidence/report fields",
                    "candidate_paths": "; ".join(str(p) for p in paths),
                }
            ]
        )
        raise FileNotFoundError(f"Missing required input: {required_name}")
    return pd.read_csv(path)


def write_missing_inputs(rows: list[dict[str, Any]]) -> None:
    ensure_out_dir()
    lines = [
        "# Missing Inputs",
        "",
        "| name | expected_schema | candidate_paths |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {expected_schema} | {candidate_paths} |".format(
                name=str(row.get("name", "")),
                expected_schema=str(row.get("expected_schema", "")).replace("|", "/"),
                candidate_paths=str(row.get("candidate_paths", "")).replace("|", "/"),
            )
        )
    (OUT_DIR / "MISSING_INPUTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_json_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def compact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def redact_outcome_and_ids(text: str) -> str:
    out = compact_text(text)
    for pat in FORBIDDEN_PATTERNS:
        out = re.sub(pat, "[REDACTED_DIAGNOSTIC_TERM]", out, flags=re.IGNORECASE)
    for pat in ID_LIKE_PATTERNS:
        out = re.sub(pat, "[REDACTED_IDENTIFIER]", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def contains_forbidden_outcome(text: str) -> bool:
    blob = compact_text(text)
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, blob, flags=re.IGNORECASE):
            return True
    return False


def contains_identifier_like_text(text: str) -> bool:
    blob = compact_text(text)
    for pat in ID_LIKE_PATTERNS:
        if re.search(pat, blob, flags=re.IGNORECASE):
            return True
    return False


def join_tags(values: Iterable[Any]) -> str:
    cleaned = []
    for value in values:
        item = compact_text(value).lower().replace(" ", "_")
        item = re.sub(r"[^a-z0-9_:+.-]+", "_", item).strip("_")
        if item and item not in cleaned:
            cleaned.append(item)
    return "|".join(cleaned)


def split_tags(value: Any) -> list[str]:
    text = compact_text(value)
    if not text:
        return []
    out = []
    for part in re.split(r"[|;,]", text):
        tag = compact_text(part).lower()
        if tag and tag not in out:
            out.append(tag)
    return out


def tokenize_tag_text(text: str) -> list[str]:
    blob = compact_text(text).lower()
    words = re.findall(r"[a-z0-9_:+.-]{2,}", blob)
    return words


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{digits}f}")
        else:
            view[col] = view[col].astype(str)
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for row in view.to_numpy():
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |")
    return "\n".join(lines)


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1)
    return float((precision * y_sorted).sum() / n_pos)


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = sensitivity if math.isfinite(sensitivity) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced = (sensitivity + specificity) / 2.0 if math.isfinite(sensitivity) and math.isfinite(specificity) else float("nan")
    return {
        "auroc": auc_score(y, s),
        "auprc": average_precision(y, s),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "balanced_accuracy": float(balanced),
        "threshold": float(threshold),
    }


def select_threshold_max_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    scores = np.asarray(y_score, dtype=float)
    candidates = np.unique(np.concatenate([np.linspace(0.0, 1.0, 101), scores[np.isfinite(scores)]]))
    best_thr = 0.5
    best_key = (-1.0, -1.0, 0.0)
    for thr in candidates:
        m = classification_metrics(y_true, scores, float(thr))
        key = (m["f1"], m["balanced_accuracy"], -abs(float(thr) - 0.5))
        if key > best_key:
            best_key = key
            best_thr = float(thr)
    return best_thr


def clip_prob(x: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 1e-4, 1.0 - 1e-4)


def logit(x: np.ndarray | float) -> np.ndarray:
    p = clip_prob(x)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def fused_score(model_score: np.ndarray, retrieval_prior: np.ndarray, alpha: float) -> np.ndarray:
    return sigmoid((1.0 - float(alpha)) * logit(model_score) + float(alpha) * logit(retrieval_prior))


def paired_auc_bootstrap(
    y_true: np.ndarray,
    baseline_score: np.ndarray,
    candidate_score: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 20260616,
) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    b = np.asarray(baseline_score, dtype=float)
    c = np.asarray(candidate_score, dtype=float)
    obs = auc_score(y, c) - auc_score(y, b)
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(auc_score(y[idx], c[idx]) - auc_score(y[idx], b[idx]))
    arr = np.asarray(deltas, dtype=float)
    if arr.size == 0:
        return {
            "delta_auc": float(obs),
            "delta_auc_ci_low": float("nan"),
            "delta_auc_ci_high": float("nan"),
            "paired_bootstrap_p_two_sided": float("nan"),
            "bootstrap_samples": 0,
        }
    p = 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
    if p == 0.0:
        p = 1.0 / float(arr.size)
    return {
        "delta_auc": float(obs),
        "delta_auc_ci_low": float(np.quantile(arr, 0.025)),
        "delta_auc_ci_high": float(np.quantile(arr, 0.975)),
        "paired_bootstrap_p_two_sided": float(min(1.0, p)),
        "bootstrap_samples": int(arr.size),
    }


def empty_csv(path: Path, columns: list[str]) -> None:
    ensure_out_dir()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
