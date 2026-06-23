#!/usr/bin/env python3
"""Build train-only semantic-tag retrieval predictions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from llm_semantic_common import (
    OUT_DIR,
    RETRIEVAL_COLUMNS,
    compact_text,
    empty_csv,
    ensure_out_dir,
    markdown_table,
    split_tags,
    tokenize_tag_text,
)


INPUT = OUT_DIR / "semantic_tagging_input.csv"
RULE_TAGS = OUT_DIR / "rule_semantic_tags.csv"
LLM_TAGS = OUT_DIR / "llm_semantic_tags.csv"
RULE_OUT = OUT_DIR / "rule_tag_retrieval_predictions.csv"
LLM_OUT = OUT_DIR / "llm_tag_retrieval_predictions.csv"
LLM_WEIGHTED_OUT = OUT_DIR / "llm_tag_retrieval_predictions_weighted.csv"


def _case_tag_text(row: pd.Series) -> str:
    parts = [compact_text(row.get("tag_text"))]
    for col in [
        "oct_tags",
        "colposcopy_tags",
        "clinical_tags",
        "impression_tags",
        "severity_tags",
        "modality_evidence",
        "missing_section_flags",
    ]:
        parts.extend(split_tags(row.get(col)))
    return " ".join(p for p in parts if p)


def _vectorise(train_texts: list[str], query_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    vocab_counter: Counter[str] = Counter()
    train_tokens = []
    for text in train_texts:
        tokens = tokenize_tag_text(text)
        train_tokens.append(tokens)
        vocab_counter.update(set(tokens))
    vocab = {tok: i for i, (tok, count) in enumerate(vocab_counter.items()) if count >= 1}
    if not vocab:
        return np.zeros((len(train_texts), 1), dtype=float), np.zeros((len(query_texts), 1), dtype=float)

    def make_matrix(texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), len(vocab)), dtype=float)
        for r, text in enumerate(texts):
            for tok in set(tokenize_tag_text(text)):
                idx = vocab.get(tok)
                if idx is not None:
                    mat[r, idx] = 1.0
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        mat = mat / norms[:, None]
        return mat

    return make_matrix(train_texts), make_matrix(query_texts)


def _build_for_source(tags_path: Path, out_path: Path, *, source: str, weighted_out: Path | None = None) -> pd.DataFrame:
    if not tags_path.exists():
        empty_csv(out_path, RETRIEVAL_COLUMNS)
        if weighted_out is not None:
            empty_csv(weighted_out, RETRIEVAL_COLUMNS)
        return pd.DataFrame(columns=RETRIEVAL_COLUMNS)
    tags = pd.read_csv(tags_path)
    if tags.empty or "valid_json" in tags.columns and int(tags.get("valid_json", pd.Series(dtype=int)).sum()) == 0:
        empty_csv(out_path, RETRIEVAL_COLUMNS)
        if weighted_out is not None:
            empty_csv(weighted_out, RETRIEVAL_COLUMNS)
        return pd.DataFrame(columns=RETRIEVAL_COLUMNS)
    inputs = pd.read_csv(INPUT)
    keep_cols = ["case_id", "split", "center_id", "y_true"]
    tags = tags.merge(inputs[keep_cols], on=["case_id", "split", "center_id"], how="left")
    tags["case_tag_text"] = tags.apply(_case_tag_text, axis=1)
    tags["support_score"] = pd.to_numeric(tags.get("support_score", 0), errors="coerce").fillna(0.0)
    tags["contradiction_flag"] = pd.to_numeric(tags.get("contradiction_flag", 0), errors="coerce").fillna(0).astype(int)

    train = tags[tags["split"].eq("train")].copy()
    query = tags[tags["split"].isin(["val", "test"])].copy()
    if train.empty or query.empty:
        empty_csv(out_path, RETRIEVAL_COLUMNS)
        if weighted_out is not None:
            empty_csv(weighted_out, RETRIEVAL_COLUMNS)
        return pd.DataFrame(columns=RETRIEVAL_COLUMNS)

    train_matrix, query_matrix = _vectorise(train["case_tag_text"].tolist(), query["case_tag_text"].tolist())
    sim = query_matrix @ train_matrix.T
    train_y = train["y_true"].astype(int).to_numpy()
    train_ids = train["case_id"].astype(str).to_numpy()
    prevalence = float(train_y.mean()) if len(train_y) else 0.5

    rows = []
    for q_idx, (_, qrow) in enumerate(query.iterrows()):
        sims = sim[q_idx]
        for topk in [5, 10, 20]:
            k = min(topk, len(train))
            order = np.argsort(-sims, kind="mergesort")[:k]
            top_sims = sims[order]
            top_y = train_y[order]
            if float(top_sims.sum()) > 0:
                prior = float(np.average(top_y, weights=np.maximum(top_sims, 1e-6)))
            else:
                prior = prevalence
            support = float(np.mean(top_sims)) if len(top_sims) else 0.0
            rows.append(
                {
                    "case_id": qrow["case_id"],
                    "split": qrow["split"],
                    "center_id": qrow["center_id"],
                    "source": source,
                    "topk": topk,
                    "retrieval_prior": prior,
                    "mean_similarity": support,
                    "max_similarity": float(np.max(top_sims)) if len(top_sims) else 0.0,
                    "retrieved_positive_fraction": float(np.mean(top_y)) if len(top_y) else prevalence,
                    "retrieval_support_weight": float(qrow["support_score"]) * (1.0 - 0.5 * int(qrow["contradiction_flag"])),
                    "support_score": float(qrow["support_score"]),
                    "contradiction_flag": int(qrow["contradiction_flag"]),
                    "y_true": int(qrow["y_true"]),
                    "top_train_case_ids": ";".join(train_ids[order].tolist()),
                }
            )

    out = pd.DataFrame(rows, columns=RETRIEVAL_COLUMNS)
    out.to_csv(out_path, index=False)

    if weighted_out is not None:
        weighted = out.copy()
        weighted["source"] = source + "_weighted"
        global_prior = prevalence
        w = weighted["retrieval_support_weight"].astype(float).clip(0.0, 1.0)
        weighted["retrieval_prior"] = w * weighted["retrieval_prior"].astype(float) + (1.0 - w) * global_prior
        weighted.to_csv(weighted_out, index=False)
    return out


def main() -> None:
    ensure_out_dir()
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing semantic input: {INPUT}")

    rule = _build_for_source(RULE_TAGS, RULE_OUT, source="rule")
    llm = _build_for_source(LLM_TAGS, LLM_OUT, source="llm", weighted_out=LLM_WEIGHTED_OUT)

    summaries = []
    for name, df in [("rule", rule), ("llm", llm)]:
        if df.empty:
            summaries.append({"source": name, "rows": 0, "val_test_cases": 0, "mean_prior_top10": np.nan, "mean_support_top10": np.nan})
        else:
            top10 = df[df["topk"].eq(10)]
            summaries.append(
                {
                    "source": name,
                    "rows": len(df),
                    "val_test_cases": int(top10["case_id"].nunique()),
                    "mean_prior_top10": float(top10["retrieval_prior"].mean()),
                    "mean_support_top10": float(top10["mean_similarity"].mean()),
                }
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT_DIR / "semantic_tag_retrieval_summary.csv", index=False)
    md = [
        "# Semantic Tag Retrieval Summary",
        "",
        "Retrieval banks are constructed from training cases only. Validation/test labels are used only after prediction export for metric computation.",
        "",
        markdown_table(summary),
    ]
    if llm.empty:
        md.extend(
            [
                "",
                "LLM tag retrieval is unavailable because no valid LLM semantic tag table was produced. The schema-only LLM prediction files are intentional and prevent rule outputs from being mislabeled as LLM evidence.",
            ]
        )
    (OUT_DIR / "semantic_tag_retrieval_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {RULE_OUT}")
    print(f"Wrote {LLM_OUT}")
    print(f"Wrote {LLM_WEIGHTED_OUT}")


if __name__ == "__main__":
    main()
