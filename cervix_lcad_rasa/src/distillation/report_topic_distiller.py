"""Unsupervised report-topic distillation for weak report guidance.

This module adapts the useful part of external ultrasound-report generation
workflows: derive latent report-topic labels from report text and reuse them as
auxiliary supervision. It fits topics on the training split only, then assigns
all rows through the fitted text representation to avoid validation/test text
leakage into the topic basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import re


@dataclass
class ReportTopicDistillationResult:
    manifest: pd.DataFrame
    topic_table: pd.DataFrame
    assignment_table: pd.DataFrame


def _clean_text(value: object) -> str:
    text = str(value or "").replace("\n", " ").replace("\t", " ").strip().lower()
    text = re.sub(
        r"(diagnostic_summary|oct_findings|colposcopy_findings|clinical_context|impression|recommendation|generated_sections|generated_report_text|evidence|support|supported|label|weak|supervision|finding|findings|case)",
        " ",
        text,
    )
    text = re.sub(r"m\d+_\d+_p\d+", " ", text)
    text = re.sub(r"\d{3,}", " ", text)
    text = re.sub(r"[{}\[\]\"'“”‘’、，。；;:：,.()（）/\\|<>_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return "empty_report" if not text else " ".join(text.split())


def _top_terms_for_topics(
    x_tfidf,
    topic_ids: np.ndarray,
    feature_names: np.ndarray,
    n_topics: int,
    top_k: int,
) -> list[str]:
    def _useful_term(term: str) -> bool:
        t = term.strip()
        if len(t) < 3:
            return False
        compact = t.replace(" ", "")
        if len(compact) < 3:
            return False
        clinical_ascii = ("hpv", "tct", "cin", "oct", "ascus", "lsil", "hsil", "suspicious", "negative", "positive")
        if compact.isascii() and compact.isalpha() and compact not in clinical_ascii:
            return False
        return True

    labels = []
    for topic in range(n_topics):
        idx = np.where(topic_ids == topic)[0]
        if len(idx) == 0:
            labels.append("")
            continue
        mean_vec = np.asarray(x_tfidf[idx].mean(axis=0)).reshape(-1)
        terms = []
        for i in mean_vec.argsort()[::-1]:
            if mean_vec[i] <= 0:
                break
            term = str(feature_names[i])
            if _useful_term(term):
                terms.append(term.strip())
            if len(terms) >= top_k:
                break
        labels.append(", ".join(terms))
    return labels


def build_report_topics(
    df: pd.DataFrame,
    *,
    text_col: str = "training_report_text",
    split_col: str = "split",
    train_split: str = "train",
    n_topics: int = 8,
    max_features: int = 6000,
    svd_dim: int = 48,
    top_terms: int = 12,
    random_state: int = 42,
) -> ReportTopicDistillationResult:
    """Fit train-split report topics and assign every row.

    The implementation intentionally uses character n-grams rather than jieba
    tokenisation, because this project mixes Chinese findings with English field
    names and short clinical tokens. Character n-grams are robust to both.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import Normalizer
    except ImportError as exc:  # pragma: no cover - environment-specific guard
        raise RuntimeError("report topic distillation requires scikit-learn") from exc

    if text_col not in df.columns:
        raise KeyError(f"missing text column: {text_col}")

    out = df.copy()
    text = out[text_col].map(_clean_text)
    has_text = text.ne("empty_report")
    if split_col in out.columns:
        fit_mask = out[split_col].astype(str).eq(train_split) & has_text
    else:
        fit_mask = has_text
    if fit_mask.sum() < 3:
        raise ValueError(f"not enough train rows with report text to fit topics: {int(fit_mask.sum())}")

    effective_topics = int(min(max(2, n_topics), fit_mask.sum()))
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )
    x_train_tfidf = vectorizer.fit_transform(text[fit_mask])
    x_all_tfidf = vectorizer.transform(text)

    max_svd = max(2, min(svd_dim, x_train_tfidf.shape[0] - 1, x_train_tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=max_svd, random_state=random_state)
    norm = Normalizer(copy=False)
    x_train = norm.fit_transform(svd.fit_transform(x_train_tfidf))
    x_all = norm.transform(svd.transform(x_all_tfidf))

    km = KMeans(n_clusters=effective_topics, n_init=25, random_state=random_state)
    train_topic = km.fit_predict(x_train)
    all_topic = km.predict(x_all)
    distances = km.transform(x_all)
    if effective_topics > 1:
        sorted_dist = np.sort(distances, axis=1)
        confidence = (sorted_dist[:, 1] - sorted_dist[:, 0]) / np.maximum(sorted_dist[:, 1], 1e-6)
        confidence = np.clip(confidence, 0.0, 1.0)
    else:
        confidence = np.ones(len(out), dtype=float)

    all_topic = all_topic.astype(int)
    all_topic[~has_text.to_numpy()] = -1
    confidence[~has_text.to_numpy()] = 0.0

    feature_names = vectorizer.get_feature_names_out()
    topic_terms = _top_terms_for_topics(x_train_tfidf, train_topic, feature_names, effective_topics, top_terms)
    train_text = text[fit_mask].reset_index(drop=True)
    topic_rows = []
    for topic in range(effective_topics):
        idx = np.where(train_topic == topic)[0]
        examples = train_text.iloc[idx[:3]].tolist() if len(idx) else []
        topic_rows.append(
            {
                "report_topic_id": topic,
                "n_train": int(len(idx)),
                "top_terms": topic_terms[topic],
                "example_text": " || ".join(examples),
            }
        )
    topic_table = pd.DataFrame(topic_rows).sort_values(["n_train", "report_topic_id"], ascending=[False, True])

    out["report_topic_id"] = all_topic
    out["report_topic_confidence"] = confidence.astype(float)
    out["report_topic_source"] = "train_split_char_tfidf_svd_kmeans"
    out["report_topic_fit_split"] = train_split

    assignment_cols = [
        c
        for c in ["case_id", split_col, "center_id", "training_report_type", "binary_label", "report_topic_id", "report_topic_confidence"]
        if c in out.columns
    ]
    assignment_table = out[assignment_cols].copy()
    return ReportTopicDistillationResult(out, topic_table, assignment_table)


def write_report_topic_outputs(
    result: ReportTopicDistillationResult,
    *,
    manifest_out: Path,
    output_dir: Path,
    source_manifest: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    result.manifest.to_csv(manifest_out, index=False)
    topic_path = tables / "report_topic_catalog.csv"
    assign_path = tables / "report_topic_case_assignments.csv"
    result.topic_table.to_csv(topic_path, index=False)
    result.assignment_table.to_csv(assign_path, index=False)

    topic_counts = result.manifest["report_topic_id"].value_counts(dropna=False).sort_index()
    summary = [
        "# Report Topic Distiller Summary",
        "",
        "Purpose: unsupervised report-text topic labels for auxiliary report-generation supervision.",
        "",
        f"- Source manifest: `{source_manifest}`",
        f"- Output manifest: `{manifest_out}`",
        f"- Topic catalog: `{topic_path}`",
        f"- Assignment table: `{assign_path}`",
        f"- Rows: {len(result.manifest)}",
        "",
        "## Topic Counts",
        "",
    ]
    for topic, count in topic_counts.items():
        summary.append(f"- Topic {topic}: {int(count)} cases")
    summary += [
        "",
        "## Topic Labels",
        "",
    ]
    for row in result.topic_table.itertuples(index=False):
        summary.append(f"- Topic {row.report_topic_id} (train n={row.n_train}): {row.top_terms}")
    summary_path = output_dir / "REPORT_TOPIC_DISTILLER_SUMMARY.md"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return {
        "manifest": manifest_out,
        "topic_catalog": topic_path,
        "assignments": assign_path,
        "summary": summary_path,
    }
