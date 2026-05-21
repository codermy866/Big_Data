"""Text and clinical consistency metrics."""

from __future__ import annotations

from typing import Any


def _safe_rouge(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(ref, pred)["rougeL"].fmeasure
    except Exception:
        pred_set = set(pred.lower().split())
        ref_set = set(ref.lower().split())
        if not ref_set:
            return 0.0
        return len(pred_set & ref_set) / len(ref_set)


def label_consistency(text: str, label: int) -> float:
    text_l = text.lower()
    if label == 1:
        return 1.0 if any(k in text_l for k in ("cin2", "suspicious", "high-grade")) else 0.0
    return 1.0 if any(k in text_l for k in ("no high-grade", "negative", "nil")) else 0.5


def compute_metrics(
    predictions: list[str],
    references: list[str],
    labels: list[int],
) -> dict[str, Any]:
    rouge_scores = [_safe_rouge(p, r) for p, r in zip(predictions, references)]
    lc_scores = [label_consistency(p, l) for p, l in zip(predictions, labels)]
    return {
        "rouge_l_mean": sum(rouge_scores) / max(len(rouge_scores), 1),
        "label_consistency_mean": sum(lc_scores) / max(len(lc_scores), 1),
        "n": len(predictions),
    }
