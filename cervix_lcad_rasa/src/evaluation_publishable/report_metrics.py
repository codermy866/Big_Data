"""BLEU, ROUGE-L, METEOR, BERTScore (optional) for reference-based eval."""

from __future__ import annotations

from typing import Any


def token_bleu(pred: str, ref: str) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        ref_t = ref.lower().split()
        pred_t = pred.lower().split()
        if not ref_t:
            return 0.0
        return float(sentence_bleu([ref_t], pred_t, smoothing_function=SmoothingFunction().method1))
    except Exception:
        ps, rs = set(pred.lower().split()), set(ref.lower().split())
        return len(ps & rs) / max(len(rs), 1)


def rouge_l(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer

        s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return s.score(ref, pred)["rougeL"].fmeasure
    except Exception:
        ps, rs = set(pred.lower().split()), set(ref.lower().split())
        return len(ps & rs) / max(len(rs), 1)


def meteor_score(pred: str, ref: str) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score as ms
        from nltk.tokenize import word_tokenize

        return float(ms([word_tokenize(ref)], word_tokenize(pred)))
    except Exception:
        return rouge_l(pred, ref)


def bertscore_f1(pred: str, ref: str) -> float:
    try:
        from bert_score import score

        P, R, F1 = score([pred], [ref], lang="zh", verbose=False)
        return float(F1[0])
    except Exception:
        return 0.0


def compute_reference_metrics(pred: str, ref: str) -> dict[str, float]:
    return {
        "bleu": token_bleu(pred, ref),
        "rouge_l": rouge_l(pred, ref),
        "meteor": meteor_score(pred, ref),
        "bertscore_f1": bertscore_f1(pred, ref),
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    keys = [k for k in rows[0] if k not in ("case_id", "center_id", "eval_scope")]
    out = {"n": len(rows)}
    for k in keys:
        if isinstance(rows[0][k], (int, float)):
            out[k] = sum(r[k] for r in rows) / len(rows)
    return out
