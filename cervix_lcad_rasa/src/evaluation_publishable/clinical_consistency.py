"""Clinical consistency metrics for decoded reports."""

from __future__ import annotations

from src.evaluation.metrics import label_consistency


def clinical_metrics(pred_text: str, label: int) -> dict[str, float]:
    lc = label_consistency(pred_text, label)
    text_l = pred_text.lower()
    pos_phrases = ("cin2", "suspicious", "high-grade", "positive")
    neg_phrases = ("no definitive", "negative", "no high-grade", "nil")
    if label == 1:
        pos_c = 1.0 if any(p in text_l for p in pos_phrases) else 0.0
        neg_c = 1.0 if any(p in text_l for p in neg_phrases) else 0.0
        contradiction = 1.0 if neg_c > 0.5 else 0.0
        overdiag = 0.0
    else:
        pos_c = 1.0 if any(p in text_l for p in pos_phrases) else 0.0
        neg_c = 1.0 if any(p in text_l for p in neg_phrases) else 0.5
        contradiction = 1.0 if pos_c > 0.5 and "suspicious for cin2" in text_l else 0.0
        overdiag = 1.0 if "cin3" in text_l or "invasive" in text_l else 0.0
    return {
        "label_consistency": lc,
        "positive_consistency": pos_c if label == 1 else 1.0 - pos_c,
        "negative_consistency": neg_c if label == 0 else 1.0 - neg_c,
        "contradiction_rate": contradiction,
        "overdiagnosis_flag_rate": overdiag,
    }
