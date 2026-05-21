"""Evaluation metrics per revised method §12 and execution prompt §13."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.metrics import compute_metrics, label_consistency


def _section_completeness(text: str) -> float:
    keys = ("oct", "colposcopy", "clinical", "impression", "recommend")
    t = text.lower()
    return sum(1 for k in keys if k in t) / len(keys)


def evaluate_predictions(
    preds: list[str],
    refs: list[str],
    labels: list[int],
    case_meta: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    base = compute_metrics(preds, refs, labels)
    pos = [label_consistency(p, l) for p, l in zip(preds, labels) if l == 1]
    neg = [label_consistency(p, l) for p, l in zip(preds, labels) if l == 0]
    contradictions = sum(1 for p, l in zip(preds, labels) if label_consistency(p, l) < 0.5)
    base.update(
        {
            "positive_consistency_mean": sum(pos) / max(len(pos), 1),
            "negative_consistency_mean": sum(neg) / max(len(neg), 1),
            "contradiction_rate": contradictions / max(len(preds), 1),
            "section_completeness_mean": sum(_section_completeness(p) for p in preds) / max(len(preds), 1),
            "empty_report_rate": sum(1 for p in preds if len(p.strip()) < 30) / max(len(preds), 1),
        }
    )
    return base


def evaluate_by_groups(
    df: pd.DataFrame,
    preds_map: dict[str, str],
    refs_map: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for group_name, mask in [
        ("all_test", df["split"] == "test"),
        ("real_report_only", (df["split"] == "test") & (df["has_real_report"] == 1)),
        ("pseudo_candidate_only", (df["split"] == "test") & (df["needs_pseudo_report"] == 1)),
    ]:
        sub = df[mask]
        preds, refs, labels, ids = [], [], [], []
        for _, r in sub.iterrows():
            cid = str(r["case_id"])
            if cid not in preds_map:
                continue
            preds.append(preds_map[cid])
            refs.append(refs_map.get(cid, preds_map[cid]))
            labels.append(int(r["binary_label"]))
            ids.append(cid)
        if not preds:
            continue
        m = evaluate_predictions(preds, refs, labels)
        m["eval_group"] = group_name
        m["n"] = len(preds)
        rows.append(m)

    for cid in sorted(df["center_id"].unique()):
        sub = df[(df["split"] == "test") & (df["center_id"] == cid)]
        preds, refs, labels = [], [], []
        for _, r in sub.iterrows():
            case = str(r["case_id"])
            if case not in preds_map:
                continue
            preds.append(preds_map[case])
            refs.append(refs_map.get(case, ""))
            labels.append(int(r["binary_label"]))
        if not preds:
            continue
        m = evaluate_predictions(preds, refs, labels)
        m["eval_group"] = "per_center"
        m["center_id"] = cid
        m["has_real_report_center_rate"] = float(sub["has_real_report"].mean())
        rows.append(m)

    return pd.DataFrame(rows)
