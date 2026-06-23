#!/usr/bin/env python3
"""Generate revision-audit tables for the JBD MOSAIC manuscript.

The script is intentionally post-hoc and read-only with respect to model
training. It uses locked validation/test predictions to audit protocol
harmonisation, fusion-weight sensitivity, and the same-split contrastive
baseline comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "publishable" / "mosaic_revision_audit"
TABLES = OUT / "tables"
REPORT = OUT / "MOSAIC_JBD_MAJOR_REVISION_UPGRADE_REPORT.md"


def _clip_prob(x: np.ndarray) -> np.ndarray:
    # Match the original semantic-fusion export, which clipped probabilities
    # before logit fusion to avoid infinite logits from zero retrieval ratios.
    return np.clip(x.astype(float), 0.01, 0.99)


def _logit(x: np.ndarray) -> np.ndarray:
    x = _clip_prob(x)
    return np.log(x / (1.0 - x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _roc_auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    unique_scores, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    del unique_scores
    for group_idx, count in enumerate(counts):
        if count > 1:
            mask = inverse == group_idx
            ranks[mask] = ranks[mask].mean()
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision_score(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted == 1)
    precision_at_k = tp_cum / (np.arange(len(y_sorted)) + 1)
    return float((precision_at_k * (y_sorted == 1)).sum() / n_pos)


def _classification_counts(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return tp, fp, tn, fn


def _precision_recall_f1_balacc(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    tp, fp, tn, fn = _classification_counts(y, pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balacc = 0.5 * (recall + specificity)
    return precision, recall, f1, balacc


def _binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    precision, recall, f1, balacc = _precision_recall_f1_balacc(y, pred)
    return {
        "auc": _roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "auprc": _average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "f1": f1,
        "sensitivity": recall,
        "precision": precision,
        "balanced_accuracy": balacc if len(np.unique(y)) == 2 else np.nan,
    }


def _best_f1_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.round(np.linspace(0.01, 0.99, 99), 2):
        _, _, f1, _ = _precision_recall_f1_balacc(y, score >= t)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def _paired_auc_bootstrap(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 20260614,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(_roc_auc_score(y[idx], score_a[idx]) - _roc_auc_score(y[idx], score_b[idx]))
    deltas = np.asarray(deltas)
    observed = _roc_auc_score(y, score_a) - _roc_auc_score(y, score_b)
    p_two_sided = 2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    return {
        "delta_auc": float(observed),
        "delta_auc_ci_low": float(np.quantile(deltas, 0.025)),
        "delta_auc_ci_high": float(np.quantile(deltas, 0.975)),
        "paired_bootstrap_p_two_sided": float(min(p_two_sided, 1.0)),
        "bootstrap_samples": int(len(deltas)),
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    fusion_scores = pd.read_csv(
        ROOT
        / "outputs"
        / "publishable"
        / "kra_semantic_fusion_analysis"
        / "kra_semantic_fusion_val_test_scores.csv"
    )
    external = pd.read_csv(
        ROOT
        / "outputs"
        / "publishable"
        / "external_baselines"
        / "predictions"
        / "contrastive_multimodal_no_report_sections_test_predictions.csv"
    )
    mosaic_main = pd.read_csv(
        ROOT / "outputs" / "publishable" / "tables" / "manuscript" / "T_mosaic_main_comparison.csv"
    )
    baselines = pd.read_csv(
        ROOT
        / "outputs"
        / "publishable"
        / "tables"
        / "manuscript"
        / "T_external_baselines_same_split.csv"
    )

    sensitivity_rows = []
    for alpha in np.round(np.linspace(0.0, 1.0, 21), 2):
        for split in ["val", "test"]:
            frame = fusion_scores[fusion_scores["split"] == split].copy()
            score = _sigmoid(
                (1.0 - alpha) * _logit(frame["risk_score"].to_numpy())
                + alpha * _logit(frame["semantic_retrieval_positive_ratio"].to_numpy())
            )
            frame["score_alpha"] = score
            if split == "val":
                threshold, val_f1 = _best_f1_threshold(frame["y_true"].to_numpy(), score)
                val_auc = _roc_auc_score(frame["y_true"], score)
                sensitivity_rows.append(
                    {
                        "alpha": alpha,
                        "split": split,
                        "selected_threshold": threshold,
                        "val_f1_at_selected_threshold": val_f1,
                        "val_auc": val_auc,
                        **_binary_metrics(frame["y_true"].to_numpy(), score, threshold),
                    }
                )
            else:
                val_row = [r for r in sensitivity_rows if r["alpha"] == alpha and r["split"] == "val"][0]
                threshold = val_row["selected_threshold"]
                sensitivity_rows.append(
                    {
                        "alpha": alpha,
                        "split": split,
                        "selected_threshold": threshold,
                        "val_f1_at_selected_threshold": val_row["val_f1_at_selected_threshold"],
                        "val_auc": val_row["val_auc"],
                        **_binary_metrics(frame["y_true"].to_numpy(), score, threshold),
                    }
                )

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLES / "mosaic_alpha_threshold_sensitivity.csv", index=False)

    selected_val = sensitivity[sensitivity["split"] == "val"].sort_values("val_auc", ascending=False).iloc[0]
    selected_alpha = float(selected_val["alpha"])
    selected_test = sensitivity[
        (sensitivity["split"] == "test") & (sensitivity["alpha"] == selected_alpha)
    ].iloc[0]

    test_scores = fusion_scores[fusion_scores["split"] == "test"].copy()
    paired = test_scores.merge(
        external[["case_id", "y_true_cin2plus", "risk_score"]],
        on="case_id",
        how="inner",
        suffixes=("_mosaic", "_contrastive"),
    )
    if not np.array_equal(paired["y_true"].to_numpy(), paired["y_true_cin2plus"].to_numpy()):
        raise RuntimeError("Label mismatch between MOSAIC and contrastive prediction files.")
    paired_stats = _paired_auc_bootstrap(
        paired["y_true"].to_numpy(),
        paired["semantic_fusion_score"].to_numpy(),
        paired["risk_score_contrastive"].to_numpy(),
    )
    paired_table = pd.DataFrame(
        [
            {
                "comparison": "MOSAIC full vs CLIP-style contrastive multimodal baseline",
                "n_paired": len(paired),
                "mosaic_auc": _roc_auc_score(paired["y_true"], paired["semantic_fusion_score"]),
                "contrastive_auc": _roc_auc_score(paired["y_true"], paired["risk_score_contrastive"]),
                **paired_stats,
            }
        ]
    )
    paired_table.to_csv(TABLES / "mosaic_vs_contrastive_paired_bootstrap.csv", index=False)

    full_reference = baselines[baselines["baseline_id"] == "full_lcad_rasa_reference"].iloc[0]
    contrastive = baselines[baselines["baseline_id"] == "contrastive_multimodal_no_report_sections"].iloc[0]
    stable_backbone = mosaic_main[mosaic_main["model_id"] == "full_lcad_rasa_stablehash"].iloc[0]
    retrieval = mosaic_main[mosaic_main["model_id"] == "semantic_retrieval_positive_ratio"].iloc[0]
    mosaic_full = mosaic_main[mosaic_main["model_id"] == "kra_semantic_fusion"].iloc[0]
    protocol_rows = [
        {
            "result_label": "Pre-retrieval Full LCAD-RASA reference",
            "role": "historical internal held-out mechanism block",
            "score_source": str(full_reference["prediction_file"]),
            "auc": full_reference["auc"],
            "f1": full_reference["f1"],
            "threshold": full_reference["threshold_val_max_f1"],
            "interpretation": "Do not mix with stable-hash MOSAIC table; retained to explain earlier backbone number.",
        },
        {
            "result_label": "MOSAIC-RASA stable-hash backbone",
            "role": "backbone used for MOSAIC retrieval fusion",
            "score_source": "kra_semantic_fusion_val_test_scores.csv:risk_score",
            "auc": stable_backbone["auc"],
            "f1": stable_backbone["f1"],
            "threshold": stable_backbone["threshold"],
            "interpretation": "Primary backbone comparator for MOSAIC full.",
        },
        {
            "result_label": "Semantic retrieval prior only",
            "role": "train-only weak-oracle semantic prior",
            "score_source": "kra_semantic_fusion_val_test_scores.csv:semantic_retrieval_positive_ratio",
            "auc": retrieval["auc"],
            "f1": retrieval["f1"],
            "threshold": retrieval["threshold"],
            "interpretation": "Shows retrieved report-derived semantics carry signal but are insufficient alone.",
        },
        {
            "result_label": "MOSAIC full",
            "role": "proposed full method",
            "score_source": "kra_semantic_fusion_val_test_scores.csv:semantic_fusion_score",
            "auc": mosaic_full["auc"],
            "f1": mosaic_full["f1"],
            "threshold": mosaic_full["threshold"],
            "interpretation": "Main method: validation-calibrated fusion of backbone risk and train-only semantic prior.",
        },
        {
            "result_label": "CLIP-style contrastive baseline",
            "role": "strong same-split discriminative baseline",
            "score_source": str(contrastive["prediction_file"]),
            "auc": contrastive["auc"],
            "f1": contrastive["f1"],
            "threshold": contrastive["threshold_val_max_f1"],
            "interpretation": "Use to show MOSAIC full, not the backbone alone, is the relevant comparison.",
        },
    ]
    protocol = pd.DataFrame(protocol_rows)
    protocol.to_csv(TABLES / "mosaic_protocol_harmonization.csv", index=False)

    report = f"""# MOSAIC JBD Major-Revision Upgrade Report

## Purpose

This audit converts the editor-style critique into manuscript-safe evidence
without retraining models or changing held-out labels.

## Protocol harmonisation

The apparent backbone mismatch is a protocol issue, not a new result:

- Pre-retrieval Full LCAD--RASA reference: AUROC `{full_reference['auc']:.3f}`.
- Stable-hash MOSAIC--RASA backbone used for semantic fusion: AUROC `{stable_backbone['auc']:.3f}`.
- MOSAIC full: AUROC `{mosaic_full['auc']:.3f}`.

Use the stable-hash backbone only when discussing MOSAIC full.

## Alpha and threshold sensitivity

Validation-selected grid alpha by AUROC: `{selected_alpha:.2f}`.
The corresponding held-out test AUROC is `{selected_test['auc']:.3f}` and F1 is
`{selected_test['f1']:.3f}` at the validation-selected threshold
`{selected_test['selected_threshold']:.2f}`.

The manuscript should cite this table as a sensitivity audit, not as nested
cross-validation.

## Strong-baseline comparison

MOSAIC full versus the CLIP-style contrastive multimodal baseline on paired
test cases:

- Paired cases: `{len(paired)}`.
- MOSAIC AUROC: `{paired_table.iloc[0]['mosaic_auc']:.3f}`.
- Contrastive AUROC: `{paired_table.iloc[0]['contrastive_auc']:.3f}`.
- Delta AUROC: `{paired_table.iloc[0]['delta_auc']:.3f}`.
- 95% bootstrap CI: `[{paired_table.iloc[0]['delta_auc_ci_low']:.3f}, {paired_table.iloc[0]['delta_auc_ci_high']:.3f}]`.
- Two-sided bootstrap p-value: `{paired_table.iloc[0]['paired_bootstrap_p_two_sided']:.3f}`.

Claim boundary: this supports MOSAIC full as a competitive report-aware
semantic-prior framework. It does not rescue the pre-retrieval RASA backbone as
superior to the contrastive baseline.

## Output tables

- `tables/mosaic_protocol_harmonization.csv`
- `tables/mosaic_alpha_threshold_sensitivity.csv`
- `tables/mosaic_vs_contrastive_paired_bootstrap.csv`
"""
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
