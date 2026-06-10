#!/usr/bin/env python3
"""Validation-calibrated semantic retrieval fusion for KRA-RASA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(float), 1e-5, 1.0 - 1e-5)
    return np.log(x / (1.0 - x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (y_score >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "auprc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "threshold": float(threshold),
    }


def select_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.round(np.arange(0.01, 1.0, 0.01), 2):
        f1 = f1_score(y_true, y_score >= thr, zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr


def calibrated_fusion(model_score: np.ndarray, retrieval_positive_ratio: np.ndarray, alpha: float) -> np.ndarray:
    retrieval = retrieval_positive_ratio * 0.98 + 0.01
    return _sigmoid((1.0 - alpha) * _logit(model_score) + alpha * _logit(retrieval))


def select_alpha(val: pd.DataFrame) -> tuple[float, float]:
    y = val["y_true"].to_numpy()
    best_alpha, best_auc = 0.0, -1.0
    for alpha in np.round(np.linspace(0.0, 1.0, 101), 2):
        score = calibrated_fusion(val["risk_score"].to_numpy(), val["semantic_retrieval_positive_ratio"].to_numpy(), float(alpha))
        auc = roc_auc_score(y, score)
        if auc > best_auc:
            best_alpha, best_auc = float(alpha), float(auc)
    return best_alpha, best_auc


def paired_auc_bootstrap(y_true: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, *, n_boot: int = 1000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    obs = float(roc_auc_score(y_true, candidate) - roc_auc_score(y_true, baseline))
    deltas = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        deltas.append(float(roc_auc_score(y_true[idx], candidate[idx]) - roc_auc_score(y_true[idx], baseline[idx])))
    arr = np.asarray(deltas, dtype=float)
    p = 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0))) if arr.size else np.nan
    return {
        "delta_auc": obs,
        "delta_auc_ci_low": float(np.quantile(arr, 0.025)) if arr.size else np.nan,
        "delta_auc_ci_high": float(np.quantile(arr, 0.975)) if arr.size else np.nan,
        "paired_bootstrap_p_two_sided": float(min(1.0, max(1.0 / arr.size if p == 0 else p, p))) if arr.size else np.nan,
        "bootstrap_samples": int(arr.size),
    }


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{digits}f}")
        else:
            view[col] = view[col].astype(str)
    return "\n".join(
        [
            "| " + " | ".join(view.columns) + " |",
            "| " + " | ".join(["---"] * len(view.columns)) + " |",
            *["| " + " | ".join(str(v) for v in row) + " |" for row in view.to_numpy()],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-table",
        default="outputs/publishable/kra_rasa_stablehash_analysis/full_lcad_rasa_val_test_scores.csv",
    )
    parser.add_argument("--output-dir", default="outputs/publishable/kra_semantic_fusion_analysis")
    args = parser.parse_args()
    pred_path = Path(args.prediction_table)
    out_dir = Path(args.output_dir)
    root = Path(__file__).resolve().parents[1]
    if not pred_path.is_absolute():
        pred_path = root / pred_path
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(pred_path)
    val = pred[pred["split"].eq("val")].copy()
    test = pred[pred["split"].eq("test")].copy()
    alpha, val_auc = select_alpha(val)
    val["semantic_fusion_score"] = calibrated_fusion(val["risk_score"].to_numpy(), val["semantic_retrieval_positive_ratio"].to_numpy(), alpha)
    test["semantic_fusion_score"] = calibrated_fusion(test["risk_score"].to_numpy(), test["semantic_retrieval_positive_ratio"].to_numpy(), alpha)
    threshold = select_threshold(val["y_true"].to_numpy(), val["semantic_fusion_score"].to_numpy())

    baseline_thr = select_threshold(val["y_true"].to_numpy(), val["risk_score"].to_numpy())
    retrieval_thr = select_threshold(val["y_true"].to_numpy(), val["semantic_retrieval_positive_ratio"].to_numpy())
    comparison = pd.DataFrame(
        [
            {"model_id": "full_lcad_rasa_stablehash", **metrics(test["y_true"].to_numpy(), test["risk_score"].to_numpy(), baseline_thr)},
            {"model_id": "semantic_retrieval_positive_ratio", **metrics(test["y_true"].to_numpy(), test["semantic_retrieval_positive_ratio"].to_numpy(), retrieval_thr)},
            {"model_id": "kra_semantic_fusion", **metrics(test["y_true"].to_numpy(), test["semantic_fusion_score"].to_numpy(), threshold), "alpha_val_auc": alpha, "val_auc_selected": val_auc},
        ]
    )
    comparison.to_csv(out_dir / "kra_semantic_fusion_risk_comparison.csv", index=False)

    scored = pd.concat([val, test], ignore_index=True)
    scored.to_csv(out_dir / "kra_semantic_fusion_val_test_scores.csv", index=False)

    bootstrap = paired_auc_bootstrap(test["y_true"].to_numpy(), test["risk_score"].to_numpy(), test["semantic_fusion_score"].to_numpy())
    (out_dir / "kra_semantic_fusion_vs_full_paired_auc_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")

    center_rows = []
    for center, part in test.groupby("center_id"):
        row = {"center_id": center, "n": int(len(part))}
        if len(part["y_true"].unique()) > 1:
            row.update(metrics(part["y_true"].to_numpy(), part["semantic_fusion_score"].to_numpy(), threshold))
            row["baseline_auc"] = metrics(part["y_true"].to_numpy(), part["risk_score"].to_numpy(), baseline_thr)["auc"]
        else:
            row.update({"auc": np.nan, "auprc": np.nan, "f1": np.nan, "sensitivity": np.nan, "precision": np.nan, "balanced_accuracy": np.nan, "threshold": threshold, "baseline_auc": np.nan})
        center_rows.append(row)
    center_table = pd.DataFrame(center_rows)
    center_table.to_csv(out_dir / "kra_semantic_fusion_centerwise.csv", index=False)

    summary = [
        "# MOSAIC Retrieval-Calibration Analysis",
        "",
        "## Method",
        "",
        "MOSAIC completes the framework with a train-only report-derived semantic bank. The retrieved positive semantic prior is fused with the MOSAIC--RASA backbone risk score through a validation-calibrated logit fusion layer. The fusion weight is selected only on the validation split.",
        "",
        f"- Selected fusion alpha: {alpha:.2f}",
        f"- Validation AUROC at selected alpha: {val_auc:.4f}",
        f"- Validation-selected threshold: {threshold:.2f}",
        "",
        "## Held-Out Risk Metrics",
        "",
        markdown_table(comparison),
        "",
        "## Paired AUROC Bootstrap: MOSAIC (full) - MOSAIC--RASA backbone",
        "",
        markdown_table(pd.DataFrame([bootstrap])),
        "",
        "## Center-Wise Semantic Fusion Metrics",
        "",
        markdown_table(center_table),
        "",
        "## Decision",
        "",
    ]
    fused = comparison[comparison["model_id"].eq("kra_semantic_fusion")].iloc[0]
    base = comparison[comparison["model_id"].eq("full_lcad_rasa_stablehash")].iloc[0]
    delta_auc = float(fused["auc"] - base["auc"])
    delta_f1 = float(fused["f1"] - base["f1"])
    decision = "mosaic_main_method" if delta_auc > 0.02 and delta_f1 > 0 else "needs_revision"
    summary.extend(
        [
            f"- Delta AUROC MOSAIC (full) - MOSAIC--RASA backbone: {delta_auc:.4f}",
            f"- Delta F1 MOSAIC (full) - MOSAIC--RASA backbone: {delta_f1:.4f}",
            f"- Decision: `{decision}`",
            "",
            "MOSAIC (full) is the proposed manuscript method. The neural token-packer variant should remain an ablation until it produces stable gains.",
        ]
    )
    (out_dir / "MOSAIC_ANALYSIS.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (out_dir / "KRA_SEMANTIC_FUSION_ANALYSIS.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote semantic fusion analysis to {out_dir}")


if __name__ == "__main__":
    main()

