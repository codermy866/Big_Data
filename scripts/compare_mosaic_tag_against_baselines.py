#!/usr/bin/env python3
"""Compare MOSAIC-Tag with MOSAIC, RASA, and same-split baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from llm_semantic_common import (
    RASA_SCORE_CANDIDATES,
    classification_metrics,
    first_existing,
    markdown_table,
    paired_auc_bootstrap,
    select_threshold_max_f1,
)


ROOT = Path(__file__).resolve().parents[1]
REV_DIR = ROOT / "outputs" / "revision"
ABLATION_PRED = REV_DIR / "semantic_tag_source_ablation_predictions.csv"
ABLATION_METRICS = REV_DIR / "semantic_tag_source_ablation_metrics.csv"

BASELINE_FILES = {
    "clip_contrastive": (
        "CLIP-style contrastive baseline",
        ROOT / "cervix_lcad_rasa/outputs/publishable/external_baselines/predictions/contrastive_multimodal_no_report_sections_test_predictions.csv",
    ),
    "clinical_hgb": (
        "Clinical-only HistGradientBoosting",
        ROOT / "cervix_lcad_rasa/outputs/publishable/external_baselines/predictions/clinical_hist_gradient_boosting_test_predictions.csv",
    ),
    "cross_attention": (
        "Cross-attention multimodal transformer",
        ROOT / "cervix_lcad_rasa/outputs/publishable/external_baselines/predictions/cross_attention_multimodal_transformer_test_predictions.csv",
    ),
    "clinical_lr": (
        "Clinical-only logistic regression",
        ROOT / "cervix_lcad_rasa/outputs/publishable/external_baselines/predictions/clinical_lr_test_predictions.csv",
    ),
    "late_fusion_mlp": (
        "Late-fusion MLP",
        ROOT / "cervix_lcad_rasa/outputs/publishable/external_baselines/predictions/late_fusion_mlp_test_predictions.csv",
    ),
    "oct_only_mlp": (
        "OCT-only embedding MLP",
        ROOT / "cervix_lcad_rasa/outputs/publishable/external_baselines/predictions/oct_only_embedding_mlp_test_predictions.csv",
    ),
    "colposcopy_only_mlp": (
        "Colposcopy-only embedding MLP",
        ROOT / "cervix_lcad_rasa/outputs/publishable/external_baselines/predictions/colposcopy_only_embedding_mlp_test_predictions.csv",
    ),
}


def load_scores() -> pd.DataFrame:
    path = first_existing(RASA_SCORE_CANDIDATES)
    if path is None:
        raise FileNotFoundError("Missing RASA score table")
    return pd.read_csv(path)


def make_row(model_id: str, model: str, protocol: str, df: pd.DataFrame, score_col: str, threshold: float) -> dict[str, object]:
    y = df["y_true"].astype(int).to_numpy()
    s = df[score_col].astype(float).to_numpy()
    m = classification_metrics(y, s, threshold)
    return {
        "model_id": model_id,
        "model": model,
        "protocol": protocol,
        "n_test": int(len(df)),
        "auroc": m["auroc"],
        "auprc": m["auprc"],
        "f1": m["f1"],
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "precision": m["precision"],
        "balanced_accuracy": m["balanced_accuracy"],
        "threshold": threshold,
    }


def main() -> None:
    REV_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    val = scores[scores["split"].eq("val")].copy()
    test = scores[scores["split"].eq("test")].copy()
    rows: list[dict[str, object]] = []
    score_map: dict[str, pd.DataFrame] = {}

    rasa_thr = select_threshold_max_f1(val["y_true"].to_numpy(), val["risk_score"].to_numpy())
    rasa = test[["case_id", "center_id", "y_true", "risk_score"]].rename(columns={"risk_score": "score"})
    score_map["mosaic_rasa"] = rasa
    rows.append(make_row("mosaic_rasa", "MOSAIC-RASA", "primary held-out protocol; validation threshold", test, "risk_score", rasa_thr))

    full = test[["case_id", "center_id", "y_true", "semantic_fusion_score"]].rename(columns={"semantic_fusion_score": "score"})
    score_map["mosaic"] = full
    rows.append(make_row("mosaic", "MOSAIC", "primary held-out protocol; train-only semantic retrieval; validation-calibrated fusion", test, "semantic_fusion_score", 0.50))

    if not ABLATION_PRED.exists():
        raise FileNotFoundError("Run evaluate_semantic_tag_source_ablations.py first")
    tag_pred = pd.read_csv(ABLATION_PRED)
    tag_test = tag_pred[(tag_pred["variant_id"].eq("all_rule_tags")) & (tag_pred["split"].eq("test"))].copy()
    tag_metrics = pd.read_csv(ABLATION_METRICS)
    tag_thr = float(tag_metrics[tag_metrics["variant_id"].eq("all_rule_tags")].iloc[0]["selected_threshold"])
    tag_test = tag_test.rename(columns={"fusion_score": "score"})
    tag_test["center_id"] = tag_test["center_id"].astype(str)
    score_map["mosaic_tag"] = tag_test[["case_id", "center_id", "y_true", "score"]]
    rows.append(
        make_row(
            "mosaic_tag",
            "MOSAIC-Tag",
            "secondary semantic-tag audit; train-only tag bank; validation-calibrated fusion",
            tag_test,
            "score",
            tag_thr,
        )
    )

    for model_id, (model_name, path) in BASELINE_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path).rename(columns={"y_true_cin2plus": "y_true", "center": "center_id"})
        thr = float(df["threshold_val_selected"].iloc[0])
        score_map[model_id] = df[["case_id", "center_id", "y_true", "risk_score"]].rename(columns={"risk_score": "score"})
        rows.append(make_row(model_id, model_name, "same-split external baseline; validation max-F1 threshold", df, "risk_score", thr))

    metrics = pd.DataFrame(rows)

    comparators = {
        "delta_auc_vs_mosaic": "mosaic",
        "delta_auc_vs_clip": "clip_contrastive",
        "delta_auc_vs_rasa": "mosaic_rasa",
    }
    for metric_idx, row in metrics.iterrows():
        model_id = row["model_id"]
        candidate = score_map[model_id]
        for prefix, comp_id in comparators.items():
            comp = score_map.get(comp_id)
            if comp is None:
                continue
            merged = comp[["case_id", "y_true", "score"]].merge(
                candidate[["case_id", "y_true", "score"]],
                on="case_id",
                suffixes=("_comp", "_cand"),
            )
            if not (merged["y_true_comp"].to_numpy() == merged["y_true_cand"].to_numpy()).all():
                raise ValueError(f"y_true mismatch for {model_id} vs {comp_id}")
            boot = (
                {
                    "delta_auc": 0.0,
                    "delta_auc_ci_low": 0.0,
                    "delta_auc_ci_high": 0.0,
                    "paired_bootstrap_p_two_sided": 1.0,
                    "bootstrap_samples": 2000,
                }
                if model_id == comp_id
                else paired_auc_bootstrap(
                    merged["y_true_comp"].to_numpy(),
                    merged["score_comp"].to_numpy(),
                    merged["score_cand"].to_numpy(),
                )
            )
            metrics.loc[metric_idx, prefix] = boot["delta_auc"]
            metrics.loc[metric_idx, f"{prefix}_ci_low"] = boot["delta_auc_ci_low"]
            metrics.loc[metric_idx, f"{prefix}_ci_high"] = boot["delta_auc_ci_high"]
            metrics.loc[metric_idx, f"{prefix}_p"] = boot["paired_bootstrap_p_two_sided"]

    metrics = metrics.sort_values("auroc", ascending=False)
    metrics.to_csv(REV_DIR / "mosaic_tag_vs_baselines.csv", index=False)
    lines = [
        "# MOSAIC-Tag Versus Baselines",
        "",
        "Positive paired deltas favour the row model over the named comparator. MOSAIC-Tag is treated as a secondary semantic-tag audit, not as a replacement for the primary MOSAIC framework.",
        "",
        markdown_table(metrics),
    ]
    (REV_DIR / "mosaic_tag_vs_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REV_DIR / 'mosaic_tag_vs_baselines.csv'}")
    print(f"Wrote {REV_DIR / 'mosaic_tag_vs_baselines.md'}")


if __name__ == "__main__":
    main()
