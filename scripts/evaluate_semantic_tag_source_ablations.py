#!/usr/bin/env python3
"""Evaluate semantic-tag source ablations under train-only retrieval."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from llm_semantic_common import (
    RASA_SCORE_CANDIDATES,
    auc_score,
    average_precision,
    classification_metrics,
    first_existing,
    fused_score,
    markdown_table,
    paired_auc_bootstrap,
    select_threshold_max_f1,
    split_tags,
    tokenize_tag_text,
)


ROOT = Path(__file__).resolve().parents[1]
SEM_DIR = ROOT / "outputs" / "llm_semantic"
REV_DIR = ROOT / "outputs" / "revision"
INPUT = SEM_DIR / "semantic_tagging_input.csv"
TAGS = SEM_DIR / "rule_semantic_tags.csv"

TAG_COLUMNS = ["oct_tags", "colposcopy_tags", "clinical_tags", "impression_tags", "severity_tags"]
DIAGNOSTIC_TOKENS = {
    "hsil",
    "lsil",
    "cin",
    "cin2",
    "cin3",
    "cin2+",
    "cin3+",
    "cancer",
    "carcinoma",
    "malignant",
    "benign",
    "invasive",
    "high",
    "high_grade",
    "low",
    "low_grade",
    "low_suspicion_language",
    "abnormal_or_suspicious_language",
    "elevated_attention_language",
    "low_attention_language",
}


def load_scores() -> pd.DataFrame:
    path = first_existing(RASA_SCORE_CANDIDATES)
    if path is None:
        raise FileNotFoundError("Missing RASA score table")
    scores = pd.read_csv(path)
    required = {"case_id", "split", "center_id", "y_true", "risk_score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"RASA score table missing columns: {sorted(missing)}")
    return scores


def join_selected(row: pd.Series, cols: list[str], *, remove_diagnostic: bool = False) -> str:
    tags: list[str] = []
    for col in cols:
        tags.extend(split_tags(row.get(col)))
    cleaned: list[str] = []
    for tag in tags:
        token = tag.lower().strip()
        if remove_diagnostic and any(term in token for term in DIAGNOSTIC_TOKENS):
            continue
        if token and token not in cleaned:
            cleaned.append(token)
    return " ".join(cleaned)


def vectorise(train_texts: list[str], query_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    vocab_counter: Counter[str] = Counter()
    for text in train_texts:
        vocab_counter.update(set(tokenize_tag_text(text)))
    vocab = {tok: i for i, tok in enumerate(vocab_counter)}
    if not vocab:
        return np.zeros((len(train_texts), 1), dtype=float), np.zeros((len(query_texts), 1), dtype=float)

    def matrix(texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), len(vocab)), dtype=float)
        for r, text in enumerate(texts):
            for tok in set(tokenize_tag_text(text)):
                idx = vocab.get(tok)
                if idx is not None:
                    mat[r, idx] = 1.0
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        return mat / norms[:, None]

    return matrix(train_texts), matrix(query_texts)


def retrieve_priors(
    train: pd.DataFrame,
    query: pd.DataFrame,
    *,
    query_texts: list[str] | None = None,
    train_y_override: np.ndarray | None = None,
    same_centre_excluded: bool = False,
    seed: int = 20260616,
) -> np.ndarray:
    train_texts = train["variant_text"].tolist()
    q_texts = query["variant_text"].tolist() if query_texts is None else query_texts
    train_matrix, query_matrix = vectorise(train_texts, q_texts)
    sim = query_matrix @ train_matrix.T
    train_y = train["y_true"].astype(int).to_numpy() if train_y_override is None else np.asarray(train_y_override).astype(int)
    train_centres = train["center_id"].astype(str).to_numpy()
    query_centres = query["center_id"].astype(str).to_numpy()
    prevalence = float(train_y.mean()) if len(train_y) else 0.5
    priors = []
    for i in range(len(query)):
        sims = sim[i].copy()
        if same_centre_excluded:
            sims[train_centres == query_centres[i]] = -np.inf
        finite = np.isfinite(sims)
        if not finite.any():
            priors.append(prevalence)
            continue
        k = min(10, int(finite.sum()))
        order = np.argsort(-sims, kind="mergesort")[:k]
        top_sims = np.maximum(sims[order], 0.0)
        top_y = train_y[order]
        if float(top_sims.sum()) > 0:
            priors.append(float(np.average(top_y, weights=np.maximum(top_sims, 1e-6))))
        else:
            priors.append(float(top_y.mean()) if len(top_y) else prevalence)
    return np.asarray(priors, dtype=float)


def select_alpha(val: pd.DataFrame, retrieval_col: str) -> tuple[float, float]:
    y = val["y_true"].astype(int).to_numpy()
    best_alpha = 0.0
    best_auc = -1.0
    for alpha in np.linspace(0, 1, 101):
        score = fused_score(val["risk_score"].to_numpy(), val[retrieval_col].to_numpy(), float(alpha))
        auc = auc_score(y, score)
        if auc > best_auc:
            best_alpha = float(alpha)
            best_auc = float(auc)
    return best_alpha, best_auc


def metric_bundle(y_val: np.ndarray, s_val: np.ndarray, y_test: np.ndarray, s_test: np.ndarray) -> dict[str, float]:
    thr = select_threshold_max_f1(y_val, s_val)
    return classification_metrics(y_test, s_test, thr)


def make_variant_predictions(tags: pd.DataFrame, variant_id: str, config: dict[str, object]) -> pd.DataFrame:
    df = tags.copy()
    cols = list(config.get("cols", TAG_COLUMNS))
    remove_diag = bool(config.get("remove_diagnostic", False))
    df["variant_text"] = df.apply(lambda r: join_selected(r, cols, remove_diagnostic=remove_diag), axis=1)
    train = df[df["split"].eq("train")].copy()
    query = df[df["split"].isin(["val", "test"])].copy()
    rng = np.random.default_rng(20260616)

    if config.get("constant_train_prevalence", False):
        prior = float(train["y_true"].astype(int).mean())
        query["retrieval_prior"] = prior
    else:
        query_texts = None
        train_y_override = None
        if config.get("random_query", False):
            query_texts = query["variant_text"].sample(frac=1.0, random_state=20260616).tolist()
        if config.get("random_train_labels", False):
            train_y_override = rng.permutation(train["y_true"].astype(int).to_numpy())
        query["retrieval_prior"] = retrieve_priors(
            train,
            query,
            query_texts=query_texts,
            train_y_override=train_y_override,
            same_centre_excluded=bool(config.get("same_centre_excluded", False)),
        )
    query["variant_id"] = variant_id
    return query[["case_id", "split", "center_id", "y_true", "risk_score", "retrieval_prior", "variant_id"]].copy()


def main() -> None:
    REV_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_scores()
    inputs = pd.read_csv(INPUT)[["case_id", "split", "center_id", "y_true"]]
    tags = pd.read_csv(TAGS)
    tags = tags.merge(inputs, on=["case_id", "split", "center_id"], how="left")
    tags = tags.merge(scores[["case_id", "split", "risk_score"]], on=["case_id", "split"], how="left")

    variants = {
        "all_rule_tags": {"label": "MOSAIC-Tag all rule tags", "cols": TAG_COLUMNS},
        "no_impression_tags": {"label": "No impression tags", "cols": ["oct_tags", "colposcopy_tags", "clinical_tags", "severity_tags"]},
        "no_severity_tags": {"label": "No severity tags", "cols": ["oct_tags", "colposcopy_tags", "clinical_tags", "impression_tags"]},
        "modality_only_tags": {"label": "Modality-only tags", "cols": ["oct_tags", "colposcopy_tags"]},
        "clinical_only_tags": {"label": "Clinical-only tags", "cols": ["clinical_tags"]},
        "visual_only_tags": {"label": "Visual-only tags", "cols": ["oct_tags", "colposcopy_tags"]},
        "no_diagnostic_terminology": {"label": "No diagnostic terminology", "cols": TAG_COLUMNS, "remove_diagnostic": True},
        "train_label_prior_only": {"label": "Train-label prior only negative control", "cols": TAG_COLUMNS, "constant_train_prevalence": True},
        "random_tag_query": {"label": "Random tag query negative control", "cols": TAG_COLUMNS, "random_query": True},
        "random_train_label_bank": {"label": "Random train-label bank negative control", "cols": TAG_COLUMNS, "random_train_labels": True},
        "same_centre_excluded": {"label": "Same-centre excluded retrieval", "cols": TAG_COLUMNS, "same_centre_excluded": True},
    }

    all_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    boot_rows: list[dict[str, object]] = []
    all_rule_test_score: np.ndarray | None = None

    for variant_id, config in variants.items():
        pred = make_variant_predictions(tags, variant_id, config)
        val = pred[pred["split"].eq("val")].copy()
        test = pred[pred["split"].eq("test")].copy()
        alpha, val_auc = select_alpha(val, "retrieval_prior")
        val_fused = fused_score(val["risk_score"].to_numpy(), val["retrieval_prior"].to_numpy(), alpha)
        test_fused = fused_score(test["risk_score"].to_numpy(), test["retrieval_prior"].to_numpy(), alpha)
        val_y = val["y_true"].astype(int).to_numpy()
        test_y = test["y_true"].astype(int).to_numpy()
        retrieval_metrics = metric_bundle(val_y, val["retrieval_prior"].to_numpy(), test_y, test["retrieval_prior"].to_numpy())
        fusion_metrics = metric_bundle(val_y, val_fused, test_y, test_fused)
        pred["fusion_score"] = np.nan
        pred.loc[pred["split"].eq("val"), "fusion_score"] = val_fused
        pred.loc[pred["split"].eq("test"), "fusion_score"] = test_fused
        pred["alpha"] = alpha
        pred["threshold"] = fusion_metrics["threshold"]
        all_predictions.append(pred)

        if variant_id == "all_rule_tags":
            all_rule_test_score = test_fused.copy()

        boot_rasa = paired_auc_bootstrap(test_y, test["risk_score"].to_numpy(), test_fused)
        metric_row = {
            "variant_id": variant_id,
            "variant": config["label"],
            "protocol": "semantic-tag audit; train-only tag bank; validation-calibrated fusion; test metrics",
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "retrieval_auroc": retrieval_metrics["auroc"],
            "retrieval_auprc": retrieval_metrics["auprc"],
            "retrieval_f1": retrieval_metrics["f1"],
            "fusion_auroc": fusion_metrics["auroc"],
            "fusion_auprc": fusion_metrics["auprc"],
            "fusion_f1": fusion_metrics["f1"],
            "sensitivity": fusion_metrics["sensitivity"],
            "specificity": fusion_metrics["specificity"],
            "precision": fusion_metrics["precision"],
            "balanced_accuracy": fusion_metrics["balanced_accuracy"],
            "selected_alpha": alpha,
            "selected_threshold": fusion_metrics["threshold"],
            "validation_fusion_auroc": val_auc,
            "delta_auroc_vs_rasa": boot_rasa["delta_auc"],
            "delta_vs_rasa_ci_low": boot_rasa["delta_auc_ci_low"],
            "delta_vs_rasa_ci_high": boot_rasa["delta_auc_ci_high"],
            "p_vs_rasa": boot_rasa["paired_bootstrap_p_two_sided"],
        }
        metric_rows.append(metric_row)
        boot_rows.append({"variant_id": variant_id, "comparison": "fusion_vs_rasa_backbone", **boot_rasa})

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)

    if all_rule_test_score is not None:
        test_base = predictions[(predictions["variant_id"].eq("all_rule_tags")) & (predictions["split"].eq("test"))]
        y_base = test_base["y_true"].astype(int).to_numpy()
        for variant_id in metrics["variant_id"]:
            test = predictions[(predictions["variant_id"].eq(variant_id)) & (predictions["split"].eq("test"))]
            boot = (
                {
                    "delta_auc": 0.0,
                    "delta_auc_ci_low": 0.0,
                    "delta_auc_ci_high": 0.0,
                    "paired_bootstrap_p_two_sided": 1.0,
                    "bootstrap_samples": 2000,
                }
                if variant_id == "all_rule_tags"
                else paired_auc_bootstrap(y_base, all_rule_test_score, test["fusion_score"].to_numpy())
            )
            boot_rows.append({"variant_id": variant_id, "comparison": "fusion_vs_all_rule_tag_fusion", **boot})
            for key, col in [
                ("delta_auc", "delta_auroc_vs_all_rule"),
                ("delta_auc_ci_low", "delta_vs_all_rule_ci_low"),
                ("delta_auc_ci_high", "delta_vs_all_rule_ci_high"),
                ("paired_bootstrap_p_two_sided", "p_vs_all_rule"),
            ]:
                metrics.loc[metrics["variant_id"].eq(variant_id), col] = boot[key]

    metrics.to_csv(REV_DIR / "semantic_tag_source_ablation_metrics.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(REV_DIR / "semantic_tag_source_ablation_bootstrap.csv", index=False)
    predictions.to_csv(REV_DIR / "semantic_tag_source_ablation_predictions.csv", index=False)

    all_rule = metrics[metrics["variant_id"].eq("all_rule_tags")].iloc[0]
    no_diag = metrics[metrics["variant_id"].eq("no_diagnostic_terminology")].iloc[0]
    random_q = metrics[metrics["variant_id"].eq("random_tag_query")].iloc[0]
    random_label = metrics[metrics["variant_id"].eq("random_train_label_bank")].iloc[0]
    same_centre = metrics[metrics["variant_id"].eq("same_centre_excluded")].iloc[0]
    neg_pass = bool(random_q["fusion_auroc"] < all_rule["fusion_auroc"] - 0.02 and random_label["fusion_auroc"] < all_rule["fusion_auroc"] - 0.02)
    diag_robust = bool(no_diag["fusion_auroc"] >= all_rule["fusion_auroc"] - 0.03)
    lines = [
        "# Semantic-Tag Source Ablation Summary",
        "",
        f"All-rule MOSAIC-Tag fusion AUROC: {all_rule['fusion_auroc']:.3f}; AUPRC: {all_rule['fusion_auprc']:.3f}; F1: {all_rule['fusion_f1']:.3f}.",
        f"No-diagnostic-terminology AUROC: {no_diag['fusion_auroc']:.3f}.",
        f"Random-query negative control AUROC: {random_q['fusion_auroc']:.3f}.",
        f"Random training-label-bank negative control AUROC: {random_label['fusion_auroc']:.3f}.",
        f"Same-centre-excluded retrieval AUROC: {same_centre['fusion_auroc']:.3f}.",
        "",
        f"Negative controls passed: {'yes' if neg_pass else 'no'}.",
        f"No-diagnostic-terminology result remained close to all-rule fusion: {'yes' if diag_robust else 'no'}.",
        "",
        "## Metrics",
        "",
        markdown_table(metrics),
    ]
    if diag_robust and neg_pass:
        lines.append("\nInterpretation: MOSAIC-Tag is consistent with a lightweight semantic-prior effect under the available deterministic tag audit.")
    elif not neg_pass:
        lines.append("\nInterpretation: at least one negative control remained too strong; MOSAIC-Tag should be treated cautiously and not elevated to a primary claim.")
    else:
        lines.append("\nInterpretation: the tag result depends on high-level semantic descriptors and should be interpreted as a protocol audit rather than a replacement for MOSAIC.")
    (REV_DIR / "semantic_tag_source_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REV_DIR / 'semantic_tag_source_ablation_metrics.csv'}")
    print(f"Wrote {REV_DIR / 'semantic_tag_source_ablation_summary.md'}")


if __name__ == "__main__":
    main()
