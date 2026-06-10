#!/usr/bin/env python3
"""Analyze topic-guided LCAD-RASA against the locked full baseline."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models_publishable.lcad_rasa_model import PublishableLCADRASA, instr_vector, load_visual_emb


OUT = ROOT / "outputs/publishable/topic_aux_analysis"
MANIFEST = ROOT / "outputs/publishable/manifests/full_manifest_publishable_with_report_topics.csv"
BASELINE_CKPT = ROOT / "outputs/publishable/checkpoints/publishable_full_lcad_rasa/best.ckpt"
TOPIC_CKPT = ROOT / "outputs/publishable/checkpoints/publishable_full_lcad_rasa_topic_aux/best.ckpt"
BASELINE_TABLE = ROOT / "outputs/publishable/tables/publishable_full_lcad_rasa"
TOPIC_TABLE = ROOT / "outputs/publishable/tables/publishable_full_lcad_rasa_topic_aux"
TOPIC_CATALOG = ROOT / "outputs/publishable/report_topic_distiller/tables/report_topic_catalog.csv"


def load_model(ckpt_path: Path, device: torch.device) -> PublishableLCADRASA:
    state = torch.load(ckpt_path, map_location="cpu")
    model_state = state["model"]
    topic_weight = model_state.get("topic_head.weight")
    if topic_weight is not None:
        model = PublishableLCADRASA(use_topic_head=True, num_report_topics=int(topic_weight.shape[0]))
    else:
        model = PublishableLCADRASA()
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def score_split(model: PublishableLCADRASA, df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        oct_e = torch.tensor(load_visual_emb(str(row.get("oct_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
        col_e = torch.tensor(load_visual_emb(str(row.get("colposcopy_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
        fus_e = torch.tensor(load_visual_emb(str(row.get("fused_visual_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
        instr = torch.tensor(instr_vector(row.to_dict()), dtype=torch.float32, device=device).unsqueeze(0)
        ids = torch.zeros(1, 64, dtype=torch.long, device=device)
        lab = torch.tensor([int(row["binary_label"])], device=device)
        out = model(oct_e, col_e, fus_e, instr, ids, lab)
        risk = float(torch.sigmoid(out["risk_logit"]).item()) if out.get("risk_logit") is not None else np.nan
        topic_pred = -1
        topic_prob = np.nan
        if out.get("report_topic_logits") is not None:
            prob = F.softmax(out["report_topic_logits"], dim=-1).squeeze(0)
            topic_pred = int(prob.argmax().item())
            topic_prob = float(prob.max().item())
        rows.append(
            {
                "case_id": row["case_id"],
                "split": row["split"],
                "center_id": row["center_id"],
                "y_true": int(row["binary_label"]),
                "risk_score": risk,
                "report_topic_id": int(row.get("report_topic_id", -1)),
                "report_topic_confidence": float(row.get("report_topic_confidence", 0.0)),
                "topic_pred": topic_pred,
                "topic_pred_confidence": topic_prob,
            }
        )
    return pd.DataFrame(rows)


def threshold_grid() -> np.ndarray:
    return np.round(np.arange(0.05, 0.96, 0.01), 2)


def select_val_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for thr in threshold_grid():
        f1 = f1_score(y_true, y_score >= thr, zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr


def select_threshold_policies(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    policies = {"max_f1": 0.5, "youden": 0.5, "sens80": 0.5, "sens90": 0.5, "spec70": 0.5}
    best_f1, best_youden = -1.0, -1.0
    best_s80_spec, best_s90_spec, best_spec70_sens = -1.0, -1.0, -1.0
    for thr in threshold_grid():
        m = metrics(y_true, y_score, float(thr))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            policies["max_f1"] = float(thr)
        youden = m["sensitivity"] + m["balanced_accuracy"] * 2 - m["sensitivity"] - 1
        # Equivalent to sensitivity + specificity - 1, derived from balanced accuracy.
        if youden > best_youden:
            best_youden = youden
            policies["youden"] = float(thr)
        specificity = max(0.0, min(1.0, 2 * m["balanced_accuracy"] - m["sensitivity"]))
        if m["sensitivity"] >= 0.80 and specificity > best_s80_spec:
            best_s80_spec = specificity
            policies["sens80"] = float(thr)
        if m["sensitivity"] >= 0.90 and specificity > best_s90_spec:
            best_s90_spec = specificity
            policies["sens90"] = float(thr)
        if specificity >= 0.70 and m["sensitivity"] > best_spec70_sens:
            best_spec70_sens = m["sensitivity"]
            policies["spec70"] = float(thr)
    return policies


def metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (y_score >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "auprc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "threshold_val_max_f1": float(threshold),
    }


def paired_auc_bootstrap(
    y_true: np.ndarray,
    baseline_score: np.ndarray,
    topic_score: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs = float(roc_auc_score(y_true, topic_score) - roc_auc_score(y_true, baseline_score))
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        deltas.append(float(roc_auc_score(y_true[idx], topic_score[idx]) - roc_auc_score(y_true[idx], baseline_score[idx])))
    arr = np.asarray(deltas, dtype=float)
    if len(arr) == 0:
        return {"delta_auc": obs, "delta_auc_ci_low": np.nan, "delta_auc_ci_high": np.nan, "paired_bootstrap_p_two_sided": np.nan}
    p = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
    p = min(1.0, max(1.0 / len(arr) if p == 0 else p, p))
    return {
        "delta_auc": obs,
        "delta_auc_ci_low": float(np.quantile(arr, 0.025)),
        "delta_auc_ci_high": float(np.quantile(arr, 0.975)),
        "paired_bootstrap_p_two_sided": float(p),
        "bootstrap_samples": int(len(arr)),
    }


def load_metric_table(root: Path) -> dict[str, float]:
    out = {}
    for name in ["eval_reference_based.csv", "eval_clinical_consistency.csv", "eval_report_metrics.csv"]:
        p = root / name
        if p.is_file():
            row = pd.read_csv(p).iloc[0].to_dict()
            for k, v in row.items():
                if isinstance(v, (int, float, np.number)):
                    out[f"{p.stem}.{k}"] = float(v)
    return out


def simple_markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{float_digits}f}")
        else:
            view[col] = view[col].astype(str)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST)
    device = _select_device(args.device)
    models = {
        "full_lcad_rasa": load_model(BASELINE_CKPT, device),
        "topic_aux": load_model(TOPIC_CKPT, device),
    }
    prediction_paths = {}
    risk_rows = []
    for model_id, model in models.items():
        pred_frames = []
        for split in ["val", "test"]:
            scored = score_split(model, df[df["split"].astype(str).eq(split)].copy(), device)
            scored["model_id"] = model_id
            pred_frames.append(scored)
        pred = pd.concat(pred_frames, ignore_index=True)
        path = OUT / f"{model_id}_val_test_scores.csv"
        pred.to_csv(path, index=False)
        prediction_paths[model_id] = path
        val = pred[pred["split"].eq("val")]
        test = pred[pred["split"].eq("test")]
        thr = select_val_threshold(val["y_true"].to_numpy(), val["risk_score"].to_numpy())
        row = {
            "model_id": model_id,
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            **metrics(test["y_true"].to_numpy(), test["risk_score"].to_numpy(), thr),
        }
        if model_id == "topic_aux":
            valid = test["report_topic_id"].ge(0)
            topic_acc = (test.loc[valid, "topic_pred"] == test.loc[valid, "report_topic_id"]).mean()
            weighted = (
                ((test.loc[valid, "topic_pred"] == test.loc[valid, "report_topic_id"]).astype(float) * test.loc[valid, "report_topic_confidence"]).sum()
                / test.loc[valid, "report_topic_confidence"].sum()
            )
            row.update(
                {
                    "topic_acc_test": float(topic_acc),
                    "topic_conf_weighted_acc_test": float(weighted),
                    "mean_topic_pred_confidence_test": float(test["topic_pred_confidence"].mean()),
                }
            )
        risk_rows.append(row)
    risk_table = pd.DataFrame(risk_rows)
    risk_table.to_csv(OUT / "topic_aux_risk_comparison.csv", index=False)

    threshold_rows = []
    for model_id, pred_path in prediction_paths.items():
        pred = pd.read_csv(pred_path)
        val = pred[pred["split"].eq("val")]
        test = pred[pred["split"].eq("test")]
        policies = select_threshold_policies(val["y_true"].to_numpy(), val["risk_score"].to_numpy())
        for policy, thr in policies.items():
            threshold_rows.append(
                {
                    "model_id": model_id,
                    "threshold_policy": policy,
                    **metrics(test["y_true"].to_numpy(), test["risk_score"].to_numpy(), float(thr)),
                }
            )
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(OUT / "topic_aux_threshold_policy_sensitivity.csv", index=False)

    base_pred = pd.read_csv(prediction_paths["full_lcad_rasa"])
    topic_pred = pd.read_csv(prediction_paths["topic_aux"])
    test_base = base_pred[base_pred["split"].eq("test")].sort_values("case_id")
    test_topic = topic_pred[topic_pred["split"].eq("test")].sort_values("case_id")
    merged = test_base[["case_id", "center_id", "y_true", "risk_score"]].merge(
        test_topic[["case_id", "risk_score"]],
        on="case_id",
        suffixes=("_full", "_topic_aux"),
    )
    bootstrap = paired_auc_bootstrap(
        merged["y_true"].to_numpy(),
        merged["risk_score_full"].to_numpy(),
        merged["risk_score_topic_aux"].to_numpy(),
    )
    (OUT / "topic_aux_paired_auc_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")

    center_rows = []
    thresholds = dict(zip(risk_table["model_id"], risk_table["threshold_val_max_f1"]))
    for model_id, pred_path in prediction_paths.items():
        pred = pd.read_csv(pred_path)
        test = pred[pred["split"].eq("test")]
        thr = float(thresholds[model_id])
        for center, part in test.groupby("center_id"):
            row = {"model_id": model_id, "center_id": center, "n": int(len(part))}
            if len(part["y_true"].unique()) > 1:
                row.update(metrics(part["y_true"].to_numpy(), part["risk_score"].to_numpy(), thr))
            else:
                row.update({"auc": np.nan, "auprc": np.nan, "f1": np.nan, "sensitivity": np.nan, "precision": np.nan, "balanced_accuracy": np.nan, "threshold_val_max_f1": thr})
            center_rows.append(row)
    center_table = pd.DataFrame(center_rows)
    center_table.to_csv(OUT / "topic_aux_centerwise_risk_comparison.csv", index=False)

    text_rows = []
    for model_id, root in [("full_lcad_rasa", BASELINE_TABLE), ("topic_aux", TOPIC_TABLE)]:
        text_rows.append({"model_id": model_id, **load_metric_table(root)})
    text_table = pd.DataFrame(text_rows)
    text_table.to_csv(OUT / "topic_aux_report_metric_comparison.csv", index=False)

    curve_rows = []
    for model_id, root in [("full_lcad_rasa", BASELINE_TABLE), ("topic_aux", TOPIC_TABLE)]:
        p = root / "training_curve.csv"
        if p.is_file():
            c = pd.read_csv(p)
            c["model_id"] = model_id
            curve_rows.append(c)
    if curve_rows:
        pd.concat(curve_rows, ignore_index=True).to_csv(OUT / "topic_aux_training_curve_comparison.csv", index=False)

    catalog_text = ""
    if TOPIC_CATALOG.is_file():
        catalog = pd.read_csv(TOPIC_CATALOG)
        catalog_text = "\n".join(
            f"- Topic {int(r.report_topic_id)} (train n={int(r.n_train)}): {str(r.top_terms)[:160]}"
            for r in catalog.itertuples(index=False)
        )

    base = risk_table[risk_table["model_id"].eq("full_lcad_rasa")].iloc[0]
    topic = risk_table[risk_table["model_id"].eq("topic_aux")].iloc[0]
    delta_auc = topic["auc"] - base["auc"]
    delta_f1 = topic["f1"] - base["f1"]
    if delta_auc > 0.01 and delta_f1 >= -0.02:
        decision = "candidate_method_with_threshold_calibration"
    elif delta_auc > 0:
        decision = "mechanistic_ablation_until_threshold_or_sensitivity_recovers"
    else:
        decision = "do_not_promote_yet"

    summary = [
        "# Topic-Auxiliary LCAD-RASA Analysis",
        "",
        "## Training",
        "",
        f"- Baseline checkpoint: `{BASELINE_CKPT}`",
        f"- Topic-aux checkpoint: `{TOPIC_CKPT}`",
        f"- Prediction tables: `{prediction_paths['full_lcad_rasa']}`, `{prediction_paths['topic_aux']}`",
        "",
        "## Held-Out Risk Metrics",
        "",
        simple_markdown_table(risk_table),
        "",
        "## Paired AUROC Bootstrap",
        "",
        simple_markdown_table(pd.DataFrame([bootstrap])),
        "",
        "## Center-Wise Risk Metrics",
        "",
        simple_markdown_table(center_table),
        "",
        "## Threshold Policy Sensitivity",
        "",
        simple_markdown_table(threshold_table),
        "",
        "## Report Metrics",
        "",
        simple_markdown_table(text_table),
        "",
        "## Topic Catalog",
        "",
        catalog_text,
        "",
        "## Decision",
        "",
        f"- Delta AUROC topic_aux - full: {delta_auc:.4f}",
        f"- Delta F1 topic_aux - full: {delta_f1:.4f}",
        f"- Decision: `{decision}`",
        "",
        "Interpretation: topic auxiliary guidance can be discussed as a candidate method component only if it preserves held-out discrimination and improves report/section behavior. Otherwise keep it as a mechanistic ablation.",
    ]
    (OUT / "TOPIC_AUX_EXPERIMENT_ANALYSIS.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote analysis to {OUT}")


if __name__ == "__main__":
    main()
