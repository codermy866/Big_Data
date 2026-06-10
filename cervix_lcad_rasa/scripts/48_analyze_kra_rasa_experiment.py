#!/usr/bin/env python3
"""Analyze KRA-RASA against full LCAD-RASA and topic-aux baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models_publishable.lcad_rasa_model import PublishableLCADRASA, instr_vector, load_semantic_emb, load_visual_emb


def load_model(ckpt_path: Path, device: torch.device) -> PublishableLCADRASA:
    state = torch.load(ckpt_path, map_location="cpu")
    model_state = state["model"]
    topic_weight = model_state.get("topic_head.weight")
    use_semantic = any(k.startswith("semantic_proj.") or k.startswith("semantic_token_packer.") for k in model_state)
    model = PublishableLCADRASA(
        use_topic_head=topic_weight is not None,
        num_report_topics=int(topic_weight.shape[0]) if topic_weight is not None else 0,
        use_semantic_retrieval=use_semantic,
    )
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
        sem_e = torch.tensor(load_semantic_emb(str(row.get("semantic_retrieval_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
        instr = torch.tensor(instr_vector(row.to_dict()), dtype=torch.float32, device=device).unsqueeze(0)
        ids = torch.zeros(1, 64, dtype=torch.long, device=device)
        lab = torch.tensor([int(row["binary_label"])], device=device)
        out = model(oct_e, col_e, fus_e, instr, ids, lab, semantic_emb=sem_e)
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
                "semantic_retrieval_score": float(row.get("semantic_retrieval_score", 0.0)),
                "semantic_retrieval_positive_ratio": float(row.get("semantic_retrieval_positive_ratio", 0.0)),
                "semantic_retrieval_section_coverage": float(row.get("semantic_retrieval_section_coverage", 0.0)),
                "report_topic_id": int(row.get("report_topic_id", -1)),
                "topic_pred": topic_pred,
                "topic_pred_confidence": topic_prob,
            }
        )
    return pd.DataFrame(rows)


def threshold_grid() -> np.ndarray:
    return np.round(np.arange(0.05, 0.96, 0.01), 2)


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


def select_val_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for thr in threshold_grid():
        f1 = f1_score(y_true, y_score >= thr, zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr


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
    if arr.size == 0:
        return {"delta_auc": obs, "delta_auc_ci_low": np.nan, "delta_auc_ci_high": np.nan, "paired_bootstrap_p_two_sided": np.nan}
    p = 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
    return {
        "delta_auc": obs,
        "delta_auc_ci_low": float(np.quantile(arr, 0.025)),
        "delta_auc_ci_high": float(np.quantile(arr, 0.975)),
        "paired_bootstrap_p_two_sided": float(min(1.0, max(1.0 / arr.size if p == 0 else p, p))),
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


def load_metric_table(root: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ["eval_reference_based.csv", "eval_clinical_consistency.csv", "eval_report_metrics.csv"]:
        p = root / name
        if p.is_file():
            row = pd.read_csv(p).iloc[0].to_dict()
            for k, v in row.items():
                if isinstance(v, (int, float, np.number)):
                    out[f"{p.stem}.{k}"] = float(v)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/publishable/manifests/full_manifest_publishable_with_kra_semantic.csv")
    parser.add_argument("--output-dir", default="outputs/publishable/kra_rasa_analysis")
    parser.add_argument("--baseline-checkpoint", default="outputs/publishable/checkpoints/publishable_full_lcad_rasa/best.ckpt")
    parser.add_argument("--topic-checkpoint", default="outputs/publishable/checkpoints/publishable_full_lcad_rasa_topic_aux/best.ckpt")
    parser.add_argument("--kra-checkpoint", default="outputs/publishable/checkpoints/publishable_kra_rasa/best.ckpt")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.output_dir)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    def _resolve_checkpoint(path_text: str) -> Path:
        p = Path(path_text)
        return p if p.is_absolute() else ROOT / p

    model_specs = {
        "full_lcad_rasa": _resolve_checkpoint(args.baseline_checkpoint),
        "topic_aux": _resolve_checkpoint(args.topic_checkpoint),
        "kra_rasa": _resolve_checkpoint(args.kra_checkpoint),
    }
    preds = {}
    risk_rows = []
    for model_id, ckpt in model_specs.items():
        model = load_model(ckpt, device)
        pred = pd.concat(
            [score_split(model, df[df["split"].astype(str).eq(split)], device).assign(model_id=model_id) for split in ["val", "test"]],
            ignore_index=True,
        )
        pred_path = out_dir / f"{model_id}_val_test_scores.csv"
        pred.to_csv(pred_path, index=False)
        preds[model_id] = pred
        val = pred[pred["split"].eq("val")]
        test = pred[pred["split"].eq("test")]
        thr = select_val_threshold(val["y_true"].to_numpy(), val["risk_score"].to_numpy())
        row = {
            "model_id": model_id,
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            **metrics(test["y_true"].to_numpy(), test["risk_score"].to_numpy(), thr),
        }
        if model_id == "kra_rasa":
            row.update(
                {
                    "mean_semantic_retrieval_score_test": float(test["semantic_retrieval_score"].mean()),
                    "mean_semantic_positive_ratio_test": float(test["semantic_retrieval_positive_ratio"].mean()),
                    "mean_section_coverage_test": float(test["semantic_retrieval_section_coverage"].mean()),
                }
            )
        risk_rows.append(row)
    risk_table = pd.DataFrame(risk_rows)
    risk_table.to_csv(out_dir / "kra_rasa_risk_comparison.csv", index=False)

    base_test = preds["full_lcad_rasa"][preds["full_lcad_rasa"]["split"].eq("test")].sort_values("case_id")
    kra_test = preds["kra_rasa"][preds["kra_rasa"]["split"].eq("test")].sort_values("case_id")
    merged = base_test[["case_id", "center_id", "y_true", "risk_score"]].merge(
        kra_test[["case_id", "risk_score"]], on="case_id", suffixes=("_full", "_kra")
    )
    bootstrap = paired_auc_bootstrap(merged["y_true"].to_numpy(), merged["risk_score_full"].to_numpy(), merged["risk_score_kra"].to_numpy())
    (out_dir / "kra_vs_full_paired_auc_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")

    thresholds = dict(zip(risk_table["model_id"], risk_table["threshold_val_max_f1"]))
    center_rows = []
    for model_id, pred in preds.items():
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
    center_table.to_csv(out_dir / "kra_rasa_centerwise_risk_comparison.csv", index=False)

    report_rows = []
    for model_id in ["full_lcad_rasa", "topic_aux", "kra_rasa"]:
        table_id = model_specs[model_id].parent.name
        report_rows.append({"model_id": model_id, **load_metric_table(ROOT / "outputs/publishable/tables" / table_id)})
    report_table = pd.DataFrame(report_rows)
    report_table.to_csv(out_dir / "kra_rasa_report_metric_comparison.csv", index=False)

    base = risk_table[risk_table["model_id"].eq("full_lcad_rasa")].iloc[0]
    kra = risk_table[risk_table["model_id"].eq("kra_rasa")].iloc[0]
    delta_auc = float(kra["auc"] - base["auc"])
    delta_f1 = float(kra["f1"] - base["f1"])
    decision = "candidate_main_method" if delta_auc > 0.01 and delta_f1 >= -0.02 else "promising_but_requires_ablation" if delta_auc > 0 else "do_not_promote_without_revision"

    summary = [
        "# KRA-RASA Experiment Analysis",
        "",
        "## Method Framing",
        "",
        "KRA-RASA adapts STREAM-style semantic retrieval to cervical multimodal analytics. A train-only report-derived knowledge bank is queried by case-level clinical and visual signatures. Retrieved section entities are packed with OCT, colposcopy, fused visual, and instruction tokens using a semantic token packer.",
        "",
        "## Held-Out Risk Metrics",
        "",
        markdown_table(risk_table),
        "",
        "## Paired AUROC Bootstrap: KRA-RASA - Full LCAD-RASA",
        "",
        markdown_table(pd.DataFrame([bootstrap])),
        "",
        "## Center-Wise Risk Metrics",
        "",
        markdown_table(center_table),
        "",
        "## Report Metrics",
        "",
        markdown_table(report_table),
        "",
        "## Decision",
        "",
        f"- Delta AUROC KRA-RASA - full: {delta_auc:.4f}",
        f"- Delta F1 KRA-RASA - full: {delta_f1:.4f}",
        f"- Decision: `{decision}`",
        "",
        "Use this as a candidate main method only if the paired and center-wise analyses remain defensible after retrieval ablations.",
    ]
    (out_dir / "KRA_RASA_EXPERIMENT_ANALYSIS.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Wrote KRA-RASA analysis to {out_dir}")


if __name__ == "__main__":
    main()
