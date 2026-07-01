#!/usr/bin/env python3
"""Generate supplementary missing-control artifacts for the RASA baseline package.

This script fills controls that do not require a new public VLP implementation:

* A0_clinical_mlp: a train/validation/test clinical MLP using age, HPV, and TCT.
* D5/E2/E3/E4/E5: fixed-backbone retrieval controls from train-only claim banks.
* F5: decision-curve analysis tables.

The script intentionally does not synthesize MGCA/GLoRIA/MedCLIP results. Those
rows require official model code and weights, so they remain blocked unless real
implementations are added.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cervix_lcad_rasa"))

from src.retrieval.llm_claim_bank import embeddings_matrix, query_embedding_from_row, retrieve_prior  # noqa: E402


MANIFEST = ROOT / "cervix_lcad_rasa" / "outputs" / "publishable" / "manifests" / "full_manifest_publishable.csv"
REAL_LLM_MANIFEST = ROOT / "cervix_lcad_rasa" / "outputs" / "real_llm_full_lfsct" / "full_all_splits" / "manifest_with_real_llm_claims.csv"
RASA_SCORES = ROOT / "cervix_lcad_rasa" / "outputs" / "publishable" / "kra_rasa_analysis" / "full_lcad_rasa_val_test_scores.csv"
CLAIM_BANK_ROOT = ROOT / "cervix_lcad_rasa" / "outputs" / "real_llm_full_lfsct" / "claim_banks"
OUT = ROOT / "outputs" / "rasa_public_baselines" / "supplementary_controls"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def auc_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    pos_ranks = ranks[y == 1].sum()
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if int(y.sum()) == 0:
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    recall = tp / max(tp[-1], 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def threshold_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    pred = (s >= threshold).astype(int)
    tp = float(((pred == 1) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sens / (precision + sens) if (precision + sens) > 0 else 0.0
    return {"f1": float(f1), "balanced_accuracy": float(np.nanmean([sens, spec]))}


def best_threshold_f1(y: np.ndarray, score: np.ndarray) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.01, 0.99, 99):
        f1 = threshold_metrics(y, score, float(t))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def fuse(base: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip((1.0 - alpha) * base + alpha * prior, 0.0, 1.0)


def select_alpha(y_val: np.ndarray, base_val: np.ndarray, prior_val: np.ndarray) -> float:
    best_alpha = 0.0
    best_value = -1.0
    for alpha in np.linspace(0.0, 0.6, 31):
        score = fuse(base_val, prior_val, float(alpha))
        t = best_threshold_f1(y_val, score)
        value = threshold_metrics(y_val, score, t)["balanced_accuracy"]
        if value > best_value:
            best_value = value
            best_alpha = float(alpha)
    return best_alpha


def write_predictions(df: pd.DataFrame, experiment_id: str) -> Path:
    out_dir = OUT / "predictions" / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "all_predictions.csv"
    df.to_csv(path, index=False)
    return path


def train_clinical_mlp(seed: int) -> Path:
    import torch
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    df = pd.read_csv(MANIFEST)
    feature_cols = ["age", "hpv", "tct"]
    work = df[["case_id", "center_id", "split", "binary_label"] + feature_cols].copy()
    work["age"] = pd.to_numeric(work["age"], errors="coerce")
    for col in ["hpv", "tct"]:
        work[col] = work[col].fillna("missing").astype(str)

    train = work[work["split"].astype(str).eq("train")].copy()
    val = work[work["split"].astype(str).eq("val")].copy()
    test = work[work["split"].astype(str).eq("test")].copy()
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["age"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["hpv", "tct"]),
        ]
    )
    x_train = pre.fit_transform(train[feature_cols]).astype(np.float32)
    x_val = pre.transform(val[feature_cols]).astype(np.float32)
    x_test = pre.transform(test[feature_cols]).astype(np.float32)
    y_train = train["binary_label"].astype(int).to_numpy()
    y_val = val["binary_label"].astype(int).to_numpy()

    class ClinicalMLP(torch.nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.15),
                torch.nn.Linear(64, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClinicalMLP(x_train.shape[1]).to(device)
    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    xtr = torch.tensor(x_train, device=device)
    ytr = torch.tensor(y_train.astype(np.float32), device=device)
    best_state = None
    best_auc = -1.0
    for _ in range(240):
        model.train()
        idx = rng.permutation(len(x_train))
        for start in range(0, len(idx), 128):
            batch = idx[start : start + 128]
            logits = model(xtr[batch])
            loss = loss_fn(logits, ytr[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = sigmoid(model(torch.tensor(x_val, device=device)).detach().cpu().numpy())
        auc = auc_rank(y_val, pv)
        if math.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(x: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return sigmoid(model(torch.tensor(x, device=device)).detach().cpu().numpy())

    p_train, p_val, p_test = predict(x_train), predict(x_val), predict(x_test)
    threshold = best_threshold_f1(y_val, p_val)
    frames = []
    for split, sub, prob in [("train", train, p_train), ("val", val, p_val), ("test", test, p_test)]:
        frames.append(
            pd.DataFrame(
                {
                    "case_id": sub["case_id"].astype(str).to_numpy(),
                    "center_id": sub["center_id"].astype(str).to_numpy(),
                    "split": split,
                    "y_cin2plus": sub["binary_label"].astype(int).to_numpy(),
                    "p_cin2plus": prob,
                    "threshold": threshold,
                    "alpha": np.nan,
                }
            )
        )
    pred = pd.concat(frames, ignore_index=True)
    path = write_predictions(pred, "A0_clinical_mlp")
    meta = {
        "experiment_id": "A0_clinical_mlp",
        "features": feature_cols,
        "best_val_auc": best_auc,
        "threshold_val_f1": threshold,
        "test_auc": auc_rank(test["binary_label"].astype(int).to_numpy(), p_test),
        "test_auprc": auprc_score(test["binary_label"].astype(int).to_numpy(), p_test),
        "path": str(path),
    }
    (OUT / "clinical_mlp_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_bank(name: str, dim: int = 256) -> tuple[pd.DataFrame, np.ndarray]:
    bank_dir = CLAIM_BANK_ROOT / name
    bank = pd.read_csv(bank_dir / "claim_bank.csv")
    vec_path = bank_dir / "claim_bank_vectors.npy"
    vecs = np.load(vec_path) if vec_path.exists() else embeddings_matrix(bank, dim=dim)
    return bank, vecs.astype(np.float32)


def retrieve_for_scores(
    scores: pd.DataFrame,
    manifest: pd.DataFrame,
    bank: pd.DataFrame,
    vecs: np.ndarray,
    *,
    mode: str,
    seed: int,
    top_k: int,
    dim: int = 256,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    work_bank = bank.copy()
    work_vecs = vecs.copy()
    if mode == "shuffled_claim_bank":
        work_vecs = work_vecs[rng.permutation(len(work_vecs))]
    elif mode == "shuffled_memory_bank":
        work_bank["risk_label_for_prior"] = rng.permutation(work_bank["risk_label_for_prior"].astype(float).to_numpy())
    train_prev = float(work_bank["risk_label_for_prior"].astype(float).mean())
    merged = scores.merge(manifest, on="case_id", how="left", suffixes=("", "_manifest"))
    priors: list[float] = []
    for _, row in merged.iterrows():
        if mode == "random_retrieval":
            priors.append(float(rng.random()))
            continue
        if mode == "label_prior_retrieval":
            priors.append(train_prev)
            continue
        if mode == "centre_restricted_retrieval":
            center = str(row.get("center_id", row.get("center_id_manifest", "")))
            mask = work_bank.get("center_id_for_audit_only", pd.Series([""] * len(work_bank))).astype(str).eq(center)
            sub_bank = work_bank[mask].copy()
            sub_vecs = work_vecs[mask.to_numpy()] if mask.any() else work_vecs
            if sub_bank.empty:
                sub_bank = work_bank
            prior, _, _ = retrieve_prior(query_embedding_from_row(row, dim=dim), sub_bank, sub_vecs, top_k=top_k)
            priors.append(prior)
            continue
        prior, _, _ = retrieve_prior(query_embedding_from_row(row, dim=dim), work_bank, work_vecs, top_k=top_k)
        priors.append(prior)
    return np.asarray(priors, dtype=float)


def run_retrieval_controls(seed: int, top_k: int) -> dict[str, str]:
    manifest = pd.read_csv(REAL_LLM_MANIFEST)
    scores = pd.read_csv(RASA_SCORES)
    if "y_true" not in scores.columns and "y_true_cin2plus" in scores.columns:
        scores = scores.rename(columns={"y_true_cin2plus": "y_true"})
    scores = scores[scores["split"].astype(str).isin(["val", "test"])].copy()
    bank, vecs = load_bank("real_llm_claim_bank")
    outputs: dict[str, str] = {}
    mapping = {
        "D5_shuffled_claim_bank": "shuffled_claim_bank",
        "E2_random_retrieval": "random_retrieval",
        "E3_shuffled_memory_bank": "shuffled_memory_bank",
        "E4_label_prior_retrieval": "label_prior_retrieval",
        "E5_centre_restricted_retrieval": "centre_restricted_retrieval",
    }
    summary_rows = []
    for exp_id, mode in mapping.items():
        priors = retrieve_for_scores(scores, manifest, bank, vecs, mode=mode, seed=seed, top_k=top_k)
        val_mask = scores["split"].astype(str).eq("val").to_numpy()
        y_val = scores.loc[val_mask, "y_true"].astype(int).to_numpy()
        base_val = scores.loc[val_mask, "risk_score"].astype(float).to_numpy()
        alpha = select_alpha(y_val, base_val, priors[val_mask])
        fused = fuse(scores["risk_score"].astype(float).to_numpy(), priors, alpha)
        threshold = best_threshold_f1(y_val, fused[val_mask])
        out = pd.DataFrame(
            {
                "case_id": scores["case_id"].astype(str),
                "center_id": scores["center_id"].astype(str),
                "split": scores["split"].astype(str),
                "y_cin2plus": scores["y_true"].astype(int),
                "p_cin2plus": fused,
                "threshold": threshold,
                "alpha": alpha,
                "retrieval_prior": priors,
                "base_score": scores["risk_score"].astype(float),
            }
        )
        path = write_predictions(out, exp_id)
        outputs[exp_id] = str(path)
        test = out[out["split"].eq("test")]
        summary_rows.append(
            {
                "experiment_id": exp_id,
                "control_mode": mode,
                "alpha": alpha,
                "threshold": threshold,
                "test_auroc": auc_rank(test["y_cin2plus"].to_numpy(), test["p_cin2plus"].to_numpy()),
                "test_auprc": auprc_score(test["y_cin2plus"].to_numpy(), test["p_cin2plus"].to_numpy()),
                "path": str(path),
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUT / "retrieval_control_summary.csv", index=False)
    return outputs


def run_decision_curve(seed: int) -> Path:
    pred_paths = {
        "Clinical logistic": ROOT / "outputs/rasa_public_baselines/predictions/A0_clinical_logistic/test_predictions.csv",
        "Clinical MLP": OUT / "predictions/A0_clinical_mlp/all_predictions.csv",
        "CLIP-style baseline": ROOT / "outputs/rasa_public_baselines/predictions/B0_clip_report/test_predictions.csv",
        "RASA-only": ROOT / "outputs/rasa_public_baselines/predictions/C0_rasa_only/test_predictions.csv",
        "Full MOSAIC retrieval": ROOT / "outputs/rasa_public_baselines/predictions/E1_train_only_semantic_retrieval/test_predictions.csv",
    }
    thresholds = np.round(np.arange(0.05, 0.81, 0.05), 2)
    rows = []
    for name, path in pred_paths.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "split" in df.columns:
            df = df[df["split"].astype(str).eq("test")].copy()
        if df.empty:
            continue
        y = df["y_cin2plus"].astype(int).to_numpy()
        p = df["p_cin2plus"].astype(float).to_numpy()
        n = len(y)
        for t in thresholds:
            pred = p >= float(t)
            tp = float(((pred == 1) & (y == 1)).sum())
            fp = float(((pred == 1) & (y == 0)).sum())
            nb = tp / n - fp / n * (float(t) / (1.0 - float(t)))
            rows.append({"method_name": name, "threshold": float(t), "net_benefit": nb, "seed": seed})
    dca = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    dca_path = OUT / "decision_curve_long.csv"
    dca.to_csv(dca_path, index=False)
    summary = (
        dca.sort_values("net_benefit", ascending=False)
        .groupby("method_name", as_index=False)
        .first()
        .rename(columns={"threshold": "best_threshold_by_net_benefit", "net_benefit": "max_net_benefit"})
    )
    summary["experiment_id"] = "F5_decision_curve_analysis"
    summary_path = OUT / "decision_curve_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    clinical = train_clinical_mlp(args.seed)
    retrieval = run_retrieval_controls(args.seed, args.top_k)
    dca = run_decision_curve(args.seed)
    result = {"clinical_mlp": str(clinical), "retrieval_controls": retrieval, "decision_curve_summary": str(dca)}
    (OUT / "missing_control_run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
