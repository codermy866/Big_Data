#!/usr/bin/env python3
"""VL-DINO-inspired semantic-alignment audits for the MOSAIC/JBD workflow.

This script implements lightweight, reproducible audits for three ideas:

1. ORSA-style bidirectional contrastive section alignment.
2. QPSC-style QC-gated semantic positive construction with leakage controls.
3. VSE-style medical VLP teacher audit, kept as a gated auxiliary analysis.

The implementation is deliberately train-only and numpy/pandas based so it can
run in the public-baseline environment without PyTorch or scikit-learn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "rasa_public_baselines" / "vldino_inspired_semantic_audits"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PREDICTIONS = OUT / "predictions"

MANIFEST = ROOT / "cervix_lcad_rasa" / "outputs" / "publishable" / "manifests" / "full_manifest_publishable_with_llm_pseudo.csv"
BASELINE_PRED = ROOT / "outputs" / "rasa_public_baselines" / "predictions"
BASELINE_METRICS = ROOT / "outputs" / "rasa_public_baselines" / "all_metrics.csv"

SECTION_SPECS = [
    ("oct_findings", "oct_embedding_path", "OCT"),
    ("colposcopy_findings", "colposcopy_embedding_path", "Colposcopy"),
    ("clinical_context", "clinical_features", "Clinical"),
    ("impression", "fused_visual_embedding_path", "Fused"),
]

PALETTE = {
    "blue": "#2f5f8f",
    "light_blue": "#8fb8d8",
    "gold": "#d9a066",
    "pale_gold": "#efd7b5",
    "red": "#9e3f3a",
    "salmon": "#d47f6f",
    "gray": "#7f7f7f",
    "light_gray": "#d6d6d6",
    "green": "#3f7f5f",
}


def ensure_dirs() -> None:
    for path in [OUT, TABLES, FIGURES, PREDICTIONS]:
        path.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#3a3a3a",
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16], 16)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def is_true(value: Any) -> bool:
    text = safe_text(value).strip().lower()
    if text in {"true", "yes", "y"}:
        return True
    try:
        return int(float(text)) == 1
    except Exception:
        return False


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, eps)


def load_visual(path_text: Any) -> np.ndarray:
    path = Path(safe_text(path_text))
    if not path.is_file() and "/data2/hmy/" in str(path):
        alt = Path(str(path).replace("/data2/hmy/", "/data2/hmy_pri/"))
        if alt.is_file():
            path = alt
    if not path.is_file():
        return np.zeros(2048, dtype=np.float32)
    arr = np.load(path).astype(np.float32).reshape(-1)
    if arr.size == 2048:
        return arr
    out = np.zeros(2048, dtype=np.float32)
    out[: min(2048, arr.size)] = arr[:2048]
    return out


def clinical_features(row: pd.Series) -> np.ndarray:
    vals = np.zeros(32, dtype=np.float32)
    try:
        age = float(row.get("age", np.nan))
        if math.isfinite(age):
            vals[0] = (age - 45.0) / 20.0
    except Exception:
        pass
    fields = [
        f"hpv:{safe_text(row.get('hpv', ''))}",
        f"tct:{safe_text(row.get('tct', ''))}",
        f"other:{safe_text(row.get('other_clinical_attributes', ''))}",
        f"center:{safe_text(row.get('center_id', ''))}",
    ]
    for field in fields:
        for tok in re.findall(r"[\w+\-.]+", field.lower()):
            idx = 1 + stable_int(tok) % 31
            vals[idx] += 1.0
    return vals


def parse_jsonish(text: Any) -> dict[str, Any]:
    s = safe_text(text).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        path = Path(s)
        if path.is_file():
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
    return {}


def section_text(row: pd.Series, section: str) -> str:
    ref_col = f"reference_{section}"
    text = safe_text(row.get(ref_col, ""))
    if len(text.strip()) >= 8:
        return text
    obj = parse_jsonish(row.get("pseudo_report_text", ""))
    if not obj:
        obj = parse_jsonish(row.get("pseudo_report_path", ""))
    if section in obj and len(safe_text(obj.get(section)).strip()) >= 8:
        return safe_text(obj.get(section))
    fallback = safe_text(row.get("training_report_text", row.get("reference_report_text", "")))
    if fallback:
        return f"{section}: {fallback}"
    parts = [
        safe_text(row.get("age", "")),
        safe_text(row.get("hpv", "")),
        safe_text(row.get("tct", "")),
        safe_text(row.get("other_clinical_attributes", "")),
    ]
    return f"{section}: " + " ".join(p for p in parts if p)


def text_vector(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[\w+\-.]+", text.lower())
    if not tokens:
        return vec
    for tok in tokens:
        h = stable_int(tok)
        idx = h % dim
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[idx] += sign
    return l2_normalize(vec)


def feature_for_row(row: pd.Series, feature_col: str) -> np.ndarray:
    if feature_col == "clinical_features":
        return clinical_features(row)
    return load_visual(row.get(feature_col, ""))


def standardize_train_apply(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return ((train_x - mu) / sd).astype(np.float32), ((x - mu) / sd).astype(np.float32), (mu, sd)


def standardize_apply(x: np.ndarray, fit: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mu, sd = fit
    return ((x - mu) / sd).astype(np.float32)


def top_svd_projection(x: np.ndarray, y: np.ndarray, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    c = (x.T @ y) / max(len(x), 1)
    try:
        u, _, vt = np.linalg.svd(c, full_matrices=False)
        wx = u[:, :dim].astype(np.float32)
        wy = vt.T[:, :dim].astype(np.float32)
        if wx.shape[1] < dim:
            add = dim - wx.shape[1]
            wx = np.concatenate(
                [wx, rng.normal(0, 1.0 / math.sqrt(x.shape[1]), size=(x.shape[1], add)).astype(np.float32)],
                axis=1,
            )
            wy = np.concatenate(
                [wy, rng.normal(0, 1.0 / math.sqrt(y.shape[1]), size=(y.shape[1], add)).astype(np.float32)],
                axis=1,
            )
    except np.linalg.LinAlgError:
        wx = rng.normal(0, 1.0 / math.sqrt(x.shape[1]), size=(x.shape[1], dim)).astype(np.float32)
        wy = rng.normal(0, 1.0 / math.sqrt(y.shape[1]), size=(y.shape[1], dim)).astype(np.float32)
    return wx, wy


def softmax(z: np.ndarray, axis: int) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(np.clip(z, -40, 40))
    return ez / np.maximum(ez.sum(axis=axis, keepdims=True), 1e-8)


def train_bicontrastive_projection(
    x: np.ndarray,
    y: np.ndarray,
    *,
    out_dim: int = 64,
    epochs: int = 90,
    batch_size: int = 192,
    lr: float = 0.025,
    tau: float = 0.2,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    wx, wy = top_svd_projection(x, y, out_dim, seed)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    n = len(x)
    eye_cache: dict[int, np.ndarray] = {}
    for epoch in range(epochs):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            if len(idx) < 4:
                continue
            xb = x[idx]
            yb = y[idx]
            q = xb @ wx
            k = yb @ wy
            logits = (q @ k.T) / tau
            b = len(idx)
            eye = eye_cache.setdefault(b, np.eye(b, dtype=np.float32))
            pr = softmax(logits, axis=1)
            pc = softmax(logits, axis=0)
            loss_r = -np.log(np.maximum(np.diag(pr), 1e-8)).mean()
            loss_c = -np.log(np.maximum(np.diag(pc), 1e-8)).mean()
            losses.append(float(0.5 * (loss_r + loss_c)))
            grad_logits = 0.5 * ((pr - eye) + (pc - eye)) / b
            grad_q = grad_logits @ k / tau
            grad_k = grad_logits.T @ q / tau
            grad_wx = xb.T @ grad_q + 1e-4 * wx
            grad_wy = yb.T @ grad_k + 1e-4 * wy
            scale = max(1.0, float(np.linalg.norm(grad_wx) / 50.0), float(np.linalg.norm(grad_wy) / 50.0))
            wx -= (lr / scale) * grad_wx.astype(np.float32)
            wy -= (lr / scale) * grad_wy.astype(np.float32)
        wx = l2_normalize(wx, axis=0)
        wy = l2_normalize(wy, axis=0)
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses)) if losses else float("nan")})
    return wx.astype(np.float32), wy.astype(np.float32), history


def retrieval_metrics(q: np.ndarray, k: np.ndarray) -> dict[str, float]:
    qn = l2_normalize(q)
    kn = l2_normalize(k)
    sim = qn @ kn.T
    ranks = []
    for i in range(sim.shape[0]):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
    ranks_arr = np.asarray(ranks, dtype=float)
    off = sim.copy()
    np.fill_diagonal(off, np.nan)
    return {
        "recall_at_1": float(np.mean(ranks_arr <= 1)),
        "recall_at_5": float(np.mean(ranks_arr <= 5)),
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "positive_cosine": float(np.nanmean(np.diag(sim))),
        "negative_cosine": float(np.nanmean(off)),
        "positive_minus_negative": float(np.nanmean(np.diag(sim)) - np.nanmean(off)),
    }


@dataclass
class ProjectionBundle:
    section: str
    modality: str
    feature_col: str
    wx: np.ndarray
    wy: np.ndarray
    x_fit: tuple[np.ndarray, np.ndarray]
    y_fit: tuple[np.ndarray, np.ndarray]
    history: list[dict[str, float]]


def build_xy(df: pd.DataFrame, section: str, feature_col: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, row in df.iterrows():
        xs.append(feature_for_row(row, feature_col))
        ys.append(text_vector(section_text(row, section)))
    return np.vstack(xs).astype(np.float32), np.vstack(ys).astype(np.float32)


def fit_orsa(df: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, ProjectionBundle], pd.DataFrame, pd.DataFrame]:
    train = df[df["split"].astype(str).eq("train")].copy()
    rows = []
    hist_rows = []
    bundles: dict[str, ProjectionBundle] = {}
    for i, (section, feature_col, modality) in enumerate(SECTION_SPECS):
        x_train_raw, y_train_raw = build_xy(train, section, feature_col)
        x_train, _, x_fit = standardize_train_apply(x_train_raw, x_train_raw)
        y_train, _, y_fit = standardize_train_apply(y_train_raw, y_train_raw)
        wx, wy, history = train_bicontrastive_projection(
            x_train,
            y_train,
            out_dim=args.orsa_dim,
            epochs=args.orsa_epochs,
            batch_size=args.orsa_batch_size,
            lr=args.orsa_lr,
            tau=args.orsa_tau,
            seed=args.seed + i,
        )
        bundles[section] = ProjectionBundle(section, modality, feature_col, wx, wy, x_fit, y_fit, history)
        for h in history:
            hist_rows.append({"section": section, "modality": modality, **h})
        for split, part in df.groupby("split"):
            part = part.copy()
            x_raw, y_raw = build_xy(part, section, feature_col)
            x = standardize_apply(x_raw, x_fit)
            y = standardize_apply(y_raw, y_fit)
            q = x @ wx
            k = y @ wy
            fwd = retrieval_metrics(q, k)
            rev = retrieval_metrics(k, q)
            rows.append(
                {
                    "experiment_id": "ORSA_style_bicontrastive_section_alignment",
                    "section": section,
                    "modality": modality,
                    "split": split,
                    "n_cases": len(part),
                    "direction": "modality_to_section",
                    **fwd,
                }
            )
            rows.append(
                {
                    "experiment_id": "ORSA_style_bicontrastive_section_alignment",
                    "section": section,
                    "modality": modality,
                    "split": split,
                    "n_cases": len(part),
                    "direction": "section_to_modality",
                    **rev,
                }
            )
    detail = pd.DataFrame(rows)
    history_df = pd.DataFrame(hist_rows)
    detail.to_csv(TABLES / "T_orsa_bidirectional_section_alignment_detail.csv", index=False)
    history_df.to_csv(TABLES / "T_orsa_training_history.csv", index=False)
    macro = (
        detail.groupby(["split", "direction"], as_index=False)
        .agg(
            n_sections=("section", "count"),
            macro_recall_at_1=("recall_at_1", "mean"),
            macro_recall_at_5=("recall_at_5", "mean"),
            macro_mrr=("mrr", "mean"),
            macro_positive_minus_negative=("positive_minus_negative", "mean"),
        )
        .sort_values(["split", "direction"])
    )
    macro.to_csv(TABLES / "T_orsa_bidirectional_section_alignment_macro.csv", index=False)
    return bundles, detail, macro


def project_case_profile(df: pd.DataFrame, bundles: dict[str, ProjectionBundle]) -> pd.DataFrame:
    section_vecs = []
    meta = []
    for _, row in df.iterrows():
        parts = []
        for section, bundle in bundles.items():
            x_raw = feature_for_row(row, bundle.feature_col).reshape(1, -1)
            x = standardize_apply(x_raw, bundle.x_fit)
            parts.append(l2_normalize(x @ bundle.wx).reshape(-1))
        vec = l2_normalize(np.mean(np.vstack(parts), axis=0))
        section_vecs.append(vec)
        meta.append(
            {
                "case_id": safe_text(row.get("case_id", "")),
                "split": safe_text(row.get("split", "")),
                "center_id": safe_text(row.get("center_id", "")),
                "binary_label": int(float(row.get("binary_label", 0))),
                "has_real_report": int(float(row.get("has_real_report", 0))),
                "needs_pseudo_report": int(float(row.get("needs_pseudo_report", 0))),
                "pseudo_report_pass_qc": int(float(row.get("pseudo_report_pass_qc", 0))),
                "qc_score": float(row.get("qc_score", 0.0) if pd.notna(row.get("qc_score", np.nan)) else 0.0),
                "pseudo_training_weight": float(
                    row.get("pseudo_training_weight", 0.0) if pd.notna(row.get("pseudo_training_weight", np.nan)) else 0.0
                ),
            }
        )
    out = pd.DataFrame(meta)
    vec = np.vstack(section_vecs).astype(np.float32)
    for j in range(vec.shape[1]):
        out[f"z{j:02d}"] = vec[:, j]
    return out


def qc_valid_train(meta: pd.DataFrame) -> np.ndarray:
    real = meta["has_real_report"].astype(int).to_numpy() == 1
    pseudo = (
        (meta["needs_pseudo_report"].astype(int).to_numpy() == 1)
        & (meta["pseudo_report_pass_qc"].astype(int).to_numpy() == 1)
        & (meta["pseudo_training_weight"].astype(float).to_numpy() > 0)
    )
    return real | pseudo


def rank_prior(
    query_z: np.ndarray,
    bank_z: np.ndarray,
    bank_labels: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    sim = l2_normalize(query_z) @ l2_normalize(bank_z).T
    priors = np.zeros(sim.shape[0], dtype=np.float32)
    mean_sims = np.zeros(sim.shape[0], dtype=np.float32)
    ids: list[str] = []
    for i in range(sim.shape[0]):
        order = np.argsort(-sim[i])[:top_k]
        weights = np.exp(sim[i, order] - float(np.max(sim[i, order])))
        weights = weights / max(float(weights.sum()), 1e-8)
        priors[i] = float(np.sum(weights * bank_labels[order]))
        mean_sims[i] = float(np.mean(sim[i, order]))
        ids.append("|".join(str(int(x)) for x in order))
    return priors, ids, mean_sims


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        f = threshold_metrics(y, score, t)["f1"]
        if f > best_f:
            best_f = f
            best_t = float(t)
    return best_t


def select_alpha(y_val: np.ndarray, base_val: np.ndarray, prior_val: np.ndarray) -> tuple[float, float]:
    best_alpha, best_score = 0.0, -1.0
    for alpha in np.linspace(0.0, 0.6, 31):
        fused = np.clip((1.0 - alpha) * base_val + alpha * prior_val, 0.0, 1.0)
        t = best_threshold(y_val, fused)
        bal = threshold_metrics(y_val, fused, t)["balanced_accuracy"]
        if bal > best_score:
            best_score = bal
            best_alpha = float(alpha)
    return best_alpha, best_score


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
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
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
    recall = np.r_[0.0, tp / max(tp[-1], 1)]
    precision = np.r_[1.0, tp / np.maximum(tp + fp, 1)]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def threshold_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    pred = (s >= threshold).astype(int)
    tp = float(((pred == 1) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    if math.isnan(sensitivity) and math.isnan(specificity):
        balanced_accuracy = float("nan")
    else:
        vals = [v for v in [sensitivity, specificity] if not math.isnan(v)]
        balanced_accuracy = float(np.mean(vals)) if vals else float("nan")
    return {
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float((tp + tn) / max(len(y), 1)),
        "balanced_accuracy": balanced_accuracy,
    }


def ece_score(y_true: np.ndarray, score: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (s >= lo) & (s < hi if hi < 1.0 else s <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(y[mask].mean() - s[mask].mean()))
    return float(ece)


def brier_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
    return float(np.mean((s - y) ** 2))


def bootstrap_metric_ci(y: np.ndarray, score: np.ndarray, metric: str, *, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auc_rank(y[idx], score[idx]) if metric == "auroc" else auprc_score(y[idx], score[idx]))
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def metric_row(
    experiment_id: str,
    method_name: str,
    y: np.ndarray,
    score: np.ndarray,
    threshold: float,
    *,
    alpha: float | None,
    notes: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    auroc = auc_rank(y, score)
    auprc = auprc_score(y, score)
    m = threshold_metrics(y, score, threshold)
    auc_lo, auc_hi = bootstrap_metric_ci(y, score, "auroc", n_boot=n_boot, seed=seed)
    pr_lo, pr_hi = bootstrap_metric_ci(y, score, "auprc", n_boot=n_boot, seed=seed + 17)
    return {
        "experiment_id": experiment_id,
        "method_name": method_name,
        "seed": seed,
        "fold_id": "locked_train_val_test",
        "heldout_centre": "",
        "n_test": int(len(y)),
        "n_positive_cin2plus": int(np.asarray(y).sum()),
        "n_negative_cin2plus": int(len(y) - np.asarray(y).sum()),
        "auroc": auroc,
        "auroc_ci_low": auc_lo,
        "auroc_ci_high": auc_hi,
        "auprc": auprc,
        "auprc_ci_low": pr_lo,
        "auprc_ci_high": pr_hi,
        "f1": m["f1"],
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "ece": ece_score(y, score),
        "brier": brier_score(y, score),
        "threshold": threshold,
        "alpha": np.nan if alpha is None else alpha,
        "notes": notes,
        "status": "DONE",
    }


def read_pred(exp_id: str, split: str) -> pd.DataFrame:
    path = BASELINE_PRED / exp_id / f"{split}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def pred_arrays(exp_id: str, split: str) -> pd.DataFrame:
    df = read_pred(exp_id, split)
    return df[["case_id", "centre_id", "y_cin2plus", "y_cin3plus", "p_cin2plus", "threshold"]].copy()


def qpsc_experiments(profile: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    z_cols = [c for c in profile.columns if c.startswith("z")]
    train = profile[profile["split"].eq("train")].reset_index(drop=True)
    val = profile[profile["split"].eq("val")].reset_index(drop=True)
    test = profile[profile["split"].eq("test")].reset_index(drop=True)
    base_val = pred_arrays("C10_full_rasa", "val")
    base_test = pred_arrays("C10_full_rasa", "test")
    val = val.merge(base_val[["case_id", "p_cin2plus"]].rename(columns={"p_cin2plus": "base_score"}), on="case_id")
    test = test.merge(base_test[["case_id", "p_cin2plus"]].rename(columns={"p_cin2plus": "base_score"}), on="case_id")

    train_z = train[z_cols].to_numpy(dtype=np.float32)
    train_y = train["binary_label"].to_numpy(dtype=float)
    valid_mask = qc_valid_train(train)
    bank_z = train_z[valid_mask]
    bank_y = train_y[valid_mask]
    all_z = train_z
    all_y = train_y
    shuffled_y = bank_y.copy()
    rng.shuffle(shuffled_y)
    prevalence = float(train_y.mean())

    rows = []
    pred_frames = []
    settings = [
        ("QPSC_qc_gated_semantic_positives", "QC-gated semantic positives", bank_z, bank_y, False, "valid_train_only_qc_passed_label_free_positive_construction"),
        ("QPSC_uncurated_semantic_neighbors", "Uncurated semantic neighbors", all_z, all_y, False, "train_only_no_qc_filter_ablation"),
        ("QPSC_shuffled_label_control", "QPSC shuffled-label control", bank_z, shuffled_y, False, "train_only_semantic_neighbors_with_shuffled_train_labels_control"),
        ("QPSC_random_positive_control", "QPSC random positive control", bank_z, bank_y, False, "train_only_random_positive_control"),
        ("QPSC_label_prior_control", "QPSC label-prior control", bank_z, bank_y, False, "train_prevalence_only_control"),
        ("QPSC_oracle_label_leakage_stress", "QPSC oracle-label leakage stress", bank_z, bank_y, True, "invalid_oracle_stress_test_uses_query_label_for_neighbor_selection"),
    ]

    for exp_id, name, bz, by, is_oracle, note in settings:
        priors: dict[str, np.ndarray] = {}
        sim_scores: dict[str, np.ndarray] = {}
        neighbor_ids: dict[str, list[str]] = {}
        for split_name, part in [("val", val), ("test", test)]:
            qz = part[z_cols].to_numpy(dtype=np.float32)
            if exp_id == "QPSC_label_prior_control":
                prior = np.full(len(part), prevalence, dtype=np.float32)
                ids = ["train_prevalence"] * len(part)
                sims = np.zeros(len(part), dtype=np.float32)
            elif exp_id == "QPSC_random_positive_control":
                prior = np.zeros(len(part), dtype=np.float32)
                ids = []
                sims = np.zeros(len(part), dtype=np.float32)
                for i in range(len(part)):
                    choice = rng.choice(len(by), size=min(args.qpsc_top_k, len(by)), replace=False)
                    prior[i] = float(by[choice].mean())
                    ids.append("|".join(str(int(x)) for x in choice))
            elif is_oracle:
                prior = np.zeros(len(part), dtype=np.float32)
                ids = []
                sims = np.zeros(len(part), dtype=np.float32)
                q_labels = part["binary_label"].to_numpy(dtype=float)
                for i, lab in enumerate(q_labels):
                    candidate = np.where(by == lab)[0]
                    if len(candidate) == 0:
                        candidate = np.arange(len(by))
                    choice = candidate[: min(args.qpsc_top_k, len(candidate))]
                    prior[i] = float(by[choice].mean())
                    ids.append("|".join(str(int(x)) for x in choice))
                    sims[i] = 1.0
            else:
                prior, ids, sims = rank_prior(qz, bz, by, top_k=args.qpsc_top_k)
            priors[split_name] = prior
            sim_scores[split_name] = sims
            neighbor_ids[split_name] = ids

        alpha, _ = select_alpha(val["binary_label"].to_numpy(), val["base_score"].to_numpy(), priors["val"])
        val_fused = np.clip((1.0 - alpha) * val["base_score"].to_numpy() + alpha * priors["val"], 0.0, 1.0)
        threshold = best_threshold(val["binary_label"].to_numpy(), val_fused)
        test_fused = np.clip((1.0 - alpha) * test["base_score"].to_numpy() + alpha * priors["test"], 0.0, 1.0)
        y_test = test["binary_label"].to_numpy(dtype=int)
        rows.append(
            metric_row(
                exp_id,
                name,
                y_test,
                test_fused,
                threshold,
                alpha=alpha,
                notes=note,
                n_boot=args.bootstrap,
                seed=args.seed,
            )
        )
        for split_name, part, fused in [("val", val, val_fused), ("test", test, test_fused)]:
            pred = pd.DataFrame(
                {
                    "case_id": part["case_id"].to_numpy(),
                    "centre_id": part["center_id"].to_numpy(),
                    "split": split_name,
                    "y_cin2plus": part["binary_label"].to_numpy(dtype=int),
                    "base_score_full_rasa": part["base_score"].to_numpy(dtype=float),
                    "qpsc_semantic_prior": priors[split_name],
                    "qpsc_mean_similarity": sim_scores[split_name],
                    "p_cin2plus": fused,
                    "threshold": threshold,
                    "pred_label": (fused >= threshold).astype(int),
                    "alpha": alpha,
                    "experiment_id": exp_id,
                    "method_name": name,
                    "neighbor_indices": neighbor_ids[split_name],
                    "label_use_in_positive_construction": bool(is_oracle),
                    "valid_for_main_claim": not bool(is_oracle),
                    "notes": note,
                }
            )
            out_dir = PREDICTIONS / exp_id
            out_dir.mkdir(parents=True, exist_ok=True)
            pred.to_csv(out_dir / f"{split_name}_predictions.csv", index=False)
            if split_name == "test":
                pred_frames.append(pred)
    metrics = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    metrics.to_csv(TABLES / "T_qpsc_semantic_positive_controls_metrics.csv", index=False)
    all_preds = pd.concat(pred_frames, ignore_index=True)
    all_preds.to_csv(TABLES / "T_qpsc_test_predictions_long.csv", index=False)
    audit = pd.DataFrame(
        [
            {
                "experiment_id": r["experiment_id"],
                "train_only_bank": True,
                "qc_gated": "qc_gated" in r["experiment_id"],
                "uses_query_or_test_label_for_positive_construction": bool(r["experiment_id"] == "QPSC_oracle_label_leakage_stress"),
                "valid_for_main_claim": bool(r["experiment_id"] != "QPSC_oracle_label_leakage_stress"),
                "interpretation": "invalid stress test only" if r["experiment_id"] == "QPSC_oracle_label_leakage_stress" else "valid control/audit",
            }
            for r in rows
        ]
    )
    audit.to_csv(TABLES / "T_qpsc_leakage_boundary_audit.csv", index=False)
    return metrics, audit


def aligned_val_test(base_id: str, teacher_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    def align(split: str) -> pd.DataFrame:
        base = pred_arrays(base_id, split).rename(columns={"p_cin2plus": "base_score"})
        teacher = pred_arrays(teacher_id, split).rename(columns={"p_cin2plus": "teacher_score"})
        keep = ["case_id", "teacher_score"]
        out = base.merge(teacher[keep], on="case_id", how="inner")
        return out

    return align("val"), align("test")


def vse_experiments(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    teacher_sets = {
        "VSE_B0_CLIP_Report_teacher": ["B0_clip_report"],
        "VSE_B5_BiomedCLIP_teacher": ["B5_biomedclip_frozen"],
        "VSE_B6_UniMedCLIP_teacher": ["B6_unimedclip_frozen"],
        "VSE_medical_VLP_mean_teacher": ["B5_biomedclip_frozen", "B6_unimedclip_frozen"],
        "VSE_public_VLP_mean_teacher": ["B0_clip_report", "B5_biomedclip_frozen", "B6_unimedclip_frozen"],
    }
    refs = pd.read_csv(BASELINE_METRICS)
    ref_mosaic = refs[refs["experiment_id"].eq("E1_train_only_semantic_retrieval")].iloc[0]
    ref_rasa = refs[refs["experiment_id"].eq("C10_full_rasa")].iloc[0]
    rows = []
    preds = []
    for base_id, base_name in [("C10_full_rasa", "Full RASA backbone"), ("E1_train_only_semantic_retrieval", "Full MOSAIC")]:
        for exp_id, teachers in teacher_sets.items():
            val_parts = []
            test_parts = []
            for tid in teachers:
                val_i, test_i = aligned_val_test(base_id, tid)
                val_parts.append(val_i[["case_id", "teacher_score"]].rename(columns={"teacher_score": tid}))
                test_parts.append(test_i[["case_id", "teacher_score"]].rename(columns={"teacher_score": tid}))
            val, test = pred_arrays(base_id, "val").rename(columns={"p_cin2plus": "base_score"}), pred_arrays(base_id, "test").rename(
                columns={"p_cin2plus": "base_score"}
            )
            for part in val_parts:
                val = val.merge(part, on="case_id", how="inner")
            for part in test_parts:
                test = test.merge(part, on="case_id", how="inner")
            val["teacher_score"] = val[teachers].mean(axis=1)
            test["teacher_score"] = test[teachers].mean(axis=1)
            alpha, _ = select_alpha(val["y_cin2plus"].to_numpy(), val["base_score"].to_numpy(), val["teacher_score"].to_numpy())
            val_score = np.clip((1.0 - alpha) * val["base_score"].to_numpy() + alpha * val["teacher_score"].to_numpy(), 0.0, 1.0)
            threshold = best_threshold(val["y_cin2plus"].to_numpy(), val_score)
            test_score = np.clip((1.0 - alpha) * test["base_score"].to_numpy() + alpha * test["teacher_score"].to_numpy(), 0.0, 1.0)
            y_test = test["y_cin2plus"].to_numpy(dtype=int)
            full_exp_id = f"{exp_id}_on_{base_id}"
            row = metric_row(
                full_exp_id,
                f"{base_name} + VLP teacher audit ({'+'.join(teachers)})",
                y_test,
                test_score,
                threshold,
                alpha=alpha,
                notes="VSE-style prediction-level medical VLP teacher audit; auxiliary only, not promoted unless it beats locked Full MOSAIC and Full RASA in AUROC and F1",
                n_boot=args.bootstrap,
                seed=args.seed,
            )
            row["base_id"] = base_id
            row["teacher_ids"] = "+".join(teachers)
            row["beats_full_rasa_auc_f1"] = bool(row["auroc"] > float(ref_rasa["auroc"]) and row["f1"] > float(ref_rasa["f1"]))
            row["beats_full_mosaic_auc_f1"] = bool(row["auroc"] > float(ref_mosaic["auroc"]) and row["f1"] > float(ref_mosaic["f1"]))
            row["promotion_decision"] = "promote" if row["beats_full_rasa_auc_f1"] and row["beats_full_mosaic_auc_f1"] else "do_not_promote"
            rows.append(row)
            pred = pd.DataFrame(
                {
                    "case_id": test["case_id"],
                    "centre_id": test["centre_id"],
                    "split": "test",
                    "y_cin2plus": y_test,
                    "base_score": test["base_score"],
                    "teacher_score": test["teacher_score"],
                    "p_cin2plus": test_score,
                    "threshold": threshold,
                    "pred_label": (test_score >= threshold).astype(int),
                    "alpha": alpha,
                    "experiment_id": full_exp_id,
                    "base_id": base_id,
                    "teacher_ids": "+".join(teachers),
                    "promotion_decision": row["promotion_decision"],
                }
            )
            out_dir = PREDICTIONS / full_exp_id
            out_dir.mkdir(parents=True, exist_ok=True)
            pred.to_csv(out_dir / "test_predictions.csv", index=False)
            preds.append(pred)
    metrics = pd.DataFrame(rows).sort_values(["promotion_decision", "auroc", "f1"], ascending=[True, False, False])
    metrics.to_csv(TABLES / "T_vse_medical_vlp_teacher_audit_metrics.csv", index=False)
    pred_long = pd.concat(preds, ignore_index=True)
    pred_long.to_csv(TABLES / "T_vse_test_predictions_long.csv", index=False)
    decision = pd.DataFrame(
        [
            {
                "reference": "C10_full_rasa",
                "auroc": float(ref_rasa["auroc"]),
                "f1": float(ref_rasa["f1"]),
            },
            {
                "reference": "E1_train_only_semantic_retrieval",
                "auroc": float(ref_mosaic["auroc"]),
                "f1": float(ref_mosaic["f1"]),
            },
        ]
    )
    decision.to_csv(TABLES / "T_vse_promotion_references.csv", index=False)
    return metrics, decision


def plot_orsa(detail: pd.DataFrame) -> None:
    setup_style()
    plot_df = detail[detail["split"].eq("test")].copy()
    pair_order = {
        "OCT -> oct_findings": 0,
        "Colposcopy -> colposcopy_findings": 1,
        "Clinical -> clinical_context": 2,
        "Fused -> impression": 3,
    }
    plot_df["pair"] = plot_df["modality"] + " -> " + plot_df["section"]
    plot_df["order"] = plot_df["pair"].map(pair_order)
    plot_df = plot_df[plot_df["order"].notna()].sort_values(["order", "direction"])
    pairs = [p for p, _ in sorted(pair_order.items(), key=lambda x: x[1])]
    y_base = {p: i for i, p in enumerate(pairs)}
    offsets = {"modality_to_section": -0.11, "section_to_modality": 0.11}
    colors = {"modality_to_section": PALETTE["blue"], "section_to_modality": PALETTE["gold"]}
    markers = {"modality_to_section": "o", "section_to_modality": "s"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for direction, group in plot_df.groupby("direction"):
        ys = [y_base[p] + offsets[direction] for p in group["pair"]]
        ax.scatter(group["mrr"], ys, s=42, color=colors[direction], marker=markers[direction], edgecolor="white", linewidth=0.7, label=direction.replace("_", " "))
        for x, y in zip(group["mrr"], ys):
            ax.text(float(x) + 0.006, y, f"{float(x):.3f}", va="center", fontsize=7)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p.replace(" -> ", "\n-> ") for p in pairs])
    ax.set_xlabel("MRR")
    ax.set_title("ORSA-style bidirectional alignment on locked test cases")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_orsa_test_mrr_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / "Figure_orsa_test_mrr_heatmap.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compact_label(row: pd.Series, label_col: str) -> str:
    exp_id = safe_text(row.get("experiment_id", ""))
    if exp_id.startswith("QPSC_"):
        mapping = {
            "QPSC_qc_gated_semantic_positives": "QPSC QC-gated\nsemantic positives",
            "QPSC_uncurated_semantic_neighbors": "QPSC uncurated\nsemantic neighbors",
            "QPSC_shuffled_label_control": "QPSC shuffled-label\ncontrol",
            "QPSC_random_positive_control": "QPSC random-positive\ncontrol",
            "QPSC_label_prior_control": "QPSC label-prior\ncontrol",
            "QPSC_oracle_label_leakage_stress": "QPSC oracle leakage\nstress test",
        }
        return mapping.get(exp_id, textwrap.fill(exp_id.replace("_", " "), width=24))
    if exp_id.startswith("VSE_"):
        base = "MOSAIC" if "E1_train_only_semantic_retrieval" in exp_id else "RASA"
        if "medical_VLP_mean" in exp_id:
            teacher = "BioMed+UniMed"
        elif "public_VLP_mean" in exp_id:
            teacher = "CLIP+BioMed+UniMed"
        elif "B6_UniMedCLIP" in exp_id:
            teacher = "UniMedCLIP"
        elif "B5_BiomedCLIP" in exp_id:
            teacher = "BiomedCLIP"
        elif "B0_CLIP_Report" in exp_id:
            teacher = "CLIP-Report"
        else:
            teacher = "VLP teacher"
        return f"{base} +\n{teacher}"
    return textwrap.fill(safe_text(row.get(label_col, exp_id)), width=28)


def plot_metric_dotplot(df: pd.DataFrame, path: Path, title: str, label_col: str = "method_name") -> None:
    setup_style()
    keep = df.copy().sort_values("auroc", ascending=True)
    labels = keep.apply(lambda r: compact_label(r, label_col), axis=1)
    y = np.arange(len(keep))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, max(3.8, 0.42 * len(keep))), sharey=True)
    for ax, metric, color in [(axes[0], "auroc", PALETTE["blue"]), (axes[1], "f1", PALETTE["gold"])]:
        ax.scatter(keep[metric], y, s=34, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_xlabel(metric.upper())
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_title(title)
    axes[1].set_title("Threshold metric")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary(
    orsa_macro: pd.DataFrame,
    qpsc_metrics: pd.DataFrame,
    qpsc_audit: pd.DataFrame,
    vse_metrics: pd.DataFrame,
    vse_refs: pd.DataFrame,
) -> None:
    def rpath(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    best_qpsc = qpsc_metrics[~qpsc_metrics["experiment_id"].eq("QPSC_oracle_label_leakage_stress")].sort_values("auroc", ascending=False).iloc[0]
    best_vse = vse_metrics.sort_values(["promotion_decision", "auroc", "f1"], ascending=[True, False, False]).iloc[0]
    test_orsa = orsa_macro[orsa_macro["split"].eq("test")]
    macro_mrr = float(test_orsa["macro_mrr"].mean()) if not test_orsa.empty else float("nan")
    valid_qpsc = bool(not qpsc_audit[qpsc_audit["valid_for_main_claim"].eq(False)].empty)
    any_vse_promoted = bool(vse_metrics["promotion_decision"].eq("promote").any())
    lines = [
        "# VL-DINO-Inspired Semantic Alignment Audit Summary\n\n",
        "## Scope\n\n",
        "This package evaluates ORSA-style bidirectional section alignment, QPSC-style QC-gated semantic positives, and VSE-style medical VLP teacher auditing while keeping MOSAIC as the main method.\n\n",
        "## ORSA-style bidirectional alignment\n\n",
        f"- Test macro MRR averaged across bidirectional retrieval summaries: {macro_mrr:.4f}.\n",
        f"- Detail table: `{rpath(TABLES / 'T_orsa_bidirectional_section_alignment_detail.csv')}`.\n",
        f"- Figure: `{rpath(FIGURES / 'Figure_orsa_test_mrr_heatmap.pdf')}`.\n\n",
        "## QPSC-style semantic positives\n\n",
        f"- Best valid QPSC/control AUROC: `{best_qpsc['experiment_id']}` = {float(best_qpsc['auroc']):.4f}, F1 = {float(best_qpsc['f1']):.4f}.\n",
        "- Oracle-label stress control is written and explicitly marked invalid for main claims.\n" if valid_qpsc else "",
        f"- Metrics table: `{rpath(TABLES / 'T_qpsc_semantic_positive_controls_metrics.csv')}`.\n",
        f"- Leakage audit: `{rpath(TABLES / 'T_qpsc_leakage_boundary_audit.csv')}`.\n\n",
        "## VSE-style medical VLP teacher audit\n\n",
        f"- Best teacher-audit row: `{best_vse['experiment_id']}`; AUROC = {float(best_vse['auroc']):.4f}, F1 = {float(best_vse['f1']):.4f}, decision = {best_vse['promotion_decision']}.\n",
        f"- Promotion references: `{rpath(TABLES / 'T_vse_promotion_references.csv')}`.\n",
        f"- Promotion result: {'promote one or more teacher audits' if any_vse_promoted else 'do not promote VLP teacher audit to main MOSAIC method'}.\n\n",
        "## Claim boundary\n\n",
        "The outputs are manuscript-supporting audits and ablations. MOSAIC remains the primary LLM-augmented report-section semantic alignment and retrieval-calibrated fusion framework. VLP teacher rows are auxiliary unless the strict AUROC-and-F1 promotion gate is met.\n",
    ]
    (OUT / "VL_DINO_INSPIRED_AUDIT_SUMMARY.md").write_text("".join(lines), encoding="utf-8")
    status = {
        "orsa_style_bidirectional_alignment": True,
        "qpsc_style_qc_gated_semantic_positives": True,
        "qpsc_controls": ["random", "shuffled", "label_prior", "oracle_leakage_stress", "uncurated"],
        "vse_style_medical_vlp_teacher_audit": True,
        "vse_promote_any": any_vse_promoted,
        "mosaic_remains_primary": True,
        "summary": str(OUT / "VL_DINO_INSPIRED_AUDIT_SUMMARY.md"),
    }
    (OUT / "IMPLEMENTATION_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--orsa-dim", type=int, default=64)
    parser.add_argument("--orsa-epochs", type=int, default=80)
    parser.add_argument("--orsa-batch-size", type=int, default=192)
    parser.add_argument("--orsa-lr", type=float, default=0.025)
    parser.add_argument("--orsa-tau", type=float, default=0.2)
    parser.add_argument("--qpsc-top-k", type=int, default=12)
    args = parser.parse_args()

    ensure_dirs()
    df = pd.read_csv(MANIFEST)
    bundles, orsa_detail, orsa_macro = fit_orsa(df, args)
    profile = project_case_profile(df, bundles)
    profile.to_csv(TABLES / "T_orsa_projected_case_profiles.csv", index=False)
    qpsc_metrics, qpsc_audit = qpsc_experiments(profile, args)
    vse_metrics, vse_refs = vse_experiments(args)
    plot_orsa(orsa_detail)
    plot_metric_dotplot(qpsc_metrics, FIGURES / "Figure_qpsc_semantic_positive_controls", "QPSC-style controls")
    plot_metric_dotplot(vse_metrics, FIGURES / "Figure_vse_medical_vlp_teacher_audit", "VSE-style teacher audits")
    combined = pd.concat(
        [
            qpsc_metrics.assign(audit_family="QPSC_semantic_positive_controls"),
            vse_metrics.assign(audit_family="VSE_medical_vlp_teacher_audit"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(TABLES / "T_vldino_inspired_metrics_combined.csv", index=False)
    plot_metric_dotplot(combined, FIGURES / "Figure_vldino_inspired_metric_summary", "VL-DINO-inspired audits")
    write_summary(orsa_macro, qpsc_metrics, qpsc_audit, vse_metrics, vse_refs)
    print(json.dumps({"status": "ok", "out": str(OUT), "n_metrics": int(len(combined))}, indent=2))


if __name__ == "__main__":
    main()
