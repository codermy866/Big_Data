#!/usr/bin/env python3
"""Same-split external baselines and paired bootstrap recheck for JBD.

The block is deliberately independent from the LCAD-RASA ablation code:
all baselines use the locked train/val/test split, validation max-F1
threshold selection, held-out test scoring, and bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/publishable/external_baselines/tables"
FIGURES = ROOT / "outputs/publishable/external_baselines/figures"
PRED_DIR = ROOT / "outputs/publishable/external_baselines/predictions"
MANUSCRIPT_TABLES = ROOT / "outputs/publishable/tables/manuscript"
SUMMARY_MD = ROOT / "outputs/publishable/external_baselines/EXTERNAL_BASELINE_BLOCK_SUMMARY.md"

PALETTE = [
    "#2f5f8f",
    "#8fb8d8",
    "#d9a066",
    "#efd7b5",
    "#9e3f3a",
    "#d47f6f",
    "#7f7f7f",
    "#d6d6d6",
]
TEXT = "#343434"
GRID = "#d6d6d6"


def setup_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font="Arial",
        palette=PALETTE,
        rc={
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "axes.edgecolor": "#7f7f7f",
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
        },
    )


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def threshold_grid() -> np.ndarray:
    return np.round(np.arange(0.05, 0.96, 0.01), 2)


def select_val_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    best_thr, best_f1 = 0.5, -1.0
    for thr in threshold_grid():
        f1 = f1_score(y_true, y_score >= thr, zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr


def metric_dict(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict[str, float]:
    pred = (y_score >= thr).astype(int)
    return {
        "auc": safe_auc(y_true, y_score),
        "auprc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "threshold_val_max_f1": float(thr),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thr: float,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(safe_auc(y_true[idx], y_score[idx]))
        f1s.append(f1_score(y_true[idx], y_score[idx] >= thr, zero_division=0))
    out = {}
    for name, vals in [("auc", aucs), ("f1", f1s)]:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            out[f"{name}_ci_low"] = float("nan")
            out[f"{name}_ci_high"] = float("nan")
        else:
            out[f"{name}_ci_low"] = float(np.quantile(arr, 0.025))
            out[f"{name}_ci_high"] = float(np.quantile(arr, 0.975))
    return out


def corrected_paired_bootstrap(
    y_true: np.ndarray,
    ref_score: np.ndarray,
    cmp_score: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs = safe_auc(y_true, ref_score) - safe_auc(y_true, cmp_score)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        deltas.append(safe_auc(y_true[idx], ref_score[idx]) - safe_auc(y_true[idx], cmp_score[idx]))
    arr = np.asarray(deltas, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "delta_auc_full_minus_comparator": obs,
            "delta_auc_ci_low": float("nan"),
            "delta_auc_ci_high": float("nan"),
            "paired_bootstrap_p_two_sided": float("nan"),
            "bootstrap_samples": 0,
        }
    p = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
    p = min(1.0, max(1.0 / len(arr) if p == 0 else p, p))
    return {
        "delta_auc_full_minus_comparator": float(obs),
        "delta_auc_ci_low": float(np.quantile(arr, 0.025)),
        "delta_auc_ci_high": float(np.quantile(arr, 0.975)),
        "paired_bootstrap_p_two_sided": float(p),
        "bootstrap_samples": int(len(arr)),
    }


def resolve_emb_path(path_val: str) -> Path:
    p = Path(str(path_val))
    if p.is_file():
        return p
    name = p.name
    for sub in ("oct", "colposcopy", "fused_visual"):
        alt = ROOT / "outputs/publishable/embeddings" / sub / name
        if alt.is_file():
            return alt
    return p


def load_embedding_matrix(df: pd.DataFrame, col: str) -> np.ndarray:
    arrs = []
    for path_val in df[col].astype(str):
        p = resolve_emb_path(path_val)
        if p.is_file():
            arr = np.load(p).astype(np.float32).reshape(-1)
        else:
            arr = np.zeros(2048, dtype=np.float32)
        if arr.shape[0] != 2048:
            fixed = np.zeros(2048, dtype=np.float32)
            fixed[: min(2048, arr.shape[0])] = arr[:2048]
            arr = fixed
        arrs.append(arr)
    return np.stack(arrs)


def standardize(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (train - mu) / sd, (val - mu) / sd, (test - mu) / sd


def make_clinical_pipeline(model: str) -> Pipeline:
    clinical_cols = ["age", "hpv", "tct"]
    numeric = ["age"]
    categorical = ["hpv", "tct"]
    pre = ColumnTransformer(
        transformers=[
            ("age", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=3)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    if model == "lr":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=42)
    elif model == "gb":
        clf = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            class_weight="balanced",
            random_state=42,
        )
    else:
        raise ValueError(model)
    return Pipeline([("preprocess", pre), ("classifier", clf)])


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CrossAttentionClassifier(nn.Module):
    def __init__(self, oct_dim: int, col_dim: int, clinical_dim: int, hidden: int = 128):
        super().__init__()
        self.oct_proj = nn.Linear(oct_dim, hidden)
        self.col_proj = nn.Linear(col_dim, hidden)
        self.clin_proj = nn.Linear(clinical_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, oct_x: torch.Tensor, col_x: torch.Tensor, clin_x: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([self.oct_proj(oct_x), self.col_proj(col_x), self.clin_proj(clin_x)], dim=1)
        h, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        h = self.norm(h + tokens)
        return self.head(h.reshape(h.size(0), -1)).squeeze(-1)


class ContrastiveFusionClassifier(nn.Module):
    def __init__(self, oct_dim: int, col_dim: int, clinical_dim: int, hidden: int = 128):
        super().__init__()
        self.oct_proj = nn.Sequential(nn.Linear(oct_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.col_proj = nn.Sequential(nn.Linear(col_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.clin_proj = nn.Sequential(nn.Linear(clinical_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(0.25), nn.Linear(hidden, 1))

    def forward(self, oct_x: torch.Tensor, col_x: torch.Tensor, clin_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = F.normalize(self.oct_proj(oct_x), dim=-1)
        c = F.normalize(self.col_proj(col_x), dim=-1)
        cl = self.clin_proj(clin_x)
        logit = self.head(torch.cat([o, c, cl], dim=-1)).squeeze(-1)
        temp = self.temperature.clamp(0.03, 0.25)
        sim = (o @ c.T) / temp
        target = torch.arange(sim.size(0), device=sim.device)
        con = (F.cross_entropy(sim, target) + F.cross_entropy(sim.T, target)) * 0.5
        return logit, con


@dataclass
class TorchResult:
    val_score: np.ndarray
    test_score: np.ndarray
    train_seconds: float
    best_epoch: int


def train_torch_model(
    model: nn.Module,
    train_tensors: tuple[np.ndarray, ...],
    val_tensors: tuple[np.ndarray, ...],
    y_train: np.ndarray,
    y_val: np.ndarray,
    mode: str,
    seed: int,
    epochs: int = 80,
    batch_size: int = 128,
) -> TorchResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    xs = [torch.tensor(x, dtype=torch.float32) for x in train_tensors]
    ys = torch.tensor(y_train.astype(np.float32), dtype=torch.float32)
    ds = TensorDataset(*xs, ys)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_auc, best_epoch = None, -1.0, 0
    patience, stale = 12, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            *xb, yb = batch
            xb = [x.to(device) for x in xb]
            yb = yb.to(device)
            opt.zero_grad()
            if mode == "single" or mode == "mlp":
                logit = model(xb[0])
                loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight)
            elif mode == "cross_attention":
                logit = model(xb[0], xb[1], xb[2])
                loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight)
            elif mode == "contrastive":
                logit, con = model(xb[0], xb[1], xb[2])
                loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight) + 0.1 * con
            else:
                raise ValueError(mode)
            loss.backward()
            opt.step()
        val_score = predict_torch(model, val_tensors, mode, device)
        val_auc = safe_auc(y_val, val_score)
        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_seconds = time.time() - t0
    val_score = predict_torch(model, val_tensors, mode, device)
    return TorchResult(val_score=val_score, test_score=np.array([]), train_seconds=train_seconds, best_epoch=best_epoch)


@torch.no_grad()
def predict_torch(model: nn.Module, tensors: tuple[np.ndarray, ...], mode: str, device: torch.device) -> np.ndarray:
    model.eval()
    xs = [torch.tensor(x, dtype=torch.float32, device=device) for x in tensors]
    n = xs[0].shape[0]
    outs = []
    for start in range(0, n, 256):
        xb = [x[start : start + 256] for x in xs]
        if mode in ("single", "mlp"):
            logit = model(xb[0])
        elif mode == "cross_attention":
            logit = model(xb[0], xb[1], xb[2])
        elif mode == "contrastive":
            logit, _ = model(xb[0], xb[1], xb[2])
        else:
            raise ValueError(mode)
        outs.append(torch.sigmoid(logit).detach().cpu().numpy())
    return np.concatenate(outs)


def clinical_design(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["age", "hpv", "tct"]].copy()
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["hpv"] = out["hpv"].astype(str).replace({"nan": "missing", "None": "missing", "": "missing"})
    out["tct"] = out["tct"].astype(str).replace({"nan": "missing", "None": "missing", "": "missing"})
    return out


def transformed_clinical_arrays(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pipe = make_clinical_pipeline("lr")
    # Fit only the preprocessing part and reuse dense arrays for neural baselines.
    pre = pipe.named_steps["preprocess"]
    train_x = pre.fit_transform(clinical_design(train_df))
    val_x = pre.transform(clinical_design(val_df))
    test_x = pre.transform(clinical_design(test_df))
    return train_x.astype(np.float32), val_x.astype(np.float32), test_x.astype(np.float32)


def write_predictions(
    baseline_id: str,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    thr: float,
) -> Path:
    pred = (y_score >= thr).astype(int)
    out = pd.DataFrame(
        {
            "case_id": test_df["case_id"].astype(str).values,
            "center": test_df["center_id"].astype(str).values,
            "split": test_df["split"].astype(str).values,
            "y_true_cin2plus": y_true.astype(int),
            "risk_score": y_score.astype(float),
            "threshold_val_selected": thr,
            "pred_label": pred.astype(int),
            "correct": (pred == y_true).astype(int),
            "baseline_id": baseline_id,
            "evaluation_protocol": "same_split_val_threshold_max_f1",
        }
    )
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{baseline_id}_test_predictions.csv"
    out.to_csv(path, index=False)
    return path


def row_from_scores(
    baseline_id: str,
    display_name: str,
    family: str,
    feature_source: str,
    y_val: np.ndarray,
    val_score: np.ndarray,
    y_test: np.ndarray,
    test_score: np.ndarray,
    test_df: pd.DataFrame,
    train_seconds: float,
    best_epoch: int | None,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, object], Path]:
    thr = select_val_threshold(y_val, val_score)
    metrics = metric_dict(y_test, test_score, thr)
    ci = bootstrap_ci(y_test, test_score, thr, n_boot=n_boot, seed=seed)
    pred_path = write_predictions(baseline_id, test_df, y_test, test_score, thr)
    row = {
        "baseline_id": baseline_id,
        "model": display_name,
        "model_family": family,
        "feature_source": feature_source,
        "n_train": int(len(y_val) * 0 + 1325),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        **metrics,
        **ci,
        "train_seconds": float(train_seconds),
        "best_epoch": int(best_epoch) if best_epoch is not None else "",
        "prediction_file": str(pred_path.relative_to(ROOT)),
        "protocol": "locked train/val/test split; validation max-F1 threshold; test AUROC/F1; bootstrap CI",
    }
    return row, pred_path


def add_reference_full_row(table: pd.DataFrame) -> pd.DataFrame:
    main_path = MANUSCRIPT_TABLES / "T2_main_model_comparison_with_ci.csv"
    if not main_path.is_file():
        return table
    main = pd.read_csv(main_path)
    ref = main[main["model"].astype(str).eq("Full LCAD-RASA")]
    if ref.empty:
        return table
    r = ref.iloc[0].to_dict()
    row = {
        "baseline_id": "full_lcad_rasa_reference",
        "model": "Full LCAD-RASA (reference)",
        "model_family": "Proposed report-aware fusion",
        "feature_source": "OCT + colposcopy + clinical + structured report supervision",
        "n_train": "",
        "n_val": "",
        "n_test": int(r.get("n_test", r.get("n", 288))),
        "auc": float(r.get("auc", np.nan)),
        "auprc": np.nan,
        "f1": float(r.get("f1", np.nan)),
        "sensitivity": float(r.get("sensitivity", np.nan)) if "sensitivity" in r else np.nan,
        "precision": np.nan,
        "balanced_accuracy": np.nan,
        "threshold_val_max_f1": float(r.get("threshold_val_selected", r.get("threshold", np.nan))),
        "auc_ci_low": float(r.get("auc_ci_low", np.nan)),
        "auc_ci_high": float(r.get("auc_ci_high", np.nan)),
        "f1_ci_low": float(r.get("f1_ci_low", np.nan)),
        "f1_ci_high": float(r.get("f1_ci_high", np.nan)),
        "train_seconds": "",
        "best_epoch": "",
        "prediction_file": "outputs/publishable/predictions/final_per_case/full_lcad_rasa_test_predictions.csv",
        "protocol": "reference from locked Table 2",
    }
    return pd.concat([pd.DataFrame([row]), table], ignore_index=True)


def build_figures(table: pd.DataFrame, paired: pd.DataFrame) -> None:
    setup_theme()
    plot_df = table.copy()
    plot_df["model_short"] = plot_df["model"].str.replace(" \\(reference\\)", "", regex=True)
    order = plot_df.sort_values("auc", ascending=True)["model_short"].tolist()

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(plot_df))]
    for i, (_, r) in enumerate(plot_df.sort_values("auc", ascending=True).iterrows()):
        y = i
        ax.plot([r["auc_ci_low"], r["auc_ci_high"]], [y, y], color="#343434", lw=1.6, alpha=0.9)
        ax.scatter(r["auc"], y, s=130, color=colors[i], edgecolor="#343434", linewidth=0.85, zorder=3)
        ax.text(float(r["auc"]) + 0.012, y, f"{float(r['auc']):.3f}", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Held-out AUROC with 95% bootstrap CI")
    ax.set_title("Same-split external baselines: discrimination")
    ax.set_xlim(0.25, max(0.92, float(plot_df["auc_ci_high"].max()) + 0.03))
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    save_fig(fig, FIGURES / "Figure_external_baselines_auc_forest")

    long = plot_df.melt(
        id_vars=["model_short"],
        value_vars=["auc", "f1", "balanced_accuracy"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    sns.stripplot(
        data=long,
        x="value",
        y="model_short",
        hue="metric",
        dodge=True,
        jitter=False,
        marker="D",
        size=8,
        linewidth=0.8,
        edgecolor="#343434",
        palette=[PALETTE[0], PALETTE[2], PALETTE[4]],
        ax=ax,
    )
    ax.set_xlabel("Metric value")
    ax.set_ylabel("")
    ax.set_title("External baseline metric profile")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlim(0, 1)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    save_fig(fig, FIGURES / "Figure_external_baselines_metric_dotplot")

    if not paired.empty:
        p = paired[paired["comparator"].str.contains("Clinical|OCT|Colposcopy|Late|Cross|Contrastive", regex=True)].copy()
        if not p.empty:
            p = p.sort_values("delta_auc_full_minus_comparator", ascending=True)
            fig, ax = plt.subplots(figsize=(9.0, 4.8))
            for i, (_, r) in enumerate(p.iterrows()):
                ax.plot([r["delta_auc_ci_low"], r["delta_auc_ci_high"]], [i, i], color="#343434", lw=1.5)
                ax.scatter(
                    r["delta_auc_full_minus_comparator"],
                    i,
                    s=120,
                    color=PALETTE[i % len(PALETTE)],
                    edgecolor="#343434",
                    linewidth=0.8,
                    zorder=3,
                )
            ax.axvline(0, ls="--", color="#7f7f7f", lw=1.2)
            ax.set_yticks(range(len(p)))
            ax.set_yticklabels(p["comparator"].tolist())
            ax.set_xlabel("Paired bootstrap delta AUROC (Full LCAD-RASA - comparator)")
            ax.set_title("Corrected paired bootstrap recheck")
            sns.despine(fig=fig, ax=ax)
            fig.tight_layout()
            save_fig(fig, FIGURES / "Figure_external_baselines_paired_delta_auc")


def recheck_pairwise(new_pred_paths: dict[str, Path], n_boot: int, seed: int) -> pd.DataFrame:
    full_path = ROOT / "outputs/publishable/predictions/final_per_case/full_lcad_rasa_test_predictions.csv"
    if not full_path.is_file():
        return pd.DataFrame()
    full = pd.read_csv(full_path)[["case_id", "y_true_cin2plus", "risk_score"]].rename(columns={"risk_score": "full_score"})
    rows = []
    existing_dir = ROOT / "outputs/publishable/predictions/final_per_case"
    name_map = {
        "report_generation_without_section_alignment_test_predictions.csv": "LCAD w/o section alignment",
        "real_report_only_decoder_test_predictions.csv": "Real-report only",
        "simple_concat_fusion_test_predictions.csv": "Simple concat fusion",
        "image_only_report_generation_test_predictions.csv": "Image-only report gen.",
        "instruction_only_report_generation_test_predictions.csv": "Instruction-only report gen.",
        "multimodal_fusion_without_report_anchor_test_predictions.csv": "Fusion w/o report anchor",
        "pseudo_augmented_lcad_test_predictions.csv": "Pseudo-augmented (LCAD)",
    }
    pred_files = {name: existing_dir / fn for fn, name in name_map.items()}
    pred_files.update(new_pred_paths)
    for name, path in pred_files.items():
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        score_col = "risk_score"
        y_col = "y_true_cin2plus" if "y_true_cin2plus" in df.columns else "binary_label"
        tmp = df[["case_id", y_col, score_col]].rename(columns={score_col: "cmp_score", y_col: "cmp_y"})
        merged = full.merge(tmp, on="case_id", how="inner")
        if merged.empty:
            continue
        y = merged["y_true_cin2plus"].to_numpy(dtype=int)
        ref = merged["full_score"].to_numpy(dtype=float)
        cmp_s = merged["cmp_score"].to_numpy(dtype=float)
        stat = corrected_paired_bootstrap(y, ref, cmp_s, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "comparison": f"Full LCAD-RASA vs {name}",
                "reference": "Full LCAD-RASA",
                "comparator": name,
                "n_paired": int(len(merged)),
                "reference_auc": safe_auc(y, ref),
                "comparator_auc": safe_auc(y, cmp_s),
                **stat,
                "test": "corrected paired bootstrap on held-out test predictions",
            }
        )
    return pd.DataFrame(rows)


def write_summary(table: pd.DataFrame, paired: pd.DataFrame, elapsed: float) -> None:
    best = table[~table["model"].str.contains("Full LCAD", regex=False)].sort_values("auc", ascending=False).head(3)
    lines = [
        "# External Baseline Block Summary\n\n",
        "## Protocol\n\n",
        "- Locked split: `train=1325`, `val=284`, `test=288` from the publishable manifest.\n",
        "- Threshold: selected on validation by max-F1 for each model, then applied once to the held-out test set.\n",
        "- Confidence intervals: case-level bootstrap on the held-out test set.\n",
        "- Clinical-only input: age, HPV, and TCT only; centre, report text, pathology-like attributes, and labels are excluded.\n",
        "- Visual input: frozen ResNet50 OCT and colposcopy embeddings already generated by the project.\n",
        "- XGBoost was not available in the current environment; the clinical gradient-boosting baseline uses `HistGradientBoostingClassifier` and is explicitly labelled as such.\n\n",
        "## Top External Baselines by AUROC\n\n",
    ]
    cols = ["model", "auc", "auc_ci_low", "auc_ci_high", "f1", "threshold_val_max_f1"]
    lines.append("| Model | AUROC | AUROC CI low | AUROC CI high | F1 | Threshold |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for _, r in best[cols].iterrows():
        lines.append(
            f"| {r['model']} | {float(r['auc']):.3f} | {float(r['auc_ci_low']):.3f} | "
            f"{float(r['auc_ci_high']):.3f} | {float(r['f1']):.3f} | "
            f"{float(r['threshold_val_max_f1']):.2f} |\n"
        )
    lines.append("\n\n## Main Output Files\n\n")
    for p in [
        TABLES / "T_external_baselines_same_split.csv",
        TABLES / "T_external_baseline_paired_bootstrap_recheck.csv",
        MANUSCRIPT_TABLES / "T_external_baselines_same_split.csv",
        MANUSCRIPT_TABLES / "T_external_baseline_paired_bootstrap_recheck.csv",
        FIGURES / "Figure_external_baselines_auc_forest.pdf",
        FIGURES / "Figure_external_baselines_metric_dotplot.pdf",
        FIGURES / "Figure_external_baselines_paired_delta_auc.pdf",
    ]:
        lines.append(f"- `{p}`\n")
    lines.append(f"\nElapsed seconds: {elapsed:.1f}\n")
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/publishable/manifests/full_manifest_publishable_with_llm_pseudo.csv")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    t0 = time.time()

    manifest = ROOT / args.manifest
    df = pd.read_csv(manifest)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    y_train = train_df["binary_label"].to_numpy(dtype=int)
    y_val = val_df["binary_label"].to_numpy(dtype=int)
    y_test = test_df["binary_label"].to_numpy(dtype=int)

    oct_tr, oct_va, oct_te = [load_embedding_matrix(x, "oct_embedding_path") for x in (train_df, val_df, test_df)]
    col_tr, col_va, col_te = [load_embedding_matrix(x, "colposcopy_embedding_path") for x in (train_df, val_df, test_df)]
    oct_tr, oct_va, oct_te = standardize(oct_tr, oct_va, oct_te)
    col_tr, col_va, col_te = standardize(col_tr, col_va, col_te)
    clin_tr, clin_va, clin_te = transformed_clinical_arrays(train_df, val_df, test_df)
    fused_tr = np.concatenate([oct_tr, col_tr, clin_tr], axis=1)
    fused_va = np.concatenate([oct_va, col_va, clin_va], axis=1)
    fused_te = np.concatenate([oct_te, col_te, clin_te], axis=1)

    rows: list[dict[str, object]] = []
    pred_paths: dict[str, Path] = {}

    for baseline_id, display_name, model_key in [
        ("clinical_lr", "Clinical-only logistic regression", "lr"),
        ("clinical_hist_gradient_boosting", "Clinical-only HistGradientBoosting", "gb"),
    ]:
        start = time.time()
        pipe = make_clinical_pipeline(model_key)
        pipe.fit(clinical_design(train_df), y_train)
        val_score = pipe.predict_proba(clinical_design(val_df))[:, 1]
        test_score = pipe.predict_proba(clinical_design(test_df))[:, 1]
        row, path = row_from_scores(
            baseline_id,
            display_name,
            "Clinical-only conventional ML",
            "age + HPV + TCT",
            y_val,
            val_score,
            y_test,
            test_score,
            test_df,
            time.time() - start,
            None,
            args.bootstrap,
            args.seed,
        )
        rows.append(row)
        pred_paths[display_name] = path

    torch_baselines = [
        (
            "oct_only_embedding_mlp",
            "OCT-only embedding MLP",
            MLPClassifier(oct_tr.shape[1]),
            (oct_tr,),
            (oct_va,),
            (oct_te,),
            "single",
            "Frozen OCT ResNet50 embedding",
        ),
        (
            "colposcopy_only_embedding_mlp",
            "Colposcopy-only embedding MLP",
            MLPClassifier(col_tr.shape[1]),
            (col_tr,),
            (col_va,),
            (col_te,),
            "single",
            "Frozen colposcopy ResNet50 embedding",
        ),
        (
            "late_fusion_mlp",
            "Late-fusion MLP",
            MLPClassifier(fused_tr.shape[1], hidden=384),
            (fused_tr,),
            (fused_va,),
            (fused_te,),
            "mlp",
            "OCT + colposcopy embeddings + age/HPV/TCT",
        ),
        (
            "cross_attention_multimodal_transformer",
            "Cross-attention multimodal transformer",
            CrossAttentionClassifier(oct_tr.shape[1], col_tr.shape[1], clin_tr.shape[1]),
            (oct_tr, col_tr, clin_tr),
            (oct_va, col_va, clin_va),
            (oct_te, col_te, clin_te),
            "cross_attention",
            "OCT token + colposcopy token + clinical token",
        ),
        (
            "contrastive_multimodal_no_report_sections",
            "CLIP-style contrastive multimodal baseline",
            ContrastiveFusionClassifier(oct_tr.shape[1], col_tr.shape[1], clin_tr.shape[1]),
            (oct_tr, col_tr, clin_tr),
            (oct_va, col_va, clin_va),
            (oct_te, col_te, clin_te),
            "contrastive",
            "OCT-colposcopy contrastive alignment without report-section supervision",
        ),
    ]

    for baseline_id, display_name, model, tr, va, te, mode, feature_source in torch_baselines:
        res = train_torch_model(model, tr, va, y_train, y_val, mode, seed=args.seed, epochs=args.epochs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        test_score = predict_torch(model, te, mode, device)
        row, path = row_from_scores(
            baseline_id,
            display_name,
            "Frozen-embedding neural baseline",
            feature_source,
            y_val,
            res.val_score,
            y_test,
            test_score,
            test_df,
            res.train_seconds,
            res.best_epoch,
            args.bootstrap,
            args.seed,
        )
        rows.append(row)
        pred_paths[display_name] = path

    table = pd.DataFrame(rows)
    table["n_train"] = len(train_df)
    table = add_reference_full_row(table)
    TABLES.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "T_external_baselines_same_split.csv", index=False)
    table.to_csv(MANUSCRIPT_TABLES / "T_external_baselines_same_split.csv", index=False)

    paired = recheck_pairwise(pred_paths, n_boot=args.bootstrap, seed=args.seed)
    paired.to_csv(TABLES / "T_external_baseline_paired_bootstrap_recheck.csv", index=False)
    paired.to_csv(MANUSCRIPT_TABLES / "T_external_baseline_paired_bootstrap_recheck.csv", index=False)

    build_figures(table, paired)
    write_summary(table, paired, time.time() - t0)
    print(json.dumps({
        "status": "ok",
        "rows": len(table),
        "paired_rows": len(paired),
        "table": str(TABLES / "T_external_baselines_same_split.csv"),
        "paired": str(TABLES / "T_external_baseline_paired_bootstrap_recheck.csv"),
        "summary": str(SUMMARY_MD),
    }, indent=2))


if __name__ == "__main__":
    main()
