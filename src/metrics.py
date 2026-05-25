#!/usr/bin/env python3
"""Unified metrics with bootstrap 95% CI — required for all JBD experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

EPS = 1e-12

METRIC_COLUMNS = [
    "auc",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "balanced_accuracy",
    "brier",
    "ece",
    "calibration_slope",
    "auc_ci_low",
    "auc_ci_high",
    "sensitivity_ci_low",
    "sensitivity_ci_high",
    "specificity_ci_low",
    "specificity_ci_high",
    "ppv_ci_low",
    "ppv_ci_high",
    "npv_ci_low",
    "npv_ci_high",
    "f1_ci_low",
    "f1_ci_high",
    "balanced_accuracy_ci_low",
    "balanced_accuracy_ci_high",
    "brier_ci_low",
    "brier_ci_high",
    "ece_ci_low",
    "ece_ci_high",
    "calibration_slope_ci_low",
    "calibration_slope_ci_high",
    "threshold",
    "n",
    "positives",
    "negatives",
]


@dataclass
class MetricResult:
    auc: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1: float
    balanced_accuracy: float
    brier: float
    ece: float
    calibration_slope: float
    auc_ci_low: float
    auc_ci_high: float
    sensitivity_ci_low: float
    sensitivity_ci_high: float
    specificity_ci_low: float
    specificity_ci_high: float
    ppv_ci_low: float
    ppv_ci_high: float
    npv_ci_low: float
    npv_ci_high: float
    f1_ci_low: float
    f1_ci_high: float
    balanced_accuracy_ci_low: float
    balanced_accuracy_ci_high: float
    brier_ci_low: float
    brier_ci_high: float
    ece_ci_low: float
    ece_ci_high: float
    calibration_slope_ci_low: float
    calibration_slope_ci_high: float
    threshold: float
    n: int
    positives: int
    negatives: int

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    sp = y_prob[order]
    i = 0
    while i < len(sp):
        j = i + 1
        while j < len(sp) and sp[j] == sp[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_prob = np.clip(y_prob, 0.0, 1.0)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if b == n_bins - 1 else y_prob < hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(total)


def calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit = np.log(y_prob / (1 - y_prob))
    if logit.std() < EPS:
        return float("nan")
    coef = np.polyfit(logit, y_true.astype(float), 1)
    return float(coef[0])


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_j = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = _confusion(y_true, pred)
        sens = tp / (tp + fn + EPS)
        spec = tn / (tn + fp + EPS)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def point_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: Optional[float] = None,
) -> MetricResult:
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.clip(np.asarray(list(y_prob), dtype=float), 0.0, 1.0)
    thr = youden_threshold(yt, yp) if threshold is None else float(threshold)
    pred = (yp >= thr).astype(int)
    tn, fp, fn, tp = _confusion(yt, pred)
    sens = tp / (tp + fn + EPS)
    spec = tn / (tn + fp + EPS)
    ppv = tp / (tp + fp + EPS)
    npv = tn / (tn + fn + EPS)
    f1 = 2 * ppv * sens / (ppv + sens + EPS)
    bacc = (sens + spec) / 2.0
    nan_ci = float("nan")
    return MetricResult(
        auc=roc_auc(yt, yp),
        sensitivity=float(sens),
        specificity=float(spec),
        ppv=float(ppv),
        npv=float(npv),
        f1=float(f1),
        balanced_accuracy=float(bacc),
        brier=float(np.mean((yp - yt) ** 2)),
        ece=ece(yt, yp),
        calibration_slope=calibration_slope(yt, yp),
        auc_ci_low=nan_ci,
        auc_ci_high=nan_ci,
        sensitivity_ci_low=nan_ci,
        sensitivity_ci_high=nan_ci,
        specificity_ci_low=nan_ci,
        specificity_ci_high=nan_ci,
        ppv_ci_low=nan_ci,
        ppv_ci_high=nan_ci,
        npv_ci_low=nan_ci,
        npv_ci_high=nan_ci,
        f1_ci_low=nan_ci,
        f1_ci_high=nan_ci,
        balanced_accuracy_ci_low=nan_ci,
        balanced_accuracy_ci_high=nan_ci,
        brier_ci_low=nan_ci,
        brier_ci_high=nan_ci,
        ece_ci_low=nan_ci,
        ece_ci_high=nan_ci,
        calibration_slope_ci_low=nan_ci,
        calibration_slope_ci_high=nan_ci,
        threshold=thr,
        n=int(len(yt)),
        positives=int((yt == 1).sum()),
        negatives=int((yt == 0).sum()),
    )


def bootstrap_ci(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> MetricResult:
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.clip(np.asarray(list(y_prob), dtype=float), 0.0, 1.0)
    base = point_metrics(yt, yp)
    rng = np.random.default_rng(seed)
    n = len(yt)
    boots: List[Dict[str, float]] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = yt[idx], yp[idx]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(point_metrics(yb, pb, threshold=base.threshold).to_dict())
    if not boots:
        return base

    def pct(key: str, lo: float = alpha / 2, hi: float = 1 - alpha / 2) -> Tuple[float, float]:
        vals = [b[key] for b in boots if np.isfinite(b[key])]
        if not vals:
            return float("nan"), float("nan")
        return float(np.percentile(vals, 100 * lo)), float(np.percentile(vals, 100 * hi))

    d = base.to_dict()
    for key in (
        "auc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "balanced_accuracy",
        "brier",
        "ece",
        "calibration_slope",
    ):
        lo, hi = pct(key)
        d[f"{key}_ci_low"] = lo
        d[f"{key}_ci_high"] = hi
    return MetricResult(**d)
