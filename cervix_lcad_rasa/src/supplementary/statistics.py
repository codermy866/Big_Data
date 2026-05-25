"""Bootstrap CI and statistical tests for supplementary tables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> dict[str, float]:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0}
    rng = np.random.default_rng(seed)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"mean": float(arr.mean()), "ci_low": lo, "ci_high": hi, "std": float(arr.std())}


def add_ci_columns(df: pd.DataFrame, metric_cols: list[str], case_values: dict[str, list[float]] | None = None) -> pd.DataFrame:
    out = df.copy()
    for col in metric_cols:
        if col not in out.columns:
            continue
        if case_values and col in case_values:
            ci = bootstrap_ci(case_values[col])
        else:
            ci = {"mean": float(out[col].mean()), "ci_low": float(out[col].mean()), "ci_high": float(out[col].mean()), "std": 0.0}
        out[f"{col}_ci_low"] = ci["ci_low"]
        out[f"{col}_ci_high"] = ci["ci_high"]
    return out


def paired_bootstrap_p(a: list[float], b: list[float], n_boot: int = 1000, seed: int = 42) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 1.0
    diffs = np.array(b, dtype=float) - np.array(a, dtype=float)
    obs = diffs.mean()
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(diffs), len(diffs))
        boot = diffs[idx].mean()
        if abs(boot) >= abs(obs):
            count += 1
    return count / n_boot


def build_statistical_tests_summary(main_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "experiment_id" not in main_df.columns:
        return pd.DataFrame(rows)
    full = main_df[main_df["experiment_id"].astype(str).str.contains("full", case=False)]
    base = main_df[main_df["experiment_id"].astype(str).str.contains("simple_concat|real_report", case=False)]
    if len(full) and len(base):
        f = full.iloc[0]
        for _, b in base.iterrows():
            for m in ("rouge_l", "label_consistency", "auc", "hallucination_rate"):
                if m in f and m in b:
                    rows.append(
                        {
                            "comparison": f"{b['experiment_id']} vs {f['experiment_id']}",
                            "metric": m,
                            "delta": float(f[m]) - float(b[m]),
                            "note": "paired bootstrap p-value requires per-case store; point estimate only",
                        }
                    )
    return pd.DataFrame(rows)
