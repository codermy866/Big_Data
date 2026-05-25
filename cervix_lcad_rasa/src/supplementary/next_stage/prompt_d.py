"""Prompt D: validation-based threshold tuning."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.supplementary.io_utils import save_table
from src.supplementary.next_stage.core import collect_risk_scores, metrics_at_threshold, select_thresholds
from src.supplementary.train_eval import load_jbd_config, resolve_checkpoint


def run_threshold_tuning(project: Path, df: pd.DataFrame, tables_dir: Path, baselines_dir: Path) -> pd.DataFrame:
    cfg = load_jbd_config(project)
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    models = [
        "real_report_only_decoder",
        "simple_concat_fusion",
        "report_generation_without_section_alignment",
        "full_lcad_rasa",
    ]
    if (baselines_dir / "best_lcad_rasa" / "best.ckpt").is_file():
        models.append("best_lcad_rasa")

    tune_rows, test_rows = [], []
    for exp_id in models:
        ckpt = resolve_checkpoint(project, exp_id, cfg, baselines_dir)
        if ckpt is None:
            continue
        yv_t, yv_s = collect_risk_scores(ckpt, val)
        yt_t, yt_s = collect_risk_scores(ckpt, test)
        if not yv_t:
            continue
        thrs = select_thresholds(yv_t, yv_s)
        for ttype, thr in thrs.items():
            vm = metrics_at_threshold(yv_t, yv_s, thr)
            tm = metrics_at_threshold(yt_t, yt_s, thr)
            tune_rows.append({"experiment_id": exp_id, "threshold_type": ttype, "selected_threshold": thr, "split": "validation", **vm})
            test_rows.append(
                {
                    "experiment_id": exp_id,
                    "threshold_type": ttype,
                    "selected_threshold": thr,
                    "split": "test",
                    **tm,
                }
            )

    tune_df = pd.DataFrame(tune_rows)
    test_df = pd.DataFrame(test_rows)
    save_table(tune_df, tables_dir / "table_threshold_tuning.csv")
    save_table(test_df, tables_dir / "table_threshold_tuned_test_metrics.csv", tables_dir / "table_threshold_tuned_test_metrics.md")

    if len(test_df):
        sub = test_df[test_df["threshold_type"] == "max_f1"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(sub["experiment_id"].astype(str), sub["f1"].astype(float))
        ax.set_title("Test F1 at validation-selected max-F1 threshold")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(project / "outputs/publishable/figures/fig_threshold_f1_curves.png", dpi=150)
        plt.close(fig)

    (tables_dir / "THRESHOLD_TUNING_INTERPRETATION.md").write_text(
        "# Threshold tuning\n\nF1/sensitivity/specificity use thresholds selected **only on validation**. "
        "Report AUC (threshold-free) as primary risk metric.\n",
        encoding="utf-8",
    )
    return test_df
