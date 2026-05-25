"""Prompt E: strict LOCO retraining."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.supplementary.io_utils import save_table
from src.supplementary.next_stage.core import default_full_spec, no_section_spec, real_only_spec
from src.supplementary.train_eval import evaluate_experiment, load_jbd_config, train_experiment

CENTERS = ["enshi", "jingzhou", "xiangyang", "wuda", "shiyan"]
CENTER_LABELS = {
    "enshi": "Enshi",
    "jingzhou": "Jingzhou",
    "xiangyang": "Xiangyang",
    "wuda": "Wuhan Renmin",
    "shiyan": "Shiyan",
}


def run_loco_strict(project: Path, df: pd.DataFrame, cfg: dict, loco_dir: Path, tables_dir: Path, quick: bool) -> pd.DataFrame:
    if quick:
        cfg = {**cfg, "training": {**cfg["training"], "num_epochs": 2, "max_steps_per_epoch": 50}}
    models = {
        "real_report_only_decoder": real_only_spec(),
        "report_generation_without_section_alignment": no_section_spec(),
        "full_lcad_rasa": default_full_spec(),
    }
    audit_rows, result_rows = [], []

    for held in CENTERS:
        train_pool = df[(df["center_id"] != held) & (df["split"].isin(["train", "val"]))]
        test_pool = df[(df["center_id"] == held) & (df["split"] == "test")]
        if len(test_pool) == 0:
            test_pool = df[df["center_id"] == held]
        audit_rows.append(
            {
                "held_out_center": held,
                "train_cases": len(train_pool),
                "test_cases": len(test_pool),
                "train_real_reports": int((train_pool["has_real_report"] == 1).sum()),
                "test_real_reports": int((test_pool["has_real_report"] == 1).sum()),
                "test_center_in_train": int((train_pool["center_id"] == held).sum()),
                "excluded_from_training": 0,
            }
        )
        for model_name, spec in models.items():
            exp_id = f"loco_strict_{held}_{model_name}"
            fold_dir = loco_dir / exp_id
            tr = train_experiment(
                project,
                df,
                exp_id,
                spec,
                cfg,
                fold_dir,
                seed=42,
                train_df_override=train_pool[train_pool["split"] == "train"] if "split" in train_pool.columns else train_pool,
            )
            if tr.get("status") != "ok":
                continue
            ev = evaluate_experiment(project, df, exp_id, fold_dir / "best.ckpt", spec, test_df=test_pool)
            ev["held_out_center"] = held
            ev["center_label"] = CENTER_LABELS.get(held, held)
            ev["model"] = model_name
            ev["loco_type"] = "strict_retrain"
            result_rows.append(ev)

    audit_df = pd.DataFrame(audit_rows)
    out = pd.DataFrame(result_rows)
    loco_dir.mkdir(parents=True, exist_ok=True)
    save_table(audit_df, tables_dir / "table_loco_strict_fold_audit.csv")
    save_table(out, tables_dir / "table_loco_strict_main_results.csv", tables_dir / "table_loco_strict_main_results.md")

    if len(out) and "auc" in out.columns:
        piv = out.pivot_table(index="model", columns="held_out_center", values="auc", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(piv.fillna(0).values, cmap="YlGnBu", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_title("Strict LOCO — test AUC")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(project / "outputs/publishable/figures/fig_loco_strict_center_heatmap.png", dpi=150)
        plt.close(fig)

    (tables_dir / "LOCO_STRICT_INTERPRETATION.md").write_text(
        "# Strict LOCO\n\nEach fold retrains without the held-out centre. "
        "Compare to legacy eval-only LOCO in `table_loco_main_results.csv`.\n",
        encoding="utf-8",
    )
    return out
