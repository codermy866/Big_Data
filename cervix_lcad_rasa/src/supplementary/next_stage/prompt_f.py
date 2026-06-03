"""Prompt F: multi-seed stability."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.supplementary.io_utils import save_table
from src.supplementary.jbd_figures_seaborn import C6, PALETTE_MAIN, _setup_theme
from src.supplementary.next_stage.core import default_full_spec, no_section_spec, real_only_spec
from src.supplementary.statistics import bootstrap_ci
from src.supplementary.train_eval import evaluate_experiment, load_jbd_config, train_experiment


def run_multiseed(project: Path, df: pd.DataFrame, cfg: dict, seed_dir: Path, tables_dir: Path, seeds: list[int], quick: bool) -> pd.DataFrame:
    if quick:
        cfg = {**cfg, "training": {**cfg["training"], "num_epochs": 2, "max_steps_per_epoch": 50}}
    models = {
        "real_report_only_decoder": real_only_spec(),
        "report_generation_without_section_alignment": no_section_spec(),
        "full_lcad_rasa": default_full_spec(),
    }
    rows = []
    for seed in seeds:
        for name, spec in models.items():
            exp_id = f"{name}_seed{seed}"
            tr = train_experiment(project, df, exp_id, spec, cfg, seed_dir / exp_id, seed=seed)
            if tr.get("status") != "ok":
                continue
            ev = evaluate_experiment(project, df, exp_id, Path(tr["checkpoint"]), spec)
            ev["seed"] = seed
            ev["model"] = name
            rows.append(ev)

    raw = pd.DataFrame(rows)
    raw.to_csv(tables_dir / "table_multiseed_raw.csv", index=False)
    summary = []
    for name, g in raw.groupby("model"):
        for col in ("auc", "f1", "label_consistency", "section_completeness", "hallucination_rate"):
            if col not in g.columns:
                continue
            ci = bootstrap_ci(g[col].tolist(), seed=42)
            summary.append({"model": name, "metric": col, **ci})
    summ_df = pd.DataFrame(summary)
    save_table(summ_df, tables_dir / "table_multiseed_stability.csv", tables_dir / "table_multiseed_stability.md")

    if len(raw) and "auc" in raw.columns:
        _setup_theme()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=raw, x="model", y="auc", palette=PALETTE_MAIN, ax=ax, linewidth=0.8)
        sns.stripplot(data=raw, x="model", y="auc", color=C6, size=5, alpha=0.6, ax=ax, jitter=0.15)
        ax.set_title("Multi-seed test AUROC")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        fig.savefig(project / "outputs/publishable/figures/fig_multiseed_auc_boxplot.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    (tables_dir / "MULTISEED_STABILITY_INTERPRETATION.md").write_text(
        "# Multi-seed stability\n\nReport mean ± 95% CI. Emphasize semantic metrics stability if AUC varies.\n",
        encoding="utf-8",
    )
    return summ_df
