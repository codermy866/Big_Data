"""Prompt B: reference-stratified evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.supplementary.io_utils import save_table
from src.supplementary.train_eval import evaluate_experiment, load_jbd_config, resolve_checkpoint


def run_stratified_eval(project: Path, df: pd.DataFrame, tables_dir: Path, baselines_dir: Path, max_cases: int | None = None) -> pd.DataFrame:
    cfg = load_jbd_config(project)
    test = df[df["split"] == "test"].copy()
    models = [
        "real_report_only_decoder",
        "simple_concat_fusion",
        "report_generation_without_section_alignment",
        "full_lcad_rasa",
    ]
    if (baselines_dir / "best_lcad_rasa" / "best.ckpt").is_file():
        models.append("best_lcad_rasa")

    rows = []
    for exp_id in models:
        ckpt = resolve_checkpoint(project, exp_id, cfg, baselines_dir)
        if ckpt is None:
            continue
        for subset_name, subset_df in [
            ("with_reference", test[(test["has_real_report"] == 1) & (test["reference_report_text"].astype(str).str.len() >= 20)]),
            ("without_reference", test[(test["has_real_report"] != 1) | (test["reference_report_text"].astype(str).str.len() < 20)]),
            ("all", test),
        ]:
            if len(subset_df) == 0:
                continue
            ev = evaluate_experiment(project, df, exp_id, ckpt, max_cases=max_cases, test_df=subset_df)
            ev["subset"] = subset_name
            ev["n_subset"] = len(subset_df)
            rows.append(ev)

    out = pd.DataFrame(rows)
    save_table(out, tables_dir / "table_reference_stratified_evaluation.csv", tables_dir / "table_reference_stratified_evaluation.md")

    interp = """# Reference-stratified evaluation

## Protocol
- **with_reference**: test cases with real `reference_report_text` (length ≥ 20). ROUGE/BLEU/BERTScore valid here only.
- **without_reference**: no reliable reference text. Use label consistency, safety, section completeness, AUC — not ROUGE/BLEU.
- **all**: threshold-free risk metrics and consistency.

## Main-text metrics
| Subset | Primary metrics |
|--------|-----------------|
| With reference | ROUGE-L, BERTScore, section similarity |
| Without reference | AUC, label consistency, hallucination, EDS |
| All | AUC, training scalability |

Do **not** interpret global ROUGE≈0 as model failure.
"""
    (tables_dir / "REFERENCE_STRATIFIED_EVALUATION_INTERPRETATION.md").write_text(interp, encoding="utf-8")

    if len(out) and "auc" in out.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ref = out[out["subset"] == "with_reference"]
        if len(ref):
            for exp in ref["experiment_id"].unique():
                sub = ref[ref["experiment_id"] == exp]
                ax.bar(f"{exp}\n(ref)", sub["auc"].mean(), alpha=0.7)
        ax.set_ylabel("AUC")
        ax.set_title("AUC by model (reference-available subset)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(project / "outputs/publishable/figures/fig_reference_available_metric_comparison.png", dpi=150)
        plt.close(fig)
    return out
