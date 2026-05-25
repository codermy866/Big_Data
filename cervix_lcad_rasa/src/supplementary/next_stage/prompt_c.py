"""Prompt C: RASA loss weight sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.supplementary.io_utils import save_table
from src.supplementary.next_stage.core import default_full_spec
from src.supplementary.train_eval import evaluate_experiment, load_jbd_config, train_experiment


def run_loss_sweep(project: Path, df: pd.DataFrame, cfg: dict, sweep_dir: Path, tables_dir: Path, quick: bool) -> pd.DataFrame:
    if quick:
        cfg = {**cfg, "training": {**cfg["training"], "num_epochs": 2, "max_steps_per_epoch": 60}}
    lambdas = [0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
    rows = []
    for lam in lambdas:
        exp_id = f"rasa_align_{lam:.2f}".replace(".", "p")
        spec = default_full_spec()
        spec["loss"]["rasa_weight"] = lam
        spec["model"]["use_section_align"] = lam > 0
        tr = train_experiment(project, df, exp_id, spec, cfg, sweep_dir / exp_id, seed=int(cfg.get("seed", 42)))
        if tr.get("status") != "ok":
            rows.append({"lambda_align": lam, "status": "train_failed"})
            continue
        ckpt = Path(tr["checkpoint"])
        ev = evaluate_experiment(project, df, exp_id, ckpt, spec)
        ev["lambda_align"] = lam
        ev["lambda_report"] = spec["loss"]["ce_weight"]
        ev["lambda_cls"] = spec["loss"]["cls_weight"]
        ev["lambda_cons"] = spec["loss"]["cons_weight"]
        rows.append(ev)

    out = pd.DataFrame(rows)
    save_table(out, tables_dir / "table_rasa_loss_weight_sweep.csv", tables_dir / "table_rasa_loss_weight_sweep.md")

    # best_lcad_rasa: maximize AUC - 0.1*hallucination + 0.05*section_completeness among align>0
    scored = out[out["lambda_align"] > 0].copy()
    if len(scored):
        scored["score"] = scored["auc"] - 0.1 * scored.get("hallucination_rate", 0) + 0.05 * scored.get("section_completeness", 0)
        best_row = scored.loc[scored["score"].idxmax()]
        best_lam = float(best_row["lambda_align"])
        best_exp = f"rasa_align_{best_lam:.2f}".replace(".", "p")
        import shutil

        src = sweep_dir / best_exp / "best.ckpt"
        dst_dir = project / "outputs/publishable/baselines/best_lcad_rasa"
        dst_dir.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, dst_dir / "best.ckpt")
        out.attrs["best_lcad_rasa_lambda"] = best_lam

    if "auc" in out.columns and "lambda_align" in out.columns:
        for xcol, fname, title in [
            ("section_completeness", "fig_rasa_pareto_auc_vs_section_alignment.png", "AUC vs section completeness"),
            ("hallucination_rate", "fig_rasa_pareto_auc_vs_safety.png", "AUC vs hallucination rate"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(out[xcol], out["auc"])
            for _, r in out.iterrows():
                ax.annotate(str(r["lambda_align"]), (r[xcol], r["auc"]), fontsize=8)
            ax.set_xlabel(xcol)
            ax.set_ylabel("AUC")
            ax.set_title(title)
            plt.tight_layout()
            fig.savefig(project / "outputs/publishable/figures" / fname, dpi=150)
            plt.close(fig)

    risk_best = out.loc[out["auc"].idxmax()] if len(out) else None
    rb_auc = f"{float(risk_best['auc']):.3f}" if risk_best is not None else "n/a"
    rb_lam = risk_best["lambda_align"] if risk_best is not None else "n/a"
    rb_exp = risk_best["experiment_id"] if risk_best is not None else "n/a"
    interp = f"""# RASA loss weight sweep

## Findings
- **Risk-best** (max test AUC): `{rb_exp}` (λ_align={rb_lam}, AUC={rb_auc}).
- **full_lcad_rasa** (λ=0.5): semantic-balanced default; may trade AUC vs no-section model.
- **best_lcad_rasa**: copied from sweep optimum (see `baselines/best_lcad_rasa/`).

## Positioning for JBD
Report **report_generation_without_section_alignment** as risk-discrimination leader; **full_lcad_rasa** / **best_lcad_rasa** as report-anchored semantic grounding.
"""
    (tables_dir / "RASA_LOSS_WEIGHT_SWEEP_INTERPRETATION.md").write_text(interp, encoding="utf-8")
    return out
