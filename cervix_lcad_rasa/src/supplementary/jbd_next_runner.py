"""Execute JBD_LCAD_RASA_next_experiment_prompts.md (Prompts 0–8)."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.supplementary.io_utils import save_table
from src.supplementary.next_stage.core import collect_risk_scores, metrics_at_threshold, select_thresholds
from src.supplementary.train_eval import (
    build_model,
    evaluate_experiment,
    load_jbd_config,
    resolve_checkpoint,
    train_experiment,
)


def _merge_spec(base: dict, extra: dict) -> dict:
    out = {**base}
    for k, v in extra.items():
        if k == "model" and isinstance(v, dict):
            out["model"] = {**out.get("model", {}), **v}
        elif k == "loss" and isinstance(v, dict):
            out["loss"] = {**out.get("loss", {}), **v}
        else:
            out[k] = v
    return out


def _default_train_spec() -> dict:
    return {
        "train_filter": {"training_eligible": 1},
        "use_pseudo_report": True,
        "use_real_report": True,
        "require_qc_pass": True,
        "use_report_loss": True,
        "model": {"use_section_align": True, "use_risk_head": True},
        "loss": {"ce_weight": 1.0, "rasa_weight": 0.5, "cls_weight": 0.2, "cons_weight": 0.1},
    }


def _df_md(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or len(df) == 0:
        return "_empty_"
    sub = df.head(max_rows)
    cols = list(sub.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


MANUSCRIPT = "outputs/publishable/tables/manuscript"
PRED_DIR = "outputs/publishable/predictions/final_per_case"
FIG_DIR = "outputs/publishable/figures/jbd_final"
V2 = "outputs/publishable_jbd_submission_v2"

CORE_MODELS = [
    ("full_lcad_rasa", "Full LCAD-RASA"),
    ("report_generation_without_section_alignment", "LCAD w/o section alignment"),
    ("real_report_only_decoder", "Real-report only"),
    ("simple_concat_fusion", "Simple concat fusion"),
    ("image_only_report_generation", "Image-only report gen."),
    ("instruction_only_report_generation", "Instruction-only report gen."),
    ("multimodal_fusion_without_report_anchor", "Fusion w/o report anchor"),
]


def _tables(project: Path) -> Path:
    return project / "outputs/publishable/tables"


def _manuscript(project: Path) -> Path:
    p = project / MANUSCRIPT
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_csv(project: Path, rel: str) -> pd.DataFrame | None:
    p = project / rel
    return pd.read_csv(p) if p.is_file() else None


def _status_rows_from_main(project: Path) -> pd.DataFrame:
    p = _tables(project) / "table_main_results_for_manuscript.csv"
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p)


# --- Prompt 0 ---
def run_prompt0(project: Path) -> Path:
    ms = _manuscript(project)
    rows_main = []
    main_eligible = [
        ("T1a", "T1a_cohort_summary.csv", "main"),
        ("T1b", "T1b_centre_scale_and_supervision.csv", "main"),
        ("T2", "T2_main_model_comparison.csv", "main"),
        ("S10", "S10_masking_validation.csv", "supplement"),
        ("S6", "S6_modality_perturbation_text_decoding.csv", "main_figure"),
    ]
    for tid, fname, place in main_eligible:
        fp = ms / fname if (ms / fname).is_file() else _tables(project) / "manuscript" / fname
        if not fp.is_file():
            fp = project / "outputs/publishable/tables/manuscript" / fname
        rows_main.append({"table_id": tid, "file": fname, "placement": place, "exists": fp.is_file()})

    main_df = _status_rows_from_main(project)
    no_ckpt = []
    if len(main_df) and "status" in main_df.columns:
        for _, r in main_df[main_df["status"] == "no_checkpoint"].iterrows():
            no_ckpt.append(
                {
                    "experiment_id": r.get("experiment_id"),
                    "result_file": "table_main_results_for_manuscript.csv",
                    "action": "re-eval or remove from table",
                }
            )

    baselines_dir = project / "outputs/publishable/baselines"
    cfg = load_jbd_config(project)
    for exp_id, _ in cfg.get("baselines", {}).items():
        if resolve_checkpoint(project, exp_id, cfg, baselines_dir) is None:
            no_ckpt.append({"experiment_id": exp_id, "result_file": "baselines", "action": "retrain"})

    cohort_rows = [
        {"file": "T2_main_model_comparison.csv", "n": 288, "threshold": "validation_selected", "subset": "all_test"},
        {"file": "S6_modality_perturbation_text_decoding.csv", "n": 128, "threshold": "n/a", "subset": "report_missing_perturbation"},
    ]

    lines = [
        "# JBD Final Result Audit",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}\n",
        "## A. Main-text eligible",
        _df_md(pd.DataFrame(rows_main)) if rows_main else "",
        "\n## C. no_checkpoint / missing",
        _df_md(pd.DataFrame(no_ckpt)) if no_ckpt else "_None critical — baselines dir has checkpoints; re-run eval to refresh tables._",
        "\n## D. cohort n mismatch",
        _df_md(pd.DataFrame(cohort_rows)),
        "\n## E. threshold protocol",
        "- Table 2: validation-selected max-F1 threshold from `table_threshold_tuned_test_metrics.csv`",
        "- Do not compare default 0.5 F1 across models in main text",
        "\n## F. mock contamination",
        "- Main text must not cite `outputs/tables/` mock pipeline",
        "\n## G. final action list",
        "- Re-run Prompt 1 eval for any stale `no_checkpoint` rows in aggregated CSV",
        "- Complete Prompt 2–3 per-case stats",
        "- Prompt 5 figures → `figures/jbd_final/`",
        "- Expert scores: pending (Prompt 6)",
    ]
    out = ms / "JBD_FINAL_RESULT_AUDIT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# --- Prompt 1 ---
def run_prompt1(project: Path, df: pd.DataFrame, cfg: dict, skip_train: bool, eval_max: int, quick: bool) -> pd.DataFrame:
    baselines_dir = project / "outputs/publishable/baselines"
    tables_dir = _tables(project)
    if quick:
        cfg = {**cfg, "training": {**cfg.get("training", {}), "num_epochs": 2, "max_steps_per_epoch": 60}}
    rows = []
    targets: list[tuple[str, dict]] = []
    for exp_id, spec in cfg.get("baselines", {}).items():
        targets.append((exp_id, _merge_spec(_default_train_spec(), spec)))
    for exp_id, spec in cfg.get("rasa_ablations", {}).items():
        targets.append((exp_id, _merge_spec(_default_train_spec(), spec)))
    for exp_id, spec in cfg.get("modality_ablations", {}).items():
        targets.append((exp_id, _merge_spec(_default_train_spec(), spec)))

    import torch

    for exp_id, spec in targets:
        ckpt_p = baselines_dir / exp_id / "best.ckpt"
        log = {"experiment_id": exp_id, "training_budget": "quick" if quick else "full"}
        try:
            if not skip_train and not ckpt_p.is_file():
                tr = train_experiment(project, df, exp_id, spec, cfg, baselines_dir / exp_id, seed=int(cfg.get("seed", 42)))
                log.update(tr)
            ckpt = resolve_checkpoint(project, exp_id, cfg, baselines_dir)
            if ckpt is None:
                log["status"] = "no_checkpoint"
                rows.append(log)
                continue
            ev = evaluate_experiment(project, df, exp_id, ckpt, spec, max_cases=eval_max)
            log.update(ev)
            log["status"] = "ok"
            log["checkpoint"] = str(ckpt)
        except Exception as e:
            log["status"] = "failed"
            log["error"] = str(e)
        rows.append(log)

    out = pd.DataFrame(rows)
    resolution = tables_dir / "JBD_NO_CHECKPOINT_RESOLUTION.md"
    ok_n = int((out["status"] == "ok").sum()) if "status" in out.columns else 0
    resolution.write_text(
        f"# no_checkpoint resolution\n\nTrained/evaluated {len(out)} experiments; ok={ok_n}.\n",
        encoding="utf-8",
    )
    if "baselines" in {t[0] for t in targets[:6]}:
        base_rows = out[out["experiment_id"].isin(cfg.get("baselines", {}).keys())]
        if len(base_rows):
            save_table(base_rows, tables_dir / "table_baseline_comparison.csv", tables_dir / "table_baseline_comparison.md")
    return out


# --- Prompt 2 ---
def export_per_case_predictions(project: Path, df: pd.DataFrame, cfg: dict, device: str = "cpu") -> pd.DataFrame:
    import torch
    from src.models_publishable.lcad_rasa_model import instr_vector, load_visual_emb

    pred_root = project / PRED_DIR
    pred_root.mkdir(parents=True, exist_ok=True)
    baselines_dir = project / "outputs/publishable/baselines"
    test = df[df["split"] == "test"].copy()
    val = df[df["split"] == "validation"].copy() if "validation" in df["split"].values else df[df["split"] == "val"].copy()
    if len(val) == 0:
        val = df[df["split"] == "train"].sample(min(200, len(df)), random_state=42)

    thresh_df = _read_csv(project, "outputs/publishable/tables/table_threshold_tuned_test_metrics.csv")
    thresh_map: dict[str, float] = {}
    if thresh_df is not None and "threshold_type" in thresh_df.columns:
        sub = thresh_df[thresh_df["threshold_type"] == "max_f1"]
        for _, r in sub.iterrows():
            thresh_map[str(r["experiment_id"])] = float(r["selected_threshold"])

    dev = torch.device(device)
    index_rows = []

    for exp_id, label in CORE_MODELS:
        ckpt = resolve_checkpoint(project, exp_id, cfg, baselines_dir)
        out_csv = pred_root / f"{exp_id}_test_predictions.csv"
        if ckpt is None:
            index_rows.append({"model_name": exp_id, "status": "missing_checkpoint", "n": 0})
            continue
        state = torch.load(ckpt, map_location="cpu")
        spec = state.get("spec") or {}
        yv_t, yv_s = collect_risk_scores(ckpt, val, spec, dev)
        thrs = select_thresholds(yv_t, yv_s)
        thr = thresh_map.get(exp_id, thrs["max_f1"])

        model = build_model(spec)
        model.load_state_dict(state["model"], strict=False)
        model.to(dev)
        model.eval()

        records = []
        with torch.no_grad():
            for _, row in test.iterrows():
                oct_e = torch.tensor(load_visual_emb(str(row.get("oct_embedding_path", ""))), dtype=torch.float32, device=dev).unsqueeze(0)
                col_e = torch.tensor(load_visual_emb(str(row.get("colposcopy_embedding_path", ""))), dtype=torch.float32, device=dev).unsqueeze(0)
                fus_e = torch.tensor(load_visual_emb(str(row.get("fused_visual_embedding_path", ""))), dtype=torch.float32, device=dev).unsqueeze(0)
                instr = torch.tensor(instr_vector(row.to_dict()), dtype=torch.float32, device=dev).unsqueeze(0)
                lab = int(row["binary_label"])
                ids = torch.zeros(1, 64, dtype=torch.long, device=dev)
                out = model(oct_e, col_e, fus_e, instr, ids, torch.tensor([lab], device=dev))
                risk = float(torch.sigmoid(out["risk_logit"]).item()) if out.get("risk_logit") is not None else 0.5
                pred = int(risk >= thr)
                records.append(
                    {
                        "case_id": row["case_id"],
                        "center": row.get("center_id", ""),
                        "split": row.get("split", "test"),
                        "has_real_report": int(row.get("has_real_report", 0)),
                        "needs_pseudo_report": int(row.get("needs_pseudo_report", 0)),
                        "report_supervision_class": row.get("report_supervision_class", ""),
                        "training_report_type": row.get("training_report_type", ""),
                        "y_true_cin2plus": lab,
                        "risk_score": risk,
                        "threshold_val_selected": thr,
                        "pred_label": pred,
                        "correct": int(pred == lab),
                        "source_checkpoint": str(ckpt),
                        "seed": int(cfg.get("seed", 42)),
                        "evaluation_protocol": "test_split_val_threshold_max_f1",
                    }
                )
        pred_df = pd.DataFrame(records)
        pred_df.to_csv(out_csv, index=False)
        index_rows.append(
            {
                "model_name": exp_id,
                "display_name": label,
                "status": "ok",
                "n": len(pred_df),
                "positives": int(pred_df["y_true_cin2plus"].sum()),
                "threshold": thr,
                "path": str(out_csv.relative_to(project)),
            }
        )

    idx = pd.DataFrame(index_rows)
    idx.to_csv(pred_root / "PER_CASE_PREDICTION_INDEX.csv", index=False)
    status_md = pred_root / "PER_CASE_PREDICTION_STATUS.md"
    status_md.write_text(f"# Per-case predictions\n\n{_df_md(idx)}\n", encoding="utf-8")
    return idx


# --- Prompt 3 ---
def _auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 2000, seed: int = 20260525) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return {"auc": 0.5, "ci_low": 0.5, "ci_high": 0.5}
    auc = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    boots = []
    n = len(y_true)
    skipped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            skipped += 1
            continue
        try:
            boots.append(roc_auc_score(yt, ys))
        except Exception:
            skipped += 1
    if not boots:
        return {"auc": auc, "ci_low": auc, "ci_high": auc, "bootstrap_skipped": skipped}
    return {
        "auc": auc,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "bootstrap_skipped": skipped,
    }


def _metric_at_thr_ci(y_true: np.ndarray, y_score: np.ndarray, thr: float, n_boot: int = 2000, seed: int = 20260525) -> dict[str, float]:
    m = metrics_at_threshold(y_true.tolist(), y_score.tolist(), thr)
    rng = np.random.default_rng(seed)
    f1s, sens, spec = [], [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bm = metrics_at_threshold(y_true[idx].tolist(), y_score[idx].tolist(), thr)
        f1s.append(bm["f1"])
        sens.append(bm["sensitivity"])
        spec.append(bm["specificity"])
    return {
        **m,
        "f1_ci_low": float(np.percentile(f1s, 2.5)) if f1s else m["f1"],
        "f1_ci_high": float(np.percentile(f1s, 97.5)) if f1s else m["f1"],
        "sens_ci_low": float(np.percentile(sens, 2.5)) if sens else m["sensitivity"],
        "sens_ci_high": float(np.percentile(sens, 97.5)) if sens else m["sensitivity"],
        "spec_ci_low": float(np.percentile(spec, 2.5)) if spec else m["specificity"],
        "spec_ci_high": float(np.percentile(spec, 97.5)) if spec else m["specificity"],
    }


def _paired_bootstrap_auc_p(y_a: np.ndarray, s_a: np.ndarray, y_b: np.ndarray, s_b: np.ndarray, n_boot: int = 2000, seed: int = 20260525) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    if len(y_a) != len(y_b):
        return 0.0, 0.0
    try:
        obs = roc_auc_score(y_b, s_b) - roc_auc_score(y_a, s_a)
    except Exception:
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y_a)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            if len(np.unique(y_a[idx])) < 2:
                continue
            diffs.append(roc_auc_score(y_b[idx], s_b[idx]) - roc_auc_score(y_a[idx], s_a[idx]))
        except Exception:
            continue
    if not diffs:
        return obs, 1.0
    diffs = np.array(diffs)
    p = float((np.abs(diffs) >= np.abs(obs)).mean())
    return float(obs), p


def _mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    try:
        from scipy.stats import mcnemar

        table = np.zeros((2, 2), dtype=int)
        for yt, pa, pb in zip(y_true, pred_a, pred_b):
            table[pa, pb] += 1  # wrong orientation fix below
        # b=1 correct, a=1 correct -> use discordant pairs
        b01 = ((pred_a == 1) & (pred_b == 0)).sum()
        b10 = ((pred_a == 0) & (pred_b == 1)).sum()
        if b01 + b10 == 0:
            return 1.0
        result = mcnemar([[0, b01], [b10, 0]], exact=False)
        return float(result.pvalue)
    except Exception:
        b01 = ((pred_a == 1) & (pred_b == 0)).sum()
        b10 = ((pred_a == 0) & (pred_b == 1)).sum()
        if b01 + b10 == 0:
            return 1.0
        from scipy.stats import binom

        return float(binom.cdf(min(b01, b10), b01 + b10, 0.5) * 2)


def run_prompt3(project: Path) -> None:
    ms = _manuscript(project)
    pred_root = project / PRED_DIR
    single_rows = []
    pair_rows = []
    ref_model = "full_lcad_rasa"

    for exp_id, label in CORE_MODELS:
        p = pred_root / f"{exp_id}_test_predictions.csv"
        if not p.is_file():
            continue
        d = pd.read_csv(p)
        yt = d["y_true_cin2plus"].values.astype(int)
        ys = d["risk_score"].values.astype(float)
        thr = float(d["threshold_val_selected"].iloc[0])
        auc_ci = _auc_ci(yt, ys)
        thr_m = _metric_at_thr_ci(yt, ys, thr)
        single_rows.append(
            {
                "model": label,
                "experiment_id": exp_id,
                "n": len(d),
                **auc_ci,
                "f1": thr_m["f1"],
                "f1_ci_low": thr_m["f1_ci_low"],
                "f1_ci_high": thr_m["f1_ci_high"],
                "sensitivity": thr_m["sensitivity"],
                "sens_ci_low": thr_m["sens_ci_low"],
                "sens_ci_high": thr_m["sens_ci_high"],
                "specificity": thr_m["specificity"],
                "spec_ci_low": thr_m["spec_ci_low"],
                "spec_ci_high": thr_m["spec_ci_high"],
                "ppv": thr_m.get("ppv", 0),
                "npv": thr_m.get("npv", 0),
                "threshold": thr,
            }
        )

    ref_p = pred_root / f"{ref_model}_test_predictions.csv"
    if ref_p.is_file():
        ref_d = pd.read_csv(ref_p)
        yt = ref_d["y_true_cin2plus"].values.astype(int)
        ys_ref = ref_d["risk_score"].values.astype(float)
        pred_ref = ref_d["pred_label"].values.astype(int)
        for exp_id, label in CORE_MODELS:
            if exp_id == ref_model:
                continue
            p = pred_root / f"{exp_id}_test_predictions.csv"
            if not p.is_file():
                continue
            d = pd.read_csv(p)
            merged = ref_d[["case_id", "y_true_cin2plus", "risk_score", "pred_label"]].merge(
                d[["case_id", "risk_score", "pred_label"]],
                on="case_id",
                suffixes=("_ref", "_cmp"),
            )
            if len(merged) == 0:
                continue
            y = merged["y_true_cin2plus"].values.astype(int)
            delta, p_auc = _paired_bootstrap_auc_p(
                y, merged["risk_score_ref"].values, y, merged["risk_score_cmp"].values
            )
            mcn = _mcnemar(y, merged["pred_label_ref"].values.astype(int), merged["pred_label_cmp"].values.astype(int))
            pair_rows.append(
                {
                    "comparison": f"{label} vs Full LCAD-RASA",
                    "reference": "Full LCAD-RASA",
                    "comparator": label,
                    "delta_auc": delta,
                    "bootstrap_p_auc": p_auc,
                    "mcnemar_p_at_val_threshold": mcn,
                    "n_paired": len(merged),
                    "test": "paired bootstrap (DeLong not used)",
                }
            )

    t2_ci = pd.DataFrame(single_rows)
    t2_ci.to_csv(ms / "T2_main_model_comparison_with_ci.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(ms / "T2_pairwise_statistical_tests.csv", index=False)
    supp = project / "outputs/publishable/tables/supplementary"
    supp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pair_rows).to_csv(supp / "S_statistical_tests_full.csv", index=False)

    report = ms / "JBD_STATISTICAL_TESTS_REPORT.md"
    report.write_text(
        "# Statistical tests report\n\n"
        "- Bootstrap n=2000, seed=20260525, resampling unit=case_id.\n"
        "- AUC CI: percentile bootstrap on test set.\n"
        "- Pairwise AUC: paired bootstrap on matched case_id; **DeLong not implemented** — do not claim DeLong p-values.\n"
        "- McNemar: paired classification at validation-selected threshold.\n"
        "- Write 'statistically significant' only when bootstrap p < 0.05 **and** clinically/contextually meaningful.\n\n"
        f"## Pairwise tests\n\n{_df_md(pd.DataFrame(pair_rows)) if pair_rows else '_No pairs_'}\n",
        encoding="utf-8",
    )


# --- Prompt 4 ---
def run_prompt4(project: Path) -> None:
    ms = _manuscript(project)
    rows = []
    for p in sorted((project / "outputs/publishable/tables").rglob("*.csv")):
        if "manuscript" not in str(p) and p.name.startswith("table_"):
            try:
                df = pd.read_csv(p)
                rows.append(
                    {
                        "file_name": str(p.relative_to(project)),
                        "n_rows": len(df),
                        "columns": len(df.columns),
                        "manuscript_placement": "supplement",
                    }
                )
            except Exception:
                pass
    for name, n, thr, subset, place in [
        ("T1a_cohort_summary.csv", 1897, "n/a", "cohort", "main"),
        ("T1b_centre_scale_and_supervision.csv", 1897, "n/a", "cohort", "main"),
        ("T2_main_model_comparison.csv", 288, "validation_selected", "all_test", "main"),
        ("S6_modality_perturbation_text_decoding.csv", 128, "n/a", "report_missing", "main_figure"),
    ]:
        fp = ms / name
        rows.append(
            {
                "file_name": f"outputs/publishable/tables/manuscript/{name}",
                "evaluation_n": n,
                "threshold_protocol": thr,
                "cohort_subset": subset,
                "manuscript_placement": place,
            }
        )
    idx = pd.DataFrame(rows)
    idx.to_csv(ms / "JBD_COHORT_THRESHOLD_PROTOCOL_INDEX.csv", index=False)
    (ms / "JBD_COHORT_THRESHOLD_PROTOCOL_REPORT.md").write_text(
        f"# Cohort & threshold protocol\n\n{_df_md(idx.head(20))}\n",
        encoding="utf-8",
    )
    sec = project / "outputs/publishable/manuscript_sections"
    sec.mkdir(parents=True, exist_ok=True)
    (sec / "METHODS_EVALUATION_PROTOCOL_SAFE_TEXT.md").write_text(
        "Primary discrimination: AUROC on the held-out test set (n=288). "
        "Operating point: maximum validation F1 threshold per model. "
        "Perturbation analyses: n=128 report-missing test cases with decoded section text.\n",
        encoding="utf-8",
    )
    (sec / "RESULTS_TABLE_FOOTNOTES_SAFE_TEXT.md").write_text(
        "Table 2: test n=288; thresholds selected on validation split (max F1). "
        "Figure 3: perturbation cohort n=128 without reference reports.\n",
        encoding="utf-8",
    )


# --- Prompt 5 ---
def run_prompt5_figures(project: Path) -> None:
    from src.supplementary.jbd_figures_seaborn import generate_all_seaborn_figures

    generate_all_seaborn_figures(project)


# --- Prompt 6 ---
def run_prompt6(project: Path, df: pd.DataFrame) -> None:
    er = project / "outputs/publishable/expert_review"
    er.mkdir(parents=True, exist_ok=True)
    test = df[df["split"] == "test"]
    strat = (
        test.groupby(["center_id", "binary_label"], group_keys=False)
        .apply(lambda g: g.head(max(3, 50 // test["center_id"].nunique())))
        .head(50)
    )
    cols = [c for c in ["case_id", "center_id", "binary_label", "has_real_report", "training_report_type", "age", "hpv", "tct"] if c in strat.columns]
    pkg = strat[cols].copy()
    pkg["blinded_case_id"] = [f"BR{i+1:03d}" for i in range(len(pkg))]
    pkg.to_csv(er / "JBD_EXPERT_REVIEW_CASE_PACKAGE.csv", index=False)
    (er / "JBD_EXPERT_REVIEW_PROTOCOL.md").write_text(
        "# Expert review protocol\n\n"
        "Sample n≤50 stratified by centre and CIN2+ label. Rate 1–5: factual consistency, section completeness, "
        "clinical plausibility, impression-label consistency, hallucination risk, usefulness.\n\n"
        "**Ratings pending — do not claim expert validation in the manuscript.**\n",
        encoding="utf-8",
    )
    (er / "JBD_EXPERT_REVIEW_README.md").write_text(
        "Fill `JBD_EXPERT_REVIEW_RATING_TEMPLATE.csv` (export from xlsx if needed). "
        "Then run statistical summary script.\n",
        encoding="utf-8",
    )
    tpl = pd.DataFrame(
        columns=[
            "blinded_case_id",
            "rater_id",
            "factual_consistency",
            "section_completeness",
            "clinical_plausibility",
            "label_impression_consistency",
            "hallucination_risk",
            "usefulness",
        ]
    )
    tpl.to_csv(er / "JBD_EXPERT_REVIEW_RATING_TEMPLATE.csv", index=False)
    ms = _manuscript(project)
    pd.DataFrame([{"status": "pending", "n_cases": len(pkg)}]).to_csv(ms / "S_expert_review_summary.csv", index=False)
    (project / "outputs/publishable/manuscript_sections/EXPERT_REVIEW_RESULTS_SAFE_TEXT.md").write_text(
        "Expert review package exported; ratings pending. Manuscript should not claim expert validation.\n",
        encoding="utf-8",
    )


# --- Prompt 7 ---
def run_prompt7(project: Path) -> None:
    sec = project / "outputs/publishable/manuscript_sections"
    sec.mkdir(parents=True, exist_ok=True)
    templates = {
        "DATA_AVAILABILITY_STATEMENT.md": (
            "Individual-level clinical images and reports are not publicly released due to institutional privacy restrictions. "
            "De-identified aggregate tables, analysis schemas, and code are available as described in the Code Availability statement. "
            "Access to restricted data may be available upon reasonable request and institutional approval [TO BE FILLED BY AUTHORS]."
        ),
        "CODE_AVAILABILITY_STATEMENT.md": (
            "Source code: [GitHub URL placeholder]. Archived release: [Zenodo DOI placeholder]. "
            "Commit: [hash placeholder]. Environment: `requirements.txt`; Python 3.11; PyTorch 2.3."
        ),
        "ETHICS_APPROVAL_AND_CONSENT.md": (
            "Ethics approval: [institution placeholder]. Approval number: [placeholder]. "
            "Informed consent / waiver: [TO BE FILLED BY AUTHORS]."
        ),
        "AI_USE_STATEMENT.md": (
            "Pseudo-reports were produced by a local embedding-enhanced structured generator under label constraints; "
            "no commercial LLM API was used for the primary publishable experiments. "
            "Manuscript drafting assistance: [disclose if applicable per JBD policy]."
        ),
        "COMPETING_INTERESTS.md": "The authors declare no competing interests [TO BE CONFIRMED].",
        "AUTHOR_CONTRIBUTIONS_TEMPLATE.md": "CRediT roles: [TO BE FILLED BY AUTHORS].",
        "REPRODUCIBILITY_STATEMENT.md": (
            "Reproduce via `scripts/run_publishable_pipeline.sh`, `scripts/26_run_jbd_supplementary_experiments.py`, "
            "and `scripts/29_run_jbd_next_experiment_prompts.py`. Manifest schema in `reproducibility_package/schema/`."
        ),
        "LIMITATIONS_FINAL_SAFE_TEXT.md": (
            "Lightweight decoder; heterogeneous reference text; local structured pseudo-report generator; "
            "LOCO quick training budget; expert ratings pending."
        ),
    }
    for name, text in templates.items():
        (sec / name).write_text(text + "\n", encoding="utf-8")

    v1 = project / "outputs/publishable_jbd_submission_v1"
    bundle = []
    for f in sorted(sec.glob("*.md")):
        bundle.append(f"## {f.name}\n\n{f.read_text(encoding='utf-8')}\n")
    if v1.is_dir():
        (v1 / "JBD_DECLARATIONS_BUNDLE.md").write_text("\n".join(bundle), encoding="utf-8")
    (sec / "JBD_DECLARATIONS_BUNDLE.md").write_text("\n".join(bundle), encoding="utf-8")

    audit_lines = ["# Final privacy and path audit\n", "Scanned manuscript_sections and tables for sensitive patterns.\n"]
    bad = []
    for p in list(sec.glob("*.md")) + list((project / MANUSCRIPT).glob("*.csv"))[:30]:
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
            if "/data2/" in t or "/ssd_data/" in t:
                bad.append(str(p))
        except Exception:
            pass
    audit_lines.append(f"Files with absolute paths in content: {len(bad)}\n")
    for b in bad[:10]:
        audit_lines.append(f"- {b}\n")
    out_audit = project / "outputs/publishable_jbd_submission_v1/FINAL_PRIVACY_AND_PATH_AUDIT.md"
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    out_audit.write_text("".join(audit_lines), encoding="utf-8")


# --- Prompt 8 ---
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_prompt8(project: Path) -> dict[str, Any]:
    v2 = project / V2
    if v2.exists():
        shutil.rmtree(v2)
    for sub in ("tables", "figures", "manuscript_sections", "audit", "predictions_index"):
        (v2 / sub).mkdir(parents=True, exist_ok=True)

    ms = project / MANUSCRIPT
    fig = project / FIG_DIR
    sec = project / "outputs/publishable/manuscript_sections"

    copies: list[tuple[Path, Path]] = []
    if ms.is_dir():
        for f in ms.glob("*.csv"):
            dst = v2 / "tables" / f.name
            shutil.copy2(f, dst)
            copies.append((f, dst))
    if fig.is_dir():
        for f in fig.glob("*"):
            if f.suffix in (".png", ".pdf", ".csv", ".md"):
                dst = v2 / "figures" / f.name
                shutil.copy2(f, dst)
                copies.append((f, dst))
    if sec.is_dir():
        for f in sec.glob("*.md"):
            dst = v2 / "manuscript_sections" / f.name
            shutil.copy2(f, dst)
            copies.append((f, dst))
    idx = project / PRED_DIR / "PER_CASE_PREDICTION_INDEX.csv"
    if idx.is_file():
        shutil.copy2(idx, v2 / "predictions_index" / idx.name)

    manifest_rows = [{"relative_path": str(d.relative_to(v2)), "sha256": _sha256(d)} for _, d in copies if d.is_file()]
    pd.DataFrame(manifest_rows).to_csv(v2 / "FILE_MANIFEST.csv", index=False)
    (v2 / "SHA256SUMS.txt").write_text("\n".join(f"{r['sha256']}  {r['relative_path']}" for r in manifest_rows) + "\n", encoding="utf-8")

    checklist = [
        ("Main tables in bundle", (v2 / "tables/T2_main_model_comparison.csv").is_file() or (v2 / "tables/T2_main_model_comparison_with_ci.csv").is_file()),
        ("Figure 3 perturbation", (v2 / "figures/Figure3_modality_perturbation.png").is_file()),
        ("Statistical tests", (v2 / "tables/T2_pairwise_statistical_tests.csv").is_file()),
        ("Declarations", (v2 / "manuscript_sections/DATA_AVAILABILITY_STATEMENT.md").is_file()),
        ("Expert validation claim", False),
    ]
    n_warn = sum(1 for _, ok in checklist if not ok)
    ready = "ready with minor warnings" if n_warn <= 2 else "not ready"

    (v2 / "README.md").write_text(
        f"# JBD submission freeze v2\n\nFiles: {len(manifest_rows)}\nRecommendation: **{ready}**\n",
        encoding="utf-8",
    )
    (v2 / "SUBMISSION_READINESS_CHECKLIST.md").write_text(
        "# Checklist\n\n" + "\n".join(f"- [{'x' if ok else ' '}] {name}" for name, ok in checklist) + f"\n\n**Recommendation:** {ready}\n",
        encoding="utf-8",
    )
    shutil.copy2(project / "outputs/publishable_jbd_submission_v1/KNOWN_LIMITATIONS.md", v2 / "FINAL_KNOWN_LIMITATIONS.md") if (project / "outputs/publishable_jbd_submission_v1/KNOWN_LIMITATIONS.md").is_file() else (v2 / "FINAL_KNOWN_LIMITATIONS.md").write_text("See LIMITATIONS_FINAL_SAFE_TEXT.md\n", encoding="utf-8")
    if (project / "outputs/publishable_jbd_submission_v1/FINAL_PRIVACY_AND_PATH_AUDIT.md").is_file():
        shutil.copy2(project / "outputs/publishable_jbd_submission_v1/FINAL_PRIVACY_AND_PATH_AUDIT.md", v2 / "FINAL_PRIVACY_AUDIT.md")

    return {"files": len(manifest_rows), "warnings": n_warn, "recommendation": ready}


def run_all_prompts(
    project: Path,
    *,
    skip_train: bool = True,
    eval_max: int = 288,
    quick: bool = False,
    device: str = "cuda",
    prompts: str = "0-8",
) -> dict[str, Any]:
    cfg = load_jbd_config(project)
    df = pd.read_csv(project / cfg["manifest"])
    status: dict[str, Any] = {"started": datetime.now(timezone.utc).isoformat(), "steps": {}}
    t0 = time.time()

    def _step(name: str, fn):
        try:
            status["steps"][name] = {"result": str(fn()), "ok": True}
        except Exception as e:
            status["steps"][name] = {"ok": False, "error": str(e)}
            import traceback

            status["steps"][name]["trace"] = traceback.format_exc()

    want = set()
    if prompts in ("all", "0-8", "08"):
        want = set(range(9))
    else:
        for part in prompts.replace(" ", "").split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                want.update(range(int(a), int(b) + 1))
            else:
                want.add(int(part))

    if 0 in want:
        _step("prompt0", lambda: run_prompt0(project))
    if 1 in want:
        _step("prompt1", lambda: run_prompt1(project, df, cfg, skip_train, eval_max, quick))
    if 2 in want:
        _step("prompt2", lambda: export_per_case_predictions(project, df, cfg, device))
    if 3 in want:
        _step("prompt3", lambda: run_prompt3(project))
    if 4 in want:
        _step("prompt4", lambda: run_prompt4(project))
    if 5 in want:
        _step("prompt5", lambda: run_prompt5_figures(project))
    if 6 in want:
        _step("prompt6", lambda: run_prompt6(project, df))
    if 7 in want:
        _step("prompt7", lambda: run_prompt7(project))
    if 8 in want:
        _step("prompt8", lambda: run_prompt8(project))

    # Refresh manuscript tables
    try:
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, str(project / "scripts/28_aggregate_manuscript_result_tables.py")],
            cwd=str(project),
            check=False,
        )
        status["steps"]["aggregate_tables"] = {"ok": True}
    except Exception as e:
        status["steps"]["aggregate_tables"] = {"ok": False, "error": str(e)}

    status["elapsed_minutes"] = (time.time() - t0) / 60
    log = project / "outputs/publishable/logs/jbd_next_prompts_status.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(status, indent=2), encoding="utf-8")
    (project / "outputs/publishable/logs/JBD_NEXT_PROMPTS_STATUS.md").write_text(
        f"# JBD Next Prompts Run\n\n```json\n{json.dumps(status, indent=2)}\n```\n", encoding="utf-8"
    )
    return status
