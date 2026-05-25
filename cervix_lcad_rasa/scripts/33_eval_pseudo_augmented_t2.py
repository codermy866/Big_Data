#!/usr/bin/env python3
"""Exp 7: eval pseudo-augmented (publishable_lcad_augmented) → T2 row + LaTeX refresh."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supplementary.jbd_next_runner import (  # noqa: E402
    CORE_MODELS,
    PRED_DIR,
    _auc_ci,
    _metric_at_thr_ci,
    _paired_bootstrap_auc_p,
    _mcnemar,
)
from src.supplementary.train_eval import evaluate_experiment, load_jbd_config, resolve_checkpoint

EXP_ID = "pseudo_augmented_lcad"
DISPLAY = "Pseudo-augmented (LCAD)"
ALIAS_CKPT = "publishable_lcad_augmented"
MS = ROOT / "outputs/publishable/tables/manuscript"
TABLES = ROOT / "outputs/publishable/tables"
PRED_ROOT = ROOT / PRED_DIR


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest(project: Path) -> pd.DataFrame:
    cfg = load_jbd_config(project)
    rel = cfg.get("manifest", "outputs/publishable/manifests/full_manifest_publishable_with_llm_pseudo.csv")
    p = project / rel
    if not p.is_file():
        p = project / "outputs/publishable/manifests/full_manifest_publishable.csv"
    return pd.read_csv(p)


def _export_predictions(project: Path, df: pd.DataFrame, cfg: dict, device: str) -> pd.DataFrame:
    import torch
    from src.models_publishable.lcad_rasa_model import instr_vector, load_visual_emb
    from src.supplementary.next_stage.core import collect_risk_scores, select_thresholds
    from src.supplementary.train_eval import build_model

    cfg = dict(cfg)
    cfg.setdefault("checkpoint_aliases", {})[EXP_ID] = ALIAS_CKPT

    baselines_dir = project / "outputs/publishable/baselines"
    ckpt = resolve_checkpoint(project, EXP_ID, cfg, baselines_dir)
    if ckpt is None:
        raise FileNotFoundError(f"Checkpoint not found for {EXP_ID} -> {ALIAS_CKPT}")

    test = df[df["split"] == "test"].copy()
    val = df[df["split"] == "validation"].copy()
    if len(val) == 0:
        val = df[df["split"] == "train"].sample(min(200, len(df)), random_state=42)

    thresh_df = pd.read_csv(TABLES / "table_threshold_tuned_test_metrics.csv")
    thresh_map = {}
    if "threshold_type" in thresh_df.columns:
        sub = thresh_df[thresh_df["threshold_type"] == "max_f1"]
        for _, r in sub.iterrows():
            thresh_map[str(r["experiment_id"])] = float(r["selected_threshold"])

    dev = torch.device(device)
    state = torch.load(ckpt, map_location="cpu")
    spec = state.get("spec") or {}
    yv_t, yv_s = collect_risk_scores(ckpt, val, spec, dev)
    thrs = select_thresholds(yv_t, yv_s)
    thr = thresh_map.get(EXP_ID, thrs["max_f1"])

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
    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    out_csv = PRED_ROOT / f"{EXP_ID}_test_predictions.csv"
    pred_df.to_csv(out_csv, index=False)

    idx_path = PRED_ROOT / "PER_CASE_PREDICTION_INDEX.csv"
    if idx_path.is_file():
        idx = pd.read_csv(idx_path)
        idx = idx[idx["model_name"] != EXP_ID]
    else:
        idx = pd.DataFrame()
    new_row = pd.DataFrame(
        [
            {
                "model_name": EXP_ID,
                "display_name": DISPLAY,
                "status": "ok",
                "n": len(pred_df),
                "positives": int(pred_df["y_true_cin2plus"].sum()),
                "threshold": thr,
                "path": str(out_csv.relative_to(project)),
            }
        ]
    )
    pd.concat([idx, new_row], ignore_index=True).to_csv(idx_path, index=False)
    return pred_df


def _append_stratified_and_baseline(project: Path, df: pd.DataFrame, cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg.setdefault("checkpoint_aliases", {})[EXP_ID] = ALIAS_CKPT
    baselines_dir = project / "outputs/publishable/baselines"
    ckpt = resolve_checkpoint(project, EXP_ID, cfg, baselines_dir)
    test = df[df["split"] == "test"].copy()

    strat_rows = []
    for subset_name, subset_df in [
        ("with_reference", test[(test["has_real_report"] == 1) & (test["reference_report_text"].astype(str).str.len() >= 20)]),
        ("without_reference", test[(test["has_real_report"] != 1) | (test["reference_report_text"].astype(str).str.len() < 20)]),
        ("all", test),
    ]:
        if len(subset_df) == 0:
            continue
        ev = evaluate_experiment(project, df, EXP_ID, ckpt, test_df=subset_df)
        ev["subset"] = subset_name
        ev["n_subset"] = len(subset_df)
        strat_rows.append(ev)

    strat_path = TABLES / "table_reference_stratified_evaluation.csv"
    strat_old = pd.read_csv(strat_path)
    strat_old = strat_old[strat_old["experiment_id"] != EXP_ID]
    strat_new = pd.concat([strat_old, pd.DataFrame(strat_rows)], ignore_index=True)
    strat_new.to_csv(strat_path, index=False)

    ev_all = evaluate_experiment(project, df, EXP_ID, ckpt)
    n_train = torch_load_n_train(ckpt)

    base_path = TABLES / "table_baseline_comparison.csv"
    base_old = pd.read_csv(base_path)
    base_old = base_old[base_old["experiment_id"] != EXP_ID]
    base_row = {c: ev_all.get(c, np.nan) for c in base_old.columns}
    base_row["experiment_id"] = EXP_ID
    base_row["status"] = "ok"
    base_row["checkpoint"] = str(ckpt)
    base_row["n_train"] = n_train
    base_row["n_real"] = 520
    base_row["n_pseudo"] = 805
    base_row["train_cases"] = n_train
    pd.concat([base_old, pd.DataFrame([base_row])], ignore_index=True).to_csv(base_path, index=False)
    return ev_all


def torch_load_n_train(ckpt: Path) -> int:
    import torch

    s = torch.load(ckpt, map_location="cpu")
    return int(s.get("n_train", 1325))


def _update_t2_ci(project: Path) -> dict:
    pred = pd.read_csv(PRED_ROOT / f"{EXP_ID}_test_predictions.csv")
    yt = pred["y_true_cin2plus"].values.astype(int)
    ys = pred["risk_score"].values.astype(float)
    thr = float(pred["threshold_val_selected"].iloc[0])
    auc = _auc_ci(yt, ys)
    thr_m = _metric_at_thr_ci(yt, ys, thr)

    ci_path = MS / "T2_main_model_comparison_with_ci.csv"
    ci = pd.read_csv(ci_path)
    ci = ci[~ci["model"].str.contains("Pseudo-augmented", case=False, na=False)]
    new_row = {
        "model": DISPLAY,
        "n": len(pred),
        "auc": round(auc["auc"], 4),
        "auc_ci_low": round(auc["ci_low"], 4),
        "auc_ci_high": round(auc["ci_high"], 4),
        "f1": round(thr_m["f1"], 4),
        "f1_ci_low": round(thr_m["f1_ci_low"], 4),
        "f1_ci_high": round(thr_m["f1_ci_high"], 4),
        "threshold": thr,
    }
    ci = pd.concat([ci, pd.DataFrame([new_row])], ignore_index=True)
    ci.to_csv(ci_path, index=False)

    ref_p = PRED_ROOT / "full_lcad_rasa_test_predictions.csv"
    pair_path = MS / "T2_pairwise_statistical_tests.csv"
    if ref_p.is_file():
        ref_d = pd.read_csv(ref_p)
        merged = ref_d[["case_id", "y_true_cin2plus", "risk_score", "pred_label"]].merge(
            pred[["case_id", "risk_score", "pred_label"]],
            on="case_id",
            suffixes=("_ref", "_cmp"),
        )
        y = merged["y_true_cin2plus"].values.astype(int)
        delta, p_auc = _paired_bootstrap_auc_p(y, merged["risk_score_ref"].values, y, merged["risk_score_cmp"].values)
        mcn = _mcnemar(y, merged["pred_label_ref"].values.astype(int), merged["pred_label_cmp"].values.astype(int))
        pw = pd.read_csv(pair_path)
        pw = pw[~pw["comparison"].str.contains("Pseudo-augmented", na=False)]
        pw = pd.concat(
            [
                pw,
                pd.DataFrame(
                    [
                        {
                            "comparison": f"{DISPLAY} vs Full LCAD-RASA",
                            "reference": "Full LCAD-RASA",
                            "comparator": DISPLAY,
                            "delta_auc": delta,
                            "bootstrap_p_auc": p_auc,
                            "mcnemar_p_at_val_threshold": mcn,
                            "n_paired": len(merged),
                            "test": "paired bootstrap (DeLong not used)",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        pw.to_csv(pair_path, index=False)

    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib

    agg = importlib.import_module("28_aggregate_manuscript_result_tables")
    agg.build_t2_main_comparison()

    t2 = pd.read_csv(MS / "T2_main_model_comparison.csv")
    row = t2[t2["experiment_id"] == EXP_ID] if "experiment_id" in t2.columns else pd.DataFrame()
    return {"ci_row": new_row, "t2_row": row.to_dict("records")[0] if len(row) else {}, **auc, **{k: thr_m[k] for k in ("f1", "threshold")}}


def _patch_latex(ci_row: dict) -> None:
    tex = ROOT / "outputs/publishable/manuscript_latex/04_TABLES_AND_FIGURES.tex"
    if not tex.is_file():
        return
    text = tex.read_text(encoding="utf-8")
    if "Pseudo-augmented" in text:
        return
    insert = (
        f"    Pseudo-augmented (LCAD) & "
        f"{ci_row['auc']:.3f} [{ci_row['auc_ci_low']:.3f}, {ci_row['auc_ci_high']:.3f}]"
        f" & "
        f"{ci_row['f1']:.3f} [{ci_row['f1_ci_low']:.3f}, {ci_row['f1_ci_high']:.3f}]"
        f" & {ci_row['threshold']:.2f}"
        f" \\\\\n"
    )
    marker = "    Real-report only &"
    if marker in text:
        text = text.replace(marker, insert + "    Real-report only &")
        tex.write_text(text, encoding="utf-8")

    res_tex = ROOT / "outputs/publishable/manuscript_latex/03_RESULTS_LCAD_RASA.tex"
    if res_tex.is_file() and "Pseudo-augmented" not in res_tex.read_text(encoding="utf-8"):
        block = (
            f"Pseudo-augmented (LCAD) training (real reports plus QC-weighted pseudo reports) achieved AUROC "
            f"{ci_row['auc']:.3f} (95\\% CI {ci_row['auc_ci_low']:.3f}--{ci_row['auc_ci_high']:.3f}) and F1 "
            f"{ci_row['f1']:.3f} at threshold {ci_row['threshold']:.2f}, compared with real-report-only AUROC 0.725 "
            f"and Full LCAD--RASA AUROC 0.832 (Table~\\ref{{tab:main_comparison}}). "
            f"Paired testing did not support a superiority claim versus Full LCAD--RASA.\n"
        )
        old = "Real-report-only training yielded AUROC"
        res = res_tex.read_text(encoding="utf-8")
        res_tex.write_text(res.replace(old, block + old), encoding="utf-8")

    locked = ROOT / "outputs/publishable/manuscript_latex/00_LOCKED_NUMBERS.json"
    if locked.is_file():
        data = json.loads(locked.read_text(encoding="utf-8"))
        data["pseudo_augmented"] = {
            "auc": ci_row["auc"],
            "auc_ci": [ci_row["auc_ci_low"], ci_row["auc_ci_high"]],
            "f1": ci_row["f1"],
            "f1_ci": [ci_row["f1_ci_low"], ci_row["f1_ci_high"]],
            "threshold": ci_row["threshold"],
        }
        data["updated_utc"] = _utc()
        locked.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    project = ROOT
    cfg = load_jbd_config(project)
    cfg.setdefault("checkpoint_aliases", {})[EXP_ID] = ALIAS_CKPT

    df = _load_manifest(project)
    print(f"[1/4] Export per-case predictions ({EXP_ID}) …")
    _export_predictions(project, df, cfg, args.device)

    print("[2/4] Stratified + baseline tables …")
    _append_stratified_and_baseline(project, df, cfg)

    print("[3/4] Update T2 + pairwise …")
    stats = _update_t2_ci(project)
    print(f"  AUROC {stats['auc']:.3f} [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]  F1 {stats['f1']:.3f}  thr {stats['threshold']:.2f}")

    print("[4/4] Patch LaTeX …")
    _patch_latex(stats["ci_row"])

    print("Done. Re-run: python scripts/32_generate_manuscript_latex.py  (optional full refresh)")


if __name__ == "__main__":
    main()
