#!/usr/bin/env python3
"""Audit contrastive-teacher distillation for LCAD-RASA.

The experiment is intentionally gated: it writes an internal distillation audit
and only marks a setting as manuscript-promotable if it improves both AUROC and
F1 over the locked Full LCAD-RASA reference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models_publishable.lcad_rasa_model import PublishableLCADRASA
from src.supplementary.next_stage.core import collect_risk_scores, metrics_at_threshold, select_thresholds
from src.supplementary.train_eval import filter_train_df
from src.training.publishable_dataset import PublishableDataset

OUT = ROOT / "outputs/publishable/distillation"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PREDICTIONS = OUT / "predictions"
CHECKPOINTS = OUT / "checkpoints"
SUMMARY = OUT / "DISTILLATION_AUDIT_SUMMARY.md"
MANIFEST = ROOT / "outputs/publishable/manifests/full_manifest_publishable_with_llm_pseudo.csv"
CONFIG = ROOT / "configs/jbd_supplementary_experiments.yaml"
FULL_CKPT = ROOT / "outputs/publishable/checkpoints/publishable_full_lcad_rasa/best.ckpt"
FULL_PRED = ROOT / "outputs/publishable/predictions/final_per_case/full_lcad_rasa_test_predictions.csv"
FULL_TABLE = ROOT / "outputs/publishable/tables/manuscript/T2_main_model_comparison_with_ci.csv"
EXT_SCRIPT = ROOT / "scripts/42_run_external_baseline_block.py"

PALETTE = ["#2f5f8f", "#8fb8d8", "#d9a066", "#efd7b5", "#9e3f3a", "#d47f6f", "#7f7f7f", "#d6d6d6"]


def load_external_module():
    spec = importlib.util.spec_from_file_location("external_baseline_block", EXT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["external_baseline_block"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_student() -> PublishableLCADRASA:
    return PublishableLCADRASA(
        use_risk_head=True,
        use_section_align=True,
        use_oct=True,
        use_colposcopy=True,
        use_instruction=True,
        use_fused_visual=True,
    )


def load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def train_teacher_scores(df: pd.DataFrame, seed: int, epochs: int) -> tuple[dict[str, dict[str, float]], dict]:
    """Train the same contrastive teacher used in the external baseline block."""
    ext = load_external_module()
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    y_train = train_df["binary_label"].to_numpy(dtype=int)
    y_val = val_df["binary_label"].to_numpy(dtype=int)
    y_test = test_df["binary_label"].to_numpy(dtype=int)

    oct_tr, oct_va, oct_te = [ext.load_embedding_matrix(x, "oct_embedding_path") for x in (train_df, val_df, test_df)]
    col_tr, col_va, col_te = [ext.load_embedding_matrix(x, "colposcopy_embedding_path") for x in (train_df, val_df, test_df)]
    oct_tr, oct_va, oct_te = ext.standardize(oct_tr, oct_va, oct_te)
    col_tr, col_va, col_te = ext.standardize(col_tr, col_va, col_te)
    clin_tr, clin_va, clin_te = ext.transformed_clinical_arrays(train_df, val_df, test_df)

    teacher = ext.ContrastiveFusionClassifier(oct_tr.shape[1], col_tr.shape[1], clin_tr.shape[1])
    res = ext.train_torch_model(
        teacher,
        (oct_tr, col_tr, clin_tr),
        (oct_va, col_va, clin_va),
        y_train,
        y_val,
        mode="contrastive",
        seed=seed,
        epochs=epochs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device)
    scores = {
        "train": ext.predict_torch(teacher, (oct_tr, col_tr, clin_tr), "contrastive", device),
        "val": res.val_score,
        "test": ext.predict_torch(teacher, (oct_te, col_te, clin_te), "contrastive", device),
    }
    frames = {"train": train_df, "val": val_df, "test": test_df}
    ys = {"train": y_train, "val": y_val, "test": y_test}
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    teacher_maps: dict[str, dict[str, float]] = {}
    meta = {"teacher": "CLIP-style contrastive multimodal baseline", "teacher_epochs": epochs}
    for split, part in frames.items():
        out = pd.DataFrame(
            {
                "case_id": part["case_id"].astype(str).values,
                "split": split,
                "y_true_cin2plus": ys[split],
                "teacher_score": scores[split],
            }
        )
        out.to_csv(PREDICTIONS / f"contrastive_teacher_{split}_scores.csv", index=False)
        teacher_maps[split] = dict(zip(out["case_id"], out["teacher_score"]))
    thr = ext.select_val_threshold(y_val, scores["val"])
    test_metrics = ext.metric_dict(y_test, scores["test"], thr)
    meta.update({f"teacher_{k}": v for k, v in test_metrics.items()})
    return teacher_maps, meta


def train_distilled_student(
    df: pd.DataFrame,
    cfg: dict,
    teacher_train: dict[str, float],
    distill_weight: float,
    seed: int,
    epochs: int,
    max_steps: int,
    init_full: bool,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    spec = {
        "train_filter": {"training_eligible": 1},
        "use_pseudo_report": True,
        "use_real_report": True,
        "require_qc_pass": True,
        "use_report_loss": True,
        "model": {"use_section_align": True, "use_risk_head": True},
        "loss": {
            "ce_weight": 1.0,
            "rasa_weight": 0.5,
            "cls_weight": 0.2,
            "cons_weight": 0.1,
            "distill_weight": distill_weight,
        },
    }
    train_df = filter_train_df(df, spec)
    ds = PublishableDataset(
        train_df,
        max_len=128,
        use_pseudo_report=True,
        use_real_report=True,
        require_qc_pass=True,
        weight_mode="default",
        use_report_loss=True,
        min_weight=0.05,
    )
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_student().to(device)
    if init_full and FULL_CKPT.is_file():
        state = torch.load(FULL_CKPT, map_location="cpu")
        model.load_state_dict(state["model"], strict=False)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]) * 0.5)
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_distill, steps = 0.0, 0.0, 0
        for step, batch in enumerate(loader):
            if step >= max_steps:
                break
            case_ids = batch["case_id"]
            oct_e = batch["oct_emb"].to(device)
            col_e = batch["col_emb"].to(device)
            fus_e = batch["fused_emb"].to(device)
            instr = batch["instr"].to(device)
            ids = batch["input_ids"].to(device)
            tgt = batch["target_ids"].to(device)
            lab = batch["labels"].to(device)
            w = batch["weight"].to(device)
            out = model(oct_e, col_e, fus_e, instr, ids, lab)
            ce = F.cross_entropy(
                out["logits"].reshape(-1, out["logits"].size(-1)),
                tgt.reshape(-1),
                reduction="none",
            )
            ce = (ce.view(tgt.size(0), -1).mean(1) * w).mean()
            align = model.section_alignment_loss(out["fused"], out["hidden"])
            risk_logit = out["risk_logit"].squeeze(-1)
            risk = F.binary_cross_entropy_with_logits(risk_logit, lab.float())
            pred_risk = torch.sigmoid(risk_logit)
            cons = (pred_risk - lab.float()).abs().mean()
            teacher = torch.tensor([float(teacher_train[str(c)]) for c in case_ids], dtype=torch.float32, device=device)
            distill = F.binary_cross_entropy_with_logits(risk_logit, teacher)
            loss = ce + 0.5 * align + 0.2 * risk + 0.1 * cons + distill_weight * distill
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            epoch_distill += float(distill.item())
            steps += 1
        history.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss / max(steps, 1),
                "distill": epoch_distill / max(steps, 1),
                "steps": steps,
            }
        )

    ckpt_dir = CHECKPOINTS / f"lcad_rasa_contrastive_distill_w{distill_weight:g}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "best.ckpt"
    torch.save(
        {
            "model": model.state_dict(),
            "experiment": f"lcad_rasa_contrastive_distill_w{distill_weight:g}",
            "n_train": len(ds),
            "spec": spec,
            "loss_cfg": spec["loss"],
            "distillation": {"teacher": "contrastive_multimodal_no_report_sections", "distill_weight": distill_weight},
            "history": history,
        },
        ckpt,
    )
    return {
        "checkpoint": ckpt,
        "n_train": len(ds),
        "train_seconds": time.time() - t0,
        "final_loss": history[-1]["loss"] if history else np.nan,
        "final_distill_loss": history[-1]["distill"] if history else np.nan,
        "spec": spec,
    }


def write_test_predictions(
    experiment_id: str,
    test_df: pd.DataFrame,
    y_true: list[int],
    y_score: list[float],
    threshold: float,
    ckpt: Path,
) -> Path:
    ys = np.asarray(y_score, dtype=float)
    yt = np.asarray(y_true, dtype=int)
    pred = (ys >= threshold).astype(int)
    out = pd.DataFrame(
        {
            "case_id": test_df["case_id"].astype(str).values[: len(yt)],
            "center": test_df["center_id"].astype(str).values[: len(yt)],
            "split": "test",
            "y_true_cin2plus": yt,
            "risk_score": ys,
            "threshold_val_selected": threshold,
            "pred_label": pred,
            "correct": (pred == yt).astype(int),
            "source_checkpoint": str(ckpt),
            "evaluation_protocol": "distillation_audit_same_split_val_threshold_max_f1",
        }
    )
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS / f"{experiment_id}_test_predictions.csv"
    out.to_csv(path, index=False)
    return path


def paired_against_full(pred_path: Path, ext, bootstrap: int, seed: int) -> dict:
    full = pd.read_csv(FULL_PRED)[["case_id", "y_true_cin2plus", "risk_score"]].rename(columns={"risk_score": "full_score"})
    cmp_df = pd.read_csv(pred_path)[["case_id", "y_true_cin2plus", "risk_score"]].rename(columns={"risk_score": "distill_score"})
    merged = full.merge(cmp_df[["case_id", "distill_score"]], on="case_id", how="inner")
    y = merged["y_true_cin2plus"].to_numpy(dtype=int)
    return ext.corrected_paired_bootstrap(
        y,
        merged["full_score"].to_numpy(dtype=float),
        merged["distill_score"].to_numpy(dtype=float),
        n_boot=bootstrap,
        seed=seed,
    )


def plot_sweep(rows: pd.DataFrame) -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font="Arial",
        palette=PALETTE,
        rc={
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.55,
        },
    )
    plot = rows.melt(
        id_vars=["distill_weight", "decision"],
        value_vars=["auc", "f1"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sns.lineplot(data=plot, x="distill_weight", y="value", hue="metric", marker="o", linewidth=2.0, ax=ax)
    ax.axhline(0.8324, color=PALETTE[0], ls="--", lw=1.2, alpha=0.8)
    ax.axhline(0.6111, color=PALETTE[2], ls="--", lw=1.2, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Distillation weight")
    ax.set_ylabel("Held-out metric")
    ax.set_title("Contrastive-teacher distillation audit")
    ax.legend(title="Metric")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "Figure_distillation_weight_sweep.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "Figure_distillation_weight_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="0.05,0.10,0.20,0.50")
    p.add_argument("--teacher-epochs", type=int, default=80)
    p.add_argument("--student-epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-init-full", action="store_true")
    args = p.parse_args()
    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    TABLES.mkdir(parents=True, exist_ok=True)
    ext = load_external_module()
    cfg = load_cfg()
    df = pd.read_csv(MANIFEST)
    teacher_maps, teacher_meta = train_teacher_scores(df, seed=args.seed, epochs=args.teacher_epochs)
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    full_row = pd.read_csv(FULL_TABLE)
    ref = full_row[full_row["model"].eq("Full LCAD-RASA")].iloc[0]
    ref_auc, ref_f1 = float(ref["auc"]), float(ref["f1"])
    rows = []
    for w in weights:
        exp_id = f"lcad_rasa_contrastive_distill_w{w:g}"
        train_info = train_distilled_student(
            df,
            cfg,
            teacher_maps["train"],
            distill_weight=w,
            seed=args.seed,
            epochs=args.student_epochs,
            max_steps=args.max_steps,
            init_full=not args.no_init_full,
        )
        ckpt = Path(train_info["checkpoint"])
        yv, sv = collect_risk_scores(ckpt, val_df, train_info["spec"], torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        thr = select_thresholds(yv, sv)["max_f1"]
        yt, st = collect_risk_scores(ckpt, test_df, train_info["spec"], torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        metrics = metrics_at_threshold(yt, st, thr)
        ci = ext.bootstrap_ci(np.asarray(yt), np.asarray(st), thr, n_boot=args.bootstrap, seed=args.seed)
        pred_path = write_test_predictions(exp_id, test_df, yt, st, thr, ckpt)
        paired = paired_against_full(pred_path, ext, bootstrap=args.bootstrap, seed=args.seed)
        promote = bool(metrics["auc"] > ref_auc and metrics["f1"] > ref_f1)
        rows.append(
            {
                "experiment_id": exp_id,
                "teacher": "CLIP-style contrastive multimodal baseline",
                "distill_weight": w,
                "initialized_from_full_lcad_rasa": not args.no_init_full,
                "n_train": train_info["n_train"],
                "auc": metrics["auc"],
                "auc_ci_low": ci["auc_ci_low"],
                "auc_ci_high": ci["auc_ci_high"],
                "f1": metrics["f1"],
                "f1_ci_low": ci["f1_ci_low"],
                "f1_ci_high": ci["f1_ci_high"],
                "threshold": thr,
                "delta_auc_vs_full": metrics["auc"] - ref_auc,
                "delta_f1_vs_full": metrics["f1"] - ref_f1,
                "paired_delta_auc_full_minus_distilled": paired["delta_auc_full_minus_comparator"],
                "paired_delta_auc_ci_low": paired["delta_auc_ci_low"],
                "paired_delta_auc_ci_high": paired["delta_auc_ci_high"],
                "paired_bootstrap_p_two_sided": paired["paired_bootstrap_p_two_sided"],
                "train_seconds": train_info["train_seconds"],
                "final_loss": train_info["final_loss"],
                "final_distill_loss": train_info["final_distill_loss"],
                "checkpoint": str(ckpt.relative_to(ROOT)),
                "prediction_file": str(pred_path.relative_to(ROOT)),
                "decision": "promote_to_manuscript" if promote else "do_not_promote",
            }
        )
    out = pd.DataFrame(rows).sort_values(["decision", "auc", "f1"], ascending=[True, False, False])
    out.to_csv(TABLES / "T_contrastive_teacher_distillation_sweep.csv", index=False)
    best = out.sort_values(["auc", "f1"], ascending=False).iloc[0]
    decision = {
        "reference_full_lcad_rasa": {"auc": ref_auc, "f1": ref_f1},
        "teacher_meta": teacher_meta,
        "best_distilled": best.to_dict(),
        "promote_any": bool((out["decision"] == "promote_to_manuscript").any()),
        "rule": "promote only if distilled AUROC and F1 both exceed locked Full LCAD-RASA",
    }
    (TABLES / "T_contrastive_teacher_distillation_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    plot_sweep(out)
    lines = [
        "# Contrastive-Teacher Distillation Audit\n\n",
        "## Decision rule\n\n",
        "Promote distillation to the manuscript only if a distilled LCAD--RASA variant improves both AUROC and F1 over the locked Full LCAD--RASA reference.\n\n",
        f"- Full LCAD--RASA reference: AUROC {ref_auc:.4f}, F1 {ref_f1:.4f}\n",
        f"- Best distilled variant: `{best['experiment_id']}`; AUROC {float(best['auc']):.4f}, F1 {float(best['f1']):.4f}\n",
        f"- Decision: **{decision['best_distilled']['decision']}**\n\n",
        "## Output files\n\n",
        f"- `{TABLES / 'T_contrastive_teacher_distillation_sweep.csv'}`\n",
        f"- `{TABLES / 'T_contrastive_teacher_distillation_decision.json'}`\n",
        f"- `{FIGURES / 'Figure_distillation_weight_sweep.pdf'}`\n",
    ]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "promote_any": decision["promote_any"], "summary": str(SUMMARY), "best": decision["best_distilled"]}, indent=2))


if __name__ == "__main__":
    main()
