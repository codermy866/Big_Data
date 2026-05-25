"""Unified train/eval for JBD supplementary experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.evaluation.metrics import label_consistency
from src.evaluation_publishable.clinical_consistency import clinical_metrics
from src.evaluation_publishable.hallucination import hallucination_flags, hallucination_rates
from src.evaluation_publishable.report_metrics import aggregate_metrics, compute_reference_metrics
from src.evaluation_publishable.perturbation_metrics import section_similarity
from src.models_publishable.lcad_rasa_model import PublishableLCADRASA, instr_vector, load_visual_emb
from src.training.experiment_modes import apply_train_filter
from src.training.publishable_dataset import PublishableDataset
from src.utils.config import resolve_project_root


def load_jbd_config(project: Path) -> dict:
    p = project / "configs/jbd_supplementary_experiments.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def filter_train_df(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    flt = spec.get("train_filter", {"training_eligible": 1})
    out = df[df["split"] == "train"].copy() if "split" in df.columns else df.copy()
    if flt.get("has_real_report"):
        out = out[out["has_real_report"] == 1]
    if flt.get("center_id"):
        out = out[out["center_id"] == flt["center_id"]]
    if flt.get("center_id_in"):
        out = out[out["center_id"].isin(flt["center_id_in"])]
    if flt.get("training_eligible"):
        out = apply_train_filter(out, {"training_eligible": 1})
    return out


def build_model(spec: dict) -> PublishableLCADRASA:
    m = spec.get("model", {})
    return PublishableLCADRASA(
        use_risk_head=m.get("use_risk_head", True),
        use_section_align=m.get("use_section_align", True),
        use_oct=m.get("use_oct", True),
        use_colposcopy=m.get("use_colposcopy", True),
        use_instruction=m.get("use_instruction", True),
        use_fused_visual=m.get("use_fused_visual", True),
    )


def train_experiment(
    project: Path,
    manifest_df: pd.DataFrame,
    experiment_id: str,
    spec: dict,
    cfg: dict,
    out_ckpt_dir: Path,
    seed: int = 42,
    train_df_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    train_df = train_df_override if train_df_override is not None else filter_train_df(manifest_df, spec)
    ds = PublishableDataset(
        train_df,
        max_len=128,
        use_pseudo_report=spec.get("use_pseudo_report", True),
        use_real_report=spec.get("use_real_report", True),
        require_qc_pass=spec.get("require_qc_pass", True),
        weight_mode=spec.get("weight_mode", "default"),
        use_report_loss=spec.get("use_report_loss", True),
        min_weight=0.0 if spec.get("weight_mode") == "pseudo_all_no_qc" else 0.05,
    )
    if len(ds) == 0:
        return {"experiment_id": experiment_id, "status": "failed", "error": "empty_train_dataset"}

    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(spec).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    loss_cfg = spec.get("loss", {})
    history = []
    for epoch in range(int(cfg["training"]["num_epochs"])):
        model.train()
        eloss, steps = 0.0, 0
        for step, batch in enumerate(loader):
            if step >= int(cfg["training"]["max_steps_per_epoch"]):
                break
            oct_e = batch["oct_emb"].to(device)
            col_e = batch["col_emb"].to(device)
            fus_e = batch["fused_emb"].to(device)
            instr = batch["instr"].to(device)
            ids = batch["input_ids"].to(device)
            tgt = batch["target_ids"].to(device)
            lab = batch["labels"].to(device)
            w = batch["weight"].to(device)
            out = model(oct_e, col_e, fus_e, instr, ids, lab)
            loss = torch.tensor(0.0, device=device)
            if float(loss_cfg.get("ce_weight", 1.0)) > 0 and spec.get("use_report_loss", True):
                ce = F.cross_entropy(
                    out["logits"].reshape(-1, out["logits"].size(-1)),
                    tgt.reshape(-1),
                    reduction="none",
                )
                ce = (ce.view(tgt.size(0), -1).mean(1) * w).mean()
                loss = loss + float(loss_cfg["ce_weight"]) * ce
            if float(loss_cfg.get("rasa_weight", 0.0)) > 0:
                loss = loss + float(loss_cfg["rasa_weight"]) * model.section_alignment_loss(out["fused"], out["hidden"])
            if out.get("risk_logit") is not None and float(loss_cfg.get("cls_weight", 0.0)) > 0:
                loss = loss + float(loss_cfg["cls_weight"]) * F.binary_cross_entropy_with_logits(
                    out["risk_logit"].squeeze(-1), lab.float()
                )
            if float(loss_cfg.get("cons_weight", 0.0)) > 0:
                pred_risk = torch.sigmoid(out["risk_logit"].squeeze(-1)) if out.get("risk_logit") is not None else lab.float()
                cons = (pred_risk - lab.float()).abs().mean()
                loss = loss + float(loss_cfg["cons_weight"]) * cons
            optim.zero_grad()
            loss.backward()
            optim.step()
            eloss += float(loss.item())
            steps += 1
        history.append({"epoch": epoch + 1, "loss": eloss / max(steps, 1)})

    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_ckpt_dir / "best.ckpt"
    torch.save(
        {
            "model": model.state_dict(),
            "experiment": experiment_id,
            "n_train": len(ds),
            "spec": spec,
            "loss_cfg": loss_cfg,
        },
        ckpt,
    )
    minutes = (time.time() - t0) / 60.0
    return {
        "experiment_id": experiment_id,
        "status": "ok",
        "n_train": len(ds),
        "n_real": int((train_df["training_report_type"] == "real").sum()) if "training_report_type" in train_df else 0,
        "n_pseudo": int((train_df["training_report_type"] == "pseudo").sum()) if "training_report_type" in train_df else 0,
        "training_time_minutes": minutes,
        "final_train_loss": history[-1]["loss"] if history else None,
        "checkpoint": str(ckpt),
    }


def _risk_metrics(y_true: list[int], y_score: list[float]) -> dict[str, float]:
    if not y_true:
        return {"auc": 0.0, "sensitivity": 0.0, "specificity": 0.0, "f1": 0.0, "ece": 0.0, "brier": 0.0}
    yt = np.array(y_true)
    ys = np.array(y_score)
    pred = (ys >= 0.5).astype(int)
    tp = ((pred == 1) & (yt == 1)).sum()
    tn = ((pred == 0) & (yt == 0)).sum()
    fp = ((pred == 1) & (yt == 0)).sum()
    fn = ((pred == 0) & (yt == 1)).sum()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(yt, ys))
    except Exception:
        auc = 0.5
    brier = float(np.mean((ys - yt) ** 2))
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        m = (ys >= bins[i]) & (ys < bins[i + 1])
        if m.sum() == 0:
            continue
        ece += m.sum() / len(ys) * abs(ys[m].mean() - yt[m].mean())
    return {"auc": auc, "sensitivity": float(sens), "specificity": float(spec), "f1": float(f1), "ece": float(ece), "brier": brier}


def evaluate_experiment(
    project: Path,
    manifest_df: pd.DataFrame,
    experiment_id: str,
    ckpt_path: Path,
    spec: dict | None = None,
    test_df: pd.DataFrame | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    spec = spec or {}
    test = test_df if test_df is not None else manifest_df[manifest_df["split"] == "test"]
    if max_cases:
        test = test.head(max_cases)
    device = torch.device("cpu")
    state = torch.load(ckpt_path, map_location="cpu")
    ckpt_spec = state.get("spec") or spec or {}
    model = build_model(ckpt_spec)
    model.load_state_dict(state["model"], strict=False)
    model.to(device)
    model.eval()

    rows_ref, flags_all, inf_times = [], [], []
    y_true, y_score = [], []
    lc_scores, contra_scores = [], []
    section_scores = {k: [] for k in ("oct_findings", "colposcopy_findings", "clinical_context", "impression")}
    required_sections = ("diagnostic_summary", "oct_findings", "colposcopy_findings", "clinical_context", "impression", "recommendation")
    complete_n = 0

    for _, row in test.iterrows():
        t_case = time.time()
        oct_e = torch.tensor(load_visual_emb(str(row.get("oct_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
        col_e = torch.tensor(load_visual_emb(str(row.get("colposcopy_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
        fus_e = torch.tensor(load_visual_emb(str(row.get("fused_visual_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
        instr = torch.tensor(instr_vector(row.to_dict()), dtype=torch.float32, device=device).unsqueeze(0)
        label = int(row["binary_label"])
        gen = model.generate_structured_report(oct_e, col_e, fus_e, instr, label, row.to_dict(), {})
        sections = gen["generated_sections"]
        pred_text = gen["generated_report_text"]
        ref = str(row.get("reference_report_text", "")) if int(row.get("has_real_report", 0)) else ""
        inf_times.append(time.time() - t_case)
        flags_all.append(hallucination_flags(sections, "normal"))
        y_true.append(label)
        y_score.append(float(gen["risk_score"]))
        cm = clinical_metrics(pred_text, label)
        lc_scores.append(cm["label_consistency"])
        contra_scores.append(cm["contradiction_rate"])
        if all(len(str(sections.get(s, "")).strip()) > 20 for s in required_sections):
            complete_n += 1
        if ref and len(ref) >= 20:
            rows_ref.append({**compute_reference_metrics(pred_text, ref), "label_consistency": cm["label_consistency"]})
        for sk in section_scores:
            ref_sec = str(row.get(f"reference_{sk}", "")) or sections.get(sk, "")
            section_scores[sk].append(section_similarity(sections.get(sk, ""), ref_sec if ref_sec else sections.get(sk, "")))

    hall = hallucination_rates(flags_all)
    ref_agg = aggregate_metrics(rows_ref) if rows_ref else {}
    risk = _risk_metrics(y_true, y_score)
    n_test = max(len(test), 1)
    real_n = int(test["has_real_report"].sum()) if "has_real_report" in test.columns else 0
    pseudo_n = int(test["needs_pseudo_report"].sum()) if "needs_pseudo_report" in test.columns else 0

    return {
        "experiment_id": experiment_id,
        "test_cases": len(test),
        "real_report_cases": real_n,
        "pseudo_report_cases": pseudo_n,
        "train_cases": int(state.get("n_train", 0)),
        "rouge_l": ref_agg.get("rouge_l", 0.0),
        "bleu": ref_agg.get("bleu", 0.0),
        "meteor": ref_agg.get("meteor", 0.0),
        "bertscore_f1": ref_agg.get("bertscore_f1", 0.0),
        "section_completeness": complete_n / n_test,
        "label_consistency": float(np.mean(lc_scores)) if lc_scores else 0.0,
        "contradiction_rate": float(np.mean(contra_scores)) if contra_scores else 0.0,
        "hallucination_rate": hall.get("unsupported_specific_finding_rate", 0.0),
        "oct_section_similarity": float(np.mean(section_scores["oct_findings"])) if section_scores["oct_findings"] else 0.0,
        "colposcopy_section_similarity": float(np.mean(section_scores["colposcopy_findings"])) if section_scores["colposcopy_findings"] else 0.0,
        "clinical_context_similarity": float(np.mean(section_scores["clinical_context"])) if section_scores["clinical_context"] else 0.0,
        "impression_similarity": float(np.mean(section_scores["impression"])) if section_scores["impression"] else 0.0,
        "inference_time_per_case_seconds": float(np.mean(inf_times)) if inf_times else 0.0,
        "inference_time_p95_seconds": float(np.percentile(inf_times, 95)) if inf_times else 0.0,
        **risk,
    }


def resolve_checkpoint(project: Path, experiment_id: str, cfg: dict, baselines_dir: Path) -> Path | None:
    aliases = cfg.get("checkpoint_aliases", {})
    if experiment_id in aliases:
        p = project / "outputs/publishable/checkpoints" / aliases[experiment_id] / "best.ckpt"
        if p.is_file():
            return p
    p = baselines_dir / experiment_id / "best.ckpt"
    if p.is_file():
        return p
    pub = project / "outputs/publishable/checkpoints" / f"publishable_{experiment_id}" / "best.ckpt"
    if pub.is_file():
        return pub
    return None
