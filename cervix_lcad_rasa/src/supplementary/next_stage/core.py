"""Shared helpers for next-stage experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.models_publishable.lcad_rasa_model import instr_vector, load_visual_emb
from src.supplementary.train_eval import _risk_metrics, build_model


def count_images_in_paths(path_val: str) -> int:
    if not path_val or str(path_val) in ("nan", "None", ""):
        return 0
    try:
        arr = json.loads(str(path_val).replace("'", '"'))
        return len(arr) if isinstance(arr, list) else 0
    except Exception:
        s = str(path_val)
        return max(0, s.count(".png") + s.count(".jpg") + s.count(".jpeg"))


def collect_risk_scores(
    ckpt_path: Path,
    df: pd.DataFrame,
    spec: dict | None = None,
    device: torch.device | None = None,
) -> tuple[list[int], list[float]]:
    device = device or torch.device("cpu")
    state = torch.load(ckpt_path, map_location="cpu")
    model = build_model(state.get("spec") or spec or {})
    model.load_state_dict(state["model"], strict=False)
    model.to(device)
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            if model.risk_head is None:
                continue
            oct_e = torch.tensor(load_visual_emb(str(row.get("oct_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
            col_e = torch.tensor(load_visual_emb(str(row.get("colposcopy_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
            fus_e = torch.tensor(load_visual_emb(str(row.get("fused_visual_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
            instr = torch.tensor(instr_vector(row.to_dict()), dtype=torch.float32, device=device).unsqueeze(0)
            lab = torch.tensor([int(row["binary_label"])], device=device)
            ids = torch.zeros(1, 64, dtype=torch.long, device=device)
            out = model(oct_e, col_e, fus_e, instr, ids, lab)
            if out.get("risk_logit") is not None:
                y_true.append(int(row["binary_label"]))
                y_score.append(float(torch.sigmoid(out["risk_logit"]).item()))
    return y_true, y_score


def metrics_at_threshold(y_true: list[int], y_score: list[float], thr: float) -> dict[str, float]:
    if not y_true:
        return {}
    yt = np.array(y_true)
    ys = np.array(y_score)
    pred = (ys >= thr).astype(int)
    tp = ((pred == 1) & (yt == 1)).sum()
    tn = ((pred == 0) & (yt == 0)).sum()
    fp = ((pred == 1) & (yt == 0)).sum()
    fn = ((pred == 0) & (yt == 1)).sum()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    ba = (sens + spec) / 2
    base = _risk_metrics(y_true, y_score)
    return {**base, "threshold": thr, "sensitivity": float(sens), "specificity": float(spec), "f1": float(f1), "ppv": float(ppv), "npv": float(npv), "balanced_accuracy": float(ba)}


def select_thresholds(y_true: list[int], y_score: list[float]) -> dict[str, float]:
    if not y_true:
        return {"max_f1": 0.5, "youden": 0.5, "sens90": 0.5, "sens95": 0.5, "spec80": 0.5}
    yt = np.array(y_true)
    ys = np.array(y_score)
    best = {"max_f1": 0.5, "youden": 0.5, "sens90": 0.5, "sens95": 0.5, "spec80": 0.5}
    best_f1, best_youden = -1.0, -1.0
    best_s90_spec, best_s95_spec, best_spec80_sens = -1.0, -1.0, -1.0
    for thr in np.arange(0.05, 0.96, 0.01):
        m = metrics_at_threshold(y_true, y_score, float(thr))
        if m["f1"] > best_f1:
            best_f1, best["max_f1"] = m["f1"], float(thr)
        youden = m["sensitivity"] + m["specificity"] - 1
        if youden > best_youden:
            best_youden, best["youden"] = youden, float(thr)
        if m["sensitivity"] >= 0.90 and m["specificity"] > best_s90_spec:
            best_s90_spec, best["sens90"] = m["specificity"], float(thr)
        if m["sensitivity"] >= 0.95 and m["specificity"] > best_s95_spec:
            best_s95_spec, best["sens95"] = m["specificity"], float(thr)
        if m["specificity"] >= 0.80 and m["sensitivity"] > best_spec80_sens:
            best_spec80_sens, best["spec80"] = m["sensitivity"], float(thr)
    return best


def default_full_spec() -> dict:
    return {
        "train_filter": {"training_eligible": 1},
        "model": {"use_section_align": True, "use_risk_head": True},
        "loss": {"ce_weight": 1.0, "rasa_weight": 0.5, "cls_weight": 0.2, "cons_weight": 0.1},
    }


def no_section_spec() -> dict:
    return {
        "train_filter": {"training_eligible": 1},
        "model": {"use_section_align": False, "use_risk_head": True},
        "loss": {"ce_weight": 1.0, "rasa_weight": 0.0, "cls_weight": 0.2, "cons_weight": 0.1},
    }


def real_only_spec() -> dict:
    return {
        "train_filter": {"has_real_report": 1, "center_id_in": ["enshi", "jingzhou"]},
        "use_pseudo_report": False,
        "model": {"use_section_align": True, "use_risk_head": True},
        "loss": {"ce_weight": 1.0, "rasa_weight": 0.5, "cls_weight": 0.2, "cons_weight": 0.1},
    }
