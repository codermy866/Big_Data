"""Evaluation with trained LCAD-RASA; dual-centre metric groups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.manifest import load_manifest
from src.evaluation.comprehensive import evaluate_by_groups, evaluate_predictions
from src.evaluation.metrics import compute_metrics
from src.models.lcad_rasa import build_model
from src.training.dataset import CervixReportDataset, _pseudo_to_text
from src.utils.config import ensure_dir
from src.utils.io import read_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


def _reference_text(row: pd.Series, pseudo_root: Path) -> str:
    if int(row.get("has_real_report", 0)) == 1:
        ref = str(row.get("real_report_text", "") or "").strip()
        if len(ref) >= 20:
            return ref[:2000]
        raw = str(row.get("real_report_path", ""))
        if raw:
            from src.data.report_text import _read_text_file
            from pathlib import Path as P

            t = _read_text_file(P(raw), 2000)
            if t:
                return t
    pr = pseudo_root / str(row["center_id"]) / f"{row['case_id']}.json"
    if pr.is_file():
        rep = read_json(pr)
        return _pseudo_to_text(rep)[:2000]
    return ""


def evaluate_lcad_rasa(cfg: dict[str, Any], mock: bool = False) -> Path:
    manifest_path = Path(cfg["manifest"]["path"])
    df = load_manifest(manifest_path)
    split = cfg.get("evaluation", {}).get("split", "test")
    eval_df = df[df["split"] == split].copy() if "split" in df.columns else df.copy()

    pseudo_root = Path(cfg.get("_data", {}).get("outputs", {}).get("pseudo_reports", "outputs/pseudo_reports"))
    experiment = cfg.get("evaluation", {}).get("experiment", "full_lcad_rasa")
    gen_dir = ensure_dir(Path(cfg["outputs"]["generated_reports"]) / experiment)
    tdir = ensure_dir(Path(cfg["outputs"]["tables"]) / experiment)

    preds_map: dict[str, str] = {}
    refs_map: dict[str, str] = {}
    ckpt_path = Path(cfg["evaluation"]["checkpoint"])

    if _TORCH and ckpt_path.is_file() and ckpt_path.suffix in (".pt", ".ckpt"):
        device = torch.device("cuda" if torch.cuda.is_available() and not mock else "cpu")
        train_cfg = cfg.get("_train", cfg)
        model = build_model(train_cfg).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        ds = CervixReportDataset(eval_df, pseudo_root=pseudo_root, report_source="mixed")
        for i in range(len(ds)):
            batch = ds[i]
            with torch.no_grad():
                labels_t = batch["labels"].unsqueeze(0).to(device)
                out = model(batch["input_ids"].unsqueeze(0).to(device), labels=labels_t)
            pred_ids = out["logits"].argmax(dim=-1).squeeze(0).tolist()
            pred_text = " ".join(str(t) for t in pred_ids[:48])
            case_id = batch["case_id"]
            row = eval_df[eval_df["case_id"] == case_id].iloc[0]
            preds_map[case_id] = pred_text
            refs_map[case_id] = _reference_text(row, pseudo_root)
            with (gen_dir / f"{case_id}.json").open("w", encoding="utf-8") as gf:
                json.dump(
                    {
                        "case_id": case_id,
                        "generated_report": pred_text,
                        "has_real_report": int(row.get("has_real_report", 0)),
                    },
                    gf,
                    ensure_ascii=False,
                )
    else:
        for _, row in eval_df.iterrows():
            case_id = str(row["case_id"])
            pr = pseudo_root / str(row["center_id"]) / f"{case_id}.json"
            if pr.is_file():
                rep = read_json(pr)
                text = _pseudo_to_text(rep)
            else:
                text = str(row.get("real_report_text", ""))[:400]
            preds_map[case_id] = text[:400]
            refs_map[case_id] = _reference_text(row, pseudo_root)

    preds = list(preds_map.values())
    refs = [refs_map[k] for k in preds_map]
    labels = [int(eval_df[eval_df["case_id"] == k].iloc[0]["binary_label"]) for k in preds_map]

    metrics = evaluate_predictions(preds, refs, labels)
    metrics["mock_eval"] = not (_TORCH and ckpt_path.is_file())
    metrics["experiment"] = experiment

    pd.DataFrame([metrics]).to_csv(tdir / "eval_report_metrics.csv", index=False)
    pd.DataFrame(
        [{"label_consistency_accuracy": metrics.get("label_consistency_mean", 0), "n": metrics.get("n", 0)}]
    ).to_csv(tdir / "eval_clinical_consistency.csv", index=False)

    group_df = evaluate_by_groups(eval_df, preds_map, refs_map)
    group_df.to_csv(tdir / "eval_by_group.csv", index=False)
    group_df[group_df.get("eval_group") == "per_center"].to_csv(tdir / "eval_by_center.csv", index=False)

    real_mask = eval_df["has_real_report"] == 1
    if real_mask.any():
        rp, rr, rl = [], [], []
        for _, row in eval_df[real_mask].iterrows():
            cid = str(row["case_id"])
            if cid in preds_map:
                rp.append(preds_map[cid])
                rr.append(refs_map.get(cid, ""))
                rl.append(int(row["binary_label"]))
        if rp:
            rm = compute_metrics(rp, rr, rl)
            pd.DataFrame([{**rm, "eval_scope": "reference_based_real_report_only"}]).to_csv(
                tdir / "eval_reference_based.csv", index=False
            )

    logger.info("Evaluation %s: %s", experiment, metrics)
    return tdir / "eval_report_metrics.csv"
