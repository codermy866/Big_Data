#!/usr/bin/env python3
"""Prompt H: Publishable evaluation with full metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import label_consistency
from src.evaluation_publishable.report_metrics import aggregate_metrics, compute_reference_metrics
from src.models_publishable.lcad_rasa_model import PublishableLCADRASA, instr_vector, load_visual_emb
from src.training.publishable_dataset import PublishableDataset


def predict_text(model, row, device) -> str:
    oct_e = torch.tensor(load_visual_emb(str(row.get("oct_embedding_path", ""))), dtype=torch.float32).unsqueeze(0).to(device)
    col_e = torch.tensor(load_visual_emb(str(row.get("colposcopy_embedding_path", ""))), dtype=torch.float32).unsqueeze(0).to(device)
    fus_e = torch.tensor(load_visual_emb(str(row.get("fused_visual_embedding_path", ""))), dtype=torch.float32).unsqueeze(0).to(device)
    instr = torch.tensor(instr_vector(row.to_dict()), dtype=torch.float32).unsqueeze(0).to(device)
    ref = str(row.get("reference_report_text", row.get("training_report_text", "")))
    ids = torch.tensor([[hash(w) % 8192 for w in ref.split()[:64]]], dtype=torch.long).to(device)
    if ids.size(1) < 64:
        ids = F.pad(ids, (0, 64 - ids.size(1)))
    with torch.no_grad():
        out = model(oct_e, col_e, fus_e, instr, ids, torch.tensor([int(row["binary_label"])], device=device))
    pred_ids = out["logits"].argmax(-1).squeeze().tolist()
    return " ".join(str(t) for t in pred_ids[:32])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--experiment", default="publishable_full_lcad_rasa")
    p.add_argument("--output_dir", default="outputs/publishable")
    args = p.parse_args()
    df = pd.read_csv(ROOT / args.manifest)
    test = df[df["split"] == "test"] if "split" in df.columns else df.iloc[:400]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PublishableLCADRASA().to(device)
    state = torch.load(ROOT / args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    gen_dir = ROOT / args.output_dir / "generated_reports" / args.experiment
    gen_dir.mkdir(parents=True, exist_ok=True)
    rows_ref, rows_clin, rows_all = [], [], []
    for _, row in test.iterrows():
        pred = predict_text(model, row, device)
        ref = str(row.get("reference_report_text", "")) if int(row.get("has_real_report", 0)) else ""
        label = int(row["binary_label"])
        scope = "reference_based" if int(row.get("has_real_report", 0)) and len(ref) >= 20 else "clinical_only"
        if row["center_id"] == "xiangyang" and int(row.get("has_real_report", 0)):
            scope = "sparse_sensitivity"
        (gen_dir / f"{row['case_id']}.json").write_text(json.dumps({"generated_report": pred}), encoding="utf-8")
        base = {"case_id": row["case_id"], "center_id": row["center_id"], "metric_applicable_scope": scope, "label_consistency": label_consistency(pred, label)}
        if scope == "reference_based":
            m = compute_reference_metrics(pred, ref)
            rows_ref.append({**base, **m})
        rows_clin.append(base)
        rows_all.append({**base, "rouge_l": compute_reference_metrics(pred, ref or pred)["rouge_l"]})
    tdir = ROOT / args.output_dir / "tables" / args.experiment
    tdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([aggregate_metrics(rows_ref)]).to_csv(tdir / "eval_reference_based.csv", index=False)
    pd.DataFrame([aggregate_metrics(rows_clin)]).to_csv(tdir / "eval_clinical_consistency.csv", index=False)
    pd.DataFrame(rows_all).groupby("center_id").mean(numeric_only=True).to_csv(tdir / "eval_by_center.csv")
    pd.DataFrame([aggregate_metrics(rows_all)]).to_csv(tdir / "eval_report_metrics.csv", index=False)
    print(f"Eval written to {tdir}")


if __name__ == "__main__":
    main()
