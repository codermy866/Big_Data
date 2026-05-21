#!/usr/bin/env python3
"""Prompt G: Train publishable LCAD-RASA on visual embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models_publishable.lcad_rasa_model import PublishableLCADRASA
from src.training.experiment_modes import apply_train_filter, get_experiment_spec, load_experiment_registry
from src.training.publishable_dataset import PublishableDataset
from src.utils.config import load_config, resolve_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

PUBLISHABLE_EXPERIMENTS = {
    "publishable_enshi_real_only": {"center_id": "enshi", "has_real_report": 1, "report_source": "real"},
    "publishable_jingzhou_real_only": {"center_id": "jingzhou", "has_real_report": 1, "report_source": "real"},
    "publishable_dual_real_only": {"has_real_report": 1, "center_id_in": ["enshi", "jingzhou"]},
    "publishable_lcad_augmented": {"training_eligible": 1},
    "publishable_simple_fusion": {"training_eligible": 1, "model": {"use_section_align": False, "use_risk_head": False}},
    "publishable_fusion_plus_section_alignment": {"training_eligible": 1, "model": {"use_section_align": True, "use_risk_head": False}},
    "publishable_full_lcad_rasa": {"training_eligible": 1, "model": {"use_section_align": True, "use_risk_head": True}},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_publishable.yaml")
    p.add_argument("--manifest", default=None)
    p.add_argument("--experiment", default="publishable_full_lcad_rasa")
    args = p.parse_args()
    project = resolve_project_root()
    cfg = load_config(args.config, project)
    manifest_path = Path(args.manifest or cfg["manifest"]["path"])
    if not manifest_path.is_absolute():
        manifest_path = project / manifest_path
    spec = PUBLISHABLE_EXPERIMENTS.get(args.experiment, {"training_eligible": 1})
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"] if "split" in df.columns else df
    if "center_id" in spec:
        train_df = train_df[train_df["center_id"] == spec["center_id"]]
    if spec.get("has_real_report"):
        train_df = train_df[train_df["has_real_report"] == 1]
    if spec.get("center_id_in"):
        train_df = train_df[train_df["center_id"].isin(spec["center_id_in"])]
    if spec.get("training_eligible"):
        train_df = apply_train_filter(train_df, {"training_eligible": 1})

    mflags = spec.get("model", {})
    ds = PublishableDataset(train_df, max_len=int(cfg["training"]["max_seq_length"]))
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PublishableLCADRASA(
        use_risk_head=mflags.get("use_risk_head", True),
        use_section_align=mflags.get("use_section_align", True),
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))
    ckpt_dir = project / cfg["outputs"]["checkpoints"] / args.experiment
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(int(cfg["training"]["num_epochs"])):
        model.train()
        eloss = 0.0
        steps = 0
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
            ce = F.cross_entropy(out["logits"].reshape(-1, out["logits"].size(-1)), tgt.reshape(-1), reduction="none")
            ce = (ce.view(tgt.size(0), -1).mean(1) * w).mean()
            align = model.section_alignment_loss(out["fused"], out["hidden"])
            risk = torch.tensor(0.0, device=device)
            if out.get("risk_logit") is not None:
                risk = F.binary_cross_entropy_with_logits(out["risk_logit"].squeeze(-1), lab.float())
            loss = float(cfg["loss"]["ce_weight"]) * ce + float(cfg["loss"]["rasa_weight"]) * align + float(cfg["loss"]["cls_weight"]) * risk
            optim.zero_grad()
            loss.backward()
            optim.step()
            eloss += float(loss.item())
            steps += 1
        history.append({"epoch": epoch + 1, "loss": eloss / max(steps, 1)})
        logger.info("Epoch %d loss=%.4f n=%d", epoch + 1, history[-1]["loss"], len(ds))

    best = ckpt_dir / "best.ckpt"
    torch.save({"model": model.state_dict(), "experiment": args.experiment, "n_train": len(ds)}, best)
    curve_dir = project / cfg["outputs"]["tables"] / args.experiment
    curve_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(curve_dir / "training_curve.csv", index=False)
    (project / cfg["outputs"]["logs"] / f"{args.experiment}_history.json").write_text(json.dumps(history), encoding="utf-8")
    print(f"Saved {best}")


if __name__ == "__main__":
    main()
