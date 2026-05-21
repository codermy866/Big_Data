#!/usr/bin/env python3
"""Batch inference for report generation (report-free at inference)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest import load_manifest
from src.utils.config import ensure_dir, load_config, resolve_project_root
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    import torch
    from src.models.lcad_rasa import build_model
    from src.training.dataset import CervixReportDataset

    _TORCH = True
except ImportError:
    _TORCH = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference: multimodal inputs -> generated report.")
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--checkpoint", default=None, help="Override checkpoint path.")
    p.add_argument("--manifest", default=None, help="Override manifest CSV.")
    p.add_argument("--split", default="test", help="Split to run inference on.")
    p.add_argument("--output", default=None, help="Output JSONL path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root()
    cfg = load_config(args.config, root)
    train_cfg = load_config(cfg.get("train_config", "configs/train.yaml"), root)

    manifest_path = Path(args.manifest or cfg["manifest"]["path"])
    df = load_manifest(manifest_path)
    if "split" in df.columns:
        df = df[df["split"] == args.split]

    pseudo_path = Path(train_cfg["manifest"]["pseudo_reports_dir"]) / "pseudo_reports.jsonl"
    reports = {}
    if pseudo_path.is_file():
        with pseudo_path.open() as f:
            for line in f:
                rec = json.loads(line)
                reports[rec["exam_id"]] = rec["pseudo_report"]

    records = []
    if _TORCH:
        dataset = CervixReportDataset(df, pseudo_path)
        ckpt_path = Path(args.checkpoint or cfg["evaluation"]["checkpoint"])
        model = build_model(train_cfg)
        if ckpt_path.is_file() and ckpt_path.suffix == ".pt":
            try:
                state = torch.load(ckpt_path, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    model.load_state_dict(state["model"])
            except Exception:
                logger.warning("Could not load checkpoint %s; using random init.", ckpt_path)
        model.eval()
        for i in range(len(dataset)):
            batch = dataset[i]
            with torch.no_grad():
                out = model(batch["input_ids"].unsqueeze(0), labels=batch["labels"].unsqueeze(0))
            pred_ids = out["logits"].argmax(dim=-1).squeeze(0).tolist()[:32]
            records.append(
                {
                    "exam_id": df.iloc[i]["exam_id"],
                    "generated_report": " ".join(str(t) for t in pred_ids),
                    "label": int(batch["labels"].item()),
                }
            )
    else:
        for _, row in df.iterrows():
            exam_id = row["exam_id"]
            records.append(
                {
                    "exam_id": exam_id,
                    "generated_report": reports.get(exam_id, "[mock inference placeholder]"),
                    "label": int(row.get("cin2plus", row.get("cin2_plus", 0))),
                }
            )

    out_path = Path(args.output) if args.output else ensure_dir(cfg["outputs"]["tables"]) / f"inference_{args.split}.jsonl"
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info("Wrote %d predictions to %s", len(records), out_path)


if __name__ == "__main__":
    main()
