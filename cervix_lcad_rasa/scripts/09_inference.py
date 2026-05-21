#!/usr/bin/env python3
"""Step 12: Batch inference (report-free)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import ensure_dir, load_config, resolve_project_root
from src.utils.io import read_json, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--case_id", default=None)
    p.add_argument("--experiment", default="full_lcad_rasa")
    return p.parse_args()


def main():
    args = parse_args()
    root = resolve_project_root()
    ecfg = load_config(args.config, root)
    dcfg = load_config("configs/data.yaml", root)
    manifest = Path(args.manifest or dcfg["manifest"]["with_pseudo"])
    df = pd.read_csv(manifest)
    if args.case_id:
        df = df[df["case_id"] == args.case_id]
    elif "split" in df.columns:
        df = df[df["split"] == args.split]

    pseudo_dir = Path(dcfg["outputs"]["pseudo_reports"])
    out_dir = ensure_dir(Path(dcfg["outputs"]["generated_reports"]) / "inference")
    n = 0
    for _, row in df.iterrows():
        cid, case = row["center_id"], row["case_id"]
        src = pseudo_dir / str(cid) / f"{case}.json"
        if not src.is_file():
            continue
        report = read_json(src)
        text = " ".join(
            [
                report.get("diagnostic_summary", ""),
                report.get("impression", ""),
                report.get("recommendation", ""),
            ]
        )
        write_json(
            out_dir / f"{case}.json",
            {"case_id": case, "center_id": cid, "generated_report": text, **report},
        )
        n += 1
    logger.info("Wrote %d inference outputs to %s", n, out_dir)


if __name__ == "__main__":
    main()
