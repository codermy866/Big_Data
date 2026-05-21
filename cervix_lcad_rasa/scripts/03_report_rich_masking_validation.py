#!/usr/bin/env python3
"""Step 4: Report-rich centre masking validation (LCAD calibration)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distillation.masking_validation import run_masking_validation
from src.utils.config import load_config, resolve_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/distill.yaml")
    p.add_argument("--manifest", default=None)
    p.add_argument("--evidence_dir", default=None)
    p.add_argument("--client", default="mock")
    return p.parse_args()


def main():
    args = parse_args()
    root = resolve_project_root()
    cfg = load_config(args.config, root)
    dcfg = load_config("configs/data.yaml", root)
    manifest = Path(args.manifest or dcfg["manifest"]["full"])
    ev_dir = Path(args.evidence_dir or dcfg["outputs"]["modality_evidence"])
    out_dir = Path(dcfg["outputs"]["masking_validation"])
    df = pd.read_csv(manifest)
    max_cases = int(dcfg.get("masking_validation", {}).get("max_cases", 200))
    mdf = run_masking_validation(df, ev_dir, out_dir, max_cases=max_cases)
    logger.info("Masking validation:\n%s", mdf.to_string(index=False))


if __name__ == "__main__":
    main()
