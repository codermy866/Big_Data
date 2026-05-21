#!/usr/bin/env python3
"""Step 3: Extract per-case modality evidence JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.evidence import extract_all_evidence
from src.utils.config import load_config, resolve_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--manifest", default=None)
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, resolve_project_root())
    manifest = Path(args.manifest or cfg["manifest"]["full"])
    out_dir = Path(args.output_dir or cfg["outputs"]["modality_evidence"])
    df = pd.read_csv(manifest)
    status = extract_all_evidence(df, out_dir)
    logger.info("Evidence status: %s (%d cases)", status, len(df))


if __name__ == "__main__":
    main()
