#!/usr/bin/env python3
"""Step 2: Build unified full_manifest.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.manifest_builder import build_full_manifest
from src.utils.config import load_config, resolve_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/data")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--output", default=None)
    p.add_argument("--jbd_manifest", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, resolve_project_root())
    jbd = Path(args.jbd_manifest or cfg["jbd_modeling_csv"])
    out = Path(args.output or cfg["manifest"]["full"])
    df = build_full_manifest(jbd, out)
    logger.info("Built full manifest: %s (%d rows)", out, len(df))


if __name__ == "__main__":
    main()
