#!/usr/bin/env python3
"""Extract per-modality textual summaries for each exam."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.manifest import load_manifest
from src.data.modality import extract_modality_summaries
from src.utils.config import load_config, resolve_project_root
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract modality summaries (colpo/cyto/HPV/clinical).")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument(
        "--manifest",
        default=None,
        help="Override manifest CSV (default: outputs/manifests from config).",
    )
    p.add_argument("--mock", action="store_true", default=None)
    p.add_argument("--no-mock", dest="mock", action="store_false")
    p.set_defaults(mock=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root()
    cfg = load_config(args.config, root)
    mock = cfg.get("mock", {}).get("enabled", True) if args.mock is None else args.mock
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        from src.data.manifest import resolve_manifest_path

        manifest_path = resolve_manifest_path(cfg, mock=mock)
    df = load_manifest(manifest_path)
    out = extract_modality_summaries(df, cfg, mock=mock)
    logger.info("Modality summaries: %s", out)


if __name__ == "__main__":
    main()
