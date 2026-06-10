#!/usr/bin/env python3
"""Build KRA-RASA train-bank semantic retrieval artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.semantic_bank import build_semantic_retrieval_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="outputs/publishable/manifests/full_manifest_publishable_with_report_topics.csv",
    )
    parser.add_argument(
        "--output-manifest",
        default="outputs/publishable/manifests/full_manifest_publishable_with_kra_semantic.csv",
    )
    parser.add_argument("--output-dir", default="outputs/publishable/semantic_retrieval")
    parser.add_argument("--bank-split", default="train")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    output_manifest = Path(args.output_manifest)
    output_dir = Path(args.output_dir)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not output_manifest.is_absolute():
        output_manifest = ROOT / output_manifest
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    artifacts = build_semantic_retrieval_artifacts(
        manifest,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        bank_split=args.bank_split,
        dim=args.dim,
        top_k=args.top_k,
    )
    print(f"Wrote semantic bank with {len(artifacts.bank)} entities")
    print(f"Wrote retrievals for {len(artifacts.retrievals)} cases")
    print(f"Wrote manifest {output_manifest}")


if __name__ == "__main__":
    main()

