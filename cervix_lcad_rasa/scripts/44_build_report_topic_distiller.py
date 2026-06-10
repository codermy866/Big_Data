#!/usr/bin/env python3
"""Build unsupervised report-topic labels for LCAD-RASA auxiliary supervision."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distillation.report_topic_distiller import build_report_topics, write_report_topic_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="outputs/publishable/manifests/full_manifest_publishable_with_llm_pseudo.csv",
        help="Input manifest with training_report_text.",
    )
    parser.add_argument(
        "--output-manifest",
        default="outputs/publishable/manifests/full_manifest_publishable_with_report_topics.csv",
        help="Output manifest with report_topic_id and report_topic_confidence.",
    )
    parser.add_argument("--output-dir", default="outputs/publishable/report_topic_distiller")
    parser.add_argument("--n-topics", type=int, default=8)
    parser.add_argument("--svd-dim", type=int, default=48)
    parser.add_argument("--max-features", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    out_manifest = Path(args.output_manifest)
    if not out_manifest.is_absolute():
        out_manifest = ROOT / out_manifest
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)
    result = build_report_topics(
        df,
        n_topics=args.n_topics,
        svd_dim=args.svd_dim,
        max_features=args.max_features,
        random_state=args.seed,
    )
    paths = write_report_topic_outputs(
        result,
        manifest_out=out_manifest,
        output_dir=out_dir,
        source_manifest=manifest,
    )
    print("Built report topic distillation artifacts:")
    for key, path in paths.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
