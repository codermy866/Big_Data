#!/usr/bin/env python3
"""Prompt L: Minimal publishable training set."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_publishable.yaml")
    p.add_argument("--manifest", default="outputs/publishable/manifests/full_manifest_publishable_with_llm_pseudo.csv")
    p.add_argument("--experiments", nargs="+", required=True)
    p.add_argument("--output_dir", default="outputs/publishable")
    args = p.parse_args()
    py = sys.executable
    rows = []
    for exp in args.experiments:
        t0 = time.time()
        try:
            subprocess.run(
                [py, str(ROOT / "scripts/19_train_publishable_lcad_rasa.py"), "--experiment", exp, "--manifest", args.manifest, "--config", args.config],
                cwd=ROOT,
                check=True,
            )
            status = "completed"
        except subprocess.CalledProcessError as exc:
            status = f"failed:{exc}"
        rows.append({"experiment": exp, "status": status, "duration_sec": time.time() - t0})
    out = ROOT / args.output_dir / "tables/publishable_training_registry.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
