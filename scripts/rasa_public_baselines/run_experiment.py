#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rasa_public_baselines.core import ensure_dirs, find_experiment, load_registry, run_one_experiment, write_resolved_registry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_id", default="")
    p.add_argument("--all", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    ensure_dirs()
    write_resolved_registry()
    experiments = load_registry() if args.all else [find_experiment(args.experiment_id)]
    rows = []
    for exp in experiments:
        rows.append(run_one_experiment(exp, dry_run=args.dry_run, smoke_test=args.smoke_test))
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

