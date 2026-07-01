#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rasa_public_baselines.core import find_experiment, metric_row_from_predictions, output_paths, read_csv_if_exists


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_id", required=True)
    args = p.parse_args()
    exp = find_experiment(args.experiment_id)
    paths = output_paths(args.experiment_id)
    pred = read_csv_if_exists(paths["test"])
    row = metric_row_from_predictions(pred, exp)
    paths["metrics"].write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

