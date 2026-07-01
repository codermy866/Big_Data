#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rasa_public_baselines.core import aggregate_metrics, run_paired_bootstrap, write_final_status, write_summary


def main() -> None:
    df = aggregate_metrics()
    paired = run_paired_bootstrap()
    write_summary()
    write_final_status()
    print(f"aggregated {len(df)} metric rows; paired tests {len(paired)} rows")


if __name__ == "__main__":
    main()

