#!/usr/bin/env python3
"""Regenerate all manuscript figures with Arial typography and overlap-safe layouts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script: str) -> None:
    path = SCRIPTS / script
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, str(path)], cwd=str(ROOT), check=True)


def main() -> None:
    # Core seaborn + theme1 + ablation + API + MOSAIC + external forest
    run("41_restyle_all_experiment_figures.py")
    run("53_regenerate_external_baselines_forest.py")
    print("\nAll manuscript figures regenerated (Arial, p-values, result-visualization panels).")
    print(f"Synced figures: {ROOT.parent / 'figures'}")


if __name__ == "__main__":
    main()
