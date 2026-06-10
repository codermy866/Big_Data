#!/usr/bin/env python3
"""Regenerate all JBD result figures using Seaborn gallery-style plot types."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supplementary.jbd_figures_seaborn import generate_all_seaborn_figures
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    p = argparse.ArgumentParser(description="Regenerate JBD figures (Seaborn)")
    p.add_argument("--refresh-submission-v2", action="store_true", help="Copy figures into submission v2 bundle")
    args = p.parse_args()
    project = ROOT
    names = generate_all_seaborn_figures(project)
    logger.info("Generated %d figure groups", len(names))
    print(f"jbd_final: {project / 'outputs/publishable/figures/jbd_final'}")
    print(f"main:      {project / 'outputs/publishable/figures/main'}")
    print(f"legacy:    {project / 'outputs/publishable/figures'}")
    print("Palette: #2f5f8f #8fb8d8 #d9a066 #efd7b5 #9e3f3a #d47f6f #7f7f7f #d6d6d6")

    if args.refresh_submission_v2:
        import shutil

        v2 = project / "outputs/publishable_jbd_submission_v2/figures"
        src = project / "outputs/publishable/figures/jbd_final"
        v2.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.png"):
            shutil.copy2(f, v2 / f.name)
        for f in src.glob("*.pdf"):
            shutil.copy2(f, v2 / f.name)
        print(f"Updated: {v2}")


if __name__ == "__main__":
    main()
