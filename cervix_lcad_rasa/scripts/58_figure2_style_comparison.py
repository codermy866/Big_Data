#!/usr/bin/env python3
"""Generate Figure2 style variants for comparison, then export the recommended final."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
FINAL_FIG = PROJECT / "final_Fig"
COMPARE_DIR = FINAL_FIG / "Figure2_style_compare"

sys.path.insert(0, str(ROOT))

from src.supplementary.jbd_figure2_styles import (  # noqa: E402
    save_all_style_comparisons,
    render_figure2_final,
)


def main() -> None:
    written = save_all_style_comparisons(ROOT, COMPARE_DIR)
    print(f"Wrote {len(written)} style comparison figures to {COMPARE_DIR}")
    for p in written:
        print(f"  - {p.name}")

    out_dirs = [
        FINAL_FIG,
        PROJECT / "figures",
        ROOT / "outputs/publishable/figures/jbd_final",
    ]
    render_figure2_final(ROOT, out_dirs, style="F_conditional_kde")
    print("Exported final Figure2_centre_supervision_catplot (Style F · horizontal bar plot) to final_Fig/")


if __name__ == "__main__":
    main()
