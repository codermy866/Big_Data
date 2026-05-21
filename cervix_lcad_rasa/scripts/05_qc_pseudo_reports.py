#!/usr/bin/env python3
"""Step 6: Pseudo-report QC and reliability weighting."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "05_qc_pseudo_reports_md.py"), run_name="__main__")
