#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rasa_public_baselines.core import make_figures, make_tables, write_summary


def main() -> None:
    make_tables()
    make_figures()
    write_summary()
    print("wrote manuscript-ready tables and figures")


if __name__ == "__main__":
    main()

