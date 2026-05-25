"""Table export helpers without optional tabulate dependency."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_table(df: pd.DataFrame, csv_path: Path, md_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if md_path is None:
        md_path = csv_path.with_suffix(".md")
    try:
        md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    except ImportError:
        md_path.write_text(df.head(50).to_string(index=False), encoding="utf-8")
