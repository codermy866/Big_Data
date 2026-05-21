#!/usr/bin/env python3
"""Export tables and figures for manuscript / supplementary materials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import ensure_dir, load_config, resolve_project_root
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate metrics and export CSV/XLSX.")
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--format", choices=["csv", "xlsx", "both"], default="both")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root()
    cfg = load_config(args.config, root)
    tables_dir = Path(cfg["outputs"]["tables"])
    figures_dir = ensure_dir(cfg["outputs"]["figures"])

    rows = []
    for name in ("eval_metrics.json", "ablation_results.json"):
        path = tables_dir / name
        if not path.is_file():
            continue
        with path.open() as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                rows.append({"source": name, **item})
        else:
            rows.append({"source": name, **data})

    if not rows:
        logger.warning("No result files found in %s; writing placeholder.", tables_dir)
        rows = [{"source": "placeholder", "note": "Run 06_eval or 07_ablation first."}]

    df = pd.DataFrame(rows)
    if args.format in ("csv", "both"):
        csv_path = tables_dir / "results_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Exported %s", csv_path)
    if args.format in ("xlsx", "both"):
        xlsx_path = tables_dir / "results_summary.xlsx"
        df.to_excel(xlsx_path, index=False)
        logger.info("Exported %s", xlsx_path)

    # Placeholder figure for mock pipeline
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 3))
        if "rouge_l_mean" in df.columns:
            plt.bar(df.index.astype(str), df["rouge_l_mean"].fillna(0))
            plt.ylabel("ROUGE-L")
        else:
            plt.text(0.5, 0.5, "LCAD-RASA\n(mock export)", ha="center", va="center")
        plt.tight_layout()
        fig_path = figures_dir / "results_overview.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.info("Figure: %s", fig_path)
    except ImportError:
        logger.warning("matplotlib not available; skipped figure export.")


if __name__ == "__main__":
    main()
