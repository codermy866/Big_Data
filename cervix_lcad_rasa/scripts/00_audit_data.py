#!/usr/bin/env python3
"""Step 1: Data audit and semantic supervision profiling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.centers import REPORT_ARCHIVE_CENTERS, SEMANTIC_ANCHOR_CENTER, identify_report_archive_center
from src.data.manifest_builder import BINARY_ENDPOINT, centre_modality_summary
from src.utils.config import load_config, resolve_project_root
from src.utils.io import write_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/data")
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--output", default=None)
    p.add_argument("--jbd_manifest", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = resolve_project_root()
    cfg = load_config(args.config, root)
    jbd = Path(args.jbd_manifest or cfg["jbd_modeling_csv"])
    out_md = Path(args.output or cfg["outputs"]["data_audit_report"])
    tables = Path(cfg["outputs"]["tables"])
    tables.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    from src.data.manifest_builder import build_full_manifest

    n = 0
    centre_df = pd.DataFrame()
    mdf = pd.DataFrame()
    if jbd.is_file():
        tmp_manifest = tables / "_audit_manifest_tmp.csv"
        mdf = build_full_manifest(jbd, tmp_manifest)
        n = len(mdf)
        centre_df = centre_modality_summary(mdf)
        tmp_manifest.unlink(missing_ok=True)
    archive_c = identify_report_archive_center(mdf) if n else "n/a"

    write_csv(centre_df, tables / "centre_modality_summary.csv")
    cols = ["center_id", "has_real_report_rate"]
    if "report_archive" in centre_df.columns:
        cols.append("report_archive")
    report_avail = centre_df[cols].copy()
    report_avail.columns = ["center_id", "report_availability_rate"] + (
        ["is_report_archive_centre"] if "report_archive" in centre_df.columns else []
    )
    write_csv(report_avail, tables / "report_availability_summary.csv")

    flags = []
    if n and jbd.is_file():
        raw = pd.read_csv(jbd)
        if raw["patient_id"].duplicated().any():
            flags.append({"flag": "duplicate_patient_id", "severity": "review"})
        if int((raw.get("report_available", 0) == 1).sum()) < 50:
            flags.append({"flag": "sparse_real_reports", "severity": "info"})
    write_csv(pd.DataFrame(flags or [{"flag": "none", "severity": "ok"}]), tables / "data_quality_flags.csv")
    write_csv(
        pd.DataFrame(
            [
                {
                    "binary_label_endpoint": BINARY_ENDPOINT,
                    "definition_status": "confirmed_from_jbd_exp0",
                    "notes": "cin2plus column in modeling manifest",
                }
            ]
        ),
        tables / "binary_label_endpoint_definition.csv",
    )

    lines = [
        "# LCAD-RASA Data Audit Report",
        "",
        f"- Data root scanned: `{data_root}` (exists={data_root.is_dir()})",
        f"- Locked modeling manifest: `{jbd}` (n={n})",
        f"- Dual report archive centres: **{', '.join(REPORT_ARCHIVE_CENTERS)}**",
        f"- Primary archive centre (count): **{archive_c}**",
        f"- Semantic anchor centre: **{SEMANTIC_ANCHOR_CENTER}**",
        f"- Real-report cases (case-level): **{int(mdf['has_real_report'].sum()) if n else 0}**",
        f"- Pseudo-report candidates: **{int(mdf['needs_pseudo_report'].sum()) if n and 'needs_pseudo_report' in mdf.columns else 0}**",
        f"- Binary endpoint: **{BINARY_ENDPOINT}**",
        "",
        "## Centre modality summary",
        "",
        centre_df.to_string(index=False) if len(centre_df) else "_no data_",
        "",
        "## Privacy",
        "",
        "Logs use case_id only. No patient names in exports.",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_md)


if __name__ == "__main__":
    main()
