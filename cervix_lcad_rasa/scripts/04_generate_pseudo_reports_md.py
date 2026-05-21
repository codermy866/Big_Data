#!/usr/bin/env python3
"""Step 5: LCAD pseudo-report generation for report-missing centres."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distillation.agent_client import get_client
from src.utils.config import load_config, resolve_project_root
from src.utils.io import write_csv, write_json, read_json
from src.utils.logger import get_logger

logger = get_logger(__name__)
REPORT_RICH = "enshi"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/distill.yaml")
    p.add_argument("--manifest", default=None)
    p.add_argument("--evidence_dir", default=None)
    p.add_argument("--client", default="mock")
    p.add_argument("--setting", default="modality_plus_label")
    return p.parse_args()


def main():
    args = parse_args()
    root = resolve_project_root()
    cfg = load_config(args.config, root)
    dcfg = load_config("configs/data.yaml", root)
    manifest = Path(args.manifest or dcfg["manifest"]["full"])
    ev_dir = Path(args.evidence_dir or dcfg["outputs"]["modality_evidence"])
    out_root = Path(dcfg["outputs"]["pseudo_reports"])
    setting = args.setting.replace("modality_plus_label", "modality_plus_label_agent")
    if not setting.endswith("_agent"):
        setting = setting + "_agent" if setting in ("label_only", "modality_only") else "modality_plus_label_agent"
    client = get_client(args.client, setting, cfg)

    df = pd.read_csv(manifest)
    weak = df[df["center_id"] != REPORT_RICH]
    status = []
    for _, row in weak.iterrows():
        ev_path = ev_dir / str(row["center_id"]) / f"{row['case_id']}.json"
        if not ev_path.is_file():
            status.append({"case_id": row["case_id"], "status": "missing_evidence"})
            continue
        ev = read_json(ev_path)
        report = client.generate(ev, row.to_dict())
        out_path = out_root / str(row["center_id"]) / f"{row['case_id']}.json"
        write_json(out_path, report)
        status.append({"case_id": row["case_id"], "status": "ok", "center_id": row["center_id"]})
    write_csv(pd.DataFrame(status), Path(dcfg["outputs"]["tables"]) / "pseudo_report_generation_status.csv")
    logger.info("Generated %d pseudo reports for weak-label centres", len(status))


if __name__ == "__main__":
    main()
