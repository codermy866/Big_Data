#!/usr/bin/env python3
"""Step 5: LCAD pseudo-report generation (report-missing centres)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.centers import weak_label_mask
from src.distillation.agent_client import get_client
from src.utils.config import load_config, resolve_project_root
from src.utils.io import read_json, write_csv, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)
SETTING_MAP = {
    "label_only": "label_only_agent",
    "modality_only": "modality_only_agent",
    "modality_plus_label": "modality_plus_label_agent",
}


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
    setting = SETTING_MAP.get(args.setting, args.setting)

    client = get_client(args.client, setting, cfg)
    df = pd.read_csv(manifest)
    weak = df[weak_label_mask(df)]
    status = []
    for _, row in weak.iterrows():
        ev_path = ev_dir / str(row["center_id"]) / f"{row['case_id']}.json"
        if not ev_path.is_file():
            status.append({"case_id": row["case_id"], "status": "missing_evidence"})
            continue
        ev = read_json(ev_path)
        report = client.generate(ev, row.to_dict())
        write_json(out_root / str(row["center_id"]) / f"{row['case_id']}.json", report)
        status.append({"case_id": row["case_id"], "status": "ok", "center_id": row["center_id"]})
    write_csv(pd.DataFrame(status), Path(dcfg["outputs"]["tables"]) / "pseudo_report_generation_status.csv")
    logger.info("Generated pseudo reports for %d weak-centre cases", len([s for s in status if s["status"] == "ok"]))


if __name__ == "__main__":
    main()
