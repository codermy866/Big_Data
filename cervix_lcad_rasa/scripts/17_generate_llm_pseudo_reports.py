#!/usr/bin/env python3
"""Prompt E: LLM/local_llm pseudo-report generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distillation.llm_agent_client import get_llm_client
from src.utils.io import read_json, write_csv, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)
SETTING_MAP = {
    "label_only": "label_only_agent",
    "modality_only": "modality_only_agent",
    "modality_plus_label": "modality_plus_label_agent",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="outputs/publishable/manifests/full_manifest_publishable.csv")
    p.add_argument("--evidence_dir", default="outputs/publishable/modality_evidence")
    p.add_argument("--client", default="local_llm")
    p.add_argument("--setting", default="modality_plus_label")
    p.add_argument("--output_dir", default="outputs/publishable/pseudo_reports_llm")
    args = p.parse_args()
    df = pd.read_csv(ROOT / args.manifest)
    sub = df[(df["needs_pseudo_report"] == 1) & (df.get("has_visual_embedding", 1) == 1)]
    client = get_llm_client(args.client, SETTING_MAP.get(args.setting, args.setting))
    out_root = ROOT / args.output_dir
    status = []
    for _, row in sub.iterrows():
        ev_path = ROOT / args.evidence_dir / str(row["center_id"]) / f"{row['case_id']}.json"
        if not ev_path.is_file():
            status.append({"case_id": row["case_id"], "status": "missing_evidence"})
            continue
        ev = read_json(ev_path)
        report = client.generate(ev, row.to_dict())
        write_json(out_root / str(row["center_id"]) / f"{row['case_id']}.json", report)
        status.append({"case_id": row["case_id"], "center_id": row["center_id"], "status": "ok"})
    write_csv(pd.DataFrame(status), ROOT / "outputs/publishable/tables/llm_pseudo_report_generation_status.csv")
    logger.info("LLM pseudo reports: %d/%d", sum(s["status"] == "ok" for s in status), len(sub))


if __name__ == "__main__":
    main()
