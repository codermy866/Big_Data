#!/usr/bin/env python3
"""Prompt F: QC LLM pseudo reports + manifest update."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distillation.quality_control import qc_pseudo_report
from src.utils.io import read_json, write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="outputs/publishable/manifests/full_manifest_publishable.csv")
    p.add_argument("--pseudo_report_dir", default="outputs/publishable/pseudo_reports_llm")
    p.add_argument("--output_manifest", default="outputs/publishable/manifests/full_manifest_publishable_with_llm_pseudo.csv")
    args = p.parse_args()
    df = pd.read_csv(ROOT / args.manifest)
    pseudo_dir = ROOT / args.pseudo_report_dir
    qc_rows = []
    for i, row in df.iterrows():
        if int(row.get("has_real_report", 0)) == 1:
            continue
        pr = pseudo_dir / str(row["center_id"]) / f"{row['case_id']}.json"
        if not pr.is_file():
            continue
        report = read_json(pr)
        qc = qc_pseudo_report(report, row.to_dict())
        text = json.dumps(report, ensure_ascii=False)[:3000]
        df.at[i, "has_pseudo_report"] = 1
        df.at[i, "pseudo_report_text"] = text
        df.at[i, "pseudo_report_path"] = str(pr)
        for k, v in qc.items():
            df.at[i, k] = v
        if int(qc["pseudo_report_pass_qc"]) == 1:
            df.at[i, "training_report_type"] = "pseudo"
            df.at[i, "training_report_text"] = text
        qc_rows.append({"case_id": row["case_id"], **qc})
    out = ROOT / args.output_manifest
    df.to_csv(out, index=False)
    (ROOT / "outputs/publishable/qc").mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame(qc_rows), ROOT / "outputs/publishable/qc/llm_pseudo_report_qc_cases.csv")
    if qc_rows:
        write_csv(
            pd.DataFrame([{"n_qc": len(qc_rows), "pass_rate": pd.DataFrame(qc_rows)["pseudo_report_pass_qc"].mean()}]),
            ROOT / "outputs/publishable/tables/llm_pseudo_report_quality_summary.csv",
        )
    mock_qc = pd.read_csv(ROOT / "outputs/qc/pseudo_report_qc_cases.csv") if (ROOT / "outputs/qc/pseudo_report_qc_cases.csv").is_file() else pd.DataFrame()
    cmp = pd.DataFrame(
        [
            {"source": "mock", "n": len(mock_qc), "pass_rate": mock_qc["pseudo_report_pass_qc"].mean() if len(mock_qc) else 0},
            {"source": "llm", "n": len(qc_rows), "pass_rate": pd.DataFrame(qc_rows)["pseudo_report_pass_qc"].mean() if qc_rows else 0},
        ]
    )
    write_csv(cmp, ROOT / "outputs/publishable/tables/llm_vs_mock_pseudo_qc_comparison.csv")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
