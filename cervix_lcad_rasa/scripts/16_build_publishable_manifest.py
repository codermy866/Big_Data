#!/usr/bin/env python3
"""Prompt D: Merge normalized reports + embeddings + pseudo/QC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_report_manifest", required=True)
    p.add_argument("--visual_manifest", required=True)
    p.add_argument("--pseudo_manifest", required=True)
    p.add_argument("--qc_table", default="outputs/qc/pseudo_report_qc_cases.csv")
    p.add_argument("--output", default="outputs/publishable/manifests/full_manifest_publishable.csv")
    args = p.parse_args()
    real = pd.read_csv(ROOT / args.real_report_manifest)
    vis = pd.read_csv(ROOT / args.visual_manifest)
    pseudo = pd.read_csv(ROOT / args.pseudo_manifest)
    qc = pd.read_csv(ROOT / args.qc_table) if (ROOT / args.qc_table).is_file() else pd.DataFrame()

    base = pseudo.copy()
    vis_cols = ["has_visual_embedding", "oct_embedding_path", "colposcopy_embedding_path", "fused_visual_embedding_path", "missing_embedding"]
    for c in vis_cols:
        if c in vis.columns:
            base[c] = vis.set_index("case_id")[c].reindex(base["case_id"]).values

    ref_cols = [c for c in real.columns if c.startswith("reference_") or c in ("reference_report_text", "real_report_source_type")]
    for c in ref_cols:
        base[c] = real.set_index("case_id")[c].reindex(base["case_id"]).values

    def training_row(r):
        if int(r.get("has_real_report", 0)) == 1 and len(str(r.get("reference_report_text", ""))) >= 20:
            return "real", str(r["reference_report_text"])[:4000]
        if int(r.get("pseudo_report_pass_qc", 0)) == 1:
            return "pseudo", str(r.get("pseudo_report_text", ""))[:4000]
        return "none", ""

    tt, tx = zip(*[training_row(r) for _, r in base.iterrows()])
    base["training_report_type"] = tt
    base["training_report_text"] = tx
    base.loc[base["center_id"] == "xiangyang", "report_supervision_class"] = base.loc[
        base["center_id"] == "xiangyang", "report_supervision_class"
    ].fillna("sparse_report_case")

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(out, index=False)
    summary = base.groupby(["center_id", "training_report_type"]).size().reset_index(name="n")
    summary.to_csv(out.parent.parent / "tables/publishable_manifest_summary.csv", index=False)
    print(f"Publishable manifest: {out}")
    print(base["training_report_type"].value_counts())


if __name__ == "__main__":
    main()
