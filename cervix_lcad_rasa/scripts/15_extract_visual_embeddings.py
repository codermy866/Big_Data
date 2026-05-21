#!/usr/bin/env python3
"""Prompt C: ResNet50 visual embeddings for OCT and colposcopy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.visual_encoder import encode_image_paths
from src.utils.io import write_csv, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="outputs/publishable/manifests/full_manifest_with_normalized_real_reports.csv")
    p.add_argument("--output_manifest", default="outputs/publishable/manifests/full_manifest_with_visual_embeddings.csv")
    p.add_argument("--output_dir", default="outputs/publishable/embeddings")
    p.add_argument("--encoder", default="resnet50")
    p.add_argument("--pooling", default="mean")
    p.add_argument("--max_images", type=int, default=6)
    args = p.parse_args()
    df = pd.read_csv(ROOT / args.manifest)
    emb_root = ROOT / args.output_dir
    ev_root = ROOT / "outputs/publishable/modality_evidence"
    status = []
    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        center_id = str(row["center_id"])
        oct_paths = json.loads(row["oct_paths"]) if isinstance(row["oct_paths"], str) else []
        col_paths = json.loads(row["colposcopy_paths"]) if isinstance(row["colposcopy_paths"], str) else []
        oct_emb, oct_meta = encode_image_paths(oct_paths, args.encoder, args.pooling, args.max_images)
        col_emb, col_meta = encode_image_paths(col_paths, args.encoder, args.pooling, args.max_images)
        fused = ((oct_emb + col_emb) / 2.0).astype(np.float32)
        for sub, arr in [("oct", oct_emb), ("colposcopy", col_emb), ("fused_visual", fused)]:
            d = emb_root / sub
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f"{case_id}.npy", arr)
        df.at[row.name, "oct_embedding_path"] = str(emb_root / "oct" / f"{case_id}.npy")
        df.at[row.name, "colposcopy_embedding_path"] = str(emb_root / "colposcopy" / f"{case_id}.npy")
        df.at[row.name, "fused_visual_embedding_path"] = str(emb_root / "fused_visual" / f"{case_id}.npy")
        has_emb = int(oct_meta["readable"] > 0 or col_meta["readable"] > 0)
        df.at[row.name, "has_visual_embedding"] = has_emb
        df.at[row.name, "missing_embedding"] = int(not has_emb)
        ev = {
            "case_id": case_id,
            "center_id": center_id,
            "oct_evidence": {
                "available": len(oct_paths) > 0,
                "readable_images": oct_meta["readable"],
                "evidence_source": "local_encoder",
                "embedding_path": df.at[row.name, "oct_embedding_path"],
                "evidence_reliability": min(1.0, 0.3 + 0.1 * oct_meta["readable"]),
            },
            "colposcopy_evidence": {
                "available": len(col_paths) > 0,
                "readable_images": col_meta["readable"],
                "evidence_source": "local_encoder",
                "embedding_path": df.at[row.name, "colposcopy_embedding_path"],
                "evidence_reliability": min(1.0, 0.3 + 0.1 * col_meta["readable"]),
            },
            "instruction_evidence": {
                "age": str(row.get("age", "")),
                "hpv": str(row.get("hpv", "")),
                "tct": str(row.get("tct", "")),
            },
            "available_modalities": {"oct": len(oct_paths) > 0, "colposcopy": len(col_paths) > 0, "instruction": True},
            "visual_summary": f"ResNet embeddings oct_read={oct_meta['readable']} col_read={col_meta['readable']}",
        }
        write_json(ev_root / center_id / f"{case_id}.json", ev)
        status.append({"case_id": case_id, "center_id": center_id, "has_visual_embedding": has_emb})
    out_m = ROOT / args.output_manifest
    out_m.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_m, index=False)
    write_csv(pd.DataFrame(status), ROOT / "outputs/publishable/tables/visual_embedding_status.csv")
    print(f"Embeddings: {out_m} | success rate {pd.DataFrame(status)['has_visual_embedding'].mean():.1%}")


if __name__ == "__main__":
    main()
