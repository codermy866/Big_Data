"""Scalability and pipeline efficiency statistics (Prompt 7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _count_images(path_val: str) -> int:
    if not path_val or path_val == "nan":
        return 0
    try:
        arr = json.loads(path_val.replace("'", '"')) if path_val.startswith("[") else []
        return len(arr) if isinstance(arr, list) else 0
    except Exception:
        return max(1, str(path_val).count(".png") + str(path_val).count(".jpg"))


def compute_scalability_stats(manifest_path: Path, project: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(manifest_path)
    df["n_oct"] = df["oct_paths"].astype(str).map(_count_images) if "oct_paths" in df.columns else 0
    df["n_colpo"] = df["colposcopy_paths"].astype(str).map(_count_images) if "colposcopy_paths" in df.columns else 0
    total_oct = int(df["n_oct"].sum())
    total_colpo = int(df["n_colpo"].sum())
    imgs = total_oct + total_colpo

    pipeline_rows = [
        {"metric": "total_cases", "value": len(df)},
        {"metric": "total_centers", "value": df["center_id"].nunique()},
        {"metric": "total_images", "value": imgs},
        {"metric": "total_oct_images", "value": total_oct},
        {"metric": "total_colposcopy_images", "value": total_colpo},
        {"metric": "mean_images_per_case", "value": float((df["n_oct"] + df["n_colpo"]).mean())},
        {"metric": "median_images_per_case", "value": float((df["n_oct"] + df["n_colpo"]).median())},
        {"metric": "max_images_per_case", "value": float((df["n_oct"] + df["n_colpo"]).max())},
        {"metric": "real_report_cases", "value": int(df["has_real_report"].sum())},
        {"metric": "pseudo_report_cases", "value": int(df["needs_pseudo_report"].sum())},
        {"metric": "missing_report_rate", "value": 1 - int(df["has_real_report"].sum()) / max(len(df), 1)},
    ]
    pipeline_df = pd.DataFrame(pipeline_rows)

    center_rows = []
    for cid, g in df.groupby("center_id"):
        center_rows.append(
            {
                "center": cid,
                "total_cases": len(g),
                "real_report_cases": int(g["has_real_report"].sum()),
                "pseudo_report_cases": int(g["needs_pseudo_report"].sum()),
                "oct_image_count": int(g["n_oct"].sum()),
                "colposcopy_image_count": int(g["n_colpo"].sum()),
                "CIN2_positive_cases": int((g["binary_label"] == 1).sum()),
                "CIN2_negative_cases": int((g["binary_label"] == 0).sum()),
                "report_supervision_density": float(g["has_real_report"].mean()),
                "missing_modality_rate": float((g.get("missing_oct", 0) == 1).mean()) if "missing_oct" in g else 0.0,
            }
        )
    center_df = pd.DataFrame(center_rows)

    emb_dir = project / "outputs/publishable/embeddings"
    emb_bytes = sum(p.stat().st_size for p in emb_dir.rglob("*.npy")) if emb_dir.is_dir() else 0
    runtime_rows = [
        {"step": "embedding_storage_mb", "value": emb_bytes / 1e6},
        {"metric": "embedding_files", "value": len(list(emb_dir.rglob("*.npy"))) if emb_dir.is_dir() else 0},
    ]
    for exp_dir in (project / "outputs/publishable/checkpoints").glob("*/best.ckpt"):
        runtime_rows.append({"step": f"checkpoint_{exp_dir.parent.name}_mb", "value": exp_dir.stat().st_size / 1e6})
    runtime_df = pd.DataFrame(runtime_rows)
    return pipeline_df, center_df, runtime_df
