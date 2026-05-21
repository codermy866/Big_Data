"""Per-case modality evidence extraction (metadata + optional image stats)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import write_csv, write_json
from src.utils.privacy import sanitize_text


def _image_stats(paths: list[str]) -> dict[str, Any]:
    readable = 0
    shapes = []
    for p in paths[:5]:
        fp = Path(p)
        if not fp.is_file():
            continue
        try:
            from PIL import Image

            with Image.open(fp) as im:
                shapes.append(list(im.size))
                readable += 1
        except Exception:
            continue
    source = "local_encoder" if readable > 0 else "metadata_only"
    reliability = min(1.0, 0.3 + 0.1 * readable)
    summary = f"{readable} readable images"
    if shapes:
        summary += f"; typical size {shapes[0]}"
    return {
        "available": len(paths) > 0,
        "num_images": len(paths),
        "readable_images": readable,
        "image_shapes": shapes,
        "evidence_source": source,
        "visual_summary": summary,
        "embedding_path": "",
        "evidence_reliability": reliability,
        "quality_flags": [] if readable else ["no_readable_images"],
    }


def extract_evidence_for_case(row: pd.Series) -> dict[str, Any]:
    oct_paths = json.loads(row["oct_paths"]) if row["oct_paths"] else []
    colpo_paths = json.loads(row["colposcopy_paths"]) if row["colposcopy_paths"] else []
    case_id = row["case_id"]
    center_id = row["center_id"]
    instr = {
        "age": str(row.get("age", "")),
        "hpv": sanitize_text(str(row.get("hpv", ""))),
        "tct": sanitize_text(str(row.get("tct", ""))),
        "other_clinical_context": sanitize_text(str(row.get("other_clinical_attributes", ""))[:300]),
        "missing_fields": [],
    }
    for k in ("age", "hpv", "tct"):
        if not instr[k] or instr[k] == "nan":
            instr["missing_fields"].append(k)
    return {
        "case_id": case_id,
        "center_id": center_id,
        "oct_evidence": _image_stats(oct_paths),
        "colposcopy_evidence": _image_stats(colpo_paths),
        "instruction_evidence": instr,
        "available_modalities": {
            "oct": len(oct_paths) > 0,
            "colposcopy": len(colpo_paths) > 0,
            "instruction": len(instr["missing_fields"]) < 3,
        },
        "privacy_flags": [],
    }


def extract_all_evidence(manifest_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    status_rows = []
    for _, row in manifest_df.iterrows():
        ev = extract_evidence_for_case(row)
        cid = row["center_id"]
        out_path = output_dir / str(cid) / f"{row['case_id']}.json"
        write_json(out_path, ev)
        status_rows.append(
            {
                "case_id": row["case_id"],
                "center_id": cid,
                "oct_source": ev["oct_evidence"]["evidence_source"],
                "colpo_source": ev["colposcopy_evidence"]["evidence_source"],
                "oct_reliability": ev["oct_evidence"]["evidence_reliability"],
                "colpo_reliability": ev["colposcopy_evidence"]["evidence_reliability"],
            }
        )
    status_csv = output_dir.parent / "tables" / "modality_evidence_status.csv"
    write_csv(pd.DataFrame(status_rows), status_csv)
    return status_csv
