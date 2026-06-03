"""Build full_manifest.csv from the locked modeling cohort (dual report centres)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.centers import REPORT_ARCHIVE_CENTERS, SEMANTIC_ANCHOR_CENTER, identify_report_archive_center
from src.data.report_discovery import discover_report_path
from src.data.report_supervision import (
    case_has_real_report,
    needs_pseudo_report,
    report_supervision_class,
    training_report_type,
)
from src.data.report_text import reference_report_from_row
from src.utils.io import write_csv

BINARY_ENDPOINT = "CIN2+"
CENTERS = ["wuda", "xiangyang", "shiyan", "jingzhou", "enshi"]


def _paths_to_json(path_str: Any) -> str:
    if path_str is None or (isinstance(path_str, float) and np.isnan(path_str)):
        return json.dumps([])
    s = str(path_str).strip()
    if not s:
        return json.dumps([])
    parts = [p.strip() for p in s.split(";") if p.strip()]
    return json.dumps(parts, ensure_ascii=False)


def _resolve_real_report_path(r: pd.Series) -> str:
    raw = str(r.get("raw_report_path", "") or "").strip()
    if raw and Path(raw).is_file():
        return raw
    colpo = r.get("colpo_paths", r.get("colposcopy_paths", ""))
    discovered = discover_report_path(colpo if colpo else _paths_to_json(colpo))
    return discovered or raw


def build_full_manifest(jbd_csv: Path, out_csv: Path, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(jbd_csv)
    rows = []
    for _, r in df.iterrows():
        case_id = str(r["exam_id"])
        center_id = str(r.get("center", "unknown"))
        tier = str(r.get("report_archive_tier", "") or "")
        real_path = _resolve_real_report_path(r)
        row_dict = {
            "center": center_id,
            "center_id": center_id,
            "report_archive_tier": tier,
            "report_available": int(r.get("report_available", 0)),
            "raw_report_path": real_path,
            "standardized_report_path": str(r.get("standardized_report_path", "")),
            "case_id": case_id,
            "exam_id": case_id,
            "pathology_raw": r.get("pathology_raw", ""),
            "pathology_grade": r.get("pathology_grade", ""),
            "treatment_text": r.get("treatment_text", ""),
            "hpv": str(r.get("hpv", "")),
            "tct": str(r.get("tct", "")),
            "hpv_class": r.get("hpv_class", ""),
            "tct_class": r.get("tct_class", ""),
            "cin_grade": r.get("cin_grade", ""),
        }
        has_report = int(case_has_real_report(row_dict))
        label = int(r.get("cin2plus", r.get("cin2_plus", r.get("label", 0))))
        oct_paths = _paths_to_json(r.get("oct_paths"))
        colpo_paths = _paths_to_json(r.get("colpo_paths"))
        flags = []
        if not json.loads(oct_paths):
            flags.append("missing_oct")
        if not json.loads(colpo_paths):
            flags.append("missing_colposcopy")
        if has_report and not real_path:
            flags.append("report_path_pending_discovery")

        ref_row = {**row_dict, "real_report_path": real_path}
        real_text = reference_report_from_row(ref_row) if has_report else ""
        sup_class = report_supervision_class(
            {**row_dict, "has_real_report": has_report, "missing_label": 0}
        )
        train_type = training_report_type({**row_dict, "has_real_report": has_report})
        need_pseudo = int(needs_pseudo_report({**row_dict, "has_real_report": has_report}))

        rows.append(
            {
                "case_id": case_id,
                "patient_id": str(r.get("patient_id", case_id)),
                "center_id": center_id,
                "hospital_name": str(r.get("center_name", center_id)),
                "report_archive_tier": tier,
                "oct_paths": oct_paths,
                "colposcopy_paths": colpo_paths,
                "instruction_path": "",
                "instruction_text": "",
                "age": r.get("age", ""),
                "hpv": str(r.get("hpv", "")),
                "tct": str(r.get("tct", "")),
                "other_clinical_attributes": str(r.get("pathology_raw", ""))[:200],
                "binary_label": label,
                "binary_label_endpoint": BINARY_ENDPOINT,
                "binary_label_text": f"{BINARY_ENDPOINT}={'positive' if label else 'negative'}",
                "has_real_report": has_report,
                "report_supervision_class": sup_class,
                "needs_pseudo_report": need_pseudo,
                "training_report_type": train_type,
                "real_report_path": real_path,
                "standardized_report_path": str(r.get("standardized_report_path", "")),
                "real_report_text": real_text,
                "has_pseudo_report": 0,
                "pseudo_report_path": "",
                "pseudo_report_text": "",
                "pseudo_report_pass_qc": 0,
                "pseudo_report_confidence": 0.0,
                "qc_score": 0.0,
                "pseudo_training_weight": 0.0,
                "split": str(r.get("split", "train")),
                "fold_id": f"loco_{center_id}",
                "missing_oct": int(not json.loads(oct_paths)),
                "missing_colposcopy": int(not json.loads(colpo_paths)),
                "missing_instruction": int(
                    pd.isna(r.get("age")) and not str(r.get("hpv", "")).strip()
                ),
                "missing_label": 0,
                "missing_report": int(not has_report),
                "audit_flags": ";".join(flags) if flags else "",
                "privacy_flags": "",
            }
        )
    out = pd.DataFrame(rows)
    archive_center = identify_report_archive_center(out)
    out["report_archive_center"] = archive_center
    out["report_archive_centers"] = ",".join(REPORT_ARCHIVE_CENTERS)
    out["semantic_anchor_center"] = SEMANTIC_ANCHOR_CENTER
    out["is_dual_report_centre"] = out["center_id"].isin(REPORT_ARCHIVE_CENTERS).astype(int)
    write_csv(out, out_csv)
    return out


def centre_modality_summary(df: pd.DataFrame) -> pd.DataFrame:
    archive = identify_report_archive_center(df)
    rows = []
    for cid in sorted(df["center_id"].unique()):
        sub = df[df["center_id"] == cid]
        rows.append(
            {
                "center_id": cid,
                "n_cases": len(sub),
                "has_real_report_n": int(sub["has_real_report"].sum()),
                "has_real_report_rate": sub["has_real_report"].mean(),
                "discovered_report_path_n": int(
                    sub["real_report_path"].astype(str).str.len().gt(0).sum()
                ),
                "oct_available_rate": 1 - sub["missing_oct"].mean(),
                "colposcopy_available_rate": 1 - sub["missing_colposcopy"].mean(),
                "cin2plus_rate": sub["binary_label"].mean(),
                "report_archive": int(cid == archive),
                "is_dual_report_centre": int(cid in REPORT_ARCHIVE_CENTERS),
                "semantic_anchor": int(cid == SEMANTIC_ANCHOR_CENTER),
            }
        )
    return pd.DataFrame(rows)
