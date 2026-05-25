#!/usr/bin/env python3
"""Build patient-level manifest from All_3000_5cens with enforced patient splits."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

JBD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JBD_ROOT))

from src.split_policy import assign_patient_stratified_splits, validate_patient_splits

REPO = JBD_ROOT.parents[1]
MULTIMODAL = REPO / "data" / "All_3000_5cens"
REGISTRY = REPO / "data" / "colposcopy_3000" / "3000_nums.xlsx"
OUT = JBD_ROOT / "manifests" / "patient_manifest_v1.csv"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

CENTER_ID = {
    "武大人民医院": "wuda",
    "恩施州中心医院": "enshi",
    "襄阳市中心医院": "xiangyang",
    "十堰市人民医院": "shiyan",
    "荆州市第一人民医院": "jingzhou",
}


def collect_images(folder: Path) -> str:
    if not folder.exists():
        return ""
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return ";".join(str(p.resolve()) for p in files)


def find_report(col_dir: Path) -> tuple[bool, str, str]:
    if not col_dir.exists():
        return False, "", ""
    raw = ""
    for p in col_dir.rglob("*"):
        if not p.is_file():
            continue
        n = p.name.lower()
        if p.suffix.lower() in {".pdf", ".xml", ".doc", ".docx"} or n in {"report.jpg", "report.ini"} or "检查报告" in p.name:
            raw = str(p.resolve())
            return True, p.suffix.lstrip(".").lower() or "jpg", raw
    return False, "", ""


def infer_cin(path_result, label: int) -> tuple[str, int, int]:
    text = "" if pd.isna(path_result) else str(path_result)
    cin_grade = ""
    cin2plus = 0
    cin3plus = 0
    if re.search(r"浸润|恶性肿瘤|鳞癌|腺癌", text):
        cin_grade = "invasive"
        cin2plus = cin3plus = 1
    elif re.search(r"CIN\s*3|HSIL|高度", text, re.I):
        cin_grade = "CIN3+"
        cin2plus = cin3plus = 1
    elif re.search(r"CIN\s*2", text, re.I):
        cin_grade = "CIN2"
        cin2plus = 1
    elif text:
        cin_grade = "CIN0-1/benign"
    elif label == 1:
        cin2plus = 0
    return cin_grade, cin2plus, cin3plus


def main() -> None:
    tr = pd.read_csv(MULTIMODAL / "train_labels.csv")
    te = pd.read_csv(MULTIMODAL / "test_labels.csv")
    labels = pd.concat([tr, te], ignore_index=True)

    mi = pd.read_excel(REGISTRY, sheet_name="MedicalInfo").drop_duplicates("OCT图像Id")
    mi = mi.rename(columns={"OCT图像Id": "OCT", "医院": "center_name", "病理结果": "pathology_raw"})
    labels = labels.merge(mi[["OCT", "pathology_raw"]], on="OCT", how="left")

    pat_agg = (
        labels.groupby("ID", as_index=False)
        .agg(center_name=("center_name", "first"), label=("label", "max"))
        .rename(columns={"ID": "patient_id"})
    )
    pat_agg["center"] = pat_agg["center_name"].map(CENTER_ID)
    pat_agg["label"] = pat_agg["label"].astype(int)

    split_map = assign_patient_stratified_splits(
        pat_agg,
        patient_id_col="patient_id",
        center_col="center",
        label_col="label",
        seed=2026,
    )

    rows = []
    for _, row in labels.iterrows():
        pid = str(row["ID"])
        exam_id = str(row["OCT"])
        split = split_map[pid]
        center = CENTER_ID.get(str(row["center_name"]), "unknown")

        col_dir = col_oct = None
        for sp in ("train", "test"):
            cp = MULTIMODAL / sp / "col" / pid
            op = MULTIMODAL / sp / "oct" / exam_id
            if cp.exists():
                col_dir = cp.resolve()
            if op.exists():
                col_oct = op.resolve()

        rep_avail, rep_src, raw_rep = find_report(col_dir) if col_dir else (False, "", "")
        cin_grade, cin2plus, cin3plus = infer_cin(row.get("pathology_raw"), int(row["label"]))

        rows.append(
            {
                "patient_id": pid,
                "center": center,
                "exam_id": exam_id,
                "colpo_paths": collect_images(col_dir) if col_dir else "",
                "oct_paths": collect_images(col_oct) if col_oct else "",
                "age": row.get("AGE", ""),
                "hpv": row.get("HPV清洗", ""),
                "tct": row.get("TCT清洗", ""),
                "label": int(row["label"]),
                "report_available": int(rep_avail),
                "report_source": rep_src,
                "raw_report_path": raw_rep,
                "standardized_report_path": "",
                "pathology_raw": row.get("pathology_raw", ""),
                "cin_grade": cin_grade,
                "cin2plus": cin2plus,
                "cin3plus": cin3plus,
                "split": split,
                "fold_id": "main",
            }
        )

    manifest = pd.DataFrame(rows)
    ok, errs = validate_patient_splits(manifest)
    if not ok:
        raise SystemExit("Patient split validation failed:\n" + "\n".join(errs))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"Wrote {OUT} ({len(manifest)} exams, {manifest.patient_id.nunique()} patients)")
    print(manifest["split"].value_counts().to_string())
    print("report_available rate:", manifest.report_available.mean())


if __name__ == "__main__":
    main()
