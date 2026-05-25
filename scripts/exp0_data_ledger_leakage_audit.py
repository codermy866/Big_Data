#!/usr/bin/env python3
"""Exp0: Data Ledger and Leakage Audit for RA-HyDRA-LLM (JBD 2026)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

JBD_ROOT = Path(__file__).resolve().parents[1]
REPO = JBD_ROOT.parents[1]
sys.path.insert(0, str(JBD_ROOT))
sys.path.insert(0, str(REPO / "experiments/exp_infofusion_2026/paper_revision/scripts"))

from src.harmonization import (
    CENTER_ID,
    REPORT_ARCHIVE,
    classify_histology,
    classify_hpv,
    classify_tct,
    hist_to_endpoints,
    infer_oct_abnormal,
)
from src.split_policy import assign_patient_stratified_splits, validate_patient_splits

REGISTRY = REPO / "data/colposcopy_3000/3000_nums.xlsx"
MULTIMODAL = REPO / "data/All_3000_5cens"
LEGACY_985 = REPO / "data/5centers_multi"
COL_RAW = REPO / "data/colposcopy_3000"
OCT_REMOTE = REPO / "data/cervix_oct_original"
STD_REPORT_DIR = JBD_ROOT / "data/standardized_reports"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

LEAKAGE_CATEGORIES = {
    "pathology": [r"病理", r"pathology", r"histology"],
    "biopsy_result": [r"活检结果", r"biopsy result"],
    "cin_grade": [r"CIN\s*[0123III]+", r"CIN\s*I{1,3}", r"CIN[0123]", r"CIN II", r"CIN III", r"CINⅡ", r"CINⅢ"],
    "cin2_cin3": [r"CIN\s*2", r"CIN\s*3", r"CINII", r"CINIII", r"CIN II", r"CIN III"],
    "invasive_cancer": [r"浸润癌", r"浸润", r"invasive cancer", r"鳞癌", r"腺癌", r"恶性肿瘤"],
    "final_diagnosis_pathology": [r"最终诊断", r"病理诊断", r"病理提示", r"final diagnosis"],
    "histology_confirms": [r"病理证实", r"histology confirms", r"活检证实"],
    "cancer_generic": [r"宫颈癌", r"癌\b"],
}

LEAKAGE_FLAT_TERMS = [
    "pathology", "biopsy result", "CIN grade", "CIN2", "CIN3", "invasive cancer",
    "final diagnosis after pathology", "histology confirms", "活检结果", "病理提示",
    "病理诊断", "CIN II", "CIN III", "癌",
]


def collect_images(folder: Optional[Path]) -> Tuple[str, int]:
    if folder is None or not folder.exists():
        return "", 0
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return ";".join(str(p.resolve()) for p in files), len(files)


def extract_report_text(path: Path, max_chars: int = 8000) -> str:
    if not path or not Path(path).exists():
        return ""
    p = Path(path)
    suf = p.suffix.lower()
    try:
        if suf == ".xml":
            raw = p.read_text(encoding="utf-8", errors="ignore")
            return re.sub(r"<[^>]+>", " ", raw)[:max_chars]
        if suf in {".txt", ".ini"}:
            return p.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        if suf == ".pdf":
            try:
                import fitz  # pymupdf

                doc = fitz.open(str(p))
                parts = [page.get_text() for page in doc]
                return "\n".join(parts)[:max_chars]
            except ImportError:
                try:
                    from PyPDF2 import PdfReader

                    reader = PdfReader(str(p))
                    parts = [pg.extract_text() or "" for pg in reader.pages[:20]]
                    return "\n".join(parts)[:max_chars]
                except Exception:
                    return "[PDF_BINARY_UNPARSED]"
        if suf in {".jpg", ".jpeg", ".png"}:
            return "[IMAGE_REPORT_OCR_NOT_RUN]"
    except Exception as exc:
        return f"[READ_ERROR:{exc}]"
    return ""


def audit_leakage_text(text: str) -> Dict[str, int]:
    if not text or text.startswith("["):
        return {k: 0 for k in LEAKAGE_CATEGORIES}
    hits = {}
    for cat, patterns in LEAKAGE_CATEGORIES.items():
        hits[cat] = sum(1 for pat in patterns if re.search(pat, text, re.I))
    return hits


def flagged_terms_list(text: str) -> List[str]:
    found = []
    for term in LEAKAGE_FLAT_TERMS:
        if term.lower() in ("癌",) and re.search(r"癌", text):
            if "癌" not in found:
                found.append("癌")
        elif re.search(re.escape(term), text, re.I):
            found.append(term)
    return found


def load_registry() -> pd.DataFrame:
    mi = pd.read_excel(REGISTRY, sheet_name="MedicalInfo")
    oct_img = pd.read_excel(REGISTRY, sheet_name="OCTImages")
    mi = mi.drop_duplicates(subset=["OCT图像Id"], keep="first")
    oct_cols = [c for c in oct_img.columns if c in ("OCT图像Id", "OCT二次判读", "OCT实时判读", "二次判读高级别", "二次判读疑似")]
    oct_img = oct_img[oct_cols].rename(columns={"OCT二次判读": "OCT二次判读_img"})
    df = mi.merge(oct_img, on="OCT图像Id", how="left", suffixes=("", "_dup"))
    df = df.drop_duplicates(subset=["OCT图像Id"], keep="first")
    df = df.rename(
        columns={
            "OCT图像Id": "exam_id",
            "医院": "center_name",
            "年龄": "age",
            "HPV清洗（高亮表示阳性）": "hpv",
            "TCT清洗（高亮表示阳性）": "tct",
            "病理结果": "pathology_raw",
            "病理级别": "pathology_grade",
            "后续治疗": "treatment_text",
            "OCT二次判读": "oct_read_mi",
        }
    )
    df["center"] = df["center_name"].map(CENTER_ID)
    return df


def oct_abnormal_from_row(row: pd.Series) -> Optional[int]:
    for col in ("OCT二次判读_img", "oct_read_mi", "OCT实时判读"):
        if col in row.index:
            v = infer_oct_abnormal(row[col])
            if v is not None:
                return v
    return None


def enrich_clinical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hpv_class"] = out["hpv"].map(classify_hpv)
    out["tct_class"] = out["tct"].map(classify_tct)
    out["oct_abnormal"] = out.apply(oct_abnormal_from_row, axis=1)
    hist = []
    c2, c3, inv = [], [], []
    for _, r in out.iterrows():
        h = classify_histology(
            r.get("pathology_raw"),
            r.get("pathology_grade"),
            r.get("treatment_text"),
            r.get("oct_abnormal"),
        )
        hist.append(h)
        a, b, c = hist_to_endpoints(h)
        c2.append(a)
        c3.append(b)
        inv.append(c)
    out["cin_grade"] = hist
    out["cin2plus"] = c2
    out["cin3plus"] = c3
    out["invasive_cancer"] = inv
    out["pathology_label"] = out["oct_abnormal"].fillna(-1).astype(int)  # proxy when path missing
    return out


def build_modeling_paths() -> pd.DataFrame:
    tr = pd.read_csv(MULTIMODAL / "train_labels.csv")
    te = pd.read_csv(MULTIMODAL / "test_labels.csv")
    lab = pd.concat([tr, te], ignore_index=True)
    rows = []
    for _, row in lab.iterrows():
        pid, eid = str(row["ID"]), str(row["OCT"])
        col_dir = oct_dir = None
        for sp in ("train", "test"):
            cp, op = MULTIMODAL / sp / "col" / pid, MULTIMODAL / sp / "oct" / eid
            if cp.exists():
                col_dir = cp.resolve()
            if op.exists():
                oct_dir = op.resolve()
        col_s, col_n = collect_images(col_dir)
        oct_s, oct_n = collect_images(oct_dir)
        rep_path = ""
        rep_avail = 0
        rep_src = ""
        if col_dir:
            priority = []
            for p in Path(col_dir).rglob("*"):
                if not p.is_file():
                    continue
                nlow = p.name.lower()
                if p.suffix.lower() == ".pdf" or "检查报告" in p.name:
                    priority.insert(0, p)
                elif p.suffix.lower() == ".xml" or nlow == "report.jpg":
                    priority.append(p)
                elif nlow == "report.ini":
                    continue
            if priority:
                p = priority[0]
                rep_avail, rep_src, rep_path = 1, p.suffix.lstrip(".").lower(), str(p.resolve())
        rows.append(
            {
                "patient_id": pid,
                "exam_id": eid,
                "center_name": row["center_name"],
                "center": CENTER_ID.get(str(row["center_name"]), ""),
                "colpo_paths": col_s,
                "colpo_image_count": col_n,
                "oct_paths": oct_s,
                "oct_bscan_count": oct_n,
                "age": row.get("AGE", ""),
                "hpv": row.get("HPV清洗", row.get("hpv", "")),
                "tct": row.get("TCT清洗", row.get("tct", "")),
                "label": int(row["label"]),
                "report_available": rep_avail,
                "report_source": rep_src,
                "raw_report_path": rep_path,
                "in_modeling_cohort": 1,
                "in_985_balanced": 0,
            }
        )
    return pd.DataFrame(rows)


def build_985_manifest(modeling_lookup: pd.DataFrame) -> pd.DataFrame:
    tr = pd.read_csv(LEGACY_985 / "train_labels.csv")
    te = pd.read_csv(LEGACY_985 / "test_labels.csv")
    tr["legacy_split"] = "train"
    te["legacy_split"] = "test"
    lab = pd.concat([tr, te], ignore_index=True)
    m = modeling_lookup.set_index("exam_id")
    rows = []
    for _, row in lab.iterrows():
        eid = str(row["OCT"])
        base = m.loc[eid].to_dict() if eid in m.index else {}
        rows.append(
            {
                **base,
                "patient_id": str(row["ID"]),
                "exam_id": eid,
                "age": row.get("AGE", base.get("age", "")),
                "hpv": row.get("HPV清洗", ""),
                "tct": row.get("TCT清洗", ""),
                "label": int(row["label"]),
                "split": row["legacy_split"],
                "fold_id": "985_balanced",
                "cohort_layer": "985_balanced_comparability",
                "in_modeling_cohort": int(eid in m.index),
                "in_985_balanced": 1,
                "primary_analysis_allowed": 0,
            }
        )
    return pd.DataFrame(rows)


def assign_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    m = manifest.copy()
    if "label" not in m.columns:
        m["label"] = m.get("pathology_label", 0)
    m["label"] = pd.to_numeric(m["label"], errors="coerce").fillna(0).astype(int)
    pat = (
        m.groupby("patient_id", as_index=False)
        .agg(center=("center", "first"), label=("label", "max"))
    )
    split_map = assign_patient_stratified_splits(pat, seed=2026)
    m["split"] = m["patient_id"].map(split_map)
    m["fold_id"] = m.get("fold_id", "main")
    if "fold_id" not in m.columns:
        m["fold_id"] = "main"
    m.loc[m["fold_id"].isna(), "fold_id"] = "main"
    return m


def build_full_manifest(registry: pd.DataFrame, modeling: pd.DataFrame) -> pd.DataFrame:
    reg = enrich_clinical(registry)
    m_idx = modeling.set_index("exam_id")
    pid_map = modeling.drop_duplicates("exam_id").set_index("exam_id")["patient_id"].to_dict()

    rows = []
    for _, r in reg.iterrows():
        eid = str(r["exam_id"])
        in_mod = int(eid in m_idx.index)
        if in_mod:
            md = m_idx.loc[eid]
            pid = md["patient_id"]
            colpo_paths = md["colpo_paths"]
            oct_paths = md["oct_paths"]
            col_n = md["colpo_image_count"]
            oct_n = md["oct_bscan_count"]
            rep_a = md["report_available"]
            rep_p = md["raw_report_path"]
            rep_s = md["report_source"]
            label = int(md["label"])
        else:
            pid = pid_map.get(eid, eid)
            colpo_paths = oct_paths = ""
            col_n = oct_n = 0
            rep_a, rep_p, rep_s = 0, "", ""
            label = int(r["oct_abnormal"]) if pd.notna(r["oct_abnormal"]) else -1

        std_path = ""
        if rep_p:
            cand = STD_REPORT_DIR / f"{eid}.json"
            if cand.exists():
                std_path = str(cand)

        rows.append(
            {
                "patient_id": pid,
                "center": r["center"],
                "exam_id": eid,
                "colpo_paths": colpo_paths,
                "colpo_image_count": col_n,
                "oct_paths": oct_paths,
                "oct_bscan_count": oct_n,
                "age": r["age"],
                "hpv": r["hpv"],
                "tct": r["tct"],
                "pathology_raw": r.get("pathology_raw", ""),
                "label": label if label >= 0 else 0,
                "pathology_label": label,
                "cin_grade": r["cin_grade"],
                "cin2plus": r["cin2plus"],
                "cin3plus": r["cin3plus"],
                "invasive_cancer": r["invasive_cancer"],
                "report_available": rep_a,
                "report_archive_tier": REPORT_ARCHIVE.get(str(r["center"]), ""),
                "report_source": rep_s,
                "raw_report_path": rep_p,
                "standardized_report_path": std_path,
                "hpv_class": r["hpv_class"],
                "tct_class": r["tct_class"],
                "oct_abnormal": r["oct_abnormal"],
                "in_modeling_cohort": in_mod,
                "in_985_balanced": 0,
                "cohort_layer": "registry_full",
                "primary_analysis_allowed": 0,
                "linkage_attrition": "" if in_mod else "no_multimodal_path",
            }
        )
    full = pd.DataFrame(rows)
    full.loc[full.in_modeling_cohort == 1, "cohort_layer"] = "registry_full"
    return full


def ledger_row(center: str, df: pd.DataFrame, modeling_df: pd.DataFrame) -> dict:
    sub = df[df.center == center]
    mod = modeling_df[modeling_df.center == center]
    n_reg = len(sub)
    n_mod = len(mod)
    rep_tier = REPORT_ARCHIVE.get(center, "")

    shiyan_note = ""
    if center == "shiyan":
        shiyan_note = (
            "NOT excluded. Negative-enriched specificity and calibration cohort; "
            f"CIN2+ events n={int(sub.cin2plus.sum())} (extremely low)."
        )

    return {
        "center": center,
        "registry_n": n_reg,
        "modeling_n": n_mod,
        "linkage_attrition_n": n_reg - n_mod,
        "linkage_attrition_pct": round((n_reg - n_mod) / n_reg * 100, 1) if n_reg else 0,
        "report_archive_tier": rep_tier,
        "colposcopy_images_total": int(mod["colpo_image_count"].sum()) if len(mod) else 0,
        "oct_bscans_total": int(mod["oct_bscan_count"].sum()) if len(mod) else 0,
        "hpv_hr_positive_n": int((sub.hpv_class == "hr_positive").sum()),
        "hpv_negative_n": int((sub.hpv_class == "negative").sum()),
        "hpv_unclassifiable_n": int((sub.hpv_class == "unclassifiable").sum()),
        "cytology_negative_n": int((sub.tct_class == "negative").sum()),
        "cytology_ascus_n": int((sub.tct_class == "asc_us").sum()),
        "cytology_lsil_or_worse_n": int((sub.tct_class == "lsil_or_worse").sum()),
        "cytology_missing_n": int((sub.tct_class == "missing").sum()),
        "cytology_missing_pct": round((sub.tct_class == "missing").mean() * 100, 1),
        "cin0_1_n": int((sub.cin_grade == "cin0_1").sum()),
        "cin2_n": int((sub.cin_grade == "cin2").sum()),
        "cin3_n": int((sub.cin_grade == "cin3").sum()),
        "invasive_n": int((sub.cin_grade == "invasive").sum()),
        "histology_missing_n": int((sub.cin_grade == "missing").sum()),
        "cin2plus_n": int(sub.cin2plus.sum()),
        "cin3plus_n": int(sub.cin3plus.sum()),
        "oct_abnormal_n": int(sub.oct_abnormal.eq(1).sum()),
        "oct_normal_n": int(sub.oct_abnormal.eq(0).sum()),
        "oct_read_missing_n": int(sub.oct_abnormal.isna().sum()),
        "reports_in_modeling_n": int(mod.report_available.sum()) if len(mod) else 0,
        "analysis_positioning_note": shiyan_note,
    }


def run_leakage_audit(manifest: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    rep = manifest[manifest.report_available == 1].copy()
    audit_rows = []
    review_rows = []

    for _, row in rep.iterrows():
        raw_p = row.get("raw_report_path", "")
        std_p = row.get("standardized_report_path", "")
        raw_txt = extract_report_text(Path(raw_p)) if raw_p else ""
        std_txt = extract_report_text(Path(std_p)) if std_p else ""
        combined = raw_txt + "\n" + std_txt

        raw_hits = audit_leakage_text(raw_txt)
        std_hits = audit_leakage_text(std_txt) if std_txt else {k: 0 for k in LEAKAGE_CATEGORIES}
        comb_hits = audit_leakage_text(combined)

        flags = flagged_terms_list(combined)
        n_flags = len(flags)
        total_hits = sum(comb_hits.values())
        unstructured = raw_txt.startswith("[IMAGE_REPORT") or raw_txt.startswith("[PDF_BINARY") or raw_txt.startswith("[READ_ERROR")
        if unstructured:
            pass_fail = "review"
        elif total_hits == 0 and not flags:
            pass_fail = "pass"
        elif total_hits >= 3 or comb_hits.get("invasive_cancer", 0) > 0 or comb_hits.get("cin_grade", 0) > 2:
            pass_fail = "fail"
        elif total_hits > 0 or flags:
            pass_fail = "review"
        else:
            pass_fail = "pass"

        audit_rows.append(
            {
                "exam_id": row["exam_id"],
                "patient_id": row["patient_id"],
                "center": row["center"],
                "raw_report_path": raw_p,
                "standardized_report_path": std_p,
                **{f"hit_{k}": comb_hits[k] for k in LEAKAGE_CATEGORIES},
                "total_category_hits": sum(comb_hits.values()),
                "pass_fail": pass_fail,
            }
        )

    audit_df = pd.DataFrame(audit_rows)

    sample = rep.sample(n=min(50, len(rep)), random_state=42) if len(rep) else rep
    for _, row in sample.iterrows():
        raw_p = row.get("raw_report_path", "")
        txt = extract_report_text(Path(raw_p)) if raw_p else ""
        flags = flagged_terms_list(txt)
        excerpt = re.sub(r"\s+", " ", txt)[:400]
        hits = audit_leakage_text(txt)
        total_hits = sum(hits.values())
        if txt.startswith("[IMAGE_REPORT") or txt.startswith("[PDF_BINARY"):
            pf = "review"
        elif total_hits == 0 and not flags:
            pf = "pass"
        elif total_hits >= 5:
            pf = "fail"
        else:
            pf = "review"
        review_rows.append(
            {
                "patient_id": row["patient_id"],
                "center": row["center"],
                "exam_id": row["exam_id"],
                "raw_report_excerpt": excerpt,
                "flagged_terms": ";".join(flags),
                "pass_fail": pf,
                "reviewer_note": "Auto-screened; image/PDF reports may need manual confirmation.",
            }
        )

    n = len(audit_df)
    n_pass = int((audit_df.pass_fail == "pass").sum()) if n else 0
    n_fail = int((audit_df.pass_fail == "fail").sum()) if n else 0
    n_review = int((audit_df.pass_fail == "review").sum()) if n else 0

    cat_totals = {k: int(audit_df[f"hit_{k}"].sum()) for k in LEAKAGE_CATEGORIES} if n else {}

    md = f"""# Exp0 Leakage Audit Report

## Scope
Report-available examinations in **modeling cohort** (n={len(rep)} with `report_available=1`).
RA-HyDRA-LLM uses reports **only as training-time semantic anchors**; inference is report-free.

## Summary
| Metric | Value |
|--------|-------|
| Reports audited | {n} |
| Pass (no flagged terms) | {n_pass} ({(n_pass/n*100) if n else 0:.1f}%) |
| Review (1–2 terms) | {n_review} |
| Fail (≥3 terms or high-risk) | {n_fail} |
| Standardized reports present | {int((manifest.standardized_report_path.fillna('')!='').sum())} |

## Category hit counts (combined raw + standardized text)
"""
    for k, v in cat_totals.items():
        md += f"- **{k}**: {v}\n"

    md += """
## Interpretation
- High hit counts are **expected** for diagnostic colposcopy reports (they describe pathology).
- For RA-HyDRA-LLM, risk is **label leakage into report-anchor branch** at inference — mitigated by report-free test protocol.
- Image-only reports (`report.jpg`) without OCR are flagged for **manual review** (see `leakage_review_50cases.csv`).

## Policy
- Train: Enshi (full reports) + optional Jingzhou `report.jpg` as semantic anchor only.
- Test: **never** pass `raw_report_path` / `standardized_report_path` to the model.
"""
    return audit_df, pd.DataFrame(review_rows), md


def figure1_flow_input(registry_n: int, modeling_n: int, balanced_n: int, ledger: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"stage_id": "A", "stage": "Administrative registry export", "n": 3010, "cohort": "registry"},
        {"stage_id": "B", "stage": "Unique OCT examination IDs", "n": registry_n, "cohort": "registry"},
        {"stage_id": "C", "stage": "Multimodal linkage (OCT+colposcopy paths)", "n": modeling_n, "cohort": "modeling"},
        {"stage_id": "D", "stage": "Excluded from modeling (linkage attrition)", "n": registry_n - modeling_n, "cohort": "registry"},
        {"stage_id": "E", "stage": "985 balanced comparability subset", "n": balanced_n, "cohort": "985_balanced"},
    ]
    for _, r in ledger.iterrows():
        rows.append(
            {
                "stage_id": f"L_{r['center']}",
                "stage": f"Modeling linked — {r['center']}",
                "n": int(r["modeling_n"]),
                "cohort": "modeling",
            }
        )
    return pd.DataFrame(rows)


def table_985_provenance() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subset": "985_balanced",
                "source_dataset": "data/5centers_multi",
                "formation": "Legacy curated release with verified OCT+colposcopy+clinical rows",
                "n_examinations": 985,
                "n_negative": 663,
                "n_positive": 322,
                "neg_pos_ratio": round(663 / 322, 2),
                "intended_use": "Sensitivity/comparability analyses only",
                "primary_conclusion_allowed": 0,
                "note": "NOT the primary analytic cohort for JBD main tables; use modeling cohort (1897) with patient-level splits.",
            }
        ]
    )


def main() -> None:
    out_m = JBD_ROOT / "manifests"
    out_t = JBD_ROOT / "tables"
    out_r = JBD_ROOT / "reports"
    out_f = JBD_ROOT / "figures"
    for d in (out_m, out_t, out_r, out_f):
        d.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    modeling_paths = build_modeling_paths()
    modeling_paths = modeling_paths.merge(
        registry[["exam_id", "pathology_raw", "pathology_grade", "treatment_text"]],
        on="exam_id",
        how="left",
    )
    modeling = enrich_clinical(modeling_paths)
    modeling["cohort_layer"] = "modeling"
    modeling["primary_analysis_allowed"] = 1
    modeling["in_modeling_cohort"] = 1
    modeling["report_archive_tier"] = modeling["center"].map(REPORT_ARCHIVE)
    modeling = assign_splits(modeling)
    modeling["standardized_report_path"] = ""

    full = build_full_manifest(registry, modeling)
    full = assign_splits(full)

    balanced = build_985_manifest(modeling)
    balanced = enrich_clinical(
        balanced.merge(registry[["exam_id", "pathology_raw", "pathology_grade", "treatment_text"]], on="exam_id", how="left")
    )

    for name, df in [("full", full), ("modeling", modeling)]:
        ok, errs = validate_patient_splits(df[df.split.isin(["train", "val", "test"])])
        if not ok:
            raise SystemExit(f"Split leakage in {name}:\n" + "\n".join(errs))

    full.to_csv(out_m / "patient_manifest_full.csv", index=False, encoding="utf-8-sig")
    modeling.to_csv(out_m / "patient_manifest_modeling.csv", index=False, encoding="utf-8-sig")
    balanced.to_csv(out_m / "patient_manifest_985_balanced.csv", index=False, encoding="utf-8-sig")

    ledger_rows = [ledger_row(c, full, modeling) for c in sorted(full.center.unique())]
    ledger = pd.DataFrame(ledger_rows)
    ledger.loc[len(ledger)] = {
        "center": "overall",
        "registry_n": len(full),
        "modeling_n": len(modeling),
        "linkage_attrition_n": len(full) - len(modeling),
        "linkage_attrition_pct": round((len(full) - len(modeling)) / len(full) * 100, 1),
        "report_archive_tier": "mixed",
        "colposcopy_images_total": int(modeling.colpo_image_count.sum()),
        "oct_bscans_total": int(modeling.oct_bscan_count.sum()),
        "hpv_hr_positive_n": int((full.hpv_class == "hr_positive").sum()),
        "hpv_negative_n": int((full.hpv_class == "negative").sum()),
        "hpv_unclassifiable_n": int((full.hpv_class == "unclassifiable").sum()),
        "cytology_negative_n": int((full.tct_class == "negative").sum()),
        "cytology_ascus_n": int((full.tct_class == "asc_us").sum()),
        "cytology_lsil_or_worse_n": int((full.tct_class == "lsil_or_worse").sum()),
        "cytology_missing_n": int((full.tct_class == "missing").sum()),
        "cytology_missing_pct": round((full.tct_class == "missing").mean() * 100, 1),
        "cin0_1_n": int((full.cin_grade == "cin0_1").sum()),
        "cin2_n": int((full.cin_grade == "cin2").sum()),
        "cin3_n": int((full.cin_grade == "cin3").sum()),
        "invasive_n": int((full.cin_grade == "invasive").sum()),
        "histology_missing_n": int((full.cin_grade == "missing").sum()),
        "cin2plus_n": int(full.cin2plus.sum()),
        "cin3plus_n": int(full.cin3plus.sum()),
        "oct_abnormal_n": int(full.oct_abnormal.eq(1).sum()),
        "oct_normal_n": int(full.oct_abnormal.eq(0).sum()),
        "oct_read_missing_n": int(full.oct_abnormal.isna().sum()),
        "reports_in_modeling_n": int(modeling.report_available.sum()),
        "analysis_positioning_note": "Primary modeling cohort for JBD main analysis.",
    }
    ledger.to_csv(out_t / "Table1_data_ledger.csv", index=False, encoding="utf-8-sig")

    reg_table = full.groupby("center").agg(
        registry_n=("exam_id", "count"),
        modeling_n=("in_modeling_cohort", "sum"),
        cin2plus=("cin2plus", "sum"),
        cin3plus=("cin3plus", "sum"),
    ).reset_index()
    reg_table.to_csv(out_t / "TableS1_registry_full.csv", index=False, encoding="utf-8-sig")

    table_985_provenance().to_csv(out_t / "Table_985_balanced_provenance.csv", index=False, encoding="utf-8-sig")

    audit_df, review_df, leak_md = run_leakage_audit(modeling)
    audit_df.to_csv(out_r / "leakage_audit_summary.csv", index=False, encoding="utf-8-sig")
    review_df.to_csv(out_r / "leakage_review_50cases.csv", index=False, encoding="utf-8-sig")
    (out_r / "leakage_audit_report.md").write_text(leak_md, encoding="utf-8")

    fig1 = figure1_flow_input(len(full), len(modeling), len(balanced), ledger[ledger.center != "overall"])
    fig1.to_csv(out_f / "Figure1_data_flow_input.csv", index=False, encoding="utf-8-sig")

    summary = {
        "registry_n": len(full),
        "modeling_n": len(modeling),
        "balanced_n": len(balanced),
        "patients_modeling": int(modeling.patient_id.nunique()),
        "leakage_audited": len(audit_df),
        "leakage_pass_rate": float((audit_df.pass_fail == "pass").mean()) if len(audit_df) else None,
        "leakage_review_rate": float((audit_df.pass_fail == "review").mean()) if len(audit_df) else None,
        "leakage_fail_rate": float((audit_df.pass_fail == "fail").mean()) if len(audit_df) else None,
    }
    (out_r / "exp0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Exp0 complete.")


if __name__ == "__main__":
    main()
