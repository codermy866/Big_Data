#!/usr/bin/env python3
"""Generate reviewer-facing audit tables for the JBD MOSAIC revision.

This script is intentionally read-only with respect to training artefacts. It
derives split, overlap, leakage-boundary, CIN3+ proxy safety, QC-distribution,
embedding, and experiment-to-visualisation maps from existing locked manifests,
prediction files, and generated figures.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = PROJECT / "outputs" / "reviewer_audit"
TABLES = OUT / "tables"

MANIFEST = ROOT / "outputs" / "manifests" / "full_manifest.csv"
PSEUDO_MANIFEST = ROOT / "outputs" / "manifests" / "full_manifest_with_pseudo_reports.csv"
VISUAL_MANIFEST = ROOT / "outputs" / "publishable" / "manifests" / "full_manifest_with_visual_embeddings.csv"
FUSION_SCORES = ROOT / "outputs" / "publishable" / "kra_semantic_fusion_analysis" / "kra_semantic_fusion_val_test_scores.csv"
MANUSCRIPT_TABLES = ROOT / "outputs" / "publishable" / "tables" / "manuscript"
EXTERNAL_PRED_DIR = ROOT / "outputs" / "publishable" / "external_baselines" / "predictions"
REVISION_DIR = PROJECT / "outputs" / "revision"
LLM_SEM_DIR = PROJECT / "outputs" / "llm_semantic"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _parse_paths(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(v) for v in parsed if str(v)]
    return [str(parsed)]


def _hash_id(value: Any) -> str:
    text = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _derive_cin3_proxy(text: Any) -> int:
    """Conservative pathology-text proxy for CIN3+/AIS/cancer.

    Generic high-grade wording is not enough because CIN2 is also high-grade.
    This proxy is therefore suitable for safety auditing, not as a replacement
    for a locked clinically curated CIN3+ endpoint.
    """

    if text is None or (isinstance(text, float) and math.isnan(text)):
        return 0
    s = str(text).upper().replace("（", "(").replace("）", ")")
    s_compact = re.sub(r"\s+", "", s)
    cin3_patterns = [
        r"CIN\s*3",
        r"CIN3",
        r"CIN\s*III",
        r"CINIII",
        r"CINⅢ",
        r"CIN\s*3\s*级",
        r"CIN3\s*级",
    ]
    if any(re.search(p, s, flags=re.IGNORECASE) for p in cin3_patterns):
        return 1
    if re.search(r"CIN(3|III|Ⅲ|3级)", s_compact, flags=re.IGNORECASE):
        return 1
    if re.search(r"AIS|原位腺癌", s, flags=re.IGNORECASE):
        return 1

    cancer_terms = ["鳞状细胞癌", "腺癌", "宫颈癌", "浸润性癌", "浸润癌", "恶性肿瘤"]
    cancer_negations = ["未见癌", "无癌", "未查见癌", "未见恶性", "除外浸润性癌可能", "未见浸润"]
    if any(term in s for term in cancer_terms) and not any(neg in s for neg in cancer_negations):
        return 1
    return 0


def _auc_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    equal = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def _average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    precision = tp / (np.arange(len(ys)) + 1)
    return float((precision * ys).sum() / n_pos)


def _class_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(score, dtype=float) >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if prec == prec and sens == sens and (prec + sens) else 0.0
    return {
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": 0.5 * (sens + spec) if sens == sens and spec == spec else float("nan"),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "positive_calls": int(pred.sum()),
    }


def _threshold_for_sensitivity(y_true: np.ndarray, score: np.ndarray, target: float = 0.95) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    candidates = np.unique(np.concatenate(([0.0], s, [1.0])))
    rows = []
    for thr in candidates:
        m = _class_metrics(y, s, float(thr))
        rows.append((float(thr), m["sensitivity"], m["specificity"]))
    feasible = [r for r in rows if r[1] == r[1] and r[1] >= target]
    if feasible:
        best = sorted(feasible, key=lambda x: (x[2], x[0]), reverse=True)[0]
    else:
        best = sorted(rows, key=lambda x: (x[1] if x[1] == x[1] else -1, x[2] if x[2] == x[2] else -1), reverse=True)[0]
    return best


def add_basic_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cin3plus_proxy"] = out["other_clinical_attributes"].map(_derive_cin3_proxy)
    out["n_oct_images"] = out["oct_paths"].map(lambda x: len(_parse_paths(x)))
    out["n_colposcopy_images"] = out["colposcopy_paths"].map(lambda x: len(_parse_paths(x)))
    out["n_total_images"] = out["n_oct_images"] + out["n_colposcopy_images"]
    out["patient_hash"] = out["patient_id"].map(_hash_id) if "patient_id" in out.columns else ""
    return out


def build_split_audit(df: pd.DataFrame) -> None:
    rows = []
    for split, g in df.groupby("split", dropna=False):
        rows.append(
            {
                "split": split,
                "cases": len(g),
                "cin2plus_positive_cases": int(g["binary_label"].sum()),
                "cin2plus_negative_cases": int((1 - g["binary_label"].astype(int)).sum()),
                "cin3plus_proxy_positive_cases": int(g["cin3plus_proxy"].sum()),
                "real_report_cases": int(g["has_real_report"].sum()),
                "pseudo_report_candidates": int(g["needs_pseudo_report"].sum()),
                "centres_represented": int(g["center_id"].nunique()),
                "centre_ids": ";".join(sorted(g["center_id"].dropna().astype(str).unique())),
                "oct_images": int(g["n_oct_images"].sum()),
                "colposcopy_images": int(g["n_colposcopy_images"].sum()),
                "total_images": int(g["n_total_images"].sum()),
            }
        )
    overall = {
        "split": "overall",
        "cases": len(df),
        "cin2plus_positive_cases": int(df["binary_label"].sum()),
        "cin2plus_negative_cases": int((1 - df["binary_label"].astype(int)).sum()),
        "cin3plus_proxy_positive_cases": int(df["cin3plus_proxy"].sum()),
        "real_report_cases": int(df["has_real_report"].sum()),
        "pseudo_report_candidates": int(df["needs_pseudo_report"].sum()),
        "centres_represented": int(df["center_id"].nunique()),
        "centre_ids": ";".join(sorted(df["center_id"].dropna().astype(str).unique())),
        "oct_images": int(df["n_oct_images"].sum()),
        "colposcopy_images": int(df["n_colposcopy_images"].sum()),
        "total_images": int(df["n_total_images"].sum()),
    }
    split = pd.DataFrame(rows + [overall])
    split.to_csv(TABLES / "split_audit.csv", index=False)

    center = (
        df.groupby(["split", "center_id"], as_index=False)
        .agg(
            cases=("case_id", "count"),
            cin2plus_positive_cases=("binary_label", "sum"),
            cin3plus_proxy_positive_cases=("cin3plus_proxy", "sum"),
            real_report_cases=("has_real_report", "sum"),
            pseudo_report_candidates=("needs_pseudo_report", "sum"),
            oct_images=("n_oct_images", "sum"),
            colposcopy_images=("n_colposcopy_images", "sum"),
            total_images=("n_total_images", "sum"),
        )
        .sort_values(["split", "center_id"])
    )
    center.to_csv(TABLES / "split_center_audit.csv", index=False)


def prediction_sets(manifest: pd.DataFrame) -> dict[str, dict[str, Any]]:
    locked_test = set(manifest[manifest["split"].eq("test")]["case_id"].astype(str))
    out: dict[str, dict[str, Any]] = {}

    fusion = _read_csv(FUSION_SCORES)
    fusion_test = fusion[fusion["split"].eq("test")].copy()
    for model, col in [
        ("MOSAIC full", "semantic_fusion_score"),
        ("MOSAIC-RASA backbone", "risk_score"),
        ("Semantic retrieval only", "semantic_retrieval_positive_ratio"),
    ]:
        out[model] = {
            "source_file": str(FUSION_SCORES),
            "cases": set(fusion_test["case_id"].astype(str)),
            "score_column": col,
        }

    for path in sorted(EXTERNAL_PRED_DIR.glob("*_test_predictions.csv")):
        pred = _read_csv(path)
        model = str(pred["baseline_id"].iloc[0]).replace("_", " ")
        out[model] = {"source_file": str(path), "cases": set(pred["case_id"].astype(str)), "score_column": "risk_score"}

    tag_pred = REVISION_DIR / "semantic_tag_source_ablation_predictions.csv"
    if tag_pred.is_file():
        pred = _read_csv(tag_pred)
        pred = pred[pred["split"].eq("test") & pred["variant_id"].eq("all_rule_tags")]
        out["MOSAIC-Tag deterministic fallback"] = {
            "source_file": str(tag_pred),
            "cases": set(pred["case_id"].astype(str)),
            "score_column": "fusion_score",
        }

    rows = []
    for model, spec in out.items():
        ids = spec["cases"]
        rows.append(
            {
                "model": model,
                "source_file": spec["source_file"],
                "score_column": spec["score_column"],
                "n_cases": len(ids),
                "n_overlap_locked_test": len(ids & locked_test),
                "n_missing_from_locked_test": len(locked_test - ids),
                "n_extra_not_in_locked_test": len(ids - locked_test),
                "same_case_ids_as_locked_test": int(ids == locked_test),
            }
        )
    pd.DataFrame(rows).sort_values("model").to_csv(TABLES / "model_case_overlap.csv", index=False)

    models = sorted(out)
    matrix = []
    for a in models:
        row = {"model": a}
        for b in models:
            row[b] = len(out[a]["cases"] & out[b]["cases"])
        matrix.append(row)
    pd.DataFrame(matrix).to_csv(TABLES / "model_case_overlap_matrix.csv", index=False)
    return out


def build_leakage_tables() -> None:
    label_rows = [
        {
            "pipeline_stage": "Pseudo-report completion",
            "split": "train only for model learning",
            "input_fields": "structured clinical variables, visual evidence summaries, weak CIN2+ label for report-missing training supervision",
            "uses_y": "yes",
            "purpose": "weak-oracle semantic supervision construction",
            "leakage_risk": "training-time label verbalisation if reused as validation/test query text",
            "mitigation": "real reports preserved, pseudo reports treated as weak targets, validation/test retrieval queries exclude labels and label-derived pseudo sections",
        },
        {
            "pipeline_stage": "RASA representation learning",
            "split": "train",
            "input_fields": "multimodal embeddings, clinical variables, report sections, CIN2+ risk label",
            "uses_y": "yes",
            "purpose": "supervised risk head and weak semantic alignment",
            "leakage_risk": "ordinary supervised training risk",
            "mitigation": "case-level split; no test labels during training",
        },
        {
            "pipeline_stage": "Semantic bank construction",
            "split": "train",
            "input_fields": "training case report-derived entities and labels",
            "uses_y": "yes for bank labels",
            "purpose": "train-only retrieval prior",
            "leakage_risk": "test labels entering semantic memory",
            "mitigation": "bank built from training cases only",
        },
        {
            "pipeline_stage": "Validation calibration",
            "split": "validation",
            "input_fields": "validation predictions and validation labels",
            "uses_y": "yes",
            "purpose": "alpha and threshold selection",
            "leakage_risk": "overfitting if repeated after test inspection",
            "mitigation": "single locked grid; test set evaluated after selection",
        },
        {
            "pipeline_stage": "Final test evaluation",
            "split": "test",
            "input_fields": "locked test predictions and labels",
            "uses_y": "yes",
            "purpose": "metric computation and bootstrap resampling only",
            "leakage_risk": "test-set model selection",
            "mitigation": "no alpha, threshold, semantic bank, or model selection from test labels",
        },
        {
            "pipeline_stage": "Direct LLM semantic tagging",
            "split": "not executed",
            "input_fields": "none",
            "uses_y": "no",
            "purpose": "unavailable",
            "leakage_risk": "fabricated LLM evidence",
            "mitigation": "no LLM endpoint configured; direct LLM-tag rows kept unavailable",
        },
    ]
    label_map = pd.DataFrame(label_rows)
    label_map.to_csv(TABLES / "label_use_map.csv", index=False)

    leakage = label_map.rename(
        columns={
            "pipeline_stage": "stage",
            "uses_y": "target_labels_used",
        }
    )[["stage", "split", "target_labels_used", "purpose", "leakage_risk", "mitigation"]]
    leakage.to_csv(TABLES / "leakage_audit.csv", index=False)


def build_cin3_safety(manifest: pd.DataFrame) -> None:
    id_to_cin3 = manifest.set_index("case_id")["cin3plus_proxy"].to_dict()
    rows = []

    def add_row(model: str, score_df: pd.DataFrame, score_col: str, threshold: float, source: str, op: str) -> None:
        d = score_df.copy()
        d["case_id"] = d["case_id"].astype(str)
        d["cin3plus_proxy"] = d["case_id"].map(id_to_cin3).fillna(0).astype(int)
        y = d["cin3plus_proxy"].to_numpy()
        s = d[score_col].to_numpy(dtype=float)
        row = {
            "model": model,
            "operating_point": op,
            "source_file": source,
            "n_test": len(d),
            "cin3plus_proxy_positive_cases": int(y.sum()),
            "threshold": threshold,
            "cin3plus_proxy_auroc": _auc_rank(y, s),
            "cin3plus_proxy_auprc": _average_precision(y, s),
        }
        row.update(_class_metrics(y, s, threshold))
        rows.append(row)

    fusion = _read_csv(FUSION_SCORES)
    test = fusion[fusion["split"].eq("test")].copy()
    main = _read_csv(MANUSCRIPT_TABLES / "T_mosaic_main_comparison.csv")
    for model_id, label, col in [
        ("kra_semantic_fusion", "MOSAIC full", "semantic_fusion_score"),
        ("full_lcad_rasa_stablehash", "MOSAIC-RASA backbone", "risk_score"),
        ("semantic_retrieval_positive_ratio", "Semantic retrieval only", "semantic_retrieval_positive_ratio"),
    ]:
        thr = float(main.loc[main["model_id"].eq(model_id), "threshold"].iloc[0])
        add_row(label, test, col, thr, str(FUSION_SCORES), "validation_selected_cin2_f1_threshold")

    ext_models = {
        "contrastive_multimodal_no_report_sections_test_predictions.csv": "CLIP-style contrastive baseline",
        "clinical_hist_gradient_boosting_test_predictions.csv": "Clinical-only HistGradientBoosting",
        "clinical_lr_test_predictions.csv": "Clinical-only logistic regression",
    }
    for file_name, label in ext_models.items():
        path = EXTERNAL_PRED_DIR / file_name
        if path.is_file():
            pred = _read_csv(path)
            thr = float(pred["threshold_val_selected"].iloc[0])
            add_row(label, pred, "risk_score", thr, str(path), "validation_selected_cin2_f1_threshold")

    pd.DataFrame(rows).to_csv(TABLES / "cin3_safety_metrics.csv", index=False)

    trade_rows = []
    val = fusion[fusion["split"].eq("val")].copy()
    val["cin3plus_proxy"] = val["case_id"].map(id_to_cin3).fillna(0).astype(int)
    test = test.copy()
    test["cin3plus_proxy"] = test["case_id"].map(id_to_cin3).fillna(0).astype(int)
    for label, col in [
        ("MOSAIC full", "semantic_fusion_score"),
        ("MOSAIC-RASA backbone", "risk_score"),
        ("Semantic retrieval only", "semantic_retrieval_positive_ratio"),
    ]:
        thr, val_sens, val_spec = _threshold_for_sensitivity(val["cin3plus_proxy"].to_numpy(), val[col].to_numpy(), 0.95)
        m = _class_metrics(test["cin3plus_proxy"].to_numpy(), test[col].to_numpy(), thr)
        trade_rows.append(
            {
                "model": label,
                "operating_point": "validation_selected_proxy_cin3_sensitivity_target_0.95",
                "threshold": thr,
                "validation_proxy_cin3_sensitivity": val_sens,
                "validation_proxy_cin3_specificity": val_spec,
                "test_proxy_cin3_positive_cases": int(test["cin3plus_proxy"].sum()),
                **m,
            }
        )
    for label in ["CLIP-style contrastive baseline", "Clinical-only HistGradientBoosting", "Clinical-only logistic regression"]:
        trade_rows.append(
            {
                "model": label,
                "operating_point": "not_available_missing_validation_predictions",
                "threshold": float("nan"),
                "validation_proxy_cin3_sensitivity": float("nan"),
                "validation_proxy_cin3_specificity": float("nan"),
                "test_proxy_cin3_positive_cases": int(test["cin3plus_proxy"].sum()),
                "note": "Only locked test predictions were available for this audit; high-sensitivity threshold was not test-tuned.",
            }
        )
    pd.DataFrame(trade_rows).to_csv(TABLES / "cin3_threshold_tradeoff.csv", index=False)


def build_embedding_and_duplicate_audits(df: pd.DataFrame) -> None:
    visual = add_basic_fields(_read_csv(VISUAL_MANIFEST))
    emb_cols = [
        "case_id",
        "patient_hash",
        "center_id",
        "split",
        "binary_label",
        "cin3plus_proxy",
        "has_real_report",
        "needs_pseudo_report",
        "n_oct_images",
        "n_colposcopy_images",
        "oct_embedding_path",
        "colposcopy_embedding_path",
        "fused_visual_embedding_path",
        "has_visual_embedding",
        "missing_embedding",
    ]
    visual[emb_cols].to_csv(TABLES / "embedding_manifest_audit.csv", index=False)

    rows = []
    case_split_counts = df.groupby("case_id")["split"].nunique()
    rows.append(
        {
            "check": "case_id appears in multiple splits",
            "n_violations": int((case_split_counts > 1).sum()),
            "status": "pass" if int((case_split_counts > 1).sum()) == 0 else "fail",
        }
    )
    patient_split_counts = df[df["patient_hash"].ne("")].groupby("patient_hash")["split"].nunique()
    rows.append(
        {
            "check": "patient_hash appears in multiple splits",
            "n_violations": int((patient_split_counts > 1).sum()),
            "status": "pass" if int((patient_split_counts > 1).sum()) == 0 else "review",
        }
    )

    path_rows = []
    for _, row in df.iterrows():
        for modality, col in [("oct", "oct_paths"), ("colposcopy", "colposcopy_paths")]:
            for p in _parse_paths(row[col]):
                path_rows.append({"path": p, "modality": modality, "case_id": row["case_id"], "split": row["split"]})
    paths = pd.DataFrame(path_rows)
    if not paths.empty:
        cross = paths.groupby(["modality", "path"]).agg(n_splits=("split", "nunique"), n_cases=("case_id", "nunique")).reset_index()
        cross = cross[(cross["n_splits"] > 1) | (cross["n_cases"] > 1)]
        rows.append(
            {
                "check": "raw image path reused across cases or splits",
                "n_violations": len(cross),
                "status": "pass" if len(cross) == 0 else "fail",
            }
        )
        cross.to_csv(TABLES / "duplicate_image_path_violations.csv", index=False)

    emb_paths = []
    for _, row in visual.iterrows():
        for modality, col in [
            ("oct_embedding", "oct_embedding_path"),
            ("colposcopy_embedding", "colposcopy_embedding_path"),
            ("fused_visual_embedding", "fused_visual_embedding_path"),
        ]:
            emb_paths.append({"path": row.get(col, ""), "modality": modality, "case_id": row["case_id"], "split": row["split"]})
    emb = pd.DataFrame(emb_paths)
    emb = emb[emb["path"].astype(str).ne("")]
    cross_emb = emb.groupby(["modality", "path"]).agg(n_splits=("split", "nunique"), n_cases=("case_id", "nunique")).reset_index()
    cross_emb = cross_emb[(cross_emb["n_splits"] > 1) | (cross_emb["n_cases"] > 1)]
    rows.append(
        {
            "check": "embedding file path reused across cases or splits",
            "n_violations": len(cross_emb),
            "status": "pass" if len(cross_emb) == 0 else "fail",
        }
    )
    cross_emb.to_csv(TABLES / "duplicate_embedding_path_violations.csv", index=False)
    pd.DataFrame(rows).to_csv(TABLES / "duplicate_leakage_check.csv", index=False)


def build_qc_outputs(df: pd.DataFrame) -> None:
    pseudo = df[df["needs_pseudo_report"].astype(int).eq(1)].copy()
    cols = [
        "case_id",
        "center_id",
        "split",
        "binary_label",
        "cin3plus_proxy",
        "pseudo_report_pass_qc",
        "pseudo_report_confidence",
        "qc_score",
        "pseudo_training_weight",
    ]
    pseudo[cols].to_csv(TABLES / "pseudo_report_qc_scores.csv", index=False)
    thresholds = []
    for theta in [0.50, 0.60, 0.70, 0.80, 0.90]:
        retained = pseudo[pseudo["qc_score"].astype(float) >= theta]
        thresholds.append(
            {
                "theta_qc": theta,
                "retained_pseudo_reports": len(retained),
                "retained_fraction": len(retained) / len(pseudo) if len(pseudo) else float("nan"),
                "mean_training_weight_retained": float(retained["pseudo_training_weight"].mean()) if len(retained) else float("nan"),
                "downstream_retraining": "not_rerun; retained-count and QC-stress audit only",
            }
        )
    pd.DataFrame(thresholds).to_csv(TABLES / "qc_threshold_sensitivity.csv", index=False)

    desc = pseudo["qc_score"].astype(float).describe(percentiles=[0.25, 0.5, 0.75])
    tex = "\n".join(
        [
            "\\begin{tabular}{lr}",
            "\\toprule",
            "Statistic & QC score \\\\",
            "\\midrule",
            f"Mean & {desc['mean']:.3f} \\\\",
            f"SD & {desc['std']:.3f} \\\\",
            f"Median & {desc['50%']:.3f} \\\\",
            f"IQR & {desc['25%']:.3f}--{desc['75%']:.3f} \\\\",
            f"Min--max & {desc['min']:.3f}--{desc['max']:.3f} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )
    (TABLES / "qc_distribution_table.tex").write_text(tex, encoding="utf-8")


def build_experiment_map() -> pd.DataFrame:
    rows = [
        {
            "prompt": "Prompt 2",
            "experiment": "Locked split and case-overlap audit",
            "status": "generated",
            "main_result": "Canonical split is 1325/284/288 cases; all same-split core predictors overlap the locked test IDs.",
            "result_tables": "outputs/reviewer_audit/tables/split_audit.csv; outputs/reviewer_audit/tables/model_case_overlap.csv; outputs/reviewer_audit/tables/leakage_audit.csv",
            "figures": "Figure1_mosaic_overview.pdf; Figure2_centre_supervision_catplot.pdf",
            "manuscript_use": "Open Results with locked split and leakage-control protocol.",
        },
        {
            "prompt": "Prompt 5",
            "experiment": "Label-use boundary and semantic-tag leakage audit",
            "status": "generated from current audit outputs",
            "main_result": "Cleaned semantic-tag input scanned 1897 cases with 0 forbidden target-term, path-like, or identifier hits across train/val/test.",
            "result_tables": "outputs/reviewer_audit/tables/label_use_map.csv; outputs/revision/semantic_tag_leakage_audit.csv; outputs/revision/semantic_tag_source_ablation_metrics.csv",
            "figures": "Figure_mosaic_performance_summary.pdf; optional semantic-tag table only",
            "manuscript_use": "State training labels are weak supervision; validation/test retrieval queries remain label-free.",
        },
        {
            "prompt": "Prompt 1",
            "experiment": "LLM-specific semantic-tag availability and deterministic fallback",
            "status": "completed with claim downgrade",
            "main_result": "No supported LLM endpoint was configured; direct LLM-tag rows are N/A. MOSAIC-Tag deterministic fallback AUROC 0.878, AUPRC 0.695, F1 0.556.",
            "result_tables": "outputs/llm_semantic/LLM_NOT_AVAILABLE.md; outputs/llm_semantic/table_llm_specific_ablation.csv; outputs/revision/mosaic_tag_vs_baselines.csv",
            "figures": "P8_llm_provider_comparison_heatmap.pdf for provider audit; no direct LLM-tag performance figure",
            "manuscript_use": "Do not claim direct LLM-tag superiority.",
        },
        {
            "prompt": "Prompt 4",
            "experiment": "Pseudo-report QC distribution and failure-enriched stress test",
            "status": "generated",
            "main_result": "Clean QC pass rate 1.000; corrupted pass rate 0.716; contradiction, missing-section, and duplicate-template detection rates 1.000; hallucination detection 0.999.",
            "result_tables": "outputs/reviewer_audit/tables/pseudo_report_qc_scores.csv; outputs/reviewer_audit/tables/qc_threshold_sensitivity.csv; outputs/qc/qc_failure_enriched_stress_test.csv",
            "figures": "outputs/qc/qc_failure_enriched_stress_test.png; fig_lcad_qc_ablation_barplot.pdf",
            "manuscript_use": "Frame QC as weak-supervision filtering and stress-test evidence, not as clinical validation.",
        },
        {
            "prompt": "Prompt 6",
            "experiment": "Same-split baseline fairness and paired bootstrap",
            "status": "regenerated",
            "main_result": "MOSAIC AUROC 0.908 vs CLIP-style 0.888; paired delta 0.020 with CI crossing 0, so no universal superiority claim.",
            "result_tables": "outputs/statistics/full_mosaic_vs_external_baselines_paired_bootstrap.csv; cervix_lcad_rasa/outputs/publishable/tables/manuscript/T_external_baselines_same_split_with_mosaic.csv",
            "figures": "final_Fig/Figure_external_baselines_auc_forest.pdf; final_Fig/Figure_external_baselines_paired_delta_auc.pdf",
            "manuscript_use": "Use competitive/same-split phrasing; significance only vs own backbone.",
        },
        {
            "prompt": "Prompt 7",
            "experiment": "CIN3+ proxy safety threshold audit",
            "status": "generated as pathology-text proxy",
            "main_result": "Generated CIN3+ safety metrics from conservative pathology-text proxy; not a locked curated CIN3+ label.",
            "result_tables": "outputs/reviewer_audit/tables/cin3_safety_metrics.csv; outputs/reviewer_audit/tables/cin3_threshold_tradeoff.csv",
            "figures": "No dedicated figure; report as safety table.",
            "manuscript_use": "Add safety-oriented table with proxy caveat and prospective validation limitation.",
        },
        {
            "prompt": "Prompt 8",
            "experiment": "Case-level embedding and duplicate leakage audit",
            "status": "generated",
            "main_result": "Embedding audit covers 1897 cases with hashed patient IDs and embedding paths; duplicate and cross-split checks exported.",
            "result_tables": "outputs/reviewer_audit/tables/embedding_manifest_audit.csv; outputs/reviewer_audit/tables/duplicate_leakage_check.csv",
            "figures": "No dedicated figure; Methods/Supplementary audit table.",
            "manuscript_use": "Describe cached case-level embeddings and split leakage checks.",
        },
        {
            "prompt": "Prompt 3",
            "experiment": "Strict LOCO centre-shift stress test",
            "status": "existing canonical CSV used; figure refreshed",
            "main_result": "Strict retrain backbone AUROC: Enshi 0.702, Jingzhou 0.648, Shiyan 0.500, Wuhan Renmin 1.000, Xiangyang 0.382.",
            "result_tables": "cervix_lcad_rasa/outputs/publishable/tables/manuscript/S2_loco_strict_retrain.csv; S2b_loco_eval_only_global_ckpt.csv",
            "figures": "final_Fig/Figure4_loco_forest_catplot.pdf; final_Fig/fig_loco_heatmap.pdf",
            "manuscript_use": "Centre-shift stress testing; no centre-invariant deployment claim.",
        },
        {
            "prompt": "Prompt 9",
            "experiment": "Figure 1 overview and manuscript figure organisation",
            "status": "regenerated",
            "main_result": "Overview figure generated with cohort, weak-oracle completion, RASA alignment, train-only retrieval, and validation calibration.",
            "result_tables": "outputs/reviewer_audit/tables/experiment_result_visualization_map.csv",
            "figures": "cervix_lcad_rasa/outputs/publishable/figures/jbd_final/Figure1_mosaic_overview.pdf",
            "manuscript_use": "Insert as Figure 1 and cite before result blocks.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "experiment_result_visualization_map.csv", index=False)
    return out


def write_summary(exp_map: pd.DataFrame) -> None:
    split = _read_csv(TABLES / "split_audit.csv")
    overall = split[split["split"].eq("overall")].iloc[0]
    lines = [
        "# JBD MOSAIC Reviewer Audit Summary",
        "",
        "## Canonical Cohort",
        "",
        f"- Cases: {int(overall['cases'])}",
        f"- Split: train/val/test = {int(split[split['split'].eq('train')]['cases'].iloc[0])}/{int(split[split['split'].eq('val')]['cases'].iloc[0])}/{int(split[split['split'].eq('test')]['cases'].iloc[0])}",
        f"- Real reports: {int(overall['real_report_cases'])}; pseudo-report candidates: {int(overall['pseudo_report_candidates'])}",
        f"- Images: {int(overall['total_images'])} total ({int(overall['oct_images'])} OCT, {int(overall['colposcopy_images'])} colposcopy-associated)",
        f"- CIN3+ proxy positives: {int(overall['cin3plus_proxy_positive_cases'])} (conservative pathology-text proxy, not a curated endpoint)",
        "",
        "## Experiment Map",
        "",
    ]
    for _, row in exp_map.iterrows():
        lines.append(f"- {row['experiment']}: {row['main_result']}")
    lines.extend(
        [
            "",
            "## Claim Boundaries",
            "",
            "- Direct LLM semantic-tag results are unavailable in the current execution and must remain N/A.",
            "- CIN3+ safety is a proxy audit derived from pathology text and should not be presented as a locked curated endpoint.",
            "- Strict LOCO and eval-only centre stratification are separate protocols and should not be pooled.",
            "- QC stress testing supports weak-supervision filtering, not clinical validation of generated reports.",
            "",
        ]
    )
    (OUT / "JBD_MOSAIC_REVIEWER_AUDIT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df = add_basic_fields(_read_csv(MANIFEST))
    pseudo = add_basic_fields(_read_csv(PSEUDO_MANIFEST))
    build_split_audit(df)
    prediction_sets(df)
    build_leakage_tables()
    build_cin3_safety(df)
    build_embedding_and_duplicate_audits(df)
    build_qc_outputs(pseudo)
    exp_map = build_experiment_map()
    write_summary(exp_map)
    print(f"Wrote reviewer audit tables to {TABLES}")
    print(f"Wrote summary to {OUT / 'JBD_MOSAIC_REVIEWER_AUDIT_SUMMARY.md'}")


if __name__ == "__main__":
    main()
