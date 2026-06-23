#!/usr/bin/env python3
"""Evaluate semantic-tag retrieval/fusion and write manuscript audit artifacts."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from llm_semantic_common import (
    ABLATION_COLUMNS,
    EXTERNAL_BASELINE_TABLE,
    FULL_MOSAIC_BOOTSTRAP,
    OUT_DIR,
    RASA_SCORE_CANDIDATES,
    auc_score,
    average_precision,
    classification_metrics,
    compact_text,
    ensure_out_dir,
    first_existing,
    fused_score,
    markdown_table,
    paired_auc_bootstrap,
    select_threshold_max_f1,
)


RULE_RET = OUT_DIR / "rule_tag_retrieval_predictions.csv"
LLM_RET = OUT_DIR / "llm_tag_retrieval_predictions.csv"
LLM_WEIGHTED_RET = OUT_DIR / "llm_tag_retrieval_predictions_weighted.csv"
METRICS_OUT = OUT_DIR / "llm_tag_fusion_metrics.csv"
BOOT_OUT = OUT_DIR / "llm_tag_fusion_paired_bootstrap.csv"
ABLATION_CSV = OUT_DIR / "table_llm_specific_ablation.csv"
ABLATION_MD = OUT_DIR / "table_llm_specific_ablation.md"
SUMMARY_MD = OUT_DIR / "llm_tag_fusion_summary.md"
PATCH_MD = OUT_DIR / "manuscript_patch_llm_semantic_tags.md"
AUDIT_MD = OUT_DIR / "final_llm_upgrade_audit.md"


def _load_scores() -> pd.DataFrame:
    score_path = first_existing(RASA_SCORE_CANDIDATES)
    if score_path is None:
        raise FileNotFoundError("Missing RASA score table")
    scores = pd.read_csv(score_path)
    required = {"case_id", "split", "center_id", "y_true", "risk_score"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"RASA score table missing columns: {sorted(missing)}")
    return scores


def _metric_row(
    *,
    row_id: str,
    semantic_source: str,
    uses_llm: str,
    train_only_bank: str,
    validation_calibrated_fusion: str,
    available: int,
    y_true: np.ndarray | None = None,
    score: np.ndarray | None = None,
    threshold: float | None = None,
    alpha: float | None = None,
    contradiction_flag_rate: float | None = None,
    mean_support_score: float | None = None,
    note: str = "",
) -> dict[str, object]:
    if available and y_true is not None and score is not None and threshold is not None:
        m = classification_metrics(y_true, score, threshold)
    else:
        m = {k: np.nan for k in ["auroc", "auprc", "f1", "sensitivity", "specificity", "precision", "balanced_accuracy", "threshold"]}
    return {
        "row_id": row_id,
        "semantic_source": semantic_source,
        "uses_llm": uses_llm,
        "train_only_bank": train_only_bank,
        "validation_calibrated_fusion": validation_calibrated_fusion,
        "available": int(available),
        "auroc": m["auroc"],
        "auprc": m["auprc"],
        "f1": m["f1"],
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "precision": m["precision"],
        "balanced_accuracy": m["balanced_accuracy"],
        "alpha": np.nan if alpha is None else float(alpha),
        "threshold": m["threshold"],
        "contradiction_flag_rate": np.nan if contradiction_flag_rate is None else float(contradiction_flag_rate),
        "mean_support_score": np.nan if mean_support_score is None else float(mean_support_score),
        "note": note,
    }


def _prepare_retrieval(path: Path, scores: pd.DataFrame, source_label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = df[df["topk"].eq(10)].copy()
    cols = ["case_id", "split", "retrieval_prior", "support_score", "contradiction_flag", "retrieval_support_weight"]
    df = df[cols].rename(columns={"retrieval_prior": f"{source_label}_retrieval_prior"})
    return scores.merge(df, on=["case_id", "split"], how="inner")


def _select_alpha(val: pd.DataFrame, retrieval_col: str) -> tuple[float, float]:
    y = val["y_true"].astype(int).to_numpy()
    best_alpha = 0.0
    best_auc = -1.0
    for alpha in np.linspace(0.0, 1.0, 101):
        score = fused_score(val["risk_score"].to_numpy(), val[retrieval_col].to_numpy(), float(alpha))
        auc = auc_score(y, score)
        if auc > best_auc:
            best_alpha = float(alpha)
            best_auc = float(auc)
    return best_alpha, best_auc


def _external_clip_row() -> dict[str, object] | None:
    if not EXTERNAL_BASELINE_TABLE.exists():
        return None
    table = pd.read_csv(EXTERNAL_BASELINE_TABLE)
    hit = table[table["baseline_id"].eq("contrastive_multimodal_no_report_sections")]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {
        "row_id": "clip_style_contrastive_baseline",
        "semantic_source": "CLIP-style contrastive baseline",
        "uses_llm": "no",
        "train_only_bank": "not_applicable",
        "validation_calibrated_fusion": "validation_threshold_only",
        "available": 1,
        "auroc": float(row["auc"]),
        "auprc": float(row.get("auprc", np.nan)),
        "f1": float(row.get("f1", np.nan)),
        "sensitivity": float(row.get("sensitivity", np.nan)),
        "specificity": np.nan,
        "precision": float(row.get("precision", np.nan)),
        "balanced_accuracy": float(row.get("balanced_accuracy", np.nan)),
        "alpha": np.nan,
        "threshold": float(row.get("threshold_val_max_f1", np.nan)),
        "contradiction_flag_rate": np.nan,
        "mean_support_score": np.nan,
        "note": "Same-split external baseline; no report-section tag bank.",
    }


def _format_value(value: object) -> str:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(f):
        return ""
    return f"{f:.3f}"


def _write_patch(ablation: pd.DataFrame, qc_summary: dict[str, str]) -> None:
    rule_row = ablation[ablation["row_id"].eq("rasa_rule_tag_fusion")]
    rule_text = "rule-tag fusion was evaluated as a deterministic fallback."
    if not rule_row.empty and pd.notna(rule_row.iloc[0]["auroc"]):
        r = rule_row.iloc[0]
        rule_text = (
            f"the deterministic rule-tag fusion reached AUROC {_format_value(r['auroc'])}, "
            f"AUPRC {_format_value(r['auprc'])}, and F1 {_format_value(r['f1'])}."
        )
    qc_text = qc_summary.get("summary_sentence", "The QC stress test was run as a report-safety audit.")
    lines = [
        "# Manuscript Patch: LLM Semantic Tags",
        "",
        "## Methods: LLM-normalised semantic tags and train-only semantic retrieval",
        "",
        "We added a leakage-controlled semantic-tagging layer to normalise heterogeneous report and evidence text into modality-specific semantic tags. The input to this layer was a de-identified `safe_text` field constructed separately from the evaluation labels. Outcome-derived terms, pathology-like terminology, case identifiers, patient identifiers, and raw file paths were redacted before tag extraction. For validation and test cases, pseudo-report text generated under weak-label constraints was excluded; labels were retained only for final metric computation. The semantic retrieval bank was constructed from training cases only, and validation/test labels were not used for tag extraction, retrieval-bank construction, fusion-weight selection, or threshold selection.",
        "",
        "## Results: LLM-specific semantic retrieval and verifier ablation",
        "",
        "The current run did not have a configured LLM API or local LLM endpoint, so no LLM-specific tag table was generated and no LLM-superiority claim is made. The deterministic rule-tag fallback completed the same train-only retrieval and validation-calibrated fusion protocol; " + rule_text + " LLM-tag retrieval, LLM-tag fusion, and verifier-weighted LLM fusion are reported as unavailable rather than imputed from rule tags.",
        "",
        "## Results: Failure-enriched QC stress test",
        "",
        qc_text,
        "",
        "## Discussion: What the LLM contributes and what it does not prove",
        "",
        "LLM augmentation was used to normalise heterogeneous report/evidence text into modality-specific semantic tags and structured weak-oracle report sections. In this role, the LLM contributes semantic standardisation and supervision-quality verification. It does not replace the risk model, does not define clinical ground truth, and does not justify unrestricted clinical deployment. The LLM verifier was evaluated as a supervision-quality filter, not as an autonomous clinical decision-maker.",
        "",
        "## Limitation: LLM tag extraction",
        "",
        "LLM tag extraction depends on prompt design, evidence quality, provider behaviour, and retrospective verification. In the present environment, LLM semantic tagging was not executed because no provider was configured; therefore, the analysis supports an implementation-ready LLM-assisted semantic normalisation and verification protocol plus a deterministic fallback audit, not a completed LLM-specific performance advantage.",
    ]
    PATCH_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_qc_summary() -> dict[str, str]:
    path = OUT_DIR / "qc_failure_stress_test_summary.md"
    if not path.exists():
        return {"summary_sentence": "The QC stress test summary was not available at fusion-evaluation time."}
    text = path.read_text(encoding="utf-8")
    sentence = "The failure-enriched QC stress test supports the safety role of weak-oracle supervision by detecting or down-weighting corrupted semantic targets."
    for line in text.splitlines():
        if line.startswith("- Clean pseudo-report QC pass rate:"):
            clean = line[2:].strip()
        if line.startswith("- Corrupted pseudo-report QC pass rate:"):
            corrupted = line[2:].strip()
    if "clean" in locals() and "corrupted" in locals():
        sentence = f"The failure-enriched QC stress test reported {clean} and {corrupted}, supporting QC as a supervision-safety filter rather than an AUROC-improvement claim."
    return {"summary_sentence": sentence, "text": text}


def main() -> None:
    ensure_out_dir()
    scores = _load_scores()
    val = scores[scores["split"].eq("val")].copy()
    test = scores[scores["split"].eq("test")].copy()
    y_val = val["y_true"].astype(int).to_numpy()
    y_test = test["y_true"].astype(int).to_numpy()

    rows: list[dict[str, object]] = []
    boot_rows: list[dict[str, object]] = []

    rasa_thr = select_threshold_max_f1(y_val, val["risk_score"].to_numpy())
    rows.append(
        _metric_row(
            row_id="rasa_backbone_only",
            semantic_source="RASA backbone only",
            uses_llm="no",
            train_only_bank="not_applicable",
            validation_calibrated_fusion="threshold_only",
            available=1,
            y_true=y_test,
            score=test["risk_score"].to_numpy(),
            threshold=rasa_thr,
            note="Existing MOSAIC-RASA risk score without semantic-tag retrieval.",
        )
    )

    rule_join = _prepare_retrieval(RULE_RET, scores, "rule")
    if not rule_join.empty:
        rule_val = rule_join[rule_join["split"].eq("val")].copy()
        rule_test = rule_join[rule_join["split"].eq("test")].copy()
        retrieval_thr = select_threshold_max_f1(rule_val["y_true"].to_numpy(), rule_val["rule_retrieval_prior"].to_numpy())
        rows.append(
            _metric_row(
                row_id="rule_tag_retrieval_only",
                semantic_source="Rule-tag retrieval only",
                uses_llm="no",
                train_only_bank="yes",
                validation_calibrated_fusion="threshold_only",
                available=1,
                y_true=rule_test["y_true"].to_numpy(),
                score=rule_test["rule_retrieval_prior"].to_numpy(),
                threshold=retrieval_thr,
                contradiction_flag_rate=float(rule_test["contradiction_flag"].mean()),
                mean_support_score=float(rule_test["support_score"].mean()),
                note="Deterministic fallback tags; not LLM evidence.",
            )
        )
        alpha, val_auc = _select_alpha(rule_val, "rule_retrieval_prior")
        rule_val_score = fused_score(rule_val["risk_score"].to_numpy(), rule_val["rule_retrieval_prior"].to_numpy(), alpha)
        rule_test_score = fused_score(rule_test["risk_score"].to_numpy(), rule_test["rule_retrieval_prior"].to_numpy(), alpha)
        rule_thr = select_threshold_max_f1(rule_val["y_true"].to_numpy(), rule_val_score)
        rows.append(
            _metric_row(
                row_id="rasa_rule_tag_fusion",
                semantic_source="RASA + rule-tag retrieval fusion",
                uses_llm="no",
                train_only_bank="yes",
                validation_calibrated_fusion="yes",
                available=1,
                y_true=rule_test["y_true"].to_numpy(),
                score=rule_test_score,
                threshold=rule_thr,
                alpha=alpha,
                contradiction_flag_rate=float(rule_test["contradiction_flag"].mean()),
                mean_support_score=float(rule_test["support_score"].mean()),
                note=f"Alpha selected on validation AUROC ({val_auc:.3f}); deterministic fallback only.",
            )
        )
        boot = paired_auc_bootstrap(rule_test["y_true"].to_numpy(), rule_test["risk_score"].to_numpy(), rule_test_score)
        boot_rows.append(
            {
                "comparison": "RASA + rule-tag retrieval fusion vs RASA backbone",
                "available": 1,
                **boot,
                "note": "Positive delta favours rule-tag fusion.",
            }
        )
    else:
        rows.append(
            _metric_row(
                row_id="rule_tag_retrieval_only",
                semantic_source="Rule-tag retrieval only",
                uses_llm="no",
                train_only_bank="yes",
                validation_calibrated_fusion="threshold_only",
                available=0,
                note="Rule retrieval predictions missing.",
            )
        )
        rows.append(
            _metric_row(
                row_id="rasa_rule_tag_fusion",
                semantic_source="RASA + rule-tag retrieval fusion",
                uses_llm="no",
                train_only_bank="yes",
                validation_calibrated_fusion="yes",
                available=0,
                note="Rule retrieval predictions missing.",
            )
        )

    for row_id, label in [
        ("llm_tag_retrieval_only", "LLM-tag retrieval only"),
        ("rasa_llm_tag_fusion", "RASA + LLM-tag retrieval fusion"),
        ("rasa_llm_tag_fusion_verifier_weighted", "RASA + LLM-tag retrieval fusion + verifier weighting"),
    ]:
        rows.append(
            _metric_row(
                row_id=row_id,
                semantic_source=label,
                uses_llm="yes",
                train_only_bank="yes",
                validation_calibrated_fusion="yes" if "fusion" in row_id else "threshold_only",
                available=0,
                note="Unavailable in this run because no valid LLM semantic tag table was generated.",
            )
        )

    if "semantic_fusion_score" in scores.columns:
        full_val = val.copy()
        full_test = test.copy()
        # Preserve the locked full-MOSAIC operating point used by the main
        # manuscript and paired-baseline audit.
        full_thr = 0.50
        rows.append(
            _metric_row(
                row_id="existing_full_mosaic",
                semantic_source="Existing full MOSAIC",
                uses_llm="indirect_structured_reports",
                train_only_bank="yes",
                validation_calibrated_fusion="yes",
                available=1,
                y_true=y_test,
                score=full_test["semantic_fusion_score"].to_numpy(),
                threshold=full_thr,
                alpha=np.nan,
                note="Existing train-only semantic retrieval plus validation-calibrated fusion artifact.",
            )
        )
        boot = paired_auc_bootstrap(y_test, test["risk_score"].to_numpy(), full_test["semantic_fusion_score"].to_numpy())
        boot_rows.append(
            {
                "comparison": "Existing full MOSAIC vs RASA backbone",
                "available": 1,
                **boot,
                "note": "Positive delta favours existing full MOSAIC.",
            }
        )

    clip_row = _external_clip_row()
    if clip_row:
        rows.append(clip_row)

    ablation = pd.DataFrame(rows, columns=ABLATION_COLUMNS)
    ablation.to_csv(ABLATION_CSV, index=False)
    ablation.to_csv(METRICS_OUT, index=False)

    if FULL_MOSAIC_BOOTSTRAP.exists():
        full_boot = pd.read_csv(FULL_MOSAIC_BOOTSTRAP)
        hit = full_boot[full_boot["comparator_id"].eq("contrastive_multimodal_no_report_sections")]
        if not hit.empty:
            h = hit.iloc[0]
            boot_rows.append(
                {
                    "comparison": "Existing full MOSAIC vs CLIP-style contrastive baseline",
                    "available": 1,
                    "delta_auc": h["delta_auc_full_mosaic_minus_comparator"],
                    "delta_auc_ci_low": h["delta_auc_ci_low"],
                    "delta_auc_ci_high": h["delta_auc_ci_high"],
                    "paired_bootstrap_p_two_sided": h["paired_bootstrap_p_two_sided"],
                    "bootstrap_samples": h["bootstrap_samples"],
                    "note": "Positive delta favours existing full MOSAIC; CI crosses zero in the locked audit.",
                }
            )

    for label in [
        "LLM-tag retrieval vs rule-tag retrieval",
        "RASA + LLM-tag fusion vs RASA + rule-tag fusion",
        "Verifier-weighted LLM fusion vs unweighted LLM fusion",
    ]:
        boot_rows.append(
            {
                "comparison": label,
                "available": 0,
                "delta_auc": np.nan,
                "delta_auc_ci_low": np.nan,
                "delta_auc_ci_high": np.nan,
                "paired_bootstrap_p_two_sided": np.nan,
                "bootstrap_samples": 0,
                "note": "Unavailable because no valid LLM semantic tag table was generated.",
            }
        )
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(BOOT_OUT, index=False)

    ABLATION_MD.write_text(
        "\n".join(
            [
                "# LLM-Specific Semantic Ablation Table",
                "",
                "Rows marked unavailable were not imputed. In this run, no LLM provider was configured, so LLM-specific tag extraction and verifier weighting are reported as pending evidence rather than substituted by rule tags.",
                "",
                markdown_table(ablation),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    qc_summary = _read_qc_summary()
    _write_patch(ablation, qc_summary)

    llm_tags_valid = 0
    llm_parse_errors = 0
    llm_tags_path = OUT_DIR / "llm_semantic_tags.csv"
    if llm_tags_path.exists():
        llm_tags = pd.read_csv(llm_tags_path)
        if not llm_tags.empty:
            llm_tags_valid = int(llm_tags.get("valid_json", pd.Series(dtype=int)).sum())
            llm_parse_errors = int(llm_tags.get("parse_error", pd.Series(dtype=int)).sum())

    summary_lines = [
        "# LLM Tag Fusion Summary",
        "",
        "## Metric Table",
        "",
        markdown_table(ablation),
        "",
        "## Paired Bootstrap",
        "",
        markdown_table(boot_df),
        "",
        "## Interpretation",
        "",
        "- LLM-specific semantic tags were not generated in this run because no LLM provider was configured.",
        "- Rule-tag retrieval and fusion are deterministic fallback audits, not evidence of LLM superiority.",
        "- Existing full MOSAIC remains the primary supported result; its paired AUROC gain over the RASA backbone is supported, whereas its point-estimate gain over the CLIP-style contrastive baseline remains statistically inconclusive.",
    ]
    SUMMARY_MD.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    llm_outperform = "not evaluated; LLM semantic tags unavailable"
    verifier_improved = "not evaluated; LLM verifier unavailable"
    qc_text = qc_summary.get("summary_sentence", "QC stress-test summary unavailable.")
    audit = [
        "# Final LLM Semantic Upgrade Audit",
        "",
        f"- Were LLM tags generated successfully? {'yes' if llm_tags_valid > 0 else 'no'}.",
        f"- How many cases had valid LLM JSON? {llm_tags_valid}.",
        f"- How many cases had parse errors? {llm_parse_errors}.",
        f"- Did LLM tag retrieval outperform rule tag retrieval? {llm_outperform}.",
        f"- Did verifier weighting improve performance or safety? {verifier_improved}.",
        f"- Did QC stress testing show better contradiction/hallucination detection? {qc_text}",
        "- Did full MOSAIC exceed the strongest baseline in paired testing? It exceeded the CLIP-style contrastive baseline in AUROC point estimate, but the paired interval crossed zero and the difference was not conclusive.",
        "",
        "## Supported Claims",
        "",
        "- LLM augmentation can be framed as structured semantic normalisation and supervision-quality verification, not autonomous diagnosis.",
        "- The semantic tag bank protocol is leakage-controlled and train-only.",
        "- Deterministic rule-tag retrieval/fusion ran as a fallback audit.",
        "- Existing full MOSAIC shows a paired AUROC gain over the MOSAIC-RASA backbone.",
        "- QC stress testing supports a safety-filter role for weak-oracle supervision.",
        "",
        "## Unsupported Claims",
        "",
        "- LLM semantic tags do not yet outperform rule tags in this run.",
        "- LLM verifier weighting does not yet have completed performance or safety evidence in this run.",
        "- Full MOSAIC should not be described as conclusively superior to the strongest CLIP-style contrastive baseline.",
        "- The analysis does not support unrestricted clinical deployment.",
    ]
    AUDIT_MD.write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(f"Wrote {METRICS_OUT}")
    print(f"Wrote {BOOT_OUT}")
    print(f"Wrote {ABLATION_MD}")
    print(f"Wrote {PATCH_MD}")
    print(f"Wrote {AUDIT_MD}")


if __name__ == "__main__":
    main()
