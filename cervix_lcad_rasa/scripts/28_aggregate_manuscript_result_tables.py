#!/usr/bin/env python3
"""Aggregate existing experiment CSVs into manuscript-ready tables + index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TABLES = ROOT / "outputs/publishable/tables"
OUT = TABLES / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "real_report_only_decoder": "Real-report only",
    "pseudo_augmented_lcad": "Pseudo-augmented (LCAD)",
    "simple_concat_fusion": "Simple concat fusion",
    "report_generation_without_section_alignment": "LCAD w/o section alignment",
    "full_lcad_rasa": "Full LCAD-RASA",
    "best_lcad_rasa": "Best LCAD-RASA (λ_align=0.2)",
    "multimodal_fusion_without_report_anchor": "Fusion w/o report anchor",
    "instruction_only_report_generation": "Instruction-only report gen.",
    "image_only_report_generation": "Image-only report gen.",
}


def _read(name: str) -> pd.DataFrame:
    p = TABLES / name
    if not p.is_file():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def _round_df(df: pd.DataFrame, cols: list[str], ndigits: int = 3) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(ndigits)
    return out


def build_t1_dataset() -> pd.DataFrame:
    centre = _read("table_centerwise_image_count_audit.csv")
    stats = _read("table_final_dataset_statistics_for_manuscript.csv")
    stat_map = dict(zip(stats["metric"], stats["value"]))
    centre = centre.rename(
        columns={
            "center": "Centre",
            "cases": "Cases",
            "oct_images": "OCT images",
            "colposcopy_images": "Colposcopy images",
            "total_images": "Total images",
            "real_report_cases": "Real reports",
            "pseudo_candidates": "Pseudo-report candidates",
            "missing_oct_rate": "Missing OCT rate",
        }
    )
    summary = pd.DataFrame(
        [
            {
                "Metric": "Total cases",
                "Value": int(stat_map.get("total_cases", 0)),
            },
            {
                "Metric": "Centres",
                "Value": int(stat_map.get("total_centers", 0)),
            },
            {
                "Metric": "Evaluable images (pipeline)",
                "Value": int(stat_map.get("total_images_evaluable", 0)),
            },
            {
                "Metric": "OCT images",
                "Value": int(stat_map.get("total_oct_images", 0)),
            },
            {
                "Metric": "Colposcopy images",
                "Value": int(stat_map.get("total_colposcopy_images", 0)),
            },
            {
                "Metric": "Real reports",
                "Value": int(stat_map.get("real_report_cases", 0)),
            },
            {
                "Metric": "Pseudo-report candidates",
                "Value": int(stat_map.get("pseudo_report_candidates", 0)),
            },
            {
                "Metric": "Test cases",
                "Value": int(stat_map.get("analytic_test_cases", 0)),
            },
        ]
    )
    summary.to_csv(OUT / "T1a_cohort_summary.csv", index=False)
    centre.to_csv(OUT / "T1b_centre_scale_and_supervision.csv", index=False)
    return summary, centre


def build_t2_main_comparison() -> pd.DataFrame:
    baseline = _read("table_baseline_comparison.csv")
    strat = _read("table_reference_stratified_evaluation.csv")
    thresh = _read("table_threshold_tuned_test_metrics.csv")
    thresh = thresh[thresh["threshold_type"] == "max_f1"].copy()

    core_ids = [
        "real_report_only_decoder",
        "pseudo_augmented_lcad",
        "simple_concat_fusion",
        "report_generation_without_section_alignment",
        "full_lcad_rasa",
        "best_lcad_rasa",
    ]
    baseline = baseline[baseline["experiment_id"].isin(core_ids)].copy()
    strat_all = strat[strat["subset"] == "all"][["experiment_id", "auc", "f1", "label_consistency", "n_subset"]].rename(
        columns={"auc": "auc_all", "f1": "f1_strat_all", "label_consistency": "label_consistency_strat", "n_subset": "n_test"}
    )
    strat_ref = strat[strat["subset"] == "with_reference"][
        ["experiment_id", "auc", "f1", "label_consistency", "n_subset"]
    ].rename(
        columns={
            "auc": "auc_with_reference",
            "f1": "f1_with_reference",
            "label_consistency": "label_consistency_with_reference",
            "n_subset": "n_with_reference",
        }
    )
    strat_wo = strat[strat["subset"] == "without_reference"][
        ["experiment_id", "auc", "f1", "label_consistency", "n_subset"]
    ].rename(
        columns={
            "auc": "auc_without_reference",
            "f1": "f1_without_reference",
            "label_consistency": "label_consistency_without_reference",
            "n_subset": "n_without_reference",
        }
    )
    thresh = thresh[thresh["experiment_id"].isin(core_ids)][
        ["experiment_id", "selected_threshold", "auc", "f1", "sensitivity", "specificity"]
    ].rename(
        columns={
            "selected_threshold": "val_selected_threshold",
            "auc": "auc_at_val_threshold",
            "f1": "f1_at_val_threshold",
            "sensitivity": "sens_at_val_threshold",
            "specificity": "spec_at_val_threshold",
        }
    )

    df = baseline[
        [
            "experiment_id",
            "auc",
            "f1",
            "sensitivity",
            "specificity",
            "label_consistency",
            "section_completeness",
            "test_cases",
        ]
    ].rename(columns={"auc": "auc_baseline", "f1": "f1_default_0p5", "label_consistency": "label_consistency_baseline"})
    df = df.merge(strat_all, on="experiment_id", how="left")
    df["auc_all"] = df["auc_all"].fillna(df["auc_baseline"])
    df["label_consistency"] = df["label_consistency_strat"].fillna(df["label_consistency_baseline"])
    df = df.merge(strat_ref, on="experiment_id", how="left")
    df = df.merge(strat_wo, on="experiment_id", how="left")
    df = df.merge(thresh, on="experiment_id", how="left")
    df.insert(0, "Model", df["experiment_id"].map(lambda x: MODEL_LABELS.get(x, x)))
    df = df.sort_values("auc_all", ascending=False)
    num_cols = [c for c in df.columns if c not in ("Model", "experiment_id")]
    df = _round_df(df, num_cols)
    df.to_csv(OUT / "T2_main_model_comparison.csv", index=False)
    return df


def build_supplementary_tables() -> dict[str, Path]:
    paths: dict[str, Path] = {}

    rasa = _read("table_rasa_loss_weight_sweep.csv")[
        ["experiment_id", "lambda_align", "auc", "f1", "label_consistency", "section_completeness"]
    ]
    rasa = _round_df(rasa, ["lambda_align", "auc", "f1", "label_consistency", "section_completeness"])
    p = OUT / "S1_rasa_lambda_align_sweep.csv"
    rasa.to_csv(p, index=False)
    paths["S1"] = p

    loco = _read("table_loco_strict_main_results.csv")
    loco = loco[
        [
            "held_out_center",
            "center_label",
            "model",
            "test_cases",
            "auc",
            "f1",
            "sensitivity",
            "specificity",
            "label_consistency",
        ]
    ].sort_values(["held_out_center", "model"])
    loco = _round_df(loco, ["auc", "f1", "sensitivity", "specificity", "label_consistency"])
    p = OUT / "S2_loco_strict_retrain.csv"
    loco.to_csv(p, index=False)
    paths["S2"] = p

    loco_eval = _read("table_loco_main_results.csv")
    p = OUT / "S2b_loco_eval_only_global_ckpt.csv"
    loco_eval.to_csv(p, index=False)
    paths["S2b"] = p

    mod = _read("table_modality_ablation.csv")[
        ["experiment_id", "auc", "f1", "label_consistency", "sensitivity", "specificity"]
    ].sort_values("auc", ascending=False)
    mod = _round_df(mod, ["auc", "f1", "label_consistency", "sensitivity", "specificity"])
    p = OUT / "S3_modality_ablation.csv"
    mod.to_csv(p, index=False)
    paths["S3"] = p

    qc = _read("table_lcad_qc_ablation.csv")[
        ["experiment_id", "auc", "f1", "label_consistency", "sensitivity", "specificity"]
    ].sort_values("auc", ascending=False)
    qc = _round_df(qc, ["auc", "f1", "label_consistency", "sensitivity", "specificity"])
    p = OUT / "S4_lcad_qc_ablation.csv"
    qc.to_csv(p, index=False)
    paths["S4"] = p

    rasa_comp = _read("table_rasa_component_ablation.csv")[
        ["experiment_id", "auc", "f1", "label_consistency", "sensitivity", "specificity"]
    ].sort_values("auc", ascending=False)
    rasa_comp = _round_df(rasa_comp, ["auc", "f1", "label_consistency", "sensitivity", "specificity"])
    p = OUT / "S5_rasa_component_ablation.csv"
    rasa_comp.to_csv(p, index=False)
    paths["S5"] = p

    pert = _read("modality_perturbation_text_decoding_summary.csv")
    p = OUT / "S6_modality_perturbation_text_decoding.csv"
    pert.to_csv(p, index=False)
    paths["S6"] = p

    pert_ext = _read("table_modality_perturbation_extended.csv")
    p = OUT / "S6b_modality_perturbation_extended.csv"
    pert_ext.to_csv(p, index=False)
    paths["S6b"] = p

    seed = _read("table_multiseed_stability.csv")
    p = OUT / "S7_multiseed_stability.csv"
    seed.to_csv(p, index=False)
    paths["S7"] = p

    llm_cmp = _read("llm_vs_mock_pseudo_qc_comparison.csv")
    llm_qc = _read("llm_pseudo_report_quality_summary.csv")
    p = OUT / "S8_llm_pseudo_report_qc.csv"
    pd.concat(
        [
            llm_cmp.assign(table="mock_vs_llm_qc"),
            llm_qc.assign(table="llm_quality_summary"),
        ],
        ignore_index=True,
    ).to_csv(p, index=False)
    paths["S8"] = p

    safety = _read("table_report_safety_metrics.csv")
    p = OUT / "S9_report_safety_metrics.csv"
    safety.to_csv(p, index=False)
    paths["S9"] = p

    mask = _read("masking_validation_publishable_metrics.csv")
    p = OUT / "S10_masking_validation.csv"
    mask.to_csv(p, index=False)
    paths["S10"] = p

    scale = _read("table_scalability_pipeline_statistics.csv")
    runtime = _read("table_runtime_efficiency.csv")
    p = OUT / "S11_scalability_and_runtime.csv"
    pd.concat(
        [
            scale.assign(section="pipeline_scale"),
            runtime.assign(section="runtime"),
        ],
        ignore_index=True,
    ).to_csv(p, index=False)
    paths["S11"] = p

    return paths


def write_index(t2: pd.DataFrame, sup_paths: dict[str, Path]) -> Path:
    idx = OUT / "MASTER_RESULTS_TABLE_INDEX.md"
    lines = [
        "# LCAD-RASA — Master Results Table Index",
        "",
        "Generated by `scripts/28_aggregate_manuscript_result_tables.py`.",
        "Source tables remain under `outputs/publishable/tables/`; curated copies are in `manuscript/`.",
        "",
        "## Main text",
        "",
        "| Table | File | Source experiment |",
        "|-------|------|-------------------|",
        f"| **Table 1a** Cohort summary | [`T1a_cohort_summary.csv`](T1a_cohort_summary.csv) | Next-Stage A / dataset audit |",
        f"| **Table 1b** Centre scale & supervision | [`T1b_centre_scale_and_supervision.csv`](T1b_centre_scale_and_supervision.csv) | `table_centerwise_image_count_audit.csv` |",
        f"| **Table 2** Main model comparison | [`T2_main_model_comparison.csv`](T2_main_model_comparison.csv) | Baselines + stratified eval + val threshold |",
        "",
        "### Table 2 preview (test n=288)",
        "",
    ]
    preview_cols = [c for c in ["Model", "auc_all", "f1_default_0p5", "f1_at_val_threshold", "label_consistency", "val_selected_threshold"] if c in t2.columns]
    t2_preview = t2[preview_cols].copy()
    try:
        lines.append(t2_preview.to_markdown(index=False))
    except ImportError:
        lines.append(t2_preview.to_string(index=False))
    lines.extend(
        [
            "",
            "## Supplementary",
            "",
            "| Table | File | Task |",
            "|-------|------|------|",
        ]
    )
    sup_meta = {
        "S1": "RASA λ_align sweep (Next-Stage C)",
        "S2": "Strict LOCO — retrain per fold (Next-Stage E)",
        "S2b": "LOCO eval-only — global checkpoint (Supplementary 3)",
        "S3": "Modality ablation (Supplementary 4)",
        "S4": "LCAD QC weight ablation (Supplementary 2)",
        "S5": "RASA component ablation (Supplementary 5)",
        "S6": "Modality perturbation — text decoding (Prompt I)",
        "S6b": "Modality perturbation extended + seeds (Supplementary 6)",
        "S7": "Multi-seed stability (Next-Stage F)",
        "S8": "LLM vs mock pseudo-report QC (Prompt E/F)",
        "S9": "Report safety metrics (Supplementary 9)",
        "S10": "Masking validation — Enshi (Prompt J)",
        "S11": "Scalability & runtime (Supplementary 7)",
    }
    for key, p in sup_paths.items():
        task = sup_meta.get(key, "")
        lines.append(f"| **{key}** | [`{p.name}`]({p.name}) | {task} |")
    lines.extend(
        [
            "",
            "## Raw / full exports (unchanged)",
            "",
            "| Purpose | File |",
            "|---------|------|",
            "| All experiments long format | [`../table_supplementary_all_experiments.csv`](../table_supplementary_all_experiments.csv) |",
            "| Manuscript-wide metrics | [`../table_main_results_for_manuscript.csv`](../table_main_results_for_manuscript.csv) |",
            "| Next-Stage summary | [`../NEXT_STAGE_FINAL_SUMMARY.md`](../NEXT_STAGE_FINAL_SUMMARY.md) |",
            "| Submission v1 bundle | [`../../publishable_jbd_submission_v1/RESULT_FILE_INDEX.md`](../../publishable_jbd_submission_v1/RESULT_FILE_INDEX.md) |",
            "",
            "## Regenerate",
            "",
            "```bash",
            "cd cervix_lcad_rasa",
            "python scripts/28_aggregate_manuscript_result_tables.py",
            "```",
            "",
        ]
    )
    idx.write_text("\n".join(lines), encoding="utf-8")
    return idx


def main():
    global TABLES, OUT
    p = argparse.ArgumentParser()
    p.add_argument("--tables-dir", default="outputs/publishable/tables")
    args = p.parse_args()
    tables_dir = Path(args.tables_dir)
    if not tables_dir.is_absolute():
        tables_dir = ROOT / tables_dir
    TABLES = tables_dir
    OUT = tables_dir / "manuscript"
    OUT.mkdir(parents=True, exist_ok=True)

    build_t1_dataset()
    t2 = build_t2_main_comparison()
    sup_paths = build_supplementary_tables()
    idx = write_index(t2, sup_paths)
    print(f"Wrote manuscript tables to {OUT}")
    print(f"Index: {idx}")


if __name__ == "__main__":
    main()
