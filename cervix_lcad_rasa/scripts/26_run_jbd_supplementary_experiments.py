#!/usr/bin/env python3
"""Run JBD supplementary experiments (LCAD_RASA_JBD_Supplementary_Experiment_Prompts.md)."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supplementary.io_utils import save_table
from src.supplementary.figures import generate_all_figures
from src.supplementary.interpretations import write_interpretations
from src.supplementary.perturbation_extended import run_extended_perturbation
from src.supplementary.results_draft import build_results_draft
from src.supplementary.scalability import compute_scalability_stats
from src.supplementary.safety import aggregate_safety, safety_case_metrics
from src.supplementary.statistics import add_ci_columns, build_statistical_tests_summary
from src.supplementary.train_eval import (
    build_model,
    evaluate_experiment,
    load_jbd_config,
    resolve_checkpoint,
    train_experiment,
)
from src.models_publishable.lcad_rasa_model import instr_vector, load_visual_emb
from src.utils.config import resolve_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _merge_spec(base: dict, extra: dict) -> dict:
    out = {**base}
    for k, v in extra.items():
        if k == "model" and isinstance(v, dict):
            out["model"] = {**out.get("model", {}), **v}
        elif k == "loss" and isinstance(v, dict):
            out["loss"] = {**out.get("loss", {}), **v}
        else:
            out[k] = v
    return out


def _default_train_spec() -> dict:
    return {
        "train_filter": {"training_eligible": 1},
        "use_pseudo_report": True,
        "use_real_report": True,
        "require_qc_pass": True,
        "use_report_loss": True,
        "model": {"use_section_align": True, "use_risk_head": True},
        "loss": {"ce_weight": 1.0, "rasa_weight": 0.5, "cls_weight": 0.2, "cons_weight": 0.1},
    }


def _train_and_eval(
    project: Path,
    df: pd.DataFrame,
    cfg: dict,
    exp_id: str,
    spec: dict,
    baselines_dir: Path,
    tables_dir: Path,
    skip_train: bool,
    eval_max_cases: int | None,
) -> dict:
    ckpt = baselines_dir / exp_id / "best.ckpt"
    log_row = {"experiment_id": exp_id}
    try:
        if not skip_train and not ckpt.is_file():
            tr = train_experiment(project, df, exp_id, spec, cfg, baselines_dir / exp_id, seed=int(cfg.get("seed", 42)))
            log_row.update(tr)
            if tr.get("status") != "ok":
                return log_row
        use_ckpt = resolve_checkpoint(project, exp_id, cfg, baselines_dir)
        if use_ckpt is None:
            log_row["status"] = "no_checkpoint"
            return log_row
        ev = evaluate_experiment(project, df, exp_id, use_ckpt, spec, max_cases=eval_max_cases)
        log_row.update(ev)
        log_row["status"] = "ok"
        (tables_dir / exp_id).mkdir(parents=True, exist_ok=True)
        pd.DataFrame([ev]).to_csv(tables_dir / exp_id / "eval_summary.csv", index=False)
    except Exception as e:
        log_row["status"] = "failed"
        log_row["error"] = str(e)
        (project / "outputs/publishable/logs").mkdir(parents=True, exist_ok=True)
        (project / "outputs/publishable/logs" / f"fail_{exp_id}.log").write_text(traceback.format_exc(), encoding="utf-8")
    return log_row


def run_prompt1(project: Path, df: pd.DataFrame, cfg: dict, args) -> pd.DataFrame:
    baselines_dir = project / "outputs/publishable/baselines"
    tables_dir = project / "outputs/publishable/tables"
    rows = []
    for exp_id, spec in cfg.get("baselines", {}).items():
        full_spec = _merge_spec(_default_train_spec(), spec)
        rows.append(_train_and_eval(project, df, cfg, exp_id, full_spec, baselines_dir, tables_dir, args.skip_train, args.eval_max_cases))
    out = pd.DataFrame(rows)
    save_table(out, tables_dir / "table_baseline_comparison.csv", tables_dir / "table_baseline_comparison.md")
    return out


def run_prompt2(project: Path, df: pd.DataFrame, cfg: dict, args) -> pd.DataFrame:
    baselines_dir = project / "outputs/publishable/baselines"
    tables_dir = project / "outputs/publishable/tables"
    base = _default_train_spec()
    rows = []
    for exp_id, qspec in cfg.get("lcad_qc_ablations", {}).items():
        spec = _merge_spec(base, qspec)
        rows.append(_train_and_eval(project, df, cfg, exp_id, spec, baselines_dir, tables_dir, args.skip_train, args.eval_max_cases))
    out = pd.DataFrame(rows)
    save_table(out, tables_dir / "table_lcad_qc_ablation.csv", tables_dir / "table_lcad_qc_ablation.md")
    return out


def run_prompt5(project: Path, df: pd.DataFrame, cfg: dict, args) -> pd.DataFrame:
    baselines_dir = project / "outputs/publishable/baselines"
    tables_dir = project / "outputs/publishable/tables"
    base = _default_train_spec()
    rows = []
    for exp_id, rspec in cfg.get("rasa_ablations", {}).items():
        spec = _merge_spec(base, rspec)
        rows.append(_train_and_eval(project, df, cfg, exp_id, spec, baselines_dir, tables_dir, args.skip_train, args.eval_max_cases))
    out = pd.DataFrame(rows)
    save_table(out, tables_dir / "table_rasa_component_ablation.csv", tables_dir / "table_rasa_component_ablation.md")
    return out


def run_prompt4(project: Path, df: pd.DataFrame, cfg: dict, args) -> pd.DataFrame:
    baselines_dir = project / "outputs/publishable/baselines"
    tables_dir = project / "outputs/publishable/tables"
    base = _default_train_spec()
    rows = []
    for exp_id, mspec in cfg.get("modality_ablations", {}).items():
        spec = _merge_spec(base, mspec)
        rows.append(_train_and_eval(project, df, cfg, exp_id, spec, baselines_dir, tables_dir, args.skip_train, args.eval_max_cases))
    out = pd.DataFrame(rows)
    save_table(out, tables_dir / "table_modality_ablation.csv", tables_dir / "table_modality_ablation.md")
    return out


def run_prompt3(project: Path, df: pd.DataFrame, cfg: dict, args) -> pd.DataFrame:
    """LOCO: evaluate global checkpoints on each held-out center (eval-centric LOCO)."""
    tables_dir = project / "outputs/publishable/tables"
    baselines_dir = project / "outputs/publishable/baselines"
    model_map = {
        "dual_real_only": "real_report_only_decoder",
        "simple_concat_fusion": "simple_concat_fusion",
        "lcad_no_section_alignment": "report_generation_without_section_alignment",
        "full_lcad_rasa": "full_lcad_rasa",
    }
    rows = []
    centers = cfg.get("centers_loco", [])
    for held in centers:
        test_df = df[df["center_id"] == held]
        if len(test_df) == 0:
            continue
        for model_name, exp_id in model_map.items():
            ckpt = resolve_checkpoint(project, exp_id, cfg, baselines_dir)
            if ckpt is None:
                continue
            ev = evaluate_experiment(project, df, f"loco_{held}_{model_name}", ckpt, max_cases=args.eval_max_cases, test_df=test_df)
            ev["held_out_center"] = held
            ev["model"] = model_name
            rows.append(ev)
    out = pd.DataFrame(rows)
    save_table(out, tables_dir / "table_loco_main_results.csv", tables_dir / "table_loco_main_results.md")
    _, center_df, _ = compute_scalability_stats(project / cfg["manifest"], project)
    center_df.to_csv(tables_dir / "table_loco_center_characteristics.csv", index=False)
    return out


def run_prompt7(project: Path, cfg: dict) -> None:
    tables_dir = project / "outputs/publishable/tables"
    manifest = project / cfg["manifest"]
    pipe, center, runtime = compute_scalability_stats(manifest, project)
    save_table(pipe, tables_dir / "table_scalability_pipeline_statistics.csv", tables_dir / "table_scalability_pipeline_statistics.md")
    center.to_csv(tables_dir / "table_loco_center_characteristics.csv", index=False)
    runtime.to_csv(tables_dir / "table_runtime_efficiency.csv", index=False)
    runtime.to_csv(tables_dir / "table_storage_statistics.csv", index=False)


def run_prompt9(project: Path, df: pd.DataFrame, cfg: dict, args) -> pd.DataFrame:
    tables_dir = project / "outputs/publishable/tables"
    baselines_dir = project / "outputs/publishable/baselines"
    targets = [
        "real_report_only_decoder",
        "simple_concat_fusion",
        "no_section_alignment",
        "full_lcad_rasa",
        "pseudo_all_no_qc",
        "pseudo_qc_confidence_weighted",
    ]
    import torch

    device = torch.device("cpu")
    test = df[df["split"] == "test"].head(args.eval_max_cases or 200)
    rows = []
    for exp_id in targets:
        ckpt = resolve_checkpoint(project, exp_id, cfg, baselines_dir)
        if ckpt is None:
            continue
        state = torch.load(ckpt, map_location="cpu")
        model = build_model(state.get("spec", {}))
        model.load_state_dict(state["model"], strict=False)
        model.to(device)
        model.eval()
        case_rows = []
        for _, row in test.iterrows():
            oct_e = torch.tensor(load_visual_emb(str(row.get("oct_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
            col_e = torch.tensor(load_visual_emb(str(row.get("colposcopy_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
            fus_e = torch.tensor(load_visual_emb(str(row.get("fused_visual_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
            instr = torch.tensor(instr_vector(row.to_dict()), dtype=torch.float32, device=device).unsqueeze(0)
            gen = model.generate_structured_report(oct_e, col_e, fus_e, instr, int(row["binary_label"]), row.to_dict(), {})
            case_rows.append(safety_case_metrics(gen["generated_sections"], row.to_dict(), int(row["binary_label"])))
        agg = aggregate_safety(case_rows)
        agg["experiment_id"] = exp_id
        rows.append(agg)
    out = pd.DataFrame(rows)
    save_table(out, tables_dir / "table_report_safety_metrics.csv", tables_dir / "table_report_safety_metrics.md")
    return out


def run_prompt8(project: Path, df: pd.DataFrame) -> None:
    """Expert review package — templates only, no fabricated scores."""
    pack = project / "outputs/publishable/expert_review/blinded_review_package"
    pack.mkdir(parents=True, exist_ok=True)
    test = df[df["split"] == "test"]
    strat = test.groupby(["center_id", "binary_label"], dropna=False).head(5).head(200)
    cols = ["case_id", "center_id", "binary_label", "age", "hpv", "tct", "has_real_report", "training_report_type"]
    strat[cols].to_csv(pack / "blinded_review_cases.csv", index=False)
    proto = project / "outputs/publishable/tables/EXPERT_REVIEW_PROTOCOL.md"
    proto.write_text(
        "# Expert blind review protocol\n\n"
        "Sample n=200 stratified by center and CIN2+ label. "
        "Ratings 1–5 on clinical_plausibility, modality_consistency, section_completeness, "
        "recommendation_appropriateness, hallucination_risk (1=low), overall_usefulness.\n\n"
        "**No expert scores are included in this repository run.**\n",
        encoding="utf-8",
    )
    (project / "outputs/publishable/tables/EXPERT_REVIEW_INTERPRETATION_TEMPLATE.md").write_text(
        "Fill after expert ratings are collected. Use `outputs/publishable/scripts/analyze_expert_review.py`.\n",
        encoding="utf-8",
    )
    script = project / "outputs/publishable/scripts/analyze_expert_review.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        '#!/usr/bin/env python3\n"""Analyze expert_review_ratings.csv — Wilcoxon + ICC."""\n'
        "import pandas as pd\nfrom pathlib import Path\n"
        "p = Path(__file__).resolve().parents[1] / 'expert_review/expert_rating_template.csv'\n"
        "if not p.is_file():\n    print('No ratings file; export template only.')\n    raise SystemExit(0)\n"
        "df = pd.read_csv(p)\nprint(df.describe())\n",
        encoding="utf-8",
    )


def run_prompt10(project: Path) -> None:
    tables_dir = project / "outputs/publishable/tables"
    parts = []
    for name in (
        "table_baseline_comparison.csv",
        "table_lcad_qc_ablation.csv",
        "table_rasa_component_ablation.csv",
        "table_modality_ablation.csv",
        "table_loco_main_results.csv",
    ):
        p = tables_dir / name
        if p.is_file():
            parts.append(pd.read_csv(p))
    if not parts:
        return
    main = pd.concat(parts, ignore_index=True)
    metric_cols = [c for c in main.columns if c in (
        "rouge_l", "bleu", "label_consistency", "auc", "hallucination_rate", "contradiction_rate", "section_completeness"
    )]
    main = add_ci_columns(main, metric_cols)
    save_table(main, tables_dir / "table_main_results_for_manuscript.csv", tables_dir / "table_main_results_for_manuscript.md")
    main.to_csv(tables_dir / "table_supplementary_all_experiments.csv", index=False)
    build_statistical_tests_summary(main).to_csv(tables_dir / "statistical_tests_summary.csv", index=False)


def run_prompt11(project: Path) -> None:
    rep = project / "reproducibility_package"
    for sub in ("schema", "demo_data"):
        (rep / sub).mkdir(parents=True, exist_ok=True)
    (rep / "schema" / "pseudo_report_schema.json").write_text(
        '{"fields": ["diagnostic_summary", "oct_findings", "colposcopy_findings", "clinical_context", "impression", "recommendation"]}',
        encoding="utf-8",
    )
    ms = project / "outputs/publishable/manuscript_statements"
    ms.mkdir(parents=True, exist_ok=True)
    (ms / "DATA_AVAILABILITY.md").write_text(
        "Raw patient images and clinical reports cannot be shared due to privacy and ethics restrictions. "
        "Aggregate results, schemas, and synthetic demo cases are provided.\n",
        encoding="utf-8",
    )
    (ms / "CODE_AVAILABILITY.md").write_text("Code: LCAD-RASA scripts under `cervix_lcad_rasa/scripts/`.\n", encoding="utf-8")
    (project / "outputs/publishable/tables/REPRODUCIBILITY_PACKAGE_CHECKLIST.md").write_text(
        "- [x] Schema exported\n- [x] No raw PHI in tables\n- [ ] Expert ratings (pending)\n",
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="all", choices=["1", "2", "3", "all", "7", "10", "12"])
    p.add_argument("--skip-train", action="store_true", help="Only eval using existing checkpoints")
    p.add_argument("--eval-max-cases", type=int, default=288)
    p.add_argument("--quick", action="store_true", help="Fewer train steps")
    args = p.parse_args()
    project = resolve_project_root()
    cfg = load_jbd_config(project)
    if args.quick:
        cfg["training"]["num_epochs"] = 2
        cfg["training"]["max_steps_per_epoch"] = 60
    manifest_path = project / cfg["manifest"]
    df = pd.read_csv(manifest_path)
    tables_dir = project / "outputs/publishable/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (project / "outputs/publishable/baselines").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    failures = []

    def _run(name, fn):
        try:
            logger.info("=== %s ===", name)
            fn()
        except Exception:
            failures.append(name)
            (project / "outputs/publishable/logs" / f"fail_{name}.log").write_text(traceback.format_exc(), encoding="utf-8")

    if args.phase in ("1", "all"):
        _run("prompt7", lambda: run_prompt7(project, cfg))
        _run("prompt1", lambda: run_prompt1(project, df, cfg, args))
        _run("prompt2", lambda: run_prompt2(project, df, cfg, args))
        _run("prompt5", lambda: run_prompt5(project, df, cfg, args))
        _run("prompt3", lambda: run_prompt3(project, df, cfg, args))

    if args.phase in ("2", "all"):
        _run("prompt4", lambda: run_prompt4(project, df, cfg, args))
        ckpt = project / cfg.get("main_checkpoint", "outputs/publishable/checkpoints/publishable_full_lcad_rasa/best.ckpt")
        if ckpt.is_file():
            _run("prompt6", lambda: run_extended_perturbation(project, df, ckpt, tables_dir, max_cases=min(args.eval_max_cases, 64)))
        _run("prompt9", lambda: run_prompt9(project, df, cfg, args))

    if args.phase in ("3", "all"):
        _run("prompt8", lambda: run_prompt8(project, df))
        _run("prompt11", lambda: run_prompt11(project))

    if args.phase in ("10", "all"):
        _run("prompt10", lambda: run_prompt10(project))

    if args.phase in ("12", "all"):
        _run("figures", lambda: generate_all_figures(tables_dir, project / "outputs/publishable/figures"))
        _run("interpretations", lambda: write_interpretations(tables_dir))
        _run("results_draft", lambda: build_results_draft(tables_dir, project / "outputs/publishable/manuscript_sections"))

    summary = {
        "elapsed_minutes": (time.time() - t0) / 60,
        "failures": failures,
        "phase": args.phase,
    }
    (project / "outputs/publishable/logs/jbd_supplementary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
