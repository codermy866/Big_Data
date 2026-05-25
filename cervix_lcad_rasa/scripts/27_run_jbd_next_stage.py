#!/usr/bin/env python3
"""JBD next-stage pre-submission pipeline (Prompts A–I)."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supplementary.next_stage.manuscript import freeze_submission_v1, run_figure_table_plan, write_final_manuscript
from src.supplementary.next_stage.prompt_a import run_image_audit
from src.supplementary.next_stage.prompt_b import run_stratified_eval
from src.supplementary.next_stage.prompt_c import run_loss_sweep
from src.supplementary.next_stage.prompt_d import run_threshold_tuning
from src.supplementary.next_stage.prompt_e import run_loco_strict
from src.supplementary.next_stage.prompt_f import run_multiseed
from src.supplementary.train_eval import load_jbd_config
from src.utils.config import resolve_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)
PYTHON = "/data2/hmy_pri/VLM_Caus_Rm_Mics/my_retfound/bin/python"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="all", help="A|B|C|D|E|F|G|H|I|all|core (A-E)")
    p.add_argument("--quick", action="store_true", help="Shorter training for C/E/F")
    p.add_argument("--eval-max-cases", type=int, default=None)
    p.add_argument("--seeds", default="2026,2027,2028")
    args = p.parse_args()

    project = resolve_project_root()
    cfg = load_jbd_config(project)
    manifest = project / cfg["manifest"]
    df = __import__("pandas").read_csv(manifest)
    tables = project / "outputs/publishable/tables"
    tables.mkdir(parents=True, exist_ok=True)
    baselines = project / "outputs/publishable/baselines"
    baselines.mkdir(parents=True, exist_ok=True)
    ms = project / "outputs/publishable/manuscript_sections"
    ms.mkdir(parents=True, exist_ok=True)
    failures = []

    def run(name: str, fn):
        try:
            logger.info("=== Prompt %s ===", name)
            fn()
        except Exception:
            failures.append(name)
            (project / "outputs/publishable/logs" / f"fail_next_{name}.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )

    prompts = args.prompt.upper()
    if prompts in ("ALL", "CORE", "A"):
        run("A", lambda: run_image_audit(project, manifest, tables))
    if prompts in ("ALL", "CORE", "B"):
        run("B", lambda: run_stratified_eval(project, df, tables, baselines, args.eval_max_cases))
    if prompts in ("ALL", "CORE", "C"):
        run("C", lambda: run_loss_sweep(project, df, cfg, project / "outputs/publishable/rasa_sweep", tables, args.quick))
    if prompts in ("ALL", "CORE", "D"):
        run("D", lambda: run_threshold_tuning(project, df, tables, baselines))
    if prompts in ("ALL", "CORE", "E"):
        run("E", lambda: run_loco_strict(project, df, cfg, project / "outputs/publishable/loco_strict", tables, args.quick))
    if prompts in ("ALL", "F"):
        seeds = [int(s) for s in args.seeds.split(",")]
        run("F", lambda: run_multiseed(project, df, cfg, project / "outputs/publishable/multiseed", tables, seeds, args.quick))
    if prompts in ("ALL", "G"):
        run("G", lambda: run_figure_table_plan(project, tables, ms))
    if prompts in ("ALL", "H"):
        run("H", lambda: write_final_manuscript(project, tables, ms))
    if prompts in ("ALL", "I"):
        run("I", lambda: freeze_submission_v1(project, tables, ms))

    summary = {"failures": failures, "prompt": args.prompt}
    log = project / "outputs/publishable/logs/jbd_next_stage_summary.json"
    log.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
