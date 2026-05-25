#!/usr/bin/env python3
"""Run JBD_LCAD_RASA_next_experiment_prompts.md (Prompts 0–8)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supplementary.jbd_next_runner import run_all_prompts
from src.utils.config import resolve_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    p = argparse.ArgumentParser(description="JBD next-stage experiment prompts 0–8")
    p.add_argument("--prompts", default="0-8", help="e.g. 0-8, all, or 2,3,5")
    p.add_argument("--skip-train", action="store_true", default=True, help="Prompt 1: eval only (default)")
    p.add_argument("--train", action="store_true", help="Prompt 1: train missing checkpoints")
    p.add_argument("--quick", action="store_true", help="Prompt 1: reduced training budget")
    p.add_argument("--eval-max-cases", type=int, default=288)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()
    project = resolve_project_root()
    status = run_all_prompts(
        project,
        skip_train=not args.train,
        eval_max=args.eval_max_cases,
        quick=args.quick,
        device=args.device,
        prompts=args.prompts,
    )
    failed = [k for k, v in status.get("steps", {}).items() if not v.get("ok", True)]
    if failed:
        logger.error("Failed steps: %s", failed)
        sys.exit(1)
    logger.info("Done in %.1f min", status.get("elapsed_minutes", 0))
    print(f"Status: outputs/publishable/logs/JBD_NEXT_PROMPTS_STATUS.md")
    print(f"Figures: outputs/publishable/figures/jbd_final/")
    print(f"Freeze v2: outputs/publishable_jbd_submission_v2/")


if __name__ == "__main__":
    main()
