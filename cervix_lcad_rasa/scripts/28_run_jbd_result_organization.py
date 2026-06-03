#!/usr/bin/env python3
"""JBD result organization & remaining experiments (R1-R3, S1-S3, E1-E4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supplementary.result_organization.runner import run_all
from src.utils.config import resolve_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    p = argparse.ArgumentParser(description="LCAD-RASA result organization prompts")
    p.add_argument(
        "--prompt",
        default="priority",
        help="R1,R2,... or priority (R1-R3,S1-S2,E3,E4,S3,E1,E2) or all",
    )
    args = p.parse_args()
    project = resolve_project_root()
    failures = run_all(project, args.prompt)
    if failures:
        logger.error("Failures: %s", failures)
        sys.exit(1)
    logger.info("Result organization complete.")


if __name__ == "__main__":
    main()
