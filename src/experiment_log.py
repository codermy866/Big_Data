#!/usr/bin/env python3
"""Mandatory experiment run metadata logging."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_run_record(
    *,
    experiment_id: str,
    model_variant: str,
    config_path: str,
    seed: int,
    fold_id: str,
    train_centers: List[str],
    test_center: Optional[str],
    enshi_reports_used: bool,
    report_input_at_inference: bool,
    repo_root: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if report_input_at_inference:
        raise ValueError("report_input_at_inference must be False for all test-stage runs")
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "model_variant": model_variant,
        "config_path": config_path,
        "git_commit": git_commit(repo_root),
        "seed": seed,
        "fold_id": fold_id,
        "train_centers": train_centers,
        "test_center": test_center,
        "enshi_reports_used_training": bool(enshi_reports_used),
        "report_input_at_inference": False,
    }
    if extra:
        record.update(extra)
    return record


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
