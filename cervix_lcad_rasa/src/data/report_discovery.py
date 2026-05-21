"""Discover archived report files under colposcopy folders (Exp0-compatible)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _colpo_dirs(colpo_paths: Any) -> list[Path]:
    if colpo_paths is None:
        return []
    if isinstance(colpo_paths, str):
        s = colpo_paths.strip()
        if s.startswith("["):
            try:
                paths = json.loads(s)
            except json.JSONDecodeError:
                paths = [p.strip() for p in s.split(";") if p.strip()]
        else:
            paths = [p.strip() for p in s.split(";") if p.strip()]
    else:
        paths = list(colpo_paths)
    dirs: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        if not p:
            continue
        parent = Path(p).parent.resolve()
        key = str(parent)
        if key not in seen:
            seen.add(key)
            dirs.append(parent)
    return dirs


def discover_report_path(colpo_paths: Any) -> str:
    """Return best report file path under colposcopy case folder(s)."""
    priority: list[Path] = []
    for col_dir in _colpo_dirs(colpo_paths):
        if not col_dir.is_dir():
            continue
        for p in col_dir.rglob("*"):
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
        return str(priority[0].resolve())
    return ""
