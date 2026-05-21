"""Quality control for pseudo-reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

LABEL_KEYWORDS = {
    0: ["no high-grade", "cin2+", "negative"],
    1: ["suspicious", "cin2+", "high-grade"],
}


def run_pseudo_report_qc(reports_path: Path, cfg: dict[str, Any]) -> Path:
    qc_cfg = cfg.get("qc", {})
    min_len = int(qc_cfg.get("min_length", 20))
    max_len = int(qc_cfg.get("max_length", 512))
    require_kw = bool(qc_cfg.get("require_label_keywords", True))
    out_dir = ensure_dir(cfg["outputs"]["qc"])

    rows = []
    with reports_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec["pseudo_report"]
            label = int(rec["label"])
            length_ok = min_len <= len(text) <= max_len
            kw_ok = True
            if require_kw:
                kws = LABEL_KEYWORDS.get(label, [])
                kw_ok = any(k.lower() in text.lower() for k in kws)
            rows.append(
                {
                    "exam_id": rec["exam_id"],
                    "label": label,
                    "length": len(text),
                    "length_ok": length_ok,
                    "label_keywords_ok": kw_ok,
                    "passed": length_ok and kw_ok,
                }
            )

    df = pd.DataFrame(rows)
    out_path = out_dir / "pseudo_report_qc.csv"
    df.to_csv(out_path, index=False)
    n_pass = int(df["passed"].sum())
    logger.info("QC: %d/%d passed -> %s", n_pass, len(df), out_path)
    return out_path
