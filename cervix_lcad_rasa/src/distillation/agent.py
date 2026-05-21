"""Label-Constrained Agent for pseudo-report generation (no external API by default)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import ensure_dir
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

LABEL_TEXT = {0: "No high-grade cervical intraepithelial neoplasia (CIN2+).", 1: "Findings suspicious for CIN2+."}


class LabelConstrainedAgent:
    """Template-based agent; swap with real VLM/LLM when api.enabled=true."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.api_enabled = bool(cfg.get("api", {}).get("enabled", False))
        self.template = cfg.get("mock", {}).get("template", "{colpo_summary}")

    def generate_report(self, summaries: dict[str, str], label: int) -> str:
        if self.api_enabled:
            raise NotImplementedError(
                "External API distillation is disabled in the default scaffold. "
                "Set api.enabled=false or implement your provider."
            )
        label_text = LABEL_TEXT.get(int(label), LABEL_TEXT[0])
        return self.template.format(
            colpo_summary=summaries.get("colposcopy", "N/A"),
            cyto_summary=summaries.get("cytology", "N/A"),
            hpv_summary=summaries.get("hpv", "N/A"),
            clinical_summary=summaries.get("clinical", "N/A"),
            label_text=label_text,
        ).strip()


def generate_pseudo_reports(
    summaries_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> Path:
    out_dir = ensure_dir(cfg["outputs"]["pseudo_reports"])
    agent = LabelConstrainedAgent(cfg)

    grouped = summaries_df.groupby("exam_id")
    records = []
    for exam_id, grp in grouped:
        label = int(grp["label"].iloc[0])
        mod_map = {row["modality"]: row["summary"] for _, row in grp.iterrows()}
        text = agent.generate_report(mod_map, label)
        records.append({"exam_id": exam_id, "label": label, "pseudo_report": text})

    out_path = out_dir / "pseudo_reports.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d pseudo reports to %s", len(records), out_path)
    return out_path
