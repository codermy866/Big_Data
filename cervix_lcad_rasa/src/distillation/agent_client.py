"""Mock and API agent clients for LCAD distillation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.distillation.pseudo_report_schema import build_pseudo_report
from src.utils.io import read_json


class MockAgentClient:
    def __init__(self, setting: str = "modality_plus_label"):
        self.setting = setting

    def generate(self, evidence: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        label = int(row.get("binary_label", 0))
        if self.setting == "modality_only_agent":
            label = 0
        oct_s = evidence["oct_evidence"].get("visual_summary", "")
        colpo_s = evidence["colposcopy_evidence"].get("visual_summary", "")
        instr = evidence["instruction_evidence"]
        clinical = f"Age {instr.get('age')}; HPV {instr.get('hpv')}; TCT {instr.get('tct')}."
        if self.setting == "label_only_agent":
            oct_s = colpo_s = "Not used in label-only setting."
            clinical = f"Label endpoint {row.get('binary_label_endpoint')}; label={label}."
        conf = 0.5 * evidence["oct_evidence"]["evidence_reliability"] + 0.5 * evidence["colposcopy_evidence"]["evidence_reliability"]
        return build_pseudo_report(
            case_id=str(row["case_id"]),
            center_id=str(row["center_id"]),
            label=label if self.setting != "modality_only_agent" else int(row.get("binary_label", 0)),
            endpoint=str(row.get("binary_label_endpoint", "CIN2+")),
            oct_sum=oct_s,
            colpo_sum=colpo_s,
            clinical=clinical,
            setting=self.setting,
            confidence=min(1.0, max(0.2, conf)),
        )


def get_client(name: str, setting: str, cfg: dict[str, Any]) -> MockAgentClient:
    if name == "api" and cfg.get("api", {}).get("enabled"):
        raise NotImplementedError("APIAgentClient disabled by default.")
    return MockAgentClient(setting=setting)
