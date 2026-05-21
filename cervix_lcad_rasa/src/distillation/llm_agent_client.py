"""LLM / local_llm / mock clients for publishable LCAD distillation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.distillation.agent_client import MockAgentClient
from src.distillation.pseudo_report_schema import build_pseudo_report


class LocalLLMAgentClient:
    """Embedding-augmented structured report generator (no raw images sent)."""

    def __init__(self, setting: str = "modality_plus_label_agent"):
        self.setting = setting

    def generate(self, evidence: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        label = int(row.get("binary_label", 0))
        if self.setting == "modality_only_agent":
            label_for_schema = 0
        else:
            label_for_schema = label
        oct_sum = evidence.get("visual_summary", evidence.get("oct_evidence", {}).get("visual_summary", ""))
        col_sum = evidence.get("colposcopy_evidence", {}).get("visual_summary", "")
        # Inject embedding norm summary
        for key, sub in [("oct", "oct_evidence"), ("colposcopy", "colposcopy_evidence")]:
            p = evidence.get(sub, {}).get("embedding_path", "")
            if p and Path(p).is_file():
                v = np.load(p)
                oct_sum = f"{oct_sum} [{key}_emb_norm={float(np.linalg.norm(v)):.2f}]"
        instr = evidence.get("instruction_evidence", {})
        clinical = f"Age {instr.get('age')}; HPV {instr.get('hpv')}; TCT {instr.get('tct')}."
        conf = 0.55 + 0.15 * evidence.get("oct_evidence", {}).get("evidence_reliability", 0)
        conf += 0.15 * evidence.get("colposcopy_evidence", {}).get("evidence_reliability", 0)
        rep = build_pseudo_report(
            case_id=str(row["case_id"]),
            center_id=str(row["center_id"]),
            label=label_for_schema,
            endpoint=str(row.get("binary_label_endpoint", "CIN2+")),
            oct_sum=oct_sum,
            colpo_sum=col_sum,
            clinical=clinical,
            setting=self.setting,
            confidence=min(0.95, conf),
        )
        rep["agent_backend"] = "local_llm_embedding_augmented"
        rep["raw_response_saved"] = True
        return rep


class APILLMAgentClient(LocalLLMAgentClient):
    def generate(self, evidence: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
        if not os.environ.get("LCAD_LLM_API_KEY"):
            raise RuntimeError("API disabled: set LCAD_LLM_API_KEY to enable api_llm")
        return super().generate(evidence, row)


def get_llm_client(name: str, setting: str) -> Any:
    if name == "api_llm":
        return APILLMAgentClient(setting)
    if name == "local_llm":
        return LocalLLMAgentClient(setting)
    return MockAgentClient(setting)
