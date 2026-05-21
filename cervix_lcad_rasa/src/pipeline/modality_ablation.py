"""Modality subset and perturbation ablations (clinical consistency on test)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from src.distillation.agent_client import MockAgentClient
from src.evaluation.metrics import compute_metrics
from src.utils.io import read_json, write_csv


MODALITY_CONFIGS = {
    "oct_only": {"oct": True, "colposcopy": False, "instruction": False},
    "colposcopy_only": {"oct": False, "colposcopy": True, "instruction": False},
    "instruction_only": {"oct": False, "colposcopy": False, "instruction": True},
    "oct_colposcopy": {"oct": True, "colposcopy": True, "instruction": False},
    "oct_instruction": {"oct": True, "colposcopy": False, "instruction": True},
    "colposcopy_instruction": {"oct": False, "colposcopy": True, "instruction": True},
    "all_modalities": {"oct": True, "colposcopy": True, "instruction": True},
}

PERTURBATIONS = [
    "normal",
    "mask_oct",
    "mask_colposcopy",
    "mask_instruction",
    "shuffle_oct",
    "shuffle_colposcopy",
    "randomized_label",
]


def _mask_evidence(ev: dict, cfg: dict) -> dict:
    import copy

    e = copy.deepcopy(ev)
    if not cfg.get("oct"):
        e["oct_evidence"] = {
            "available": False,
            "visual_summary": "OCT evidence unavailable.",
            "evidence_reliability": 0.0,
            "evidence_source": "masked",
        }
    if not cfg.get("colposcopy"):
        e["colposcopy_evidence"] = {
            "available": False,
            "visual_summary": "Colposcopy evidence unavailable.",
            "evidence_reliability": 0.0,
            "evidence_source": "masked",
        }
    if not cfg.get("instruction"):
        e["instruction_evidence"] = {
            "age": "",
            "hpv": "",
            "tct": "",
            "other_clinical_context": "",
            "missing_fields": ["age", "hpv", "tct"],
        }
    return e


def _apply_perturbation(ev: dict, name: str, rng: random.Random) -> dict:
    e = _mask_evidence(ev, MODALITY_CONFIGS["all_modalities"])
    if name == "mask_oct":
        return _mask_evidence(e, {"oct": False, "colposcopy": True, "instruction": True})
    if name == "mask_colposcopy":
        return _mask_evidence(e, {"oct": True, "colposcopy": False, "instruction": True})
    if name == "mask_instruction":
        return _mask_evidence(e, {"oct": True, "colposcopy": True, "instruction": False})
    if name == "shuffle_oct":
        e["oct_evidence"]["visual_summary"] = f"shuffled_{rng.randint(0, 99999)}"
    if name == "shuffle_colposcopy":
        e["colposcopy_evidence"]["visual_summary"] = f"shuffled_{rng.randint(0, 99999)}"
    return e


def run_modality_ablations(
    manifest: Path,
    evidence_dir: Path,
    tables_dir: Path,
    split: str = "test",
    max_cases: int = 128,
) -> None:
    df = pd.read_csv(manifest)
    test = df[df["split"] == split] if "split" in df.columns else df
    test = test[test["has_real_report"] == 0]
    if len(test) > max_cases:
        test = test.sample(n=max_cases, random_state=42)

    mod_rows, pert_rows = [], []
    client = MockAgentClient(setting="modality_plus_label_agent")
    rng = random.Random(42)

    for mod_name, mcfg in MODALITY_CONFIGS.items():
        preds, labels = [], []
        for _, row in test.iterrows():
            ev_path = evidence_dir / str(row["center_id"]) / f"{row['case_id']}.json"
            if not ev_path.is_file():
                continue
            ev = _mask_evidence(read_json(ev_path), mcfg)
            rep = client.generate(ev, row.to_dict())
            preds.append(rep.get("impression", ""))
            labels.append(int(row["binary_label"]))
        m = compute_metrics(preds, preds, labels) if preds else {}
        mod_rows.append(
            {
                "modality_setting": mod_name,
                "label_consistency_mean": m.get("label_consistency_mean", 0),
                "n": len(preds),
            }
        )

    for pert in PERTURBATIONS:
        preds, labels = [], []
        for _, row in test.iterrows():
            ev_path = evidence_dir / str(row["center_id"]) / f"{row['case_id']}.json"
            if not ev_path.is_file():
                continue
            ev = _apply_perturbation(read_json(ev_path), pert, rng)
            rdict = row.to_dict()
            if pert == "randomized_label":
                rdict["binary_label"] = int(rng.randint(0, 1))
            rep = client.generate(ev, rdict)
            preds.append(rep.get("impression", ""))
            labels.append(int(row["binary_label"]))
        m = compute_metrics(preds, preds, labels) if preds else {}
        pert_rows.append(
            {
                "perturbation": pert,
                "label_consistency_mean": m.get("label_consistency_mean", 0),
                "n": len(preds),
            }
        )

    write_csv(pd.DataFrame(mod_rows), tables_dir / "modality_ablation_summary.csv")
    write_csv(pd.DataFrame(pert_rows), tables_dir / "modality_perturbation_summary.csv")
