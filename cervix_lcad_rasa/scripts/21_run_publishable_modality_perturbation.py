#!/usr/bin/env python3
"""Prompt I (revised): Modality perturbation with structured text decoding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_publishable.hallucination import hallucination_flags
from src.evaluation_publishable.perturbation_metrics import aggregate_condition, by_section_rows
from src.evaluation_publishable.section_consistency import section_supported_scores
from src.models_publishable.lcad_rasa_model import PublishableLCADRASA, instr_vector, load_visual_emb
from src.utils.io import write_csv

DEFAULT_CONDITIONS = [
    "normal",
    "mask_oct",
    "mask_colposcopy",
    "mask_instruction",
    "shuffle_oct",
    "shuffle_colposcopy",
    "shuffle_instruction",
    "mask_visual",
    "label_only_inference",
    "randomize_label",
]

CONDITION_MASKS = {
    "normal": {},
    "mask_oct": {"mask_oct": True},
    "mask_colposcopy": {"mask_colposcopy": True},
    "mask_instruction": {"mask_instruction": True},
    "shuffle_oct": {"shuffle_oct": True},
    "shuffle_colposcopy": {"shuffle_colposcopy": True},
    "shuffle_instruction": {"shuffle_instruction": True},
    "mask_visual": {"mask_oct": True, "mask_colposcopy": True},
    "label_only_inference": {"mask_oct": True, "mask_colposcopy": True, "mask_instruction": True},
    "randomize_label": {"randomize_label": True},
}


def _prepare_batch_embeddings(test_df: pd.DataFrame) -> dict[str, dict]:
    store = {}
    for _, row in test_df.iterrows():
        cid = str(row["case_id"])
        store[cid] = {
            "oct": load_visual_emb(str(row.get("oct_embedding_path", ""))),
            "col": load_visual_emb(str(row.get("colposcopy_embedding_path", ""))),
            "fus": load_visual_emb(str(row.get("fused_visual_embedding_path", ""))),
            "instr": instr_vector(row.to_dict()),
            "row": row.to_dict(),
        }
    return store


def _apply_condition(
    store: dict,
    case_ids: list[str],
    condition: str,
    rng: np.random.Generator,
) -> dict[str, dict]:
    mask = CONDITION_MASKS.get(condition, {})
    out = {}
    oct_pool = [store[c]["oct"] for c in case_ids]
    col_pool = [store[c]["col"] for c in case_ids]
    ins_pool = [store[c]["instr"] for c in case_ids]
    for i, cid in enumerate(case_ids):
        s = store[cid]
        oct_e = s["oct"].copy()
        col_e = s["col"].copy()
        ins_e = s["instr"].copy()
        if mask.get("mask_oct") or condition == "mask_visual":
            oct_e = np.zeros_like(oct_e)
        if mask.get("mask_colposcopy") or condition == "mask_visual":
            col_e = np.zeros_like(col_e)
        if mask.get("mask_instruction") or condition == "label_only_inference":
            ins_e = np.zeros_like(ins_e)
        if mask.get("shuffle_oct"):
            oct_e = oct_pool[rng.integers(0, len(oct_pool))].copy()
        if mask.get("shuffle_colposcopy"):
            col_e = col_pool[rng.integers(0, len(col_pool))].copy()
        if mask.get("shuffle_instruction"):
            ins_e = ins_pool[rng.integers(0, len(ins_pool))].copy()
        fus_e = ((oct_e + col_e) / 2.0).astype(np.float32)
        out[cid] = {"oct": oct_e, "col": col_e, "fus": fus_e, "instr": ins_e, "row": s["row"]}
    return out


def _infer_case(model, emb: dict, device: torch.device, condition: str) -> dict:
    mask = CONDITION_MASKS.get(condition, {})
    oct_t = torch.tensor(emb["oct"], dtype=torch.float32).unsqueeze(0).to(device)
    col_t = torch.tensor(emb["col"], dtype=torch.float32).unsqueeze(0).to(device)
    fus_t = torch.tensor(emb["fus"], dtype=torch.float32).unsqueeze(0).to(device)
    ins_t = torch.tensor(emb["instr"], dtype=torch.float32).unsqueeze(0).to(device)
    label = int(emb["row"].get("binary_label", 0))
    return model.generate_structured_report(
        oct_t, col_t, fus_t, ins_t, label, row=emb["row"], modality_mask=mask
    )


def _find_checkpoint(project: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        return project / p if not p.is_absolute() else p
    for name in ("publishable_full_lcad_rasa", "full_lcad_rasa"):
        c = project / "outputs/publishable/checkpoints" / name / "best.ckpt"
        if c.is_file():
            return c
    ckpts = list((project / "outputs/publishable/checkpoints").glob("*/best.ckpt"))
    if ckpts:
        return sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1]
    raise FileNotFoundError("No publishable checkpoint found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output_dir", default="outputs/publishable")
    p.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    p.add_argument("--max_cases", type=int, default=128)
    p.add_argument("--decode_text", default="true")
    p.add_argument("--save_case_outputs", default="true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(ROOT / args.manifest)
    test = df[df["split"] == "test"] if "split" in df.columns else df
    test = test[test["needs_pseudo_report"] == 1].head(args.max_cases)
    case_ids = [str(c) for c in test["case_id"]]

    ckpt = _find_checkpoint(ROOT, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PublishableLCADRASA().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    store = _prepare_batch_embeddings(test)
    refs = {
        str(r["case_id"]): str(r.get("reference_report_text", ""))
        for _, r in test.iterrows()
        if int(r.get("has_real_report", 0)) == 1
    }
    rng = np.random.default_rng(args.seed)
    out_dir = ROOT / args.output_dir
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    gen_root = out_dir / "generated_reports/perturbation"
    save_cases = str(args.save_case_outputs).lower() == "true"

    summary_rows, section_rows, risk_rows, example_rows = [], [], [], []
    all_by_condition: dict[str, list[dict]] = {}

    # Normal first
    normal_emb = _apply_condition(store, case_ids, "normal", rng)
    normal_sections = {}
    normal_records = []
    for cid in case_ids:
        out = _infer_case(model, normal_emb[cid], device, "normal")
        sec = out["generated_sections"]
        normal_sections[cid] = sec
        rec = {
            "case_id": cid,
            "center_id": normal_emb[cid]["row"].get("center_id", ""),
            "binary_label": int(normal_emb[cid]["row"].get("binary_label", 0)),
            "condition": "normal",
            "generated_sections": sec,
            "risk_score": out["risk_score"],
            **section_supported_scores(sec, "normal"),
        }
        normal_records.append(rec)
        if save_cases:
            d = gen_root / "normal"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{cid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    all_by_condition["normal"] = normal_records
    normal_risk_mean = float(np.mean([r["risk_score"] for r in normal_records]))
    normal_risk_by_case = {r["case_id"]: r["risk_score"] for r in normal_records}

    for condition in args.conditions:
        if condition == "normal":
            continue
        pert_emb = _apply_condition(store, case_ids, condition, rng)
        records = []
        for cid in case_ids:
            out = _infer_case(model, pert_emb[cid], device, condition)
            sec = out["generated_sections"]
            rec = {
                "case_id": cid,
                "center_id": pert_emb[cid]["row"].get("center_id", ""),
                "binary_label": int(pert_emb[cid]["row"].get("binary_label", 0)),
                "condition": condition,
                "generated_sections": sec,
                "risk_score": out["risk_score"],
                **section_supported_scores(sec, condition),
            }
            records.append(rec)
            if save_cases:
                d = gen_root / condition
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{cid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            if condition != "normal" and cid in normal_sections:
                ns = normal_sections[cid]
                example_rows.append(
                    {
                        "case_id": cid,
                        "center_id": rec["center_id"],
                        "condition": condition,
                        "normal_oct_findings": ns.get("oct_findings", "")[:200],
                        "perturbed_oct_findings": sec.get("oct_findings", "")[:200],
                        "normal_colposcopy_findings": ns.get("colposcopy_findings", "")[:200],
                        "perturbed_colposcopy_findings": sec.get("colposcopy_findings", "")[:200],
                        "normal_clinical_context": ns.get("clinical_context", "")[:200],
                        "perturbed_clinical_context": sec.get("clinical_context", "")[:200],
                        "normal_impression": ns.get("impression", "")[:200],
                        "perturbed_impression": sec.get("impression", "")[:200],
                        "risk_score_normal": normal_risk_by_case.get(cid, np.nan),
                        "risk_score_perturbed": out["risk_score"],
                        "hallucination_flags": ";".join(hallucination_flags(sec, condition)),
                    }
                )
        all_by_condition[condition] = records

    # Aggregate all conditions including normal
    all_by_condition["normal"] = normal_records
    for condition in args.conditions:
        recs = all_by_condition.get(condition, [])
        row = aggregate_condition(condition, recs, normal_sections, refs)
        row["risk_score_delta_vs_normal"] = row["mean_risk_score"] - normal_risk_mean
        row["risk_score_absolute_delta_vs_normal"] = abs(row["risk_score_delta_vs_normal"])
        summary_rows.append(row)
        section_rows.extend(by_section_rows(condition, recs, normal_sections))
        risk_rows.append(
            {
                "condition": condition,
                "mean_risk_score": row["mean_risk_score"],
                "risk_score_std": row["risk_score_std"],
                "risk_score_delta_vs_normal": row["risk_score_delta_vs_normal"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables / "modality_perturbation_text_decoding_summary.csv", index=False)
    summary_df.to_csv(tables / "main_table_modality_perturbation.csv", index=False)
    pd.DataFrame(section_rows).to_csv(tables / "modality_perturbation_by_section.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(tables / "modality_perturbation_risk_sensitivity.csv", index=False)
    hal_cols = [c for c in summary_df.columns if "hallucination" in c or "unsupported" in c]
    summary_df[["condition", "n_cases"] + hal_cols].to_csv(tables / "modality_perturbation_hallucination.csv", index=False)
    pd.DataFrame(example_rows[:200]).to_csv(tables / "modality_perturbation_case_examples.csv", index=False)

    # Interpretation flags
    interp = []
    if summary_df[summary_df["condition"] == "mask_oct"]["oct_findings_similarity_to_normal"].iloc[0] < 0.95:
        interp.append("mask_oct reduces oct_findings similarity — supports OCT use")
    else:
        interp.append("mask_oct: limited oct_findings degradation")
    interp.append(
        f"max section drop: condition={summary_df.loc[summary_df['oct_findings_similarity_to_normal'].idxmin(), 'condition']}"
        if "oct_findings_similarity_to_normal" in summary_df.columns
        else ""
    )
    (tables / "modality_perturbation_interpretation.txt").write_text("\n".join(interp), encoding="utf-8")

    print(summary_df[["condition", "oct_findings_similarity_to_normal", "colposcopy_findings_similarity_to_normal", "clinical_context_similarity_to_normal", "label_consistency", "mean_risk_score", "risk_score_delta_vs_normal"]].to_string(index=False))
    print(f"\nWrote tables to {tables}")


if __name__ == "__main__":
    main()
