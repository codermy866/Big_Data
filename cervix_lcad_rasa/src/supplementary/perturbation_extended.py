"""Extended modality perturbation with multi-seed EDS (Prompt 6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation_publishable.perturbation_metrics import section_similarity
from src.models_publishable.lcad_rasa_model import PublishableLCADRASA, instr_vector, load_visual_emb


def _noise_emb(v: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return v + rng.normal(0, sigma, v.shape).astype(np.float32)


def _partial_drop(v: np.ndarray, drop_frac: float, rng: np.random.Generator) -> np.ndarray:
    if drop_frac <= 0:
        return v
    mask = rng.random(v.shape) > drop_frac
    out = v * mask
    return out if out.sum() > 0 else v * 0.01


def run_extended_perturbation(
    project: Path,
    manifest_df: pd.DataFrame,
    ckpt_path: Path,
    out_tables: Path,
    seeds: list[int] | None = None,
    max_cases: int = 64,
) -> pd.DataFrame:
    seeds = seeds or [42, 43, 44, 45, 46]
    test = manifest_df[manifest_df["split"] == "test"]
    if "needs_pseudo_report" in test.columns:
        sub = test[test["needs_pseudo_report"] == 1]
        test = sub if len(sub) >= 16 else test
    test = test.head(max_cases)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device)
    model = PublishableLCADRASA().to(device)
    model.load_state_dict(state["model"])
    model.eval()

    store = {}
    for _, row in test.iterrows():
        cid = str(row["case_id"])
        store[cid] = {
            "oct": load_visual_emb(str(row.get("oct_embedding_path", ""))),
            "col": load_visual_emb(str(row.get("colposcopy_embedding_path", ""))),
            "fus": load_visual_emb(str(row.get("fused_visual_embedding_path", ""))),
            "instr": instr_vector(row.to_dict()),
            "row": row.to_dict(),
            "label": int(row["binary_label"]),
        }

    conditions = ["normal", "mask_oct", "mask_colposcopy", "mask_instruction", "label_only_inference"]
    for sig in (0.1, 0.3, 0.5, 1.0):
        conditions.append(f"gaussian_noise_oct_{sig}")
        conditions.append(f"gaussian_noise_colposcopy_{sig}")
    for frac in (0.25, 0.5, 0.75):
        conditions.append(f"partial_oct_drop_{frac}")
        conditions.append(f"partial_colposcopy_drop_{frac}")
    conditions.extend(["center_shuffle", "hpv_tct_shuffle"])

    all_rows = []
    case_ids = list(store.keys())

    for seed in seeds:
        rng = np.random.default_rng(seed)
        normals = {}
        for cid in case_ids:
            s = store[cid]
            oct_e = torch.tensor(s["oct"], dtype=torch.float32).unsqueeze(0).to(device)
            col_e = torch.tensor(s["col"], dtype=torch.float32).unsqueeze(0).to(device)
            fus_e = torch.tensor(s["fus"], dtype=torch.float32).unsqueeze(0).to(device)
            ins_e = torch.tensor(s["instr"], dtype=torch.float32).unsqueeze(0).to(device)
            gen = model.generate_structured_report(oct_e, col_e, fus_e, ins_e, s["label"], s["row"], {})
            normals[cid] = gen["generated_sections"]

        for cond in conditions:
            case_metrics = []
            for i, cid in enumerate(case_ids):
                s = store[cid]
                oct_v, col_v, ins_v = s["oct"].copy(), s["col"].copy(), s["instr"].copy()
                mask = {}
                if cond == "mask_oct":
                    mask["mask_oct"] = True
                elif cond == "mask_colposcopy":
                    mask["mask_colposcopy"] = True
                elif cond == "mask_instruction":
                    mask["mask_instruction"] = True
                elif cond == "label_only_inference":
                    mask = {"mask_oct": True, "mask_colposcopy": True, "mask_instruction": True}
                elif cond.startswith("gaussian_noise_oct_"):
                    sig = float(cond.split("_")[-1])
                    oct_v = _noise_emb(oct_v, sig, rng)
                elif cond.startswith("gaussian_noise_colposcopy_"):
                    sig = float(cond.split("_")[-1])
                    col_v = _noise_emb(col_v, sig, rng)
                elif cond.startswith("partial_oct_drop_"):
                    oct_v = _partial_drop(oct_v, float(cond.split("_")[-1]), rng)
                elif cond.startswith("partial_colposcopy_drop_"):
                    col_v = _partial_drop(col_v, float(cond.split("_")[-1]), rng)
                elif cond == "center_shuffle":
                    s["row"] = {**s["row"], "center_id": store[case_ids[rng.integers(0, len(case_ids))]]["row"]["center_id"]}
                elif cond == "hpv_tct_shuffle":
                    other = store[case_ids[rng.integers(0, len(case_ids))]]["row"]
                    s["row"] = {**s["row"], "hpv": other.get("hpv"), "tct": other.get("tct")}
                    ins_v = instr_vector(s["row"])

                oct_e = torch.tensor(oct_v, dtype=torch.float32).unsqueeze(0).to(device)
                col_e = torch.tensor(col_v, dtype=torch.float32).unsqueeze(0).to(device)
                fus_e = torch.tensor(s["fus"], dtype=torch.float32).unsqueeze(0).to(device)
                ins_e = torch.tensor(ins_v, dtype=torch.float32).unsqueeze(0).to(device)
                gen = model.generate_structured_report(oct_e, col_e, fus_e, ins_e, s["label"], s["row"], mask)
                pert = gen["generated_sections"]
                norm = normals[cid]
                sm = {
                    "oct_findings_similarity": section_similarity(pert.get("oct_findings", ""), norm.get("oct_findings", "")),
                    "colposcopy_findings_similarity": section_similarity(pert.get("colposcopy_findings", ""), norm.get("colposcopy_findings", "")),
                    "clinical_context_similarity": section_similarity(pert.get("clinical_context", ""), norm.get("clinical_context", "")),
                }
                sm["eds_oct_findings"] = 1.0 - sm["oct_findings_similarity"]
                sm["eds_colposcopy_findings"] = 1.0 - sm["colposcopy_findings_similarity"]
                sm["eds_clinical_context"] = 1.0 - sm["clinical_context_similarity"]
                sm["risk_delta"] = float(gen["risk_score"]) - 0.5
                case_metrics.append(sm)
            agg = {"condition": cond, "seed": seed, "n_cases": len(case_metrics)}
            for k in case_metrics[0]:
                vals = [m[k] for m in case_metrics]
                agg[k] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
            all_rows.append(agg)

    df = pd.DataFrame(all_rows)
    summary = df.groupby("condition").mean(numeric_only=True).reset_index()
    out_tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tables / "table_modality_perturbation_extended_seeds.csv", index=False)
    from src.supplementary.io_utils import save_table

    save_table(summary, out_tables / "table_modality_perturbation_extended.csv", out_tables / "table_modality_perturbation_extended.md")
    return summary
