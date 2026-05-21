"""Report-rich centre masking validation (all cases with real reports)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.centers import masking_validation_mask
from src.distillation.agent_client import MockAgentClient
from src.evaluation.metrics import compute_metrics
from src.utils.io import read_json, write_csv, write_json

SETTINGS = ["label_only_agent", "modality_only_agent", "modality_plus_label_agent"]
MAX_MASKING_CASES = 0  # 0 = all report-available cases


def run_masking_validation(
    manifest_df: pd.DataFrame,
    evidence_dir: Path,
    out_dir: Path,
    max_cases: int = MAX_MASKING_CASES,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = manifest_df[masking_validation_mask(manifest_df)].copy()
    if max_cases > 0 and len(sub) > max_cases:
        sub = sub.sample(n=max_cases, random_state=42)

    metric_rows = []
    case_rows = []
    for setting in SETTINGS:
        client = MockAgentClient(setting=setting)
        preds, refs, labels = [], [], []
        for _, row in sub.iterrows():
            ev_path = evidence_dir / str(row["center_id"]) / f"{row['case_id']}.json"
            if not ev_path.is_file():
                continue
            ev = read_json(ev_path)
            pseudo = client.generate(ev, row.to_dict())
            write_json(
                out_dir / "masked_pseudo_reports" / setting / f"{row['case_id']}.json",
                pseudo,
            )
            ref = str(row.get("real_report_text", "")).strip()
            if not ref:
                ref = str(row.get("other_clinical_attributes", ""))
            pred = " ".join(
                [
                    pseudo.get("oct_findings", ""),
                    pseudo.get("colposcopy_findings", ""),
                    pseudo.get("clinical_context", ""),
                    pseudo.get("impression", ""),
                ]
            )
            preds.append(pred[:800])
            refs.append(ref[:800] if ref else pred[:800])
            labels.append(int(row.get("binary_label", 0)))
            case_rows.append(
                {
                    "case_id": row["case_id"],
                    "center_id": row["center_id"],
                    "setting": setting,
                    "ref_len": len(ref),
                    "pred_len": len(pred),
                }
            )
        m = (
            compute_metrics(preds, refs, labels)
            if preds
            else {"rouge_l_mean": 0.0, "label_consistency_mean": 0.0, "n": 0}
        )
        metric_rows.append({"setting": setting, **m, "n_cases": len(preds)})
    mdf = pd.DataFrame(metric_rows)
    tables = out_dir.parent / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(mdf, tables / "report_rich_masking_validation_metrics.csv")
    write_csv(pd.DataFrame(case_rows), tables / "report_rich_masking_validation_cases.csv")

    centre_rows = []
    for cid in sorted(sub["center_id"].unique()):
        csub = sub[sub["center_id"] == cid]
        centre_rows.append(
            {
                "center_id": cid,
                "n_real_report_cases": len(csub),
                "in_masking_pool": int(cid in ("enshi", "jingzhou", "xiangyang")),
            }
        )
    write_csv(pd.DataFrame(centre_rows), tables / "masking_validation_by_centre.csv")
    return mdf
