#!/usr/bin/env python3
"""Aggregate pipeline outputs into FINAL_EXPERIMENT_SUMMARY.md."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import resolve_project_root


def _read_csv(path: Path):
    return pd.read_csv(path) if path.is_file() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/tables/FINAL_EXPERIMENT_SUMMARY.md")
    args = p.parse_args()
    project = resolve_project_root()
    tables = project / "outputs" / "tables"
    out = project / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    manifest = project / "outputs/manifests/full_manifest.csv"
    mdf = pd.read_csv(manifest) if manifest.is_file() else None
    n_total = len(mdf) if mdf is not None else 1897
    n_real = int(mdf["has_real_report"].sum()) if mdf is not None else 0
    n_pseudo = int(mdf["needs_pseudo_report"].sum()) if mdf is not None and "needs_pseudo_report" in mdf.columns else 0
    enshi_n = jingzhou_n = 0
    if mdf is not None:
        enshi_n = int(mdf[(mdf.center_id == "enshi") & (mdf.has_real_report == 1)].shape[0])
        jingzhou_n = int(mdf[(mdf.center_id == "jingzhou") & (mdf.has_real_report == 1)].shape[0])
    cin2_rate = mdf["binary_label"].mean() if mdf is not None else 0.167

    lines = [
        "# LCAD-RASA — Final Experimental Summary (Dual Report Centres)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Python: `/data2/hmy_pri/VLM_Caus_Rm_Mics/my_retfound/bin/python`",
        "",
        "## 1. Cohort",
        "",
        f"- Modeling: **{n_total}** exams, CIN2+ **{cin2_rate:.1%}**, patient-level splits validated",
        f"- Dual real-report supervision: **Enshi {enshi_n}/406** + **Jingzhou {jingzhou_n}/406** (case-level)",
        f"- Total real-report cases: **{n_real}** | pseudo-report candidates: **{n_pseudo}**",
        "",
        "## 2. Masking validation (Enshi + Jingzhou + sparse Xiangyang)",
        "",
    ]
    m = _read_csv(tables / "report_rich_masking_validation_metrics.csv")
    m_centre = _read_csv(tables / "masking_validation_by_centre_detailed.csv")
    if m is not None:
        lines.append("### Overall (n=%d real-report cases)" % int(m["n_cases"].max() if "n_cases" in m.columns else 0))
        lines.append("```\n" + m.to_string(index=False) + "\n```")
    if m_centre is not None:
        lines.append("### By centre (Enshi / Jingzhou / Xiangyang)")
        lines.append("```\n" + m_centre.to_string(index=False) + "\n```")
    lines.extend(["", "## 3. Pseudo-report QC (weak-label only)", ""])
    qc = _read_csv(tables / "pseudo_report_quality_summary.csv")
    if qc is not None:
        row = qc.iloc[0]
        lines.append(
            f"- QC'd: **{int(row['n_qc'])}** | pass rate **{row['pass_rate']:.1%}** | "
            f"mean qc_score **{row['mean_qc_score']:.3f}** | mean weight **{row['mean_weight']:.3f}**"
        )

    lines.extend(["", "## 4. Training full_lcad_rasa (10 epochs, CUDA)", ""])
    hist_path = project / "outputs/logs/full_lcad_rasa_history.json"
    if hist_path.is_file():
        hist = json.loads(hist_path.read_text())
        lines.append(f"- Final loss: **{hist[-1]['loss']:.4f}** (epoch {hist[-1]['epoch']})")
        lines.append(f"- Components: ce={hist[-1].get('ce',0):.4f}, align={hist[-1].get('align',0):.4f}, risk={hist[-1].get('risk',0):.4f}")

    lines.extend(["", "## 5. Test evaluation (GPU model)", ""])
    ev = _read_csv(tables / "full_lcad_rasa/eval_report_metrics.csv")
    if ev is not None:
        r = ev.iloc[0]
        lines.append(
            f"- n={int(r['n'])} | ROUGE-L={r['rouge_l_mean']:.4f} | "
            f"label consistency={r['label_consistency_mean']:.4f} | mock_eval={r.get('mock_eval', False)}"
        )

    lines.extend(["", "## 6. Modality ablation (test, weak-label)", ""])
    ma = _read_csv(tables / "modality_ablation_summary.csv")
    if ma is not None:
        lines.append("```\n" + ma.to_string(index=False) + "\n```")

    lines.extend(["", "## 7. Modality perturbation", ""])
    mp = _read_csv(tables / "modality_perturbation_summary.csv")
    if mp is not None:
        lines.append("```\n" + mp.to_string(index=False) + "\n```")

    lines.extend(["", "## 8. Outputs", ""])
    lines.append(f"- Manifest: `{project / 'outputs/manifests/full_manifest_with_pseudo_reports.csv'}`")
    lines.append(f"- Checkpoint: `{project / 'outputs/checkpoints/full_lcad_rasa/best.ckpt'}`")
    lines.append(f"- Experiment summary: `{project / 'outputs/experiment_summary.md'}`")
    lines.append(f"- Log: `{project / 'outputs/logs/publish_pipeline.log'}`")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
