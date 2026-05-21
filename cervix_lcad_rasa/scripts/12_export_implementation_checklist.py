#!/usr/bin/env python3
"""Export implementation status对照表 vs revised method + execution prompt."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CHECKLIST = [
    # Stage / Experiment, Source doc, Requirement, output artifact, status key
    ("00", "prompt §5", "Data audit + centre modality summary", "outputs/data_audit_report.md", "data_audit_report.md"),
    ("00", "method §5", "Case-level report supervision profiling", "outputs/tables/centre_modality_summary.csv", "centre_modality_summary.csv"),
    ("01", "prompt §6", "Unified full_manifest.csv", "outputs/manifests/full_manifest.csv", "full_manifest.csv"),
    ("01", "method §5.3", "report_supervision_class / needs_pseudo_report", "outputs/manifests/full_manifest.csv:cols", "manifest_dual_cols"),
    ("01", "method §2", "Enshi 406 + Jingzhou 334 + total 744 real", "manifest:counts", "manifest_744_real"),
    ("02", "prompt §7", "Modality evidence JSON per case", "outputs/modality_evidence/", "modality_evidence_dir"),
    ("03", "method §6 / prompt §8", "Masking validation 3 agent settings", "report_rich_masking_validation_metrics.csv", "masking_metrics"),
    ("03", "method §6.2", "Enshi + Jingzhou masking subsets", "masking_validation_by_centre_detailed.csv", "masking_by_centre"),
    ("04", "method §7 / prompt §9", "LCAD pseudo reports weak-label centres", "outputs/pseudo_reports/", "pseudo_reports_dir"),
    ("05", "method §8 / prompt §10", "QC + full_manifest_with_pseudo_reports", "outputs/manifests/full_manifest_with_pseudo_reports.csv", "manifest_with_pseudo"),
    ("11.1", "method §11.1", "enshi_real_only training", "checkpoints/enshi_real_only/", "ckpt_enshi_real_only"),
    ("11.1", "method §11.1", "jingzhou_real_only training", "checkpoints/jingzhou_real_only/", "ckpt_jingzhou_real_only"),
    ("11.1", "method §11.1", "dual_real_only training", "checkpoints/dual_real_only/", "ckpt_dual_real_only"),
    ("11.2", "method §11.2", "lcad_augmented training", "checkpoints/lcad_augmented/", "ckpt_lcad_augmented"),
    ("11.3", "method §11.3", "full_lcad_rasa training", "checkpoints/full_lcad_rasa/", "ckpt_full_lcad_rasa"),
    ("11.5", "method §11.5", "QC ablation table", "pseudo_report_qc_ablation.csv", "qc_ablation"),
    ("11.6", "method §11.6", "RASA component ablation", "main_experiments_performance.csv", "rasa_ablation_perf"),
    ("11.7", "method §11.7", "Modality ablation", "modality_ablation_summary.csv", "modality_ablation"),
    ("11.7", "method §11.7", "Modality perturbation", "modality_perturbation_summary.csv", "modality_perturbation"),
    ("11.8", "method §11.8", "Leave-one-centre-out", "leave_one_center_out_summary.csv", "loco_summary"),
    ("12", "method §12", "Reference-based eval (real report test)", "eval_reference_based.csv", "eval_reference_based"),
    ("12", "method §12", "Per-centre eval groups", "eval_by_center.csv", "eval_by_center"),
    ("15", "prompt §15", "Physician review pack", "outputs/physician_review/", "physician_review"),
]


def _status(project: Path, key: str) -> tuple[str, str]:
    if key == "manifest_dual_cols":
        p = project / "outputs/manifests/full_manifest.csv"
        if not p.is_file():
            return "待实现", "manifest missing"
        cols = pd.read_csv(p, nrows=0).columns.tolist()
        need = {"report_supervision_class", "needs_pseudo_report", "report_archive_tier"}
        ok = need.issubset(set(cols))
        return ("已实现", f"cols ok: {need}") if ok else ("待实现", f"missing {need - set(cols)}")

    if key == "manifest_744_real":
        p = project / "outputs/manifests/full_manifest.csv"
        if not p.is_file():
            return "待实现", "no manifest"
        m = pd.read_csv(p)
        n = int(m["has_real_report"].sum())
        en = int(m[(m.center_id == "enshi") & (m.has_real_report == 1)].shape[0])
        jz = int(m[(m.center_id == "jingzhou") & (m.has_real_report == 1)].shape[0])
        ok = n == 744 and en == 406 and jz == 334
        return ("已实现", f"real={n} enshi={en} jingzhou={jz}") if ok else ("部分实现", f"real={n} enshi={en} jingzhou={jz}")

    if key.endswith("_dir") or key in ("modality_evidence_dir", "pseudo_reports_dir", "physician_review"):
        sub = {
            "modality_evidence_dir": "outputs/modality_evidence",
            "pseudo_reports_dir": "outputs/pseudo_reports",
            "physician_review": "outputs/physician_review",
        }[key]
        d = project / sub
        if d.is_dir() and any(d.rglob("*")):
            n = len(list(d.rglob("*.json"))) if "evidence" in sub or "pseudo" in sub else len(list(d.iterdir()))
            return "已实现", f"{n} files under {sub}"
        return "待实现", f"empty/missing {sub}"

    if key.startswith("ckpt_"):
        name = key.replace("ckpt_", "")
        ck = project / "outputs/checkpoints" / name / "best.ckpt"
        return ("已实现", str(ck)) if ck.is_file() else ("待实现", "no checkpoint")

    # file under tables or outputs
    rel = next((c[3] for c in CHECKLIST if c[4] == key), key)
    candidates = [
        project / rel,
        project / "outputs/manifests" / Path(rel).name,
        project / "outputs/tables" / Path(rel).name,
        project / "outputs/tables/full_lcad_rasa" / Path(rel).name,
    ]
    for p in candidates:
        if p.is_file():
            return "已实现", str(p.relative_to(project))
    return "待实现", "artifact not found"


def main():
    project = ROOT
    rows = []
    for stage, doc, req, artifact, key in CHECKLIST:
        status, note = _status(project, key)
        rows.append(
            {
                "stage": stage,
                "source": doc,
                "requirement": req,
                "expected_artifact": artifact,
                "status": status,
                "note": note,
            }
        )

    df = pd.DataFrame(rows)
    out_csv = project / "outputs/tables/implementation_checklist.csv"
    out_md = project / "outputs/tables/IMPLEMENTATION_STATUS.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    n_ok = (df["status"] == "已实现").sum()
    n_part = (df["status"] == "部分实现").sum()
    n_todo = (df["status"] == "待实现").sum()
    lines = [
        "# LCAD-RASA Implementation Status对照表",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"Summary: **{n_ok} implemented** | **{n_part} partial** | **{n_todo} pending** (of {len(df)} tracked items)",
        "",
        "| Stage | Source | Requirement | Status | Note |",
        "|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['stage']} | {r['source']} | {r['requirement']} | **{r['status']}** | {r['note']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
