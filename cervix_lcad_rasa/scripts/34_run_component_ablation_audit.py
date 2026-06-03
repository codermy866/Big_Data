#!/usr/bin/env python3
"""Re-evaluate all ablation experiments and write component-necessity audit."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supplementary.io_utils import save_table
from src.supplementary.next_stage.core import collect_risk_scores, metrics_at_threshold, select_thresholds
from src.supplementary.train_eval import evaluate_experiment, load_jbd_config, resolve_checkpoint, train_experiment
from src.utils.config import resolve_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT = resolve_project_root()
LOG_DIR = PROJECT / "outputs/publishable/logs"
TABLES = PROJECT / "outputs/publishable/tables"
MANUSCRIPT = TABLES / "manuscript"
AUDIT_MD = PROJECT.parent / "JBD_COMPONENT_ABLATION_AUDIT.md"


def _merge_spec(base: dict, extra: dict) -> dict:
    out = {**base}
    for k, v in extra.items():
        if k == "model" and isinstance(v, dict):
            out["model"] = {**out.get("model", {}), **v}
        elif k == "loss" and isinstance(v, dict):
            out["loss"] = {**out.get("loss", {}), **v}
        else:
            out[k] = v
    return out


def _default_train_spec() -> dict:
    return {
        "train_filter": {"training_eligible": 1},
        "use_pseudo_report": True,
        "use_real_report": True,
        "require_qc_pass": True,
        "use_report_loss": True,
        "model": {"use_section_align": True, "use_risk_head": True},
        "loss": {"ce_weight": 1.0, "rasa_weight": 0.5, "cls_weight": 0.2, "cons_weight": 0.1},
    }


def _resolve_ckpt(exp_id: str, cfg: dict, baselines_dir: Path, baselines_only: bool) -> Path | None:
    if baselines_only:
        p = baselines_dir / exp_id / "best.ckpt"
        return p if p.is_file() else None
    return resolve_checkpoint(PROJECT, exp_id, cfg, baselines_dir)


def _apply_val_threshold_metrics(
    df: pd.DataFrame,
    ckpt_path: Path,
    spec: dict,
    eval_max: int,
) -> tuple[dict, float]:
    """Match Table 2: threshold = validation max-F1, metrics on test."""
    import torch

    val = df[df["split"] == "validation"] if "validation" in df["split"].values else df[df["split"] == "val"]
    test = df[df["split"] == "test"].head(eval_max)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yv_t, yv_s = collect_risk_scores(ckpt_path, val, spec, device)
    thr = select_thresholds(yv_t, yv_s)["max_f1"]
    yt_t, yt_s = collect_risk_scores(ckpt_path, test, spec, device)
    tm = metrics_at_threshold(yt_t, yt_s, thr)
    return tm, thr


def _train_and_eval(
    df: pd.DataFrame,
    cfg: dict,
    exp_id: str,
    spec: dict,
    baselines_dir: Path,
    skip_train: bool,
    force_retrain: bool,
    eval_max: int,
    unified_threshold: bool,
    baselines_only: bool,
) -> dict:
    ckpt = baselines_dir / exp_id / "best.ckpt"
    log_row = {"experiment_id": exp_id}
    try:
        if not skip_train and (force_retrain or not ckpt.is_file()):
            tr = train_experiment(PROJECT, df, exp_id, spec, cfg, baselines_dir / exp_id, seed=int(cfg.get("seed", 42)))
            log_row.update(tr)
            if tr.get("status") != "ok":
                return log_row
        use_ckpt = _resolve_ckpt(exp_id, cfg, baselines_dir, baselines_only)
        if use_ckpt is None:
            log_row["status"] = "no_checkpoint"
            return log_row
        ev = evaluate_experiment(PROJECT, df, exp_id, use_ckpt, spec, max_cases=eval_max)
        if unified_threshold:
            tm, thr = _apply_val_threshold_metrics(df, use_ckpt, spec, eval_max)
            for k in ("auc", "f1", "sensitivity", "specificity", "ece", "brier"):
                if k in tm:
                    ev[k] = tm[k]
            ev["threshold_val_max_f1"] = thr
            ev["evaluation_protocol"] = "test_auc_threshold_free_f1_val_max_f1"
        log_row.update(ev)
        log_row["status"] = "ok"
        log_row["checkpoint"] = str(use_ckpt)
    except Exception as e:
        log_row["status"] = "failed"
        log_row["error"] = str(e)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / f"fail_ablation_{exp_id}.log").write_text(traceback.format_exc(), encoding="utf-8")
    return log_row


def run_ablation_block(
    df: pd.DataFrame,
    cfg: dict,
    block_name: str,
    exp_specs: dict,
    baselines_dir: Path,
    skip_train: bool,
    force_retrain: bool,
    eval_max: int,
    unified_threshold: bool,
    baselines_only: bool,
) -> pd.DataFrame:
    base = _default_train_spec()
    rows = []
    for exp_id, spec in exp_specs.items():
        full = _merge_spec(base, spec)
        logger.info("[%s] %s", block_name, exp_id)
        rows.append(
            _train_and_eval(
                df, cfg, exp_id, full, baselines_dir, skip_train, force_retrain,
                eval_max, unified_threshold, baselines_only,
            )
        )
    return pd.DataFrame(rows)


def _verdict(delta_auc: float, delta_f1: float, component_removed: str) -> str:
    if delta_auc <= -0.08 or delta_f1 <= -0.12:
        return f"**必要** — 移除后 AUROC/F1 明显下降"
    if delta_auc <= -0.03 or delta_f1 <= -0.05:
        return "**有帮助** — 有数值贡献，建议保留"
    if delta_auc >= 0.02 and component_removed:
        return "**存疑/可简化** — 移除后指标未降或略升（检查 checkpoint 别名或阈值）"
    return "**边际** — 贡献较小，可放 Supplementary 简述"


def write_audit(
    rasa: pd.DataFrame,
    modality: pd.DataFrame,
    qc: pd.DataFrame,
    elapsed_min: float,
) -> Path:
    lines = [
        "# LCAD-RASA 组件消融审计\n",
        f"生成时间: {datetime.now(timezone.utc).isoformat()}\n",
        f"耗时: {elapsed_min:.1f} min\n\n",
        "## 评估协议\n",
        "- AUROC：测试集 threshold-free\n",
        "- F1/sensitivity/specificity：**验证集 max-F1 选阈 → 测试集报告**（与 Table 2 一致）\n",
        "- Checkpoint：`outputs/publishable/baselines/{exp_id}/` 独立训练权重（不使用 alias）\n\n",
        "## 判定规则\n",
        "- 对照：`full_lcad_rasa`（同配置完整模型）\n",
        "- **必要**：ΔAUROC ≤ −0.08 或 ΔF1 ≤ −0.12\n",
        "- **有帮助**：ΔAUROC ≤ −0.03 或 ΔF1 ≤ −0.05\n",
        "- **存疑**：移除后 AUROC 未降（可能 checkpoint 别名/阈值未对齐）\n\n",
    ]

    def _block(title: str, df: pd.DataFrame, ref_id: str = "full_lcad_rasa") -> None:
        if df is None or len(df) == 0:
            lines.append(f"## {title}\n\n无数据\n\n")
            return
        ref = df[df["experiment_id"] == ref_id]
        if len(ref) == 0:
            ref_auc, ref_f1 = float("nan"), float("nan")
        else:
            ref_auc = float(ref.iloc[0].get("auc", float("nan")))
            ref_f1 = float(ref.iloc[0].get("f1", float("nan")))
        lines.append(f"## {title}\n\n")
        lines.append(f"参照 `{ref_id}`: AUROC={ref_auc:.4f}, F1={ref_f1:.4f}\n\n")
        lines.append("| 实验 | AUROC | ΔAUROC | F1 | ΔF1 | 判定 |\n")
        lines.append("|------|-------|--------|-----|-----|------|\n")
        for _, r in df.sort_values("auc", ascending=False).iterrows():
            eid = r["experiment_id"]
            auc = float(r.get("auc", 0))
            f1 = float(r.get("f1", 0))
            d_auc = auc - ref_auc if eid != ref_id else 0.0
            d_f1 = f1 - ref_f1 if eid != ref_id else 0.0
            removed = eid != ref_id
            v = "（参照）" if not removed else _verdict(d_auc, d_f1, removed)
            lines.append(f"| `{eid}` | {auc:.4f} | {d_auc:+.4f} | {f1:.4f} | {d_f1:+.4f} | {v} |\n")
        lines.append("\n")

    _block("RASA 组件消融 (S5)", rasa)
    _block("模态消融 (S3)", modality, ref_id="full_with_fused")
    if len(modality) and "full_without_fused" in modality["experiment_id"].values:
        alt = modality[modality["experiment_id"] == "full_without_fused"].iloc[0]
        lines.append(
            f"> fused visual: `full_with_fused` AUC={float(alt.get('auc',0)):.4f} vs "
            f"`full_without_fused` — 见上表对比。\n\n"
        )
    _block("LCAD QC 权重消融 (S4)", qc, ref_id="pseudo_qc_confidence_weighted")

    lines.extend(
        [
            "## 技术栈与消融对应关系\n\n",
            "| 技术组件 | 消融实验 | 预期角色 |\n",
            "|----------|----------|----------|\n",
            "| ResNet50 视觉嵌入 | 模态消融 | 多模态输入基础 |\n",
            "| LCAD 伪报告弱监督 | QC 消融 + 主基线 | 无报告病例监督 |\n",
            "| RASA 节级对齐 | `no_section_alignment` | 报告-模态对齐 |\n",
            "| CIN2+ 风险头 | `no_risk_head` | 判别主任务 |\n",
            "| 标签一致性损失 | `no_label_consistency_loss` | 弱监督约束 |\n",
            "| 伪报告 QC 加权 | S4 五组 | 噪声伪报告过滤 |\n",
            "| OCT/阴道镜/临床指令 | S3 九组 | 模态贡献诊断 |\n\n",
            "## 输出文件\n\n",
            f"- `{TABLES / 'table_rasa_component_ablation.csv'}`\n",
            f"- `{TABLES / 'table_modality_ablation.csv'}`\n",
            f"- `{TABLES / 'table_lcad_qc_ablation.csv'}`\n",
            f"- `{MANUSCRIPT / 'S3_modality_ablation.csv'}` 等 manuscript 表\n",
            f"- `{PROJECT / 'outputs/publishable/figures/jbd_final/'}` 消融图\n",
        ]
    )
    AUDIT_MD.write_text("".join(lines), encoding="utf-8")
    return AUDIT_MD


def main():
    p = argparse.ArgumentParser(description="Run ablation re-eval + component audit")
    p.add_argument("--skip-train", action="store_true", default=True)
    p.add_argument("--train", action="store_true", help="Train ablation checkpoints in baselines/")
    p.add_argument("--force-retrain", action="store_true", help="Retrain even if checkpoint exists")
    p.add_argument("--unified-threshold", action="store_true", default=False, help="Val max-F1 threshold on test (Table 2 protocol)")
    p.add_argument("--baselines-only", action="store_true", default=False, help="Do not use checkpoint_aliases")
    p.add_argument("--full-budget", action="store_true", help="Use publishable training budget (150 steps/epoch)")
    p.add_argument("--eval-max-cases", type=int, default=288)
    p.add_argument("--aggregate", action="store_true", default=True)
    p.add_argument("--figures", action="store_true", default=True)
    args = p.parse_args()
    skip_train = not args.train

    t0 = time.time()
    cfg = load_jbd_config(PROJECT)
    if args.full_budget:
        cfg = {
            **cfg,
            "training": {**cfg.get("training", {}), "num_epochs": 5, "max_steps_per_epoch": 150},
        }
    manifest = PROJECT / cfg["manifest"]
    df = pd.read_csv(manifest)
    baselines_dir = PROJECT / "outputs/publishable/baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    common = dict(
        skip_train=skip_train,
        force_retrain=args.force_retrain,
        eval_max=args.eval_max_cases,
        unified_threshold=args.unified_threshold,
        baselines_only=args.baselines_only,
    )

    rasa = run_ablation_block(df, cfg, "rasa", cfg.get("rasa_ablations", {}), baselines_dir, **common)
    save_table(rasa, TABLES / "table_rasa_component_ablation.csv", TABLES / "table_rasa_component_ablation.md")

    qc = run_ablation_block(df, cfg, "qc", cfg.get("lcad_qc_ablations", {}), baselines_dir, **common)
    save_table(qc, TABLES / "table_lcad_qc_ablation.csv", TABLES / "table_lcad_qc_ablation.md")

    modality = run_ablation_block(df, cfg, "modality", cfg.get("modality_ablations", {}), baselines_dir, **common)
    save_table(modality, TABLES / "table_modality_ablation.csv", TABLES / "table_modality_ablation.md")

    if args.aggregate:
        import subprocess

        subprocess.run(
            [sys.executable, str(ROOT / "scripts/28_aggregate_manuscript_result_tables.py")],
            check=False,
            cwd=str(ROOT),
        )

    if args.figures:
        import subprocess

        subprocess.run(
            [sys.executable, str(ROOT / "scripts/30_regenerate_jbd_figures_seaborn.py")],
            check=False,
            cwd=str(ROOT),
        )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/35_generate_ablation_figures_and_summary.py")],
            check=False,
            cwd=str(ROOT),
        )

    elapsed = (time.time() - t0) / 60
    audit_path = write_audit(rasa, modality, qc, elapsed)

    summary = {
        "elapsed_minutes": elapsed,
        "rasa_ok": int((rasa["status"] == "ok").sum()) if "status" in rasa.columns else 0,
        "modality_ok": int((modality["status"] == "ok").sum()) if "status" in modality.columns else 0,
        "qc_ok": int((qc["status"] == "ok").sum()) if "status" in qc.columns else 0,
        "audit_md": str(audit_path),
        "skip_train": skip_train,
        "force_retrain": args.force_retrain,
        "unified_threshold": args.unified_threshold,
        "baselines_only": args.baselines_only,
        "full_budget": args.full_budget,
    }
    (LOG_DIR / "ablation_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Ablation audit done: %s", summary)
    print(json.dumps(summary, indent=2))
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
