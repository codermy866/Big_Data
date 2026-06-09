#!/usr/bin/env python3
"""Aggregate ablation tables + generate palette figures + master summary MD."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supplementary.jbd_ablation_figures import JBD_PALETTE_HEX, generate_all_ablation_figures
PROJECT = ROOT
TABLES = PROJECT / "outputs/publishable/tables"
MANUSCRIPT = TABLES / "manuscript"
FIG_ABL = PROJECT / "outputs/publishable/figures/ablation"
SUMMARY = PROJECT.parent / "JBD_ABLATION_RESULTS_SUMMARY.md"


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    hdr = "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return hdr + "\n".join(rows) + "\n"


def write_summary_md() -> Path:
    s3 = pd.read_csv(MANUSCRIPT / "S3_modality_ablation.csv")
    s4 = pd.read_csv(MANUSCRIPT / "S4_lcad_qc_ablation.csv")
    s5 = pd.read_csv(MANUSCRIPT / "S5_rasa_component_ablation.csv")
    audit = (PROJECT.parent / "JBD_COMPONENT_ABLATION_AUDIT.md").read_text(encoding="utf-8") if (PROJECT.parent / "JBD_COMPONENT_ABLATION_AUDIT.md").is_file() else ""

    lines = [
        "# JBD 消融实验结果汇总（统一阈值协议）\n",
        f"更新: {datetime.now(timezone.utc).isoformat()}\n\n",
        "## 评估协议\n",
        "- **AUROC**：测试集 n=288，threshold-free\n",
        "- **F1 / Sens / Spec**：验证集 max-F1 选阈 → 测试集（与 Table 2 一致）\n",
        "- **训练**：`baselines/{exp_id}/best.ckpt` 独立重训（`--force-retrain --baselines-only --full-budget`）\n",
        f"- **色板**：{', '.join(JBD_PALETTE_HEX)}\n\n",
        "---\n\n",
        "## Table S3 — 模态消融\n\n",
        _md_table(s3, list(s3.columns)),
        "\n**图**: `figures/ablation/AblationFig_S3_modality_barplot.{png,pdf}`\n\n",
        "---\n\n",
        "## Table S4 — LCAD QC / 伪报告权重\n\n",
        _md_table(s4, list(s4.columns)),
        "\n**图**: `figures/ablation/AblationFig_S4_qc_catplot.{png,pdf}`\n\n",
        "---\n\n",
        "## Table S5 — RASA 组件消融\n\n",
        _md_table(s5, list(s5.columns)),
        "\n**图**: `figures/ablation/AblationFig_S5_rasa_delta_bars.{png,pdf}`, `AblationFig_S5_auc_f1_scatter.{png,pdf}`\n\n",
        "---\n\n",
        "## 组件必要性结论（简表）\n\n",
        "| 组件 | 结论 | 依据 |\n",
        "|------|------|------|\n",
        "| CIN2+ 风险头 | **必要** | 移除后 AUROC→0.50 |\n",
        "| 报告 CE + 风险联合 | **必要** | `report_loss_only` 崩溃 |\n",
        "| RASA 节级对齐 | **有帮助** | 主要改善 F1/结构；AUROC 边际 |\n",
        "| 标签一致性损失 | **边际** | ΔAUROC≈0；F1 略升 |\n",
        "| 多模态（非 instruction-only） | **必要** | 单 instruction AUROC≈0.50 |\n",
        "| 阴道镜 + 临床 | **最强单模态组合** | colposcopy_instruction AUC 最高 |\n",
        "| fused 视觉分支 | **边际** | full_with ≈ full_without fused |\n",
        "| QC 加权 | **边际** | 五组 AUROC/F1 几乎相同 |\n\n",
        "---\n\n",
        "## 文件目录\n\n",
        "```\n",
        "cervix_lcad_rasa/outputs/publishable/\n",
        "├── tables/\n",
        "│   ├── table_modality_ablation.csv\n",
        "│   ├── table_lcad_qc_ablation.csv\n",
        "│   ├── table_rasa_component_ablation.csv\n",
        "│   └── manuscript/S3_modality_ablation.csv\n",
        "│       manuscript/S4_lcad_qc_ablation.csv\n",
        "│       manuscript/S5_rasa_component_ablation.csv\n",
        "├── figures/ablation/          ← 消融专用 Seaborn 图\n",
        "│   ├── AblationFig_S3_modality_barplot.png\n",
        "│   ├── AblationFig_S4_qc_catplot.png\n",
        "│   ├── AblationFig_S5_rasa_delta_bars.png\n",
        "│   ├── AblationFig_S5_auc_f1_scatter.png\n",
        "│   ├── AblationFig_combined_heatmap.png\n",
        "│   └── ABLATION_FIGURE_INDEX.md\n",
        "└── figures/jbd_final/         ← 全局图（含 fig_modality_ablation_* 等）\n",
        "```\n\n",
        "## 组件审计全文\n\n",
        "见 `JBD_COMPONENT_ABLATION_AUDIT.md`\n",
    ]
    SUMMARY.write_text("".join(lines), encoding="utf-8")
    return SUMMARY


def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/28_aggregate_manuscript_result_tables.py")], cwd=str(ROOT), check=False)
    out = generate_all_ablation_figures(PROJECT)
    summary = write_summary_md()
    meta = {
        "figures_dir": str(out),
        "summary_md": str(summary),
        "palette": JBD_PALETTE_HEX,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (PROJECT / "outputs/publishable/logs/ablation_summary_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
