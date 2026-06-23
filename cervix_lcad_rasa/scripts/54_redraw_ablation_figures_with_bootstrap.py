#!/usr/bin/env python3
"""Redraw ablation figures with paired-bootstrap annotations.

This script is intentionally separate from the historical gallery-style
ablation figures. It re-exports per-case risk scores from existing
publishable ablation checkpoints, computes paired AUROC bootstrap tests, and
draws compact manuscript-ready panels for modality, RASA-component, QC, and
alignment-weight ablations.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models_publishable.lcad_rasa_model import instr_vector, load_visual_emb
from src.supplementary.jbd_figure_typography import apply_arial_to_figure, setup_arial_rcparams
from src.supplementary.train_eval import build_model

PROJECT_FIGURES = ROOT.parent / "figures"
OUT_DIR = ROOT / "outputs/publishable/figures/ablation_revised"
PRED_DIR = ROOT / "outputs/publishable/predictions/ablation_revised"
TABLE_DIR = ROOT / "outputs/publishable/tables/manuscript"
SUMMARY_DIR = ROOT / "outputs/publishable/tables/ablation_revised"

TEXT = "#343434"
GRID = "#d6d6d6"
BLUE = "#2f5f8f"
LIGHT_BLUE = "#8fb8d8"
ORANGE = "#d9a066"
LIGHT_ORANGE = "#efd7b5"
RED = "#9e3f3a"
SALMON = "#d47f6f"
GREY = "#7f7f7f"
LIGHT_GREY = "#e8e8e8"

BOOTSTRAP_N = 2000
SEED = 20260615


def setup_theme() -> None:
    setup_arial_rcparams(
        {
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.fontsize": 8.0,
        }
    )
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})


def auc_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney AUC with explicit tie handling; robust for constant scores.
    greater = (pos[:, None] > neg[None, :]).sum()
    equal = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def bootstrap_auc_ci(y_true: np.ndarray, score: np.ndarray, *, n_boot: int = BOOTSTRAP_N, seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auc_rank(y[idx], s[idx]))
    if not vals:
        return auc_rank(y, s), float("nan"), float("nan")
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return auc_rank(y, s), float(lo), float(hi)


def paired_auc_bootstrap(
    y_true: np.ndarray,
    ref_score: np.ndarray,
    cmp_score: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=int)
    ref = np.asarray(ref_score, dtype=float)
    cmp = np.asarray(cmp_score, dtype=float)
    obs = auc_rank(y, cmp) - auc_rank(y, ref)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auc_rank(y[idx], cmp[idx]) - auc_rank(y[idx], ref[idx]))
    if not vals:
        return {
            "delta_auc_vs_ref": obs,
            "delta_auc_ci_low": float("nan"),
            "delta_auc_ci_high": float("nan"),
            "paired_bootstrap_p_two_sided": float("nan"),
        }
    arr = np.asarray(vals, dtype=float)
    if abs(obs) < 1e-12:
        p = 1.0
    else:
        p = 2.0 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
        p = min(1.0, max(1.0 / len(arr), p))
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return {
        "delta_auc_vs_ref": float(obs),
        "delta_auc_ci_low": float(lo),
        "delta_auc_ci_high": float(hi),
        "paired_bootstrap_p_two_sided": float(p),
    }


def p_label(p: float | None) -> str:
    if p is None or pd.isna(p):
        return "p n/a"
    p = float(p)
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def pretty_id(exp_id: str) -> str:
    mapping = {
        "oct_only": "OCT only",
        "colposcopy_only": "Colposcopy only",
        "instruction_only": "Clinical only",
        "oct_colposcopy": "OCT + colposcopy",
        "oct_instruction": "OCT + clinical",
        "colposcopy_instruction": "Colposcopy + clinical",
        "oct_colposcopy_instruction": "OCT + colposcopy + clinical",
        "full_with_fused": "Full visual-clinical",
        "full_without_fused": "Full visual-clinical, no fused token",
        "full_lcad_rasa": "MOSAIC--RASA backbone",
        "no_section_alignment": "No section alignment",
        "no_risk_head": "No risk head",
        "no_label_consistency_loss": "No label-consistency loss",
        "report_loss_only": "Report loss only",
        "section_alignment_only_auxiliary": "Section alignment only",
        "risk_head_only_auxiliary": "Risk-head auxiliary only",
        "pseudo_all_no_qc": "No QC filtering",
        "pseudo_qc_pass_only": "QC-pass only",
        "pseudo_confidence_only": "Pseudo-confidence only",
        "pseudo_qc_score_only": "QC-score only",
        "pseudo_qc_confidence_weighted": "QC + confidence weighting",
    }
    return mapping.get(exp_id, exp_id.replace("_", " "))


def load_manifest() -> pd.DataFrame:
    cfg = yaml.safe_load((ROOT / "configs/jbd_supplementary_experiments.yaml").read_text(encoding="utf-8"))
    df = pd.read_csv(ROOT / cfg["manifest"])
    test = df[df["split"].eq("test")].copy()
    if len(test) != 288:
        raise RuntimeError(f"Expected locked test n=288, got n={len(test)}")
    return test


def score_checkpoint(exp_id: str, test: pd.DataFrame) -> pd.DataFrame | None:
    ckpt = ROOT / "outputs/publishable/baselines" / exp_id / "best.ckpt"
    if not ckpt.is_file():
        return None
    state = torch.load(ckpt, map_location="cpu")
    spec = state.get("spec") or {}
    model = build_model(spec)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    risk_available = model.risk_head is not None
    rows = []

    def _emb_path(value: object) -> str:
        p = Path(str(value))
        return str(p if p.is_absolute() else ROOT / p)

    with torch.no_grad():
        for _, row in test.iterrows():
            label = int(row["binary_label"])
            if risk_available:
                oct_e = torch.tensor(load_visual_emb(_emb_path(row.get("oct_embedding_path", ""))), dtype=torch.float32).unsqueeze(0)
                col_e = torch.tensor(load_visual_emb(_emb_path(row.get("colposcopy_embedding_path", ""))), dtype=torch.float32).unsqueeze(0)
                fus_e = torch.tensor(load_visual_emb(_emb_path(row.get("fused_visual_embedding_path", ""))), dtype=torch.float32).unsqueeze(0)
                instr = torch.tensor(instr_vector(row.to_dict()), dtype=torch.float32).unsqueeze(0)
                lab = torch.tensor([label], dtype=torch.long)
                h0 = model.encode_modalities(oct_e, col_e, fus_e, instr, lab)
                risk = float(torch.sigmoid(model.risk_head(h0)).item())
            else:
                risk = 0.5
            rows.append(
                {
                    "case_id": row["case_id"],
                    "center": row.get("center_id", row.get("center", "")),
                    "split": row["split"],
                    "y_true_cin2plus": label,
                    "risk_score": risk,
                    "risk_available": int(risk_available),
                    "source_checkpoint": str(ckpt),
                }
            )
    pred = pd.DataFrame(rows)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_csv(PRED_DIR / f"{exp_id}_test_predictions.csv", index=False)
    return pred


def load_or_score_predictions(exp_ids: list[str], test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for exp_id in exp_ids:
        # Always re-export. The manifest stores embedding paths relative to
        # cervix_lcad_rasa, so stale caches from a different cwd can silently
        # produce a different AUROC.
        pred = score_checkpoint(exp_id, test)
        if pred is not None:
            out[exp_id] = pred
    return out


def enrich_block(table: pd.DataFrame, preds: dict[str, pd.DataFrame], ref_id: str, block: str) -> pd.DataFrame:
    rows = []
    ref = preds.get(ref_id)
    if ref is None:
        raise RuntimeError(f"Missing reference prediction for {ref_id}")
    ref_auc, ref_lo, ref_hi = bootstrap_auc_ci(ref["y_true_cin2plus"].to_numpy(), ref["risk_score"].to_numpy())
    for _, r in table.iterrows():
        exp_id = str(r["experiment_id"])
        pred = preds.get(exp_id)
        row = r.to_dict()
        row["block"] = block
        row["label"] = pretty_id(exp_id)
        row["reference_id"] = ref_id
        row["reference_auc"] = ref_auc
        if pred is None:
            row.update(
                {
                    "auc_from_predictions": float("nan"),
                    "auc_ci_low": float("nan"),
                    "auc_ci_high": float("nan"),
                    "delta_auc_vs_ref": float("nan"),
                    "delta_auc_ci_low": float("nan"),
                    "delta_auc_ci_high": float("nan"),
                    "paired_bootstrap_p_two_sided": float("nan"),
                    "risk_available": float("nan"),
                    "n_paired": 0,
                }
            )
        else:
            merged = ref[["case_id", "y_true_cin2plus", "risk_score"]].merge(
                pred[["case_id", "risk_score", "risk_available"]],
                on="case_id",
                suffixes=("_ref", "_cmp"),
            )
            y = merged["y_true_cin2plus"].to_numpy(dtype=int)
            cmp_score = merged["risk_score_cmp"].to_numpy(dtype=float)
            auc, lo, hi = bootstrap_auc_ci(y, cmp_score)
            boot = paired_auc_bootstrap(y, merged["risk_score_ref"].to_numpy(dtype=float), cmp_score)
            row.update(
                {
                    "auc_from_predictions": auc,
                    "auc_ci_low": lo,
                    "auc_ci_high": hi,
                    **boot,
                    "risk_available": int(merged["risk_available"].iloc[0]),
                    "n_paired": len(merged),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_prediction_index(preds: dict[str, pd.DataFrame]) -> None:
    rows = []
    for exp_id, pred in preds.items():
        rows.append(
            {
                "experiment_id": exp_id,
                "n": len(pred),
                "positives": int(pred["y_true_cin2plus"].sum()),
                "risk_available": int(pred["risk_available"].iloc[0]),
                "path": str(PRED_DIR / f"{exp_id}_test_predictions.csv"),
            }
        )
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("experiment_id").to_csv(SUMMARY_DIR / "ablation_prediction_index.csv", index=False)


def panel_modality(ax: plt.Axes, df: pd.DataFrame) -> None:
    d = df.sort_values("auc", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    ref_auc = float(d.loc[d["experiment_id"].eq("full_with_fused"), "auc"].iloc[0])
    d["delta_auc_table"] = d["auc"] - ref_auc
    colors = [RED if x < -0.05 else (BLUE if eid == "full_with_fused" else LIGHT_BLUE) for x, eid in zip(d["delta_auc_table"], d["experiment_id"])]
    ax.hlines(y, ref_auc, d["auc"], color=LIGHT_GREY, lw=3, zorder=1)
    ax.scatter(d["auc"], y, s=48, c=colors, edgecolor=TEXT, linewidth=0.45, zorder=4)
    ax.axvline(ref_auc, color=TEXT, ls="--", lw=0.8, alpha=0.6)
    for yi, r in enumerate(d.itertuples()):
        ax.text(0.875, yi, p_label(r.paired_bootstrap_p_two_sided), va="center", ha="left", fontsize=6.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlim(0.66, 0.92)
    ax.set_xlabel("AUROC (manuscript table)")
    ax.set_title("A  Input-modality ablation")
    ax.grid(axis="x", alpha=0.55)
    ax.grid(axis="y", visible=False)
    ax.text(0.835, len(d) - 0.15, "ref", ha="center", va="bottom", fontsize=7.0)


def panel_rasa(ax: plt.Axes, df: pd.DataFrame) -> None:
    order = [
        "no_risk_head",
        "report_loss_only",
        "section_alignment_only_auxiliary",
        "no_section_alignment",
        "no_label_consistency_loss",
        "risk_head_only_auxiliary",
        "full_lcad_rasa",
    ]
    d = df.set_index("experiment_id").loc[[x for x in order if x in set(df["experiment_id"])]].reset_index()
    ref_auc = float(d.loc[d["experiment_id"].eq("full_lcad_rasa"), "auc"].iloc[0])
    d["delta_auc_table"] = d["auc"] - ref_auc
    y = np.arange(len(d))
    colors = [RED if x < -0.08 else (BLUE if eid == "full_lcad_rasa" else ORANGE) for x, eid in zip(d["delta_auc_table"], d["experiment_id"])]
    ax.axvline(0, color=TEXT, lw=0.85, alpha=0.75)
    ax.hlines(y, 0, d["delta_auc_table"], color=LIGHT_GREY, lw=3, zorder=1)
    ax.scatter(d["delta_auc_table"], y, s=48, c=colors, edgecolor=TEXT, linewidth=0.45, zorder=3)
    for yi, r in enumerate(d.itertuples()):
        suffix = "constant risk" if int(r.risk_available) == 0 else p_label(r.paired_bootstrap_p_two_sided)
        ax.text(0.021, yi, suffix, va="center", ha="left", fontsize=6.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlim(-0.39, 0.10)
    ax.set_xlabel("Delta AUROC vs MOSAIC--RASA backbone (table)")
    ax.set_title("B  RASA-component ablation")
    ax.grid(axis="x", alpha=0.55)
    ax.grid(axis="y", visible=False)


def panel_qc(ax: plt.Axes, df: pd.DataFrame) -> None:
    ref_auc = float(df.loc[df["experiment_id"].eq("pseudo_qc_confidence_weighted"), "auc"].iloc[0])
    d = df.copy()
    d["delta_auc_table"] = d["auc"] - ref_auc
    d = d.sort_values("delta_auc_table", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    colors = [BLUE if eid == "pseudo_qc_confidence_weighted" else LIGHT_BLUE for eid in d["experiment_id"]]
    ax.axvline(0, color=TEXT, lw=0.85, alpha=0.75)
    ax.hlines(y, 0, d["delta_auc_table"], color=LIGHT_GREY, lw=3, zorder=1)
    ax.scatter(d["delta_auc_table"], y, s=52, c=colors, edgecolor=TEXT, linewidth=0.45, zorder=3)
    for yi, r in enumerate(d.itertuples()):
        ax.text(0.0028, yi, p_label(r.paired_bootstrap_p_two_sided), va="center", ha="left", fontsize=6.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlim(-0.005, 0.018)
    ax.set_xlabel("Delta AUROC vs QC + confidence weighting (table)")
    ax.set_title("C  LCAD QC / weighting ablation")
    ax.grid(axis="x", alpha=0.55)
    ax.grid(axis="y", visible=False)


def panel_lambda(ax: plt.Axes) -> None:
    p = TABLE_DIR / "S1_rasa_lambda_align_sweep.csv"
    if not p.is_file():
        ax.axis("off")
        return
    d = pd.read_csv(p).sort_values("lambda_align")
    x = np.arange(len(d))
    ax.plot(x, d["auc"], color=BLUE, lw=1.8, marker="o", ms=4.8, mec=TEXT, mfc="white")
    best = d.loc[d["auc"].idxmax()]
    best_idx = int(d.index.get_loc(best.name))
    ax.scatter([best_idx], [best["auc"]], s=78, marker="D", c=ORANGE, edgecolor=TEXT, linewidth=0.55, zorder=5)
    ax.axhline(float(d[d["lambda_align"].eq(0)]["auc"].iloc[0]), color=GREY, lw=0.85, ls="--")
    ax.fill_between(x, d["auc"], float(d["auc"].min()) - 0.001, color=LIGHT_BLUE, alpha=0.22)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in d["lambda_align"]])
    ax.set_ylim(0.758, 0.790)
    ax.set_xlabel("Section-alignment weight")
    ax.set_ylabel("Held-out AUROC")
    ax.set_title("D  Section-alignment weight sweep")
    ax.grid(axis="both", alpha=0.45)
    ax.text(best_idx + 0.15, float(best["auc"]) + 0.002, f"best = {best['auc']:.3f}", fontsize=7.3, va="bottom")


def draw_composite(modality: pd.DataFrame, rasa: pd.DataFrame, qc: pd.DataFrame) -> Path:
    setup_theme()
    fig = plt.figure(figsize=(12.6, 8.0))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.08, 1.0],
        height_ratios=[1.16, 1.0],
        left=0.125,
        right=0.975,
        top=0.925,
        bottom=0.105,
        wspace=0.45,
        hspace=0.50,
    )
    panel_modality(fig.add_subplot(gs[0, 0]), modality)
    panel_rasa(fig.add_subplot(gs[0, 1]), rasa)
    panel_qc(fig.add_subplot(gs[1, 0]), qc)
    panel_lambda(fig.add_subplot(gs[1, 1]))
    fig.suptitle("Ablation evidence with paired-bootstrap AUROC audits", fontsize=12.2, fontweight="bold", y=0.985)
    fig.text(
        0.125,
        0.026,
        "Points use manuscript-table AUROC values; p values use paired bootstrap on re-exported matched test predictions (n=288). Variants without a risk head use constant risk = 0.5.",
        fontsize=7.2,
        color=GREY,
        ha="left",
    )
    return save_all(fig, "Figure_ablation_revised_modality_rasa_qc")


def draw_individual(block_df: pd.DataFrame, block: str) -> Path:
    setup_theme()
    if block == "modality":
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        panel_modality(ax, block_df)
        stem = "fig_modality_ablation_revised_bootstrap"
    elif block == "rasa":
        fig, ax = plt.subplots(figsize=(7.3, 4.3))
        panel_rasa(ax, block_df)
        stem = "fig_rasa_component_revised_bootstrap"
    elif block == "qc":
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        panel_qc(ax, block_df)
        stem = "fig_lcad_qc_revised_bootstrap"
    else:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        panel_lambda(ax)
        stem = "fig_rasa_lambda_revised"
    fig.tight_layout()
    return save_all(fig, stem)


def save_all(fig: plt.Figure, stem: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_FIGURES.mkdir(parents=True, exist_ok=True)
    apply_arial_to_figure(fig)
    out_base = OUT_DIR / stem
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(out_base.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    for suffix in (".png", ".pdf", ".svg"):
        shutil.copy2(out_base.with_suffix(suffix), PROJECT_FIGURES / f"{stem}{suffix}")
    return out_base.with_suffix(".pdf")


def write_qa(paths: list[Path], stats: pd.DataFrame) -> None:
    from PIL import Image

    def _markdown(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in df.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append("" if pd.isna(v) else f"{v:.4g}")
                else:
                    vals.append(str(v))
            out.append("| " + " | ".join(vals) + " |")
        return "\n".join(out)

    lines = [
        "# Revised Ablation Figure QA\n\n",
        "## Files\n\n",
        "| File | Pixels | Status |\n",
        "|---|---:|---|\n",
    ]
    for path in paths:
        png = path.with_suffix(".png")
        if not png.is_file():
            lines.append(f"| `{png}` | missing | failed |\n")
            continue
        im = Image.open(png)
        status = "pass" if im.size[0] >= 1600 and im.size[1] >= 1000 else "check"
        lines.append(f"| `{png}` | {im.size[0]} x {im.size[1]} | {status} |\n")
    lines.extend(
        [
            "\n## Statistical Audit\n\n",
            "- Figure points use the manuscript-table AUROC values to remain consistent with current LaTeX tables.\n",
            "- Paired bootstrap tests were recomputed from exported per-case risk scores.\n",
            "- Labels were taken from the locked test split only; model selection was not performed here.\n",
            "- RASA variants without a risk head are plotted as constant-risk fallback outputs.\n\n",
            "## Bootstrap Summary\n\n",
            _markdown(stats[
                [
                    "block",
                    "experiment_id",
                    "reference_id",
                    "auc_from_predictions",
                    "auc_ci_low",
                    "auc_ci_high",
                    "delta_auc_vs_ref",
                    "paired_bootstrap_p_two_sided",
                    "risk_available",
                    "n_paired",
                ]
            ]),
            "\n",
        ]
    )
    (OUT_DIR / "ABlATION_REVISED_FIGURE_QA.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    test = load_manifest()
    s3 = pd.read_csv(TABLE_DIR / "S3_modality_ablation.csv")
    s4 = pd.read_csv(TABLE_DIR / "S4_lcad_qc_ablation.csv")
    s5 = pd.read_csv(TABLE_DIR / "S5_rasa_component_ablation.csv")
    all_ids = sorted(set(s3["experiment_id"]) | set(s4["experiment_id"]) | set(s5["experiment_id"]))
    preds = load_or_score_predictions(all_ids, test)
    write_prediction_index(preds)

    modality = enrich_block(s3, preds, "full_with_fused", "modality")
    qc = enrich_block(s4, preds, "pseudo_qc_confidence_weighted", "qc")
    rasa = enrich_block(s5, preds, "full_lcad_rasa", "rasa")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    modality.to_csv(SUMMARY_DIR / "S3_modality_ablation_paired_bootstrap.csv", index=False)
    qc.to_csv(SUMMARY_DIR / "S4_lcad_qc_ablation_paired_bootstrap.csv", index=False)
    rasa.to_csv(SUMMARY_DIR / "S5_rasa_component_ablation_paired_bootstrap.csv", index=False)
    all_stats = pd.concat([modality, qc, rasa], ignore_index=True)
    all_stats.to_csv(SUMMARY_DIR / "ablation_paired_bootstrap_summary.csv", index=False)

    paths = [
        draw_composite(modality, rasa, qc),
        draw_individual(modality, "modality"),
        draw_individual(rasa, "rasa"),
        draw_individual(qc, "qc"),
        draw_individual(pd.DataFrame(), "lambda"),
    ]
    write_qa(paths, all_stats)
    manifest = {
        "figures": [str(p) for p in paths],
        "project_figure_copies": [str(PROJECT_FIGURES / p.name) for p in paths],
        "predictions": str(PRED_DIR),
        "tables": str(SUMMARY_DIR),
        "bootstrap_n": BOOTSTRAP_N,
        "seed": SEED,
    }
    (OUT_DIR / "ablation_revised_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
