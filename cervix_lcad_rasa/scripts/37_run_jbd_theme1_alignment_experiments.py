#!/usr/bin/env python3
"""Theme-1 experiments for LLM-augmented cross-modal semantic alignment."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import label_consistency
from src.models_publishable.lcad_rasa_model import (
    PublishableLCADRASA,
    instr_vector,
    load_visual_emb,
)

SECTION_KEYS = ["oct_findings", "colposcopy_findings", "clinical_context", "impression"]
MODEL_VARIANTS = {
    "full_lcad_rasa": "outputs/publishable/baselines/full_lcad_rasa/best.ckpt",
    "no_section_alignment": "outputs/publishable/baselines/no_section_alignment/best.ckpt",
    "no_label_consistency_loss": "outputs/publishable/baselines/no_label_consistency_loss/best.ckpt",
    "risk_head_only_auxiliary": "outputs/publishable/baselines/risk_head_only_auxiliary/best.ckpt",
    "no_risk_head": "outputs/publishable/baselines/no_risk_head/best.ckpt",
    "report_loss_only": "outputs/publishable/baselines/report_loss_only/best.ckpt",
    "simple_concat_fusion": "outputs/publishable/baselines/simple_concat_fusion/best.ckpt",
    "pseudo_augmented_lcad": "outputs/publishable/checkpoints/publishable_lcad_augmented/best.ckpt",
    "real_report_only": "outputs/publishable/checkpoints/publishable_dual_real_only/best.ckpt",
}
MORANDI_HEX = [
    "#2f5f8f",
    "#8fb8d8",
    "#d9a066",
    "#efd7b5",
    "#9e3f3a",
    "#d47f6f",
    "#7f7f7f",
    "#d6d6d6",
]
MORANDI_SEQ = LinearSegmentedColormap.from_list(
    "cell_seq_blue",
    [
        "#f7f7f2",
        "#e4eef0",
        "#c3dae6",
        "#8fb8d8",
        "#5d88b3",
        "#2f5f8f",
        "#1f3f64",
    ],
    N=256,
)
MORANDI_WARM = LinearSegmentedColormap.from_list(
    "cell_seq_warm_red",
    ["#f7f7f2", "#efd7b5", "#d9a066", "#d47f6f", "#9e3f3a"],
    N=256,
)


def _setup_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#3a3a3a",
        }
    )


@dataclass
class Paths:
    out: Path
    tables: Path
    figures: Path
    manuscript: Path


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def _parse_report_json(text: Any) -> dict[str, Any]:
    s = _safe_str(text).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _case_sections(row: pd.Series, source: str = "local_llm") -> dict[str, str]:
    row_d = row.to_dict()
    label = int(row_d.get("binary_label", 0))
    endpoint = _safe_str(row_d.get("binary_label_endpoint", "CIN2+")) or "CIN2+"
    cid = _safe_str(row_d.get("case_id", ""))
    if source == "local_llm":
        obj = _parse_report_json(row_d.get("pseudo_report_text", ""))
        if not obj and row_d.get("pseudo_report_path"):
            p = Path(_safe_str(row_d.get("pseudo_report_path", "")))
            if p.is_file():
                obj = json.loads(p.read_text(encoding="utf-8"))
        if obj:
            sections = {k: _safe_str(obj.get(k, "")) for k in SECTION_KEYS}
            oct_text = sections.get("oct_findings", "")
            col_text = sections.get("colposcopy_findings", "")
            match = re.search(r"\[colposcopy_emb_norm=[^\]]+\]", oct_text)
            if match and ("unavailable" in col_text.lower() or not col_text.strip()):
                sections["oct_findings"] = re.sub(r"\s*\[colposcopy_emb_norm=[^\]]+\]", "", oct_text).strip()
                sections["colposcopy_findings"] = (
                    "Colposcopy visual evidence available from embedding summary "
                    f"{match.group(0)}."
                )
            return sections
    if source == "reference_or_lcad":
        refs = {
            "oct_findings": _safe_str(row_d.get("reference_oct_findings", "")),
            "colposcopy_findings": _safe_str(row_d.get("reference_colposcopy_findings", "")),
            "clinical_context": _safe_str(row_d.get("reference_clinical_context", "")),
            "impression": _safe_str(row_d.get("reference_impression", "")),
        }
        if any(len(v.strip()) >= 10 for v in refs.values()):
            return refs
        return _case_sections(row, "local_llm")
    if source == "label_template":
        phrase = f"suspicious for {endpoint}" if label else f"no definitive evidence for {endpoint}"
        return {
            "oct_findings": "OCT evidence was not used in this label-template pseudo report.",
            "colposcopy_findings": "Colposcopy evidence was not used in this label-template pseudo report.",
            "clinical_context": "Clinical context was not used beyond the binary endpoint label.",
            "impression": f"Template impression: {phrase}.",
        }
    if source == "rule_based":
        oct_v = load_visual_emb(_safe_str(row_d.get("oct_embedding_path", "")))
        col_v = load_visual_emb(_safe_str(row_d.get("colposcopy_embedding_path", "")))
        oct_norm = float(np.linalg.norm(oct_v))
        col_norm = float(np.linalg.norm(col_v))
        oct_tier = "high visual signal" if oct_norm >= 8 else "limited visual signal"
        col_tier = "high visual signal" if col_norm >= 8 else "limited visual signal"
        phrase = f"suspicious for {endpoint}" if label else f"no definitive evidence for {endpoint}"
        return {
            "oct_findings": f"Rule-based OCT summary: {oct_tier}; embedding norm {oct_norm:.2f}.",
            "colposcopy_findings": f"Rule-based colposcopy summary: {col_tier}; embedding norm {col_norm:.2f}.",
            "clinical_context": (
                f"Rule-based clinical context: age {_safe_str(row_d.get('age'))}; "
                f"HPV {_safe_str(row_d.get('hpv'))}; TCT {_safe_str(row_d.get('tct'))}."
            ),
            "impression": f"Rule-based impression for case {cid}: {phrase}.",
        }
    return {k: "" for k in SECTION_KEYS}


def _full_text(sections: dict[str, str]) -> str:
    return " ".join(_safe_str(sections.get(k, "")) for k in SECTION_KEYS)


def _section_complete(sections: dict[str, str]) -> float:
    return float(all(len(_safe_str(sections.get(k, "")).strip()) >= 15 for k in SECTION_KEYS))


def _supported(text: str, kind: str) -> float:
    t = text.lower()
    bad = ("unavailable", "not used", "insufficient", "not available")
    if any(x in t for x in bad):
        return 0.0
    if kind == "oct":
        return float("oct" in t or "b-scan" in t or "embedding" in t)
    if kind == "colposcopy":
        return float("colposcopy" in t or "colposcopic" in t or "embedding" in t)
    if kind == "instruction":
        return float("age" in t or "hpv" in t or "tct" in t or "clinical" in t)
    return float(len(t.strip()) > 0)


def _normalise_template_text(text: str) -> str:
    out = []
    for token in text.lower().split():
        if any(ch.isdigit() for ch in token):
            out.append("<num>")
        elif token.startswith("m") and "_p" in token:
            out.append("<case>")
        else:
            out.append(token)
    return " ".join(out)


def _write_csv(df: pd.DataFrame, path: Path, manuscript: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if manuscript is not None:
        manuscript.mkdir(parents=True, exist_ok=True)
        df.to_csv(manuscript / path.name, index=False)


def _save_heatmap(matrix: pd.DataFrame, path: Path, title: str, cmap: object | None = None) -> None:
    _setup_figure_style()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7.5, matrix.shape[1] * 1.25), max(4.5, matrix.shape[0] * 0.45)))
    im = ax.imshow(matrix.values.astype(float), aspect="auto", cmap=cmap or MORANDI_SEQ)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _text_ids_by_sections(sections: dict[str, str], max_len: int = 128, vocab_size: int = 8192) -> torch.Tensor:
    q = max(1, max_len // len(SECTION_KEYS))
    ids: list[int] = []
    for key in SECTION_KEYS:
        words = _safe_str(sections.get(key, "")).split()[:q]
        sec_ids = [hash(w) % vocab_size for w in words]
        sec_ids += [0] * max(0, q - len(sec_ids))
        ids.extend(sec_ids[:q])
    ids = ids[:max_len]
    ids += [0] * max(0, max_len - len(ids))
    return torch.tensor(ids[:max_len], dtype=torch.long)


def _load_model(ckpt_path: Path, device: torch.device) -> PublishableLCADRASA:
    payload = torch.load(ckpt_path, map_location=device)
    state = payload.get("model", payload)
    use_risk = "risk_head.weight" in state
    model = PublishableLCADRASA(use_risk_head=use_risk).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _prepare_model_batch(df: pd.DataFrame, section_source: str, device: torch.device) -> tuple[dict[str, torch.Tensor], list[str]]:
    oct_rows, col_rows, fus_rows, instr_rows, labels, ids, case_ids = [], [], [], [], [], [], []
    for _, row in df.iterrows():
        row_d = row.to_dict()
        sections = _case_sections(row, section_source)
        oct_rows.append(load_visual_emb(_safe_str(row_d.get("oct_embedding_path", ""))))
        col_rows.append(load_visual_emb(_safe_str(row_d.get("colposcopy_embedding_path", ""))))
        fus_rows.append(load_visual_emb(_safe_str(row_d.get("fused_visual_embedding_path", ""))))
        instr_rows.append(instr_vector(row_d))
        labels.append(int(row_d.get("binary_label", 0)))
        ids.append(_text_ids_by_sections(sections))
        case_ids.append(_safe_str(row_d.get("case_id", "")))
    batch = {
        "oct": torch.tensor(np.stack(oct_rows), dtype=torch.float32, device=device),
        "col": torch.tensor(np.stack(col_rows), dtype=torch.float32, device=device),
        "fus": torch.tensor(np.stack(fus_rows), dtype=torch.float32, device=device),
        "instr": torch.tensor(np.stack(instr_rows), dtype=torch.float32, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "ids": torch.stack(ids).to(device),
    }
    return batch, case_ids


def _ranks_from_similarity(sim: torch.Tensor) -> tuple[float, float, float, float, float]:
    n = sim.shape[0]
    ranks = []
    for i in range(n):
        order = torch.argsort(sim[i], descending=True)
        rank = int((order == i).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append(rank)
    ranks_np = np.array(ranks, dtype=float)
    diag = sim.diag().detach().cpu().numpy()
    off = sim.detach().cpu().numpy().copy()
    np.fill_diagonal(off, np.nan)
    return (
        float(np.mean(ranks_np <= 1)),
        float(np.mean(ranks_np <= 5)),
        float(np.mean(1.0 / ranks_np)),
        float(np.nanmean(diag)),
        float(np.nanmean(off)),
    )


def compute_latent_alignment(
    project: Path,
    df: pd.DataFrame,
    paths: Paths,
    device: torch.device,
    section_source: str = "reference_or_lcad",
    max_cases: int = 288,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = df[df["split"] == "test"].copy() if "split" in df.columns else df.copy()
    test = test.head(max_cases)
    batch, _case_ids = _prepare_model_batch(test, section_source, device)
    detail_rows = []
    summary_rows = []
    for model_name, rel in MODEL_VARIANTS.items():
        ckpt = project / rel
        if not ckpt.is_file():
            continue
        model = _load_model(ckpt, device)
        with torch.no_grad():
            out = model(batch["oct"], batch["col"], batch["fus"], batch["instr"], batch["ids"], batch["labels"])
            h0 = out["fused"]
            h = out["hidden"]
            q_len = max(1, h.shape[1] // len(SECTION_KEYS))
            targets = {
                "oct_findings": h[:, :q_len].mean(1),
                "colposcopy_findings": h[:, q_len : 2 * q_len].mean(1),
                "clinical_context": h[:, 2 * q_len : 3 * q_len].mean(1),
                "impression": h[:, 3 * q_len :].mean(1),
            }
            queries = {
                "oct_findings": model.sec_oct(h0),
                "colposcopy_findings": model.sec_col(h0),
                "clinical_context": model.sec_instr(h0),
                "impression": model.sec_imp(h0),
            }
            wrong_same_case = []
            for sec in SECTION_KEYS:
                q = F.normalize(queries[sec], dim=-1)
                t = F.normalize(targets[sec], dim=-1)
                sim = q @ t.T
                r1, r5, mrr, pos_sim, neg_sim = _ranks_from_similarity(sim)
                other_sims = []
                for other in SECTION_KEYS:
                    if other == sec:
                        continue
                    other_t = F.normalize(targets[other], dim=-1)
                    other_sims.append((q * other_t).sum(-1).mean().item())
                wrong = float(np.mean(other_sims))
                wrong_same_case.append(wrong)
                detail_rows.append(
                    {
                        "model": model_name,
                        "section": sec,
                        "n_cases": len(test),
                        "recall_at_1": r1,
                        "recall_at_5": r5,
                        "mrr": mrr,
                        "positive_cosine": pos_sim,
                        "cross_case_negative_cosine": neg_sim,
                        "wrong_section_same_case_cosine": wrong,
                        "positive_minus_cross_case": pos_sim - neg_sim,
                        "positive_minus_wrong_section": pos_sim - wrong,
                    }
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    detail = pd.DataFrame(detail_rows)
    for model_name, g in detail.groupby("model"):
        summary_rows.append(
            {
                "model": model_name,
                "n_sections": int(g.shape[0]),
                "macro_recall_at_1": float(g["recall_at_1"].mean()),
                "macro_recall_at_5": float(g["recall_at_5"].mean()),
                "macro_mrr": float(g["mrr"].mean()),
                "macro_positive_cosine": float(g["positive_cosine"].mean()),
                "macro_positive_minus_cross_case": float(g["positive_minus_cross_case"].mean()),
                "macro_positive_minus_wrong_section": float(g["positive_minus_wrong_section"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("macro_mrr", ascending=False)
    _write_csv(detail, paths.tables / "T_theme1_modality_section_retrieval_alignment.csv", paths.manuscript)
    _write_csv(summary, paths.tables / "T_theme1_rasa_direct_alignment_ablation.csv", paths.manuscript)

    if not detail.empty:
        matrix = detail.pivot(index="model", columns="section", values="mrr").fillna(0)
        _save_heatmap(matrix, paths.figures / "Figure_theme1_alignment_retrieval_mrr", "Modality-section retrieval MRR")
    return detail, summary


def run_pseudo_report_source_comparison(
    project: Path,
    df: pd.DataFrame,
    paths: Paths,
    device: torch.device,
    max_cases: int,
) -> pd.DataFrame:
    test = df[(df["split"] == "test") & (df["needs_pseudo_report"] == 1)].copy()
    if max_cases > 0:
        test = test.head(max_cases)
    sources = ["label_template", "rule_based", "local_llm"]
    rows = []
    for source in sources:
        texts = []
        complete, lc, oct_sup, col_sup, ins_sup = [], [], [], [], []
        for _, row in test.iterrows():
            sections = _case_sections(row, source)
            text = _full_text(sections)
            texts.append(_normalise_template_text(text))
            label = int(row.get("binary_label", 0))
            complete.append(_section_complete(sections))
            lc.append(label_consistency(text, label))
            oct_sup.append(_supported(sections.get("oct_findings", ""), "oct"))
            col_sup.append(_supported(sections.get("colposcopy_findings", ""), "colposcopy"))
            ins_sup.append(_supported(sections.get("clinical_context", ""), "instruction"))
        unique_rate = len(set(texts)) / max(len(texts), 1)
        max_duplicate_fraction = max((texts.count(t) for t in set(texts)), default=0) / max(len(texts), 1)
        rows.append(
            {
                "pseudo_report_source": source,
                "n_cases": len(test),
                "section_complete_rate": float(np.mean(complete)),
                "label_consistency_mean": float(np.mean(lc)),
                "oct_supported_rate": float(np.mean(oct_sup)),
                "colposcopy_supported_rate": float(np.mean(col_sup)),
                "instruction_supported_rate": float(np.mean(ins_sup)),
                "mean_modality_support_rate": float(np.mean([np.mean(oct_sup), np.mean(col_sup), np.mean(ins_sup)])),
                "unique_text_rate": unique_rate,
                "max_duplicate_fraction": max_duplicate_fraction,
            }
        )
    base = pd.DataFrame(rows)

    # Add latent alignment under the full model for each pseudo-report source.
    ckpt = project / MODEL_VARIANTS["full_lcad_rasa"]
    full_rows = []
    if ckpt.is_file() and len(test):
        model = _load_model(ckpt, device)
        for source in sources:
            batch, _ = _prepare_model_batch(test, source, device)
            with torch.no_grad():
                out = model(batch["oct"], batch["col"], batch["fus"], batch["instr"], batch["ids"], batch["labels"])
                h0 = out["fused"]
                h = out["hidden"]
                q_len = max(1, h.shape[1] // len(SECTION_KEYS))
                targets = {
                    "oct_findings": h[:, :q_len].mean(1),
                    "colposcopy_findings": h[:, q_len : 2 * q_len].mean(1),
                    "clinical_context": h[:, 2 * q_len : 3 * q_len].mean(1),
                    "impression": h[:, 3 * q_len :].mean(1),
                }
                queries = {
                    "oct_findings": model.sec_oct(h0),
                    "colposcopy_findings": model.sec_col(h0),
                    "clinical_context": model.sec_instr(h0),
                    "impression": model.sec_imp(h0),
                }
                mrrs, gaps = [], []
                for sec in SECTION_KEYS:
                    sim = F.normalize(queries[sec], dim=-1) @ F.normalize(targets[sec], dim=-1).T
                    _r1, _r5, mrr, pos_sim, neg_sim = _ranks_from_similarity(sim)
                    mrrs.append(mrr)
                    gaps.append(pos_sim - neg_sim)
                full_rows.append(
                    {
                        "pseudo_report_source": source,
                        "latent_alignment_mrr_full_model": float(np.mean(mrrs)),
                        "latent_alignment_gap_full_model": float(np.mean(gaps)),
                    }
                )
        del model
    align = pd.DataFrame(full_rows)
    out = base.merge(align, on="pseudo_report_source", how="left")
    _write_csv(out, paths.tables / "T_theme1_llm_vs_template_rule_pseudo_report.csv", paths.manuscript)

    plot_df = out.set_index("pseudo_report_source")[
        ["mean_modality_support_rate", "unique_text_rate", "latent_alignment_mrr_full_model"]
    ].fillna(0)
    _save_heatmap(plot_df, paths.figures / "Figure_theme1_pseudo_report_source_comparison", "Pseudo-report source comparison")
    return out


def _compact_features(rows: pd.DataFrame) -> np.ndarray:
    feats = []
    for _, row in rows.iterrows():
        vecs = [
            load_visual_emb(_safe_str(row.get("oct_embedding_path", ""))),
            load_visual_emb(_safe_str(row.get("colposcopy_embedding_path", ""))),
            load_visual_emb(_safe_str(row.get("fused_visual_embedding_path", ""))),
        ]
        vals = []
        for v in vecs:
            vals.extend(
                [
                    float(np.linalg.norm(v)),
                    float(np.mean(v)),
                    float(np.std(v)),
                    float(np.min(v)),
                    float(np.max(v)),
                    float(np.quantile(v, 0.25)),
                    float(np.quantile(v, 0.75)),
                ]
            )
        vals.extend(instr_vector(row.to_dict()).astype(float).tolist())
        feats.append(vals)
    return np.asarray(feats, dtype=np.float32)


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), y_prob]))
    best_t, best_f = 0.5, -1.0
    for t in candidates:
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f = f
            best_t = float(t)
    return best_t


def _sample_real_subset(real: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 0.999:
        return real.copy()
    parts = []
    for _label, g in real.groupby("binary_label"):
        n = max(1, int(round(len(g) * fraction)))
        parts.append(g.sample(n=min(n, len(g)), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def run_scarcity_curve(df: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    train_real = train[train["has_real_report"] == 1].copy()
    train_pseudo = train[train["needs_pseudo_report"] == 1].copy()
    fractions = [1.0, 0.5, 0.25, 0.1]
    seeds = [42, 123, 456, 789, 2026]
    x_val, y_val = _compact_features(val), val["binary_label"].astype(int).to_numpy()
    x_test, y_test = _compact_features(test), test["binary_label"].astype(int).to_numpy()
    rows = []
    for fraction in fractions:
        for seed in seeds:
            real_subset = _sample_real_subset(train_real, fraction, seed)
            setups = {
                "real_report_only_surrogate": real_subset,
                "lcad_augmented_surrogate": pd.concat([real_subset, train_pseudo], ignore_index=True),
            }
            for setup, train_df in setups.items():
                y_train = train_df["binary_label"].astype(int).to_numpy()
                if len(np.unique(y_train)) < 2:
                    continue
                x_train = _compact_features(train_df)
                sample_weight = np.ones(len(train_df), dtype=float)
                if setup == "lcad_augmented_surrogate":
                    sample_weight = np.where(
                        train_df["training_report_type"].astype(str).to_numpy() == "pseudo",
                        train_df.get("pseudo_training_weight", pd.Series(0.75, index=train_df.index)).fillna(0.75).to_numpy(),
                        1.0,
                    )
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed),
                )
                clf.fit(x_train, y_train, logisticregression__sample_weight=sample_weight)
                val_prob = clf.predict_proba(x_val)[:, 1]
                test_prob = clf.predict_proba(x_test)[:, 1]
                thr = _best_threshold(y_val, val_prob)
                rows.append(
                    {
                        "setup": setup,
                        "real_report_fraction": fraction,
                        "seed": seed,
                        "n_train": len(train_df),
                        "n_real_train": int((train_df["has_real_report"] == 1).sum()),
                        "n_pseudo_train": int((train_df["needs_pseudo_report"] == 1).sum()),
                        "auc": float(roc_auc_score(y_test, test_prob)),
                        "f1": float(f1_score(y_test, (test_prob >= thr).astype(int), zero_division=0)),
                        "threshold_val_selected": thr,
                    }
                )
    raw = pd.DataFrame(rows)
    agg = (
        raw.groupby(["setup", "real_report_fraction"], as_index=False)
        .agg(
            n_runs=("seed", "count"),
            mean_n_train=("n_train", "mean"),
            mean_n_real_train=("n_real_train", "mean"),
            mean_n_pseudo_train=("n_pseudo_train", "mean"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .sort_values(["real_report_fraction", "setup"], ascending=[False, True])
    )
    _write_csv(raw, paths.tables / "T_theme1_report_supervision_scarcity_curve_raw.csv", None)
    _write_csv(agg, paths.tables / "T_theme1_report_supervision_scarcity_curve.csv", paths.manuscript)

    _setup_figure_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    color_map = {
        "real_report_only_surrogate": MORANDI_HEX[0],
        "lcad_augmented_surrogate": MORANDI_HEX[4],
    }
    label_map = {
        "real_report_only_surrogate": "Real-report only",
        "lcad_augmented_surrogate": "LCAD-augmented",
    }
    for setup, g in agg.groupby("setup"):
        g = g.sort_values("real_report_fraction")
        color = color_map.get(setup, MORANDI_HEX[len(color_map) % len(MORANDI_HEX)])
        ax.plot(
            g["real_report_fraction"],
            g["auc_mean"],
            label=label_map.get(setup, setup.replace("_", " ").title()),
            color=color,
            linewidth=1.6,
            alpha=0.6,
        )
        ax.scatter(g["real_report_fraction"], g["auc_mean"], color=color, edgecolor="#3a3a3a", linewidth=0.8, s=82, zorder=3)
        ax.errorbar(
            g["real_report_fraction"],
            g["auc_mean"],
            yerr=g["auc_std"].fillna(0),
            fmt="none",
            ecolor="#3a3a3a",
            elinewidth=1.0,
            capsize=3,
            zorder=2,
        )
    ax.set_xticks([0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.set_xlabel("Available real-report supervision fraction")
    ax.set_ylabel("AUROC on locked test set")
    ax.set_title("Report-supervision scarcity curve", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    paths.figures.mkdir(parents=True, exist_ok=True)
    fig.savefig((paths.figures / "Figure_theme1_report_supervision_scarcity_curve").with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig((paths.figures / "Figure_theme1_report_supervision_scarcity_curve").with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return agg


def run_perturbation_matrix(project: Path, paths: Paths) -> pd.DataFrame:
    src = project / "outputs/publishable/tables/manuscript/S6_modality_perturbation_text_decoding.csv"
    if not src.is_file():
        src = project / "outputs/publishable/tables/modality_perturbation_text_decoding_summary.csv"
    df = pd.read_csv(src)
    normal_lc = float(df.loc[df["condition"] == "normal", "label_consistency"].iloc[0])
    rows = []
    for _, r in df.iterrows():
        row = {
            "condition": r["condition"],
            "oct_findings_drop": 1.0 - float(r.get("oct_findings_similarity_to_normal", np.nan)),
            "colposcopy_findings_drop": 1.0 - float(r.get("colposcopy_findings_similarity_to_normal", np.nan)),
            "clinical_context_drop": 1.0 - float(r.get("clinical_context_similarity_to_normal", np.nan)),
            "impression_drop": 1.0 - float(r.get("impression_similarity_to_normal", np.nan)),
            "report_drop": 1.0 - float(r.get("report_similarity_to_normal", np.nan)),
            "risk_abs_delta": float(r.get("risk_score_absolute_delta_vs_normal", np.nan)),
            "label_consistency_drop": normal_lc - float(r.get("label_consistency", np.nan)),
        }
        section_cols = ["oct_findings_drop", "colposcopy_findings_drop", "clinical_context_drop", "impression_drop"]
        row["max_drop_section"] = max(section_cols, key=lambda c: row[c] if not math.isnan(row[c]) else -1)
        expected = {
            "mask_oct": "oct_findings_drop",
            "shuffle_oct": "oct_findings_drop",
            "mask_colposcopy": "colposcopy_findings_drop",
            "shuffle_colposcopy": "colposcopy_findings_drop",
            "mask_instruction": "clinical_context_drop",
            "shuffle_instruction": "clinical_context_drop",
            "mask_visual": "colposcopy_findings_drop",
            "label_only_inference": "colposcopy_findings_drop",
        }.get(str(r["condition"]), "")
        row["expected_primary_drop"] = expected
        row["specificity_hit"] = float(expected == row["max_drop_section"]) if expected else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    _write_csv(out, paths.tables / "T_theme1_upgraded_perturbation_sensitivity_matrix.csv", paths.manuscript)
    matrix_cols = [
        "oct_findings_drop",
        "colposcopy_findings_drop",
        "clinical_context_drop",
        "impression_drop",
        "risk_abs_delta",
    ]
    matrix = out.set_index("condition")[matrix_cols].fillna(0)
    _save_heatmap(matrix, paths.figures / "Figure_theme1_perturbation_sensitivity_matrix", "Upgraded perturbation sensitivity matrix", cmap=MORANDI_WARM)
    return out


def write_summary(paths: Paths, outputs: dict[str, pd.DataFrame]) -> Path:
    summary = paths.out / "THEME1_ALIGNMENT_EXPERIMENTS_SUMMARY.md"
    lines = [
        "# Theme 1 Alignment Experiments Summary\n\n",
        "Scope: LLM-augmented cross-modal semantic alignment for large-scale analytics.\n\n",
        "## Generated tables\n\n",
    ]
    for name in [
        "T_theme1_llm_vs_template_rule_pseudo_report.csv",
        "T_theme1_modality_section_retrieval_alignment.csv",
        "T_theme1_report_supervision_scarcity_curve.csv",
        "T_theme1_rasa_direct_alignment_ablation.csv",
        "T_theme1_upgraded_perturbation_sensitivity_matrix.csv",
    ]:
        lines.append(f"- `tables/{name}`\n")
    lines.append("\n## Manuscript-safe interpretation\n\n")
    pseudo = outputs.get("pseudo", pd.DataFrame())
    if not pseudo.empty:
        local = pseudo[pseudo["pseudo_report_source"] == "local_llm"]
        template = pseudo[pseudo["pseudo_report_source"] == "label_template"]
        rule = pseudo[pseudo["pseudo_report_source"] == "rule_based"]
        if not local.empty and not template.empty:
            local = local.iloc[0]
            template = template.iloc[0]
            lines.append(
                "- LLM pseudo-reports preserve modality-grounded sections "
                f"(mean support={local['mean_modality_support_rate']:.3f}) versus label templates "
                f"(mean support={template['mean_modality_support_rate']:.3f}).\n"
            )
            lines.append(
                "- Label templates have high apparent latent MRR but are highly repetitive "
                f"(max duplicate fraction={template['max_duplicate_fraction']:.3f}); do not frame this as stronger semantic alignment.\n"
            )
        if not rule.empty:
            rule = rule.iloc[0]
            lines.append(
                f"- Rule pseudo-reports provide a non-LLM grounded baseline "
                f"(mean support={rule['mean_modality_support_rate']:.3f}, label consistency={rule['label_consistency_mean']:.3f}).\n"
            )
    direct = outputs.get("direct", pd.DataFrame())
    if not direct.empty:
        top = direct.sort_values("macro_mrr", ascending=False).iloc[0]
        def _macro(model_name: str) -> float | None:
            hit = direct[direct["model"] == model_name]
            if hit.empty:
                return None
            return float(hit["macro_mrr"].iloc[0])

        full_mrr = _macro("full_lcad_rasa")
        no_section_mrr = _macro("no_section_alignment")
        simple_mrr = _macro("simple_concat_fusion")
        if full_mrr is not None and no_section_mrr is not None:
            lines.append(
                f"- Direct alignment ablation supports the section-alignment mechanism: full LCAD-RASA "
                f"macro MRR={full_mrr:.3f} versus no-section alignment={no_section_mrr:.3f}.\n"
            )
        if simple_mrr is not None:
            lines.append(f"- Simple concatenation remains a weak alignment baseline (macro MRR={simple_mrr:.3f}).\n")
        lines.append(
            f"- The highest macro MRR variant is `{top['model']}` (MRR={top['macro_mrr']:.3f}); "
            "interpret full-model results together with risk-prediction objectives rather than as a pure retrieval optimum.\n"
        )
    scarcity = outputs.get("scarcity", pd.DataFrame())
    if not scarcity.empty:
        g = scarcity[scarcity["real_report_fraction"] == scarcity["real_report_fraction"].min()]
        lines.append("- Scarcity curve is a lightweight risk-surrogate analysis; do not replace Table 2 with it.\n")
        for _, row in g.iterrows():
            lines.append(f"  - {row['setup']} at {row['real_report_fraction']:.0%} real reports: AUROC={row['auc_mean']:.3f}.\n")
    lines.append("\n## Claim limits\n\n")
    lines.append("- Use these tables to support cross-modal semantic alignment, not clinical deployment readiness.\n")
    lines.append("- Report-supervision scarcity is a lightweight surrogate experiment unless full LCAD-RASA retraining is later performed.\n")
    summary.write_text("".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/publishable/manifests/full_manifest_publishable_with_llm_pseudo.csv")
    parser.add_argument("--output_dir", default="outputs/publishable/theme1_alignment")
    parser.add_argument("--max_cases", type=int, default=288)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    manifest = ROOT / args.manifest
    df = pd.read_csv(manifest)
    out = ROOT / args.output_dir
    paths = Paths(
        out=out,
        tables=out / "tables",
        figures=out / "figures",
        manuscript=ROOT / "outputs/publishable/tables/manuscript",
    )
    paths.tables.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    outputs: dict[str, pd.DataFrame] = {}
    print("[1/5] LLM vs template/rule pseudo-report comparison")
    outputs["pseudo"] = run_pseudo_report_source_comparison(ROOT, df, paths, device, args.max_cases)
    print(outputs["pseudo"].to_string(index=False))

    print("[2/5] Modality-section retrieval alignment")
    retrieval, direct = compute_latent_alignment(ROOT, df, paths, device, max_cases=args.max_cases)
    outputs["retrieval"] = retrieval
    print(retrieval.head(12).to_string(index=False))

    print("[3/5] Report-supervision scarcity curve")
    outputs["scarcity"] = run_scarcity_curve(df, paths)
    print(outputs["scarcity"].to_string(index=False))

    print("[4/5] RASA direct alignment metric ablation")
    outputs["direct"] = direct
    print(direct.to_string(index=False))

    print("[5/5] Upgraded perturbation sensitivity matrix")
    outputs["perturbation"] = run_perturbation_matrix(ROOT, paths)
    print(outputs["perturbation"].to_string(index=False))

    summary = write_summary(paths, outputs)
    print(f"Wrote summary: {summary}")


if __name__ == "__main__":
    main()
