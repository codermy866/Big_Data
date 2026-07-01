#!/usr/bin/env python3
"""Run the remaining feasible public-baseline controls.

This runner fills experiments that can be executed locally without downloading
new public VLP weights:

* B7/B8: self-implemented concept-space surrogates using train-only text
  concepts. These are style controls, not official MedFILIP/RadAlign
  reproductions.
* C3-C7/C9: RASA section-anchor controls that require retraining.
* E6: isolated oracle-leakage stress artifact. This is intentionally written
  outside the main supplementary-control directory unless the registry is
  explicitly updated later.

D4 is not run here because full alternative-LLM claim generation consumes an
external API budget and requires a user-approved provider/model choice.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "cervix_lcad_rasa"
sys.path.insert(0, str(PROJECT))

from src.models_publishable.lcad_rasa_model import instr_vector, load_visual_emb  # noqa: E402
from src.supplementary.next_stage.core import metrics_at_threshold, select_thresholds  # noqa: E402
from src.supplementary.train_eval import load_jbd_config, train_experiment  # noqa: E402


MANIFEST_LEGACY = PROJECT / "outputs/publishable/manifests/full_manifest_publishable_with_llm_pseudo.csv"
RASA_SCORES = PROJECT / "outputs/publishable/kra_rasa_analysis/full_lcad_rasa_val_test_scores.csv"
OUT = ROOT / "outputs/rasa_public_baselines"
SECTION_OUT = OUT / "section_anchor_controls"
CONCEPT_OUT = OUT / "concept_surrogate_controls"
LEAK_OUT = OUT / "oracle_leakage_stress"
RUN_DIR = OUT / "unfinished_10_controls"

FORBIDDEN_RE = re.compile(
    r"\b(CIN\s*2\+?|CIN\s*3\+?|HSIL|LSIL|cancer|malignant|pathology[- ]?(positive|negative)|biopsy[- ]?proven)\b|癌|浸润",
    flags=re.IGNORECASE,
)


def auc_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    pos_ranks = ranks[y == 1].sum()
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _clean_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    return FORBIDDEN_RE.sub(" endpoint_term_removed ", text)


def _legacy_text(row: pd.Series) -> str:
    for col in ("training_report_text", "pseudo_report_text", "real_report_text", "reference_report_text"):
        text = _clean_text(row.get(col, ""))
        if text.strip() and text.strip().lower() not in {"nan", "none", "null"}:
            return text
    return "cervical multimodal examination"


def _visual_matrix(df: pd.DataFrame) -> np.ndarray:
    rows: list[np.ndarray] = []
    for _, row in df.iterrows():
        oct_v = load_visual_emb(str(row.get("oct_embedding_path", "")))
        col_v = load_visual_emb(str(row.get("colposcopy_embedding_path", "")))
        fus_v = load_visual_emb(str(row.get("fused_visual_embedding_path", "")))
        ins_v = instr_vector(row.to_dict())
        rows.append(np.concatenate([oct_v, col_v, fus_v, ins_v]).astype(np.float32))
    return np.vstack(rows)


def _best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        m = metrics_at_threshold(y.astype(int).tolist(), score.astype(float).tolist(), float(t))
        if m.get("f1", 0.0) > best_f1:
            best_f1 = float(m.get("f1", 0.0))
            best_t = float(t)
    return best_t


def _write_prediction_artifact(
    frames: list[pd.DataFrame],
    out_root: Path,
    experiment_id: str,
    threshold: float,
    source: str,
) -> Path:
    out_dir = out_root / "predictions" / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.concat(frames, ignore_index=True)
    out["threshold"] = float(threshold)
    out["pred_label"] = (out["risk_score"].astype(float) >= float(threshold)).astype(int)
    out["risk_available"] = True
    out["source_artifact"] = source
    path = out_dir / "all_predictions.csv"
    out.to_csv(path, index=False)
    return path


def _prediction_frame(df: pd.DataFrame, split: str, score: np.ndarray, experiment_id: str, source: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": df["case_id"].astype(str).to_numpy(),
            "center_id": df["center_id"].astype(str).to_numpy(),
            "split": split,
            "y_true_cin2plus": df["binary_label"].astype(int).to_numpy(),
            "risk_score": score.astype(float),
            "experiment_id": experiment_id,
            "source_checkpoint": source,
        }
    )


def run_concept_surrogates(seed: int) -> list[dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    df = pd.read_csv(MANIFEST_LEGACY)
    train = df[df["split"].astype(str).eq("train")].copy()
    val = df[df["split"].astype(str).eq("val")].copy()
    test = df[df["split"].astype(str).eq("test")].copy()
    texts_train = [_legacy_text(r) for _, r in train.iterrows()]
    vectorizer = TfidfVectorizer(max_features=128, min_df=3, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    concept_train = vectorizer.fit_transform(texts_train).astype(np.float32)
    x_train = _visual_matrix(train)
    x_val = _visual_matrix(val)
    x_test = _visual_matrix(test)
    y_train = train["binary_label"].astype(int).to_numpy()
    y_val = val["binary_label"].astype(int).to_numpy()
    y_test = test["binary_label"].astype(int).to_numpy()

    ridge = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=3.0, random_state=seed))
    ridge.fit(x_train, concept_train.toarray())
    pred_concept_train = np.maximum(ridge.predict(x_train), 0.0)
    pred_concept_val = np.maximum(ridge.predict(x_val), 0.0)
    pred_concept_test = np.maximum(ridge.predict(x_test), 0.0)

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    clf.fit(pred_concept_train, y_train)
    score_train = clf.predict_proba(pred_concept_train)[:, 1]
    score_val = clf.predict_proba(pred_concept_val)[:, 1]
    score_test = clf.predict_proba(pred_concept_test)[:, 1]
    thr_b7 = _best_threshold(y_val, score_val)
    b7_id = "B7_conceptvlp_medfilip_style"
    b7_path = _write_prediction_artifact(
        [
            _prediction_frame(train, "train", score_train, b7_id, "train_only_tfidf_concept_surrogate"),
            _prediction_frame(val, "val", score_val, b7_id, "train_only_tfidf_concept_surrogate"),
            _prediction_frame(test, "test", score_test, b7_id, "train_only_tfidf_concept_surrogate"),
        ],
        CONCEPT_OUT,
        b7_id,
        thr_b7,
        "train_only_tfidf_concept_surrogate",
    )

    concept_train_dense = concept_train.toarray().astype(np.float32)
    sim_val = cosine_similarity(pred_concept_val, concept_train_dense)
    sim_test = cosine_similarity(pred_concept_test, concept_train_dense)
    sim_train = cosine_similarity(pred_concept_train, concept_train_dense)

    def prior(sim: np.ndarray, top_k: int = 25, temperature: float = 0.15) -> np.ndarray:
        out = []
        for row in sim:
            k = min(top_k, len(row))
            idx = np.argpartition(-row, k - 1)[:k]
            vals = row[idx]
            vals = vals - vals.max()
            w = np.exp(vals / max(temperature, 1e-6))
            w = w / max(w.sum(), 1e-8)
            out.append(float((w * y_train[idx]).sum()))
        return np.asarray(out, dtype=np.float32)

    score_train_b8 = prior(sim_train)
    score_val_b8 = prior(sim_val)
    score_test_b8 = prior(sim_test)
    thr_b8 = _best_threshold(y_val, score_val_b8)
    b8_id = "B8_conceptretrieval_radalign_style"
    b8_path = _write_prediction_artifact(
        [
            _prediction_frame(train, "train", score_train_b8, b8_id, "train_only_tfidf_concept_retrieval_surrogate"),
            _prediction_frame(val, "val", score_val_b8, b8_id, "train_only_tfidf_concept_retrieval_surrogate"),
            _prediction_frame(test, "test", score_test_b8, b8_id, "train_only_tfidf_concept_retrieval_surrogate"),
        ],
        CONCEPT_OUT,
        b8_id,
        thr_b8,
        "train_only_tfidf_concept_retrieval_surrogate",
    )

    meta = {
        "seed": seed,
        "concept_features": int(len(vectorizer.get_feature_names_out())),
        "train_cases": int(len(train)),
        "val_cases": int(len(val)),
        "test_cases": int(len(test)),
        "forbidden_endpoint_terms_removed": True,
        "style_surrogate_not_official_reproduction": True,
    }
    (CONCEPT_OUT / "concept_surrogate_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return [
        {"experiment_id": b7_id, "path": str(b7_path), "threshold": thr_b7, "test_auc": auc_rank(y_test, score_test)},
        {"experiment_id": b8_id, "path": str(b8_path), "threshold": thr_b8, "test_auc": auc_rank(y_test, score_test_b8)},
    ]


def _base_rasa_spec() -> dict[str, Any]:
    return {
        "train_filter": {"training_eligible": 1},
        "use_pseudo_report": True,
        "use_real_report": True,
        "use_label_free_claims": False,
        "allow_legacy_pseudo_report": True,
        "require_qc_pass": True,
        "use_report_loss": True,
        "model": {"use_section_align": True, "use_risk_head": True},
        "loss": {"ce_weight": 1.0, "rasa_weight": 0.5, "cls_weight": 0.2, "cons_weight": 0.1},
    }


def rasa_control_specs() -> dict[str, dict[str, Any]]:
    base = _base_rasa_spec()

    def merged(**kwargs: Any) -> dict[str, Any]:
        out = json.loads(json.dumps(base))
        for k, v in kwargs.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k].update(v)
            else:
                out[k] = v
        return out

    return {
        "C3_rasa_random_section_assignment": merged(model={"random_section_assignment": True}),
        "C4_rasa_without_oct_anchor": merged(model={"active_section_anchors": ["colposcopy", "instruction", "impression"]}),
        "C5_rasa_without_colposcopy_anchor": merged(model={"active_section_anchors": ["oct", "instruction", "impression"]}),
        "C6_rasa_without_clinical_context_anchor": merged(model={"active_section_anchors": ["oct", "colposcopy", "impression"]}),
        "C7_rasa_without_diagnostic_impression_anchor": merged(model={"active_section_anchors": ["oct", "colposcopy", "instruction"]}),
        "C9_rasa_pseudo_reports_only": merged(
            train_filter={"has_real_report": 0, "needs_pseudo_report": 1, "has_pseudo_report": 1, "pseudo_report_pass_qc": 1},
            use_real_report=False,
            use_label_free_claims=False,
            allow_legacy_pseudo_report=True,
        ),
    }


def _predict_checkpoint(ckpt: Path, spec: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    import torch

    from src.supplementary.train_eval import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location="cpu")
    model = build_model(state.get("spec") or spec)
    model.load_state_dict(state["model"], strict=False)
    model.to(device)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for _, row in df.iterrows():
            oct_e = torch.tensor(load_visual_emb(str(row.get("oct_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
            col_e = torch.tensor(load_visual_emb(str(row.get("colposcopy_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
            fus_e = torch.tensor(load_visual_emb(str(row.get("fused_visual_embedding_path", ""))), dtype=torch.float32, device=device).unsqueeze(0)
            instr = torch.tensor(instr_vector(row.to_dict()), dtype=torch.float32, device=device).unsqueeze(0)
            lab = torch.tensor([int(row["binary_label"])], device=device)
            ids = torch.zeros(1, 64, dtype=torch.long, device=device)
            out = model(oct_e, col_e, fus_e, instr, ids, lab)
            if out.get("risk_logit") is None:
                scores.append(0.5)
            else:
                scores.append(float(torch.sigmoid(out["risk_logit"]).item()))
    return np.asarray(scores, dtype=np.float32)


def run_rasa_anchor_controls(seed: int, force_retrain: bool) -> list[dict[str, Any]]:
    df = pd.read_csv(MANIFEST_LEGACY)
    cfg = load_jbd_config(PROJECT)
    cfg = {**cfg, "training": {**cfg.get("training", {}), "num_epochs": 5, "max_steps_per_epoch": 120}}
    train = df[df["split"].astype(str).eq("train")].copy()
    val = df[df["split"].astype(str).eq("val")].copy()
    test = df[df["split"].astype(str).eq("test")].copy()
    results: list[dict[str, Any]] = []
    ckpt_root = SECTION_OUT / "checkpoints"
    for exp_id, spec in rasa_control_specs().items():
        t0 = time.time()
        ckpt = ckpt_root / exp_id / "best.ckpt"
        train_status: dict[str, Any]
        if force_retrain or not ckpt.is_file():
            train_status = train_experiment(PROJECT, df, exp_id, spec, cfg, ckpt_root / exp_id, seed=seed)
            if train_status.get("status") != "ok":
                results.append({"experiment_id": exp_id, "status": "failed", "error": train_status.get("error", "train_failed")})
                continue
        else:
            train_status = {"status": "cached", "checkpoint": str(ckpt)}
        score_val = _predict_checkpoint(ckpt, spec, val)
        score_test = _predict_checkpoint(ckpt, spec, test)
        score_train = _predict_checkpoint(ckpt, spec, train)
        y_val = val["binary_label"].astype(int).to_numpy()
        y_test = test["binary_label"].astype(int).to_numpy()
        thr = select_thresholds(y_val.astype(int).tolist(), score_val.astype(float).tolist()).get("max_f1", 0.5)
        pred_path = _write_prediction_artifact(
            [
                _prediction_frame(train, "train", score_train, exp_id, str(ckpt)),
                _prediction_frame(val, "val", score_val, exp_id, str(ckpt)),
                _prediction_frame(test, "test", score_test, exp_id, str(ckpt)),
            ],
            SECTION_OUT,
            exp_id,
            float(thr),
            str(ckpt),
        )
        metrics = metrics_at_threshold(y_test.astype(int).tolist(), score_test.astype(float).tolist(), float(thr))
        results.append(
            {
                "experiment_id": exp_id,
                "status": "ok",
                "prediction_path": str(pred_path),
                "checkpoint": str(ckpt),
                "threshold": float(thr),
                "test_auc": auc_rank(y_test, score_test),
                "test_f1": float(metrics.get("f1", math.nan)),
                "train_status": train_status.get("status"),
                "elapsed_minutes": (time.time() - t0) / 60.0,
            }
        )
    return results


def run_oracle_leakage_stress() -> dict[str, Any]:
    scores = pd.read_csv(RASA_SCORES)
    frames = []
    for split in ("val", "test"):
        sub = scores[scores["split"].astype(str).eq(split)].copy()
        y = sub["y_true"].astype(int).to_numpy()
        oracle = np.where(y == 1, 0.99, 0.01).astype(np.float32)
        frames.append(
            pd.DataFrame(
                {
                    "case_id": sub["case_id"].astype(str).to_numpy(),
                    "center_id": sub["center_id"].astype(str).to_numpy(),
                    "split": split,
                    "y_true_cin2plus": y,
                    "risk_score": oracle,
                    "experiment_id": "E6_oracle_leakage_retrieval",
                    "source_checkpoint": "oracle_label_leakage_stress_invalid_for_main_analysis",
                }
            )
        )
    path = _write_prediction_artifact(
        frames,
        LEAK_OUT,
        "E6_oracle_leakage_retrieval",
        0.5,
        "oracle_label_leakage_stress_invalid_for_main_analysis",
    )
    test = frames[-1]
    return {
        "experiment_id": "E6_oracle_leakage_retrieval",
        "status": "stress_only_done",
        "prediction_path": str(path),
        "test_auc": auc_rank(test["y_true_cin2plus"].to_numpy(), test["risk_score"].to_numpy()),
        "main_analysis_valid": False,
    }


def write_report(results: dict[str, Any]) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RUN_DIR / "remaining_10_execution_summary.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Remaining 10 Experiment Execution Summary\n\n",
        f"Generated at: {pd.Timestamp.now(tz='Asia/Shanghai').isoformat()}\n\n",
        "## Completed Locally\n\n",
    ]
    for block in ("concept_surrogates", "rasa_anchor_controls"):
        lines.append(f"### {block}\n\n")
        for row in results.get(block, []):
            lines.append(
                f"- `{row.get('experiment_id')}`: {row.get('status', 'ok')}; "
                f"AUROC={row.get('test_auc', 'NA')}; path={row.get('path', row.get('prediction_path', 'NA'))}\n"
            )
        lines.append("\n")
    lines.append("### oracle leakage stress\n\n")
    e6 = results.get("oracle_leakage_stress", {})
    lines.append(
        f"- `{e6.get('experiment_id')}`: {e6.get('status')}; AUROC={e6.get('test_auc')}; "
        "invalid for main analysis and should only be cited as a leakage pressure test.\n\n"
    )
    lines.append("## Still Blocked or Requires User Decision\n\n")
    lines.append("- `D4_alternative_llm_claim_bank`: requires a user-approved alternative LLM provider/model and API budget for full 1,153-case claim generation before bridge evaluation.\n")
    lines.append("- Official `B7`/`B8` checkpoints remain unavailable in this repository; the completed rows are train-only style surrogates, not official reproductions.\n\n")
    lines.append("## Registry Integration Note\n\n")
    lines.append("The generated prediction artifacts can be wired into `configs/rasa_public_baselines/experiment_registry.yaml` after verifying all outputs.\n")
    md_path = RUN_DIR / "REMAINING_10_EXECUTION_SUMMARY.md"
    md_path.write_text("".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-concept", action="store_true")
    parser.add_argument("--skip-rasa", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "seed": args.seed,
        "concept_surrogates": [],
        "rasa_anchor_controls": [],
        "oracle_leakage_stress": {},
    }
    if not args.skip_concept:
        results["concept_surrogates"] = run_concept_surrogates(args.seed)
    if not args.skip_rasa:
        results["rasa_anchor_controls"] = run_rasa_anchor_controls(args.seed, args.force_retrain)
    if not args.skip_oracle:
        results["oracle_leakage_stress"] = run_oracle_leakage_stress()
    report = write_report(results)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
