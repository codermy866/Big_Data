#!/usr/bin/env python3
"""External LLM API provider comparison for structured pseudo-report generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distillation.llm_agent_client import LocalLLMAgentClient
from src.distillation.llm_api_providers import (
    LLMProviderClient,
    default_provider_specs,
    estimate_cost_usd,
)
from src.distillation.pseudo_report_schema import build_pseudo_report
from src.distillation.quality_control import qc_pseudo_report
from src.evaluation.metrics import label_consistency
from src.models_publishable.lcad_rasa_model import PublishableLCADRASA, instr_vector, load_visual_emb

SECTION_KEYS = ["oct_findings", "colposcopy_findings", "clinical_context", "impression"]
REQUIRED_FIELDS = [
    "diagnostic_summary",
    "oct_findings",
    "colposcopy_findings",
    "clinical_context",
    "impression",
    "recommendation",
]
API_PROVIDERS = [
    "qwen",
    "glm",
    "gemini",
    "gpt",
    "minimax",
    "aihubmix",
    "aihubmix_gpt",
    "aihubmix_qwen",
    "aihubmix_glm",
    "aihubmix_gemini",
    "aihubmix_deepseek",
    "aihubmix_llama",
    "aihubmix_mimo",
]
BASELINE_PROVIDERS = ["label_template", "rule_based", "local_llm"]
MORANDI_HEX = [
    "#576fa0",
    "#a7b9d7",
    "#e3b87f",
    "#fadcb4",
    "#b57979",
    "#dea3a2",
    "#9f9f9f",
    "#cfcece",
]
TEXT_DARK = "#3a3a3a"
EDGE_DARK = "#9f9f9f"
GRID_LINE = "#cfcece"
PROVIDER_LABELS = {
    "label_template": "Template",
    "rule_based": "Rule-based",
    "local_llm": "Local embedding LLM",
    "qwen": "Qwen",
    "glm": "GLM",
    "gemini": "Gemini",
    "gpt": "GPT",
    "minimax": "MiniMax",
    "aihubmix": "API model",
    "aihubmix_gpt": "GPT-5.5",
    "aihubmix_qwen": "Qwen-Plus",
    "aihubmix_glm": "GLM-4.7-Flash",
    "aihubmix_gemini": "Gemini-3.1-Pro",
    "aihubmix_deepseek": "DeepSeek-V4-Pro",
    "aihubmix_llama": "Llama-4",
    "aihubmix_mimo": "Xiaomi-MiMo-V2.5",
}


def _setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,
            "axes.edgecolor": EDGE_DARK,
            "axes.labelcolor": TEXT_DARK,
            "text.color": TEXT_DARK,
            "grid.color": GRID_LINE,
            "grid.alpha": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _provider_label(provider: Any) -> str:
    label = PROVIDER_LABELS.get(str(provider), str(provider).replace("_", " ").title())
    for token in ("AIHubMix", "aihubmix", "Free", "free", "Preview", "preview"):
        label = label.replace(token, "").strip()
    return re.sub(r"\s+", " ", label)


def _save_plot(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(df: pd.DataFrame, path: Path, manuscript_dir: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if manuscript_dir is not None:
        manuscript_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(manuscript_dir / path.name, index=False)


def _report_text(report: dict[str, Any]) -> str:
    return " ".join(_safe_str(report.get(k, "")) for k in REQUIRED_FIELDS)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _extract_json(text: str) -> tuple[dict[str, Any], str]:
    raw = _safe_str(text).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        obj = json.loads(raw)
        return (obj if isinstance(obj, dict) else {}, "")
    except Exception as first_error:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
                return (obj if isinstance(obj, dict) else {}, "")
            except Exception as second_error:
                return {}, f"json_parse_error:{second_error}"
        return {}, f"json_parse_error:{first_error}"


def _case_uid(case_id: str, center_id: str) -> str:
    return hashlib.sha256(f"{center_id}|{case_id}".encode("utf-8")).hexdigest()[:16]


def _age_group(value: Any) -> str:
    try:
        age = float(value)
    except Exception:
        return "unknown"
    if age < 30:
        return "<30"
    if age < 40:
        return "30-39"
    if age < 50:
        return "40-49"
    if age < 60:
        return "50-59"
    return "60+"


def _sanitized_field(value: Any, max_len: int = 120) -> str:
    text = _safe_str(value)
    text = re.sub(r"/\S+", "", text)
    text = re.sub(r"\b\d{6,}\b", "[redacted_id]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def _embedding_stats(path: str) -> dict[str, float]:
    p = Path(_safe_str(path))
    if not p.is_file():
        return {"norm": 0.0, "mean": 0.0, "std": 0.0}
    v = np.load(p).astype(np.float32).reshape(-1)
    return {"norm": float(np.linalg.norm(v)), "mean": float(v.mean()), "std": float(v.std())}


def _build_safe_evidence(row: pd.Series, evidence: dict[str, Any], label_mode: str) -> dict[str, Any]:
    row_d = row.to_dict()
    oct_ev = evidence.get("oct_evidence", {}) or {}
    col_ev = evidence.get("colposcopy_evidence", {}) or {}
    instr = evidence.get("instruction_evidence", {}) or {}
    safe = {
        "case_uid": _case_uid(str(row["case_id"]), str(row["center_id"])),
        "center_uid": hashlib.sha256(str(row["center_id"]).encode("utf-8")).hexdigest()[:10],
        "endpoint": _safe_str(row.get("binary_label_endpoint", "CIN2+")),
        "age_group": _age_group(row.get("age", instr.get("age"))),
        "hpv": _sanitized_field(row.get("hpv", instr.get("hpv", "")), 80),
        "tct": _sanitized_field(row.get("tct", instr.get("tct", "")), 80),
        "available_modalities": evidence.get("available_modalities", {}),
        "oct_evidence": {
            "available": bool(oct_ev.get("available", not int(row.get("missing_oct", 0)))),
            "readable_images": int(oct_ev.get("readable_images", 0) or 0),
            "embedding": _embedding_stats(_safe_str(row.get("oct_embedding_path", oct_ev.get("embedding_path", "")))),
            "reliability": float(oct_ev.get("evidence_reliability", 0.0) or 0.0),
        },
        "colposcopy_evidence": {
            "available": bool(col_ev.get("available", not int(row.get("missing_colposcopy", 0)))),
            "readable_images": int(col_ev.get("readable_images", 0) or 0),
            "embedding": _embedding_stats(_safe_str(row.get("colposcopy_embedding_path", col_ev.get("embedding_path", "")))),
            "reliability": float(col_ev.get("evidence_reliability", 0.0) or 0.0),
        },
        "clinical_evidence": {
            "available": not int(row.get("missing_instruction", 0)),
            "age_group": _age_group(row.get("age", instr.get("age"))),
            "hpv": _sanitized_field(row.get("hpv", instr.get("hpv", "")), 80),
            "tct": _sanitized_field(row.get("tct", instr.get("tct", "")), 80),
        },
        "privacy_note": "No raw image, path, patient name, hospital name, or internal patient id is provided.",
    }
    if label_mode == "label_constrained":
        safe["weak_label"] = {
            "binary_label": int(row_d.get("binary_label", 0)),
            "label_text": _safe_str(row_d.get("binary_label_text", "")),
            "warning": "Weak supervision label for research pseudo-report generation only.",
        }
    else:
        safe["weak_label"] = "withheld"
    return safe


def _system_prompt() -> str:
    return (
        "You generate structured weak-supervision pseudo-reports for a de-identified cervical "
        "multimodal research dataset. Use only the supplied evidence. Do not invent pathology, "
        "do not mention patient identity, do not request raw images, and do not provide clinical "
        "deployment advice. Return valid JSON only."
    )


def _user_prompt(safe_evidence: dict[str, Any]) -> str:
    schema = {
        "diagnostic_summary": "short research-only summary",
        "oct_findings": "OCT-specific findings or explicit unavailable/limited evidence",
        "colposcopy_findings": "colposcopy-specific findings or explicit unavailable/limited evidence",
        "clinical_context": "age group, HPV, TCT, and other supplied structured clinical context",
        "impression": "weak-supervision impression, not a real diagnosis",
        "recommendation": "histopathology correlation / research disclaimer",
        "evidence_support": {
            "oct_supported": "boolean",
            "colposcopy_supported": "boolean",
            "instruction_supported": "boolean",
            "label_supported": "boolean",
        },
        "sentence_level_evidence": [
            {"statement": "string", "source": ["OCT", "colposcopy", "instruction", "label"], "confidence": "0..1"}
        ],
        "confidence": "0..1",
        "quality_flags": [],
        "weak_supervision_disclaimer": "Pseudo report - not a real clinical report.",
    }
    return (
        "Create one JSON object following this schema. Keep section names exactly as shown.\n"
        f"Schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"De-identified evidence:\n{json.dumps(safe_evidence, ensure_ascii=False, indent=2)}"
    )


def _normalize_report(
    report: dict[str, Any],
    row: pd.Series,
    provider: str,
    model: str,
    label_mode: str,
    parse_error: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if "generated_sections" in report and isinstance(report["generated_sections"], dict):
        merged = dict(report)
        merged.update(report["generated_sections"])
        report = merged
    out: dict[str, Any] = {}
    for field in REQUIRED_FIELDS:
        out[field] = _safe_str(report.get(field, "")).strip()
    out["case_id"] = str(row["case_id"])
    out["center_id"] = str(row["center_id"])
    out["agent_setting"] = "api_provider_label_constrained" if label_mode == "label_constrained" else "api_provider_label_blinded"
    out["agent_backend"] = provider
    out["provider_model"] = model
    out["confidence"] = _float_clipped(report.get("confidence", 0.70), default=0.70)
    out["evidence_support"] = report.get("evidence_support", {})
    if not isinstance(out["evidence_support"], dict):
        out["evidence_support"] = {}
    out["sentence_level_evidence"] = report.get("sentence_level_evidence", [])
    if not isinstance(out["sentence_level_evidence"], list):
        out["sentence_level_evidence"] = []
    out["quality_flags"] = report.get("quality_flags", [])
    if not isinstance(out["quality_flags"], list):
        out["quality_flags"] = []
    if parse_error:
        out["quality_flags"].append(parse_error)
    out["weak_supervision_disclaimer"] = _safe_str(
        report.get("weak_supervision_disclaimer", "Pseudo report - not a real clinical report.")
    )
    out["_provider_metadata"] = metadata or {}
    return out


def _float_clipped(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    return max(0.0, min(1.0, v))


def _baseline_template(row: pd.Series, label_mode: str) -> dict[str, Any]:
    label = int(row.get("binary_label", 0)) if label_mode == "label_constrained" else 0
    endpoint = _safe_str(row.get("binary_label_endpoint", "CIN2+"))
    if label == 1:
        impression = f"Template weak-supervision impression: suspicious for {endpoint}."
    else:
        impression = f"Template weak-supervision impression: negative for {endpoint}."
    return {
        "case_id": str(row["case_id"]),
        "center_id": str(row["center_id"]),
        "agent_setting": f"template_{label_mode}",
        "agent_backend": "label_template",
        "provider_model": "deterministic_template",
        "diagnostic_summary": "Fixed label-template pseudo-report for ablation.",
        "oct_findings": "Template OCT section; no modality-grounded OCT evidence is used.",
        "colposcopy_findings": "Template colposcopy section; no modality-grounded colposcopy evidence is used.",
        "clinical_context": "Template clinical context; structured clinical fields are not interpreted.",
        "impression": impression,
        "recommendation": "Research-only pseudo-report; correlate with histopathology.",
        "evidence_support": {
            "oct_supported": False,
            "colposcopy_supported": False,
            "instruction_supported": False,
            "label_supported": label_mode == "label_constrained",
        },
        "sentence_level_evidence": [],
        "confidence": 0.55,
        "quality_flags": ["template_baseline"],
        "weak_supervision_disclaimer": "Pseudo report - not a real clinical report.",
        "_provider_metadata": {"provider": "label_template", "model": "deterministic_template"},
    }


def _baseline_rule(row: pd.Series, evidence: dict[str, Any], label_mode: str) -> dict[str, Any]:
    label = int(row.get("binary_label", 0)) if label_mode == "label_constrained" else int(row.get("binary_label", 0))
    endpoint = _safe_str(row.get("binary_label_endpoint", "CIN2+"))
    oct_stats = _embedding_stats(_safe_str(row.get("oct_embedding_path", evidence.get("oct_evidence", {}).get("embedding_path", ""))))
    col_stats = _embedding_stats(
        _safe_str(row.get("colposcopy_embedding_path", evidence.get("colposcopy_evidence", {}).get("embedding_path", "")))
    )
    instr = evidence.get("instruction_evidence", {}) or {}
    oct_text = (
        f"OCT available with {int(evidence.get('oct_evidence', {}).get('readable_images', 0) or 0)} readable images; "
        f"embedding norm {oct_stats['norm']:.2f}, mean {oct_stats['mean']:.3f}."
    )
    col_text = (
        f"Colposcopy available with {int(evidence.get('colposcopy_evidence', {}).get('readable_images', 0) or 0)} readable images; "
        f"embedding norm {col_stats['norm']:.2f}, mean {col_stats['mean']:.3f}."
    )
    clinical = f"Clinical context: age group {_age_group(row.get('age', instr.get('age')))}; HPV {_sanitized_field(row.get('hpv', instr.get('hpv', '')))}; TCT {_sanitized_field(row.get('tct', instr.get('tct', '')))}."
    impression = (
        f"Rule baseline weak-supervision impression: suspicious for {endpoint}."
        if label == 1
        else f"Rule baseline weak-supervision impression: negative for {endpoint}."
    )
    rep = build_pseudo_report(
        case_id=str(row["case_id"]),
        center_id=str(row["center_id"]),
        label=label,
        endpoint=endpoint,
        oct_sum=oct_text,
        colpo_sum=col_text,
        clinical=clinical,
        setting=f"rule_{label_mode}",
        confidence=0.70,
    )
    rep["impression"] = impression
    rep["agent_backend"] = "rule_based"
    rep["provider_model"] = "deterministic_rule"
    rep["_provider_metadata"] = {"provider": "rule_based", "model": "deterministic_rule"}
    return rep


def _baseline_local_llm(row: pd.Series, evidence: dict[str, Any], label_mode: str) -> dict[str, Any]:
    setting = "modality_plus_label_agent" if label_mode == "label_constrained" else "modality_only_agent"
    rep = LocalLLMAgentClient(setting=setting).generate(evidence, row.to_dict())
    rep["agent_backend"] = "local_llm"
    rep["provider_model"] = "local_embedding_augmented"
    rep["_provider_metadata"] = {"provider": "local_llm", "model": "local_embedding_augmented"}
    return rep


def choose_cases(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    sub = df.copy()
    if "needs_pseudo_report" in sub.columns:
        sub = sub[sub["needs_pseudo_report"].astype(int) == 1]
    if args.split != "all" and "split" in sub.columns:
        sub = sub[sub["split"].astype(str) == args.split]
    if args.require_embedding and "has_visual_embedding" in sub.columns:
        sub = sub[sub["has_visual_embedding"].fillna(0).astype(int) == 1]
    sub = sub.sort_values(["center_id", "case_id"]).sample(frac=1.0, random_state=args.seed)
    if args.sample_size > 0:
        sub = sub.head(args.sample_size)
    return sub.reset_index(drop=True)


def generate_reports(df: pd.DataFrame, sample: pd.DataFrame, paths: dict[str, Path], args: argparse.Namespace) -> pd.DataFrame:
    specs = default_provider_specs()
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    status_rows: list[dict[str, Any]] = []
    for provider in BASELINE_PROVIDERS:
        if not args.include_baselines:
            continue
        for _, row in sample.iterrows():
            ev_path = ROOT / args.evidence_dir / str(row["center_id"]) / f"{row['case_id']}.json"
            evidence = _read_json(ev_path) if ev_path.is_file() else {}
            out_path = paths["reports"] / provider / str(row["center_id"]) / f"{row['case_id']}.json"
            if out_path.is_file() and not args.force:
                status_rows.append(_status_row(row, provider, "baseline", "cached", out_path))
                continue
            if provider == "label_template":
                report = _baseline_template(row, args.label_mode)
            elif provider == "rule_based":
                report = _baseline_rule(row, evidence, args.label_mode)
            else:
                report = _baseline_local_llm(row, evidence, args.label_mode)
            _write_json(out_path, report)
            status_rows.append(_status_row(row, provider, report.get("provider_model", "baseline"), "ok", out_path))

    for provider in providers:
        if provider not in specs:
            status_rows.append({"provider": provider, "status": "unknown_provider", "case_id": "", "center_id": ""})
            continue
        spec = specs[provider]
        if not spec.api_key:
            status_rows.append(
                {
                    "provider": provider,
                    "model": spec.model,
                    "status": "missing_api_key",
                    "required_env": spec.missing_key_names,
                    "case_id": "",
                    "center_id": "",
                    "report_path": "",
                }
            )
            continue
        client = LLMProviderClient(spec, timeout=args.timeout, max_retries=args.max_retries)
        for _, row in sample.iterrows():
            ev_path = ROOT / args.evidence_dir / str(row["center_id"]) / f"{row['case_id']}.json"
            if not ev_path.is_file():
                status_rows.append(_status_row(row, provider, spec.model, "missing_evidence", Path("")))
                continue
            out_path = paths["reports"] / provider / str(row["center_id"]) / f"{row['case_id']}.json"
            if out_path.is_file() and not args.force:
                status_rows.append(_status_row(row, provider, spec.model, "cached", out_path))
                continue
            evidence = _read_json(ev_path)
            safe_ev = _build_safe_evidence(row, evidence, args.label_mode)
            try:
                response = client.generate_json_report(_system_prompt(), _user_prompt(safe_ev), temperature=args.temperature)
                parsed, parse_error = _extract_json(response.text)
                metadata = {
                    "provider": provider,
                    "model": response.model,
                    "latency_seconds": response.latency_seconds,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                    "estimated_cost_usd": estimate_cost_usd(spec, response.prompt_tokens, response.completion_tokens),
                    "label_mode": args.label_mode,
                }
                report = _normalize_report(parsed, row, provider, spec.model, args.label_mode, parse_error, metadata)
                _write_json(out_path, report)
                if args.save_raw:
                    raw_path = paths["raw"] / provider / f"{row['case_id']}.json"
                    _write_json(raw_path, {"response": response.raw, "text": response.text, "safe_evidence": safe_ev})
                status = "ok" if not parse_error else "parse_warning"
                status_rows.append(_status_row(row, provider, spec.model, status, out_path, metadata))
            except Exception as exc:
                status_rows.append(
                    {
                        "case_id": row["case_id"],
                        "center_id": row["center_id"],
                        "provider": provider,
                        "model": spec.model,
                        "status": "error",
                        "error": str(exc)[:500],
                        "report_path": str(out_path),
                    }
                )
    status = pd.DataFrame(status_rows)
    _write_csv(status, paths["tables"] / "T_api_provider_generation_status.csv", paths["manuscript"])
    missing = status[status["status"].astype(str).eq("missing_api_key")].copy() if not status.empty else pd.DataFrame()
    _write_csv(missing, paths["tables"] / "T_api_provider_missing_api_keys.csv", paths["manuscript"])
    sample_out = sample[[c for c in ["case_id", "center_id", "split", "binary_label", "binary_label_endpoint"] if c in sample.columns]]
    _write_csv(sample_out, paths["tables"] / "T_api_provider_sample_cases.csv", None)
    return status


def _status_row(
    row: pd.Series,
    provider: str,
    model: str,
    status: str,
    out_path: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = metadata or {}
    return {
        "case_id": row.get("case_id", ""),
        "center_id": row.get("center_id", ""),
        "split": row.get("split", ""),
        "provider": provider,
        "model": model,
        "status": status,
        "report_path": str(out_path) if str(out_path) else "",
        "latency_seconds": meta.get("latency_seconds", np.nan),
        "prompt_tokens": meta.get("prompt_tokens", np.nan),
        "completion_tokens": meta.get("completion_tokens", np.nan),
        "total_tokens": meta.get("total_tokens", np.nan),
        "estimated_cost_usd": meta.get("estimated_cost_usd", np.nan),
    }


def _provider_reports(provider: str, sample: pd.DataFrame, reports_dir: Path) -> list[tuple[pd.Series, dict[str, Any], Path]]:
    records = []
    for _, row in sample.iterrows():
        path = reports_dir / provider / str(row["center_id"]) / f"{row['case_id']}.json"
        if path.is_file():
            records.append((row, _read_json(path), path))
    return records


def _support_flags(report: dict[str, Any]) -> dict[str, int]:
    support = report.get("evidence_support", {})
    if not isinstance(support, dict):
        support = {}

    def flag(name: str, section: str, positive_terms: tuple[str, ...], negative_terms: tuple[str, ...]) -> int:
        if name in support:
            return int(bool(support.get(name)))
        text = _safe_str(report.get(section, "")).lower()
        if any(term in text for term in negative_terms):
            return 0
        return int(any(term in text for term in positive_terms))

    return {
        "oct_supported": flag(
            "oct_supported",
            "oct_findings",
            ("oct", "b-scan", "microstructural", "embedding", "readable"),
            ("unavailable", "no modality-grounded", "template"),
        ),
        "colposcopy_supported": flag(
            "colposcopy_supported",
            "colposcopy_findings",
            ("colposcopy", "colposcopic", "acetowhite", "vascular", "embedding", "readable"),
            ("unavailable", "no modality-grounded", "template"),
        ),
        "instruction_supported": flag(
            "instruction_supported",
            "clinical_context",
            ("age", "hpv", "tct", "clinical"),
            ("unavailable", "not interpreted", "template"),
        ),
        "label_supported": int(bool(support.get("label_supported", "label" in _report_text(report).lower()))),
    }


def _contradiction_rate_flag(text: str, label: int) -> int:
    t = text.lower()
    positive = any(k in t for k in ("suspicious", "positive", "high-grade", "cin2+ positive", "cin2 positive"))
    negative = any(k in t for k in ("negative", "no definitive", "no high-grade", "nil"))
    if positive and negative:
        return 1
    if label == 1 and negative and not positive:
        return 1
    if label == 0 and positive and not negative:
        return 1
    return 0


def evaluate_quality(sample: pd.DataFrame, reports_dir: Path, paths: dict[str, Path], status: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    providers = sorted({p.name for p in reports_dir.iterdir() if p.is_dir()}) if reports_dir.is_dir() else []
    case_rows: list[dict[str, Any]] = []
    for provider in providers:
        for row, report, path in _provider_reports(provider, sample, reports_dir):
            text = _report_text(report)
            qc = qc_pseudo_report(report, row.to_dict())
            support = _support_flags(report)
            case_rows.append(
                {
                    "case_id": row["case_id"],
                    "center_id": row["center_id"],
                    "provider": provider,
                    "model": report.get("provider_model", report.get("_provider_metadata", {}).get("model", "")),
                    "json_valid": 1,
                    "schema_valid": int(all(f in report for f in REQUIRED_FIELDS)),
                    "section_complete": int(all(_safe_str(report.get(f, "")).strip() for f in REQUIRED_FIELDS)),
                    "oct_supported": support["oct_supported"],
                    "colposcopy_supported": support["colposcopy_supported"],
                    "instruction_supported": support["instruction_supported"],
                    "mean_modality_support": np.mean(
                        [support["oct_supported"], support["colposcopy_supported"], support["instruction_supported"]]
                    ),
                    "label_consistency": label_consistency(text, int(row.get("binary_label", 0))),
                    "contradiction": _contradiction_rate_flag(text, int(row.get("binary_label", 0))),
                    "hallucination": int("pathology_hallucination" in _safe_str(qc.get("qc_flags", ""))),
                    "qc_pass": qc["pseudo_report_pass_qc"],
                    "qc_score": qc["qc_score"],
                    "report_chars": len(text),
                    "normalized_report_text": _normalize_text(text),
                    "report_path": str(path),
                    "latency_seconds": report.get("_provider_metadata", {}).get("latency_seconds", np.nan),
                    "estimated_cost_usd": report.get("_provider_metadata", {}).get("estimated_cost_usd", np.nan),
                    "prompt_tokens": report.get("_provider_metadata", {}).get("prompt_tokens", np.nan),
                    "completion_tokens": report.get("_provider_metadata", {}).get("completion_tokens", np.nan),
                }
            )
    per_case = pd.DataFrame(case_rows)
    if per_case.empty:
        agg = pd.DataFrame()
    else:
        agg_rows = []
        for provider, g in per_case.groupby("provider"):
            value_counts = g["normalized_report_text"].value_counts()
            max_dup = float(value_counts.max() / len(g)) if len(g) else np.nan
            agg_rows.append(
                {
                    "provider": provider,
                    "n_cases": len(g),
                    "schema_valid_rate": float(g["schema_valid"].mean()),
                    "section_completeness": float(g["section_complete"].mean()),
                    "oct_supported_rate": float(g["oct_supported"].mean()),
                    "colposcopy_supported_rate": float(g["colposcopy_supported"].mean()),
                    "instruction_supported_rate": float(g["instruction_supported"].mean()),
                    "mean_modality_support_rate": float(g["mean_modality_support"].mean()),
                    "label_consistency_mean": float(g["label_consistency"].mean()),
                    "contradiction_rate": float(g["contradiction"].mean()),
                    "hallucination_rate": float(g["hallucination"].mean()),
                    "qc_pass_rate": float(g["qc_pass"].mean()),
                    "qc_score_mean": float(g["qc_score"].mean()),
                    "unique_text_rate": float(g["normalized_report_text"].nunique() / len(g)),
                    "max_duplicate_fraction": max_dup,
                    "mean_latency_seconds": _nanmean(g["latency_seconds"]),
                    "mean_estimated_cost_usd": _nanmean(g["estimated_cost_usd"]),
                    "cost_per_1000_cases_usd": _nanmean(g["estimated_cost_usd"]) * 1000
                    if not math.isnan(_nanmean(g["estimated_cost_usd"]))
                    else np.nan,
                    "mean_prompt_tokens": _nanmean(g["prompt_tokens"]),
                    "mean_completion_tokens": _nanmean(g["completion_tokens"]),
                }
            )
        agg = pd.DataFrame(agg_rows)
    missing = status[status["status"].astype(str).eq("missing_api_key")].copy() if not status.empty else pd.DataFrame()
    for provider in missing["provider"].tolist() if not missing.empty else []:
        if agg.empty or provider not in set(agg["provider"]):
            agg = pd.concat(
                [
                    agg,
                    pd.DataFrame(
                        [
                            {
                                "provider": provider,
                                "n_cases": 0,
                                "schema_valid_rate": np.nan,
                                "section_completeness": np.nan,
                                "mean_modality_support_rate": np.nan,
                                "qc_pass_rate": np.nan,
                                "status": "missing_api_key",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    _write_csv(per_case.drop(columns=["normalized_report_text"], errors="ignore"), paths["tables"] / "T_api_provider_quality_per_case.csv", None)
    _write_csv(agg, paths["tables"] / "T_api_provider_quality_comparison.csv", paths["manuscript"])
    _plot_quality(agg, paths["figures"] / "Figure_api_provider_quality_comparison")
    return per_case, agg


def _nanmean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return np.nan
    return float(values.mean())


def _plot_quality(agg: pd.DataFrame, out_base: Path) -> None:
    if agg.empty:
        return
    plot = agg[pd.to_numeric(agg.get("n_cases", 0), errors="coerce").fillna(0) > 0].copy()
    if plot.empty:
        return
    _setup_plot_style()
    plot["display_provider"] = plot["provider"].map(_provider_label)
    metrics = ["schema_valid_rate", "section_completeness", "mean_modality_support_rate", "qc_pass_rate"]
    metric_labels = {
        "schema_valid_rate": "Schema valid",
        "section_completeness": "Section complete",
        "mean_modality_support_rate": "Modality support",
        "qc_pass_rate": "QC pass",
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    x = np.arange(len(plot))
    width = 0.18
    for i, metric in enumerate(metrics):
        xpos = x + (i - 1.5) * width
        ax.bar(
            xpos,
            plot[metric].astype(float),
            width=width,
            label=metric_labels.get(metric, metric),
            color=MORANDI_HEX[i],
            edgecolor=EDGE_DARK,
            linewidth=0.8,
            alpha=0.86,
            zorder=1,
        )
        ax.scatter(xpos, plot[metric].astype(float), marker="s", s=28, color=MORANDI_HEX[(i + 4) % len(MORANDI_HEX)], edgecolor=TEXT_DARK, linewidth=0.6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(plot["display_provider"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Structured pseudo-report quality by model")
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=1, title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _save_plot(fig, out_base)


def build_provider_manifests(
    base_df: pd.DataFrame,
    sample: pd.DataFrame,
    reports_dir: Path,
    paths: dict[str, Path],
) -> pd.DataFrame:
    rows = []
    providers = sorted({p.name for p in reports_dir.iterdir() if p.is_dir()}) if reports_dir.is_dir() else []
    for provider in providers:
        df = base_df.copy()
        n_written = 0
        for row, report, path in _provider_reports(provider, sample, reports_dir):
            mask = (df["case_id"].astype(str) == str(row["case_id"])) & (
                df["center_id"].astype(str) == str(row["center_id"])
            )
            if not mask.any():
                continue
            idx = df[mask].index[0]
            qc = qc_pseudo_report(report, df.loc[idx].to_dict())
            text = json.dumps(report, ensure_ascii=False)
            df.at[idx, "has_pseudo_report"] = 1
            df.at[idx, "pseudo_report_path"] = str(path)
            df.at[idx, "pseudo_report_text"] = text[:4000]
            df.at[idx, "pseudo_report_pass_qc"] = qc["pseudo_report_pass_qc"]
            df.at[idx, "pseudo_report_confidence"] = qc["pseudo_report_confidence"]
            df.at[idx, "qc_score"] = qc["qc_score"]
            df.at[idx, "pseudo_training_weight"] = qc["pseudo_training_weight"]
            df.at[idx, "qc_flags"] = qc["qc_flags"]
            df.at[idx, "api_provider_report_source"] = provider
            df.at[idx, "api_provider_report_path"] = str(path)
            if int(qc["pseudo_report_pass_qc"]) == 1:
                df.at[idx, "training_report_type"] = "pseudo"
                df.at[idx, "training_report_text"] = text[:4000]
            n_written += 1
        manifest_path = paths["manifests"] / f"full_manifest_with_{provider}_api_pseudo.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(manifest_path, index=False)
        rows.append({"provider": provider, "manifest_path": str(manifest_path), "n_overwritten_pseudo_reports": n_written})
    index = pd.DataFrame(rows)
    _write_csv(index, paths["tables"] / "T_api_provider_manifest_index.csv", paths["manuscript"])
    return index


def _checkpoint_path() -> Path:
    candidates = [
        ROOT / "outputs/publishable/baselines/full_lcad_rasa/best.ckpt",
        ROOT / "outputs/publishable/checkpoints/publishable_full_lcad_rasa/best.ckpt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def _text_ids(text: str, max_len: int = 128, vocab_size: int = 8192) -> torch.Tensor:
    ids = [hash(w) % vocab_size for w in text.split()[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids[:max_len], dtype=torch.long)


@torch.no_grad()
def evaluate_alignment(sample: pd.DataFrame, reports_dir: Path, paths: dict[str, Path], max_cases: int, device_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ckpt = _checkpoint_path()
    if not ckpt.is_file():
        per_section = pd.DataFrame()
        macro = pd.DataFrame([{"status": "missing_checkpoint", "checkpoint": str(ckpt)}])
        _write_csv(macro, paths["tables"] / "T_api_provider_alignment_comparison.csv", paths["manuscript"])
        return per_section, macro
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = PublishableLCADRASA().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    providers = sorted({p.name for p in reports_dir.iterdir() if p.is_dir()}) if reports_dir.is_dir() else []
    rows = []
    for provider in providers:
        recs = _provider_reports(provider, sample, reports_dir)
        if max_cases > 0:
            recs = recs[:max_cases]
        if len(recs) < 4:
            continue
        projs: dict[str, list[np.ndarray]] = {k: [] for k in SECTION_KEYS}
        targets: dict[str, list[np.ndarray]] = {k: [] for k in SECTION_KEYS}
        for row, report, _ in recs:
            text = _report_text(report)
            ids = _text_ids(text).unsqueeze(0).to(device)
            oct_e = torch.tensor(load_visual_emb(str(row.get("oct_embedding_path", ""))), dtype=torch.float32).unsqueeze(0).to(device)
            col_e = torch.tensor(load_visual_emb(str(row.get("colposcopy_embedding_path", ""))), dtype=torch.float32).unsqueeze(0).to(device)
            fus_e = torch.tensor(load_visual_emb(str(row.get("fused_visual_embedding_path", ""))), dtype=torch.float32).unsqueeze(0).to(device)
            ins = torch.tensor(instr_vector(row.to_dict()), dtype=torch.float32).unsqueeze(0).to(device)
            label = torch.tensor([int(row.get("binary_label", 0))], dtype=torch.long).to(device)
            out = model(oct_e, col_e, fus_e, ins, ids, label)
            h0 = out["fused"]
            hidden = out["hidden"]
            n = hidden.size(1)
            q = max(1, n // 4)
            sec_targets = {
                "oct_findings": hidden[:, :q].mean(1),
                "colposcopy_findings": hidden[:, q : 2 * q].mean(1),
                "clinical_context": hidden[:, 2 * q : 3 * q].mean(1),
                "impression": hidden[:, 3 * q :].mean(1),
            }
            sec_projs = {
                "oct_findings": model.sec_oct(h0),
                "colposcopy_findings": model.sec_col(h0),
                "clinical_context": model.sec_instr(h0),
                "impression": model.sec_imp(h0),
            }
            for section in SECTION_KEYS:
                projs[section].append(F.normalize(sec_projs[section], dim=-1).cpu().numpy()[0])
                targets[section].append(F.normalize(sec_targets[section], dim=-1).cpu().numpy()[0])
        for section in SECTION_KEYS:
            p_mat = np.vstack(projs[section])
            t_mat = np.vstack(targets[section])
            sim = p_mat @ t_mat.T
            ranks = []
            for i in range(sim.shape[0]):
                order = np.argsort(-sim[i])
                rank = int(np.where(order == i)[0][0]) + 1
                ranks.append(rank)
            ranks_arr = np.asarray(ranks)
            rows.append(
                {
                    "provider": provider,
                    "section": section,
                    "n_cases": sim.shape[0],
                    "recall_at_1": float(np.mean(ranks_arr <= 1)),
                    "recall_at_5": float(np.mean(ranks_arr <= 5)),
                    "mrr": float(np.mean(1.0 / ranks_arr)),
                    "positive_cosine": float(np.mean(np.diag(sim))),
                }
            )
    per_section = pd.DataFrame(rows)
    if per_section.empty:
        macro = pd.DataFrame()
    else:
        macro = (
            per_section.groupby("provider")
            .agg(
                n_sections=("section", "nunique"),
                mean_n_cases=("n_cases", "mean"),
                macro_recall_at_1=("recall_at_1", "mean"),
                macro_recall_at_5=("recall_at_5", "mean"),
                macro_mrr=("mrr", "mean"),
                macro_positive_cosine=("positive_cosine", "mean"),
            )
            .reset_index()
            .sort_values("macro_mrr", ascending=False)
        )
    _write_csv(per_section, paths["tables"] / "T_api_provider_alignment_by_section.csv", None)
    _write_csv(macro, paths["tables"] / "T_api_provider_alignment_comparison.csv", paths["manuscript"])
    _plot_alignment(macro, paths["figures"] / "Figure_api_provider_alignment_mrr")
    return per_section, macro


def _plot_alignment(macro: pd.DataFrame, out_base: Path) -> None:
    if macro.empty or "macro_mrr" not in macro.columns:
        return
    plot = macro.copy()
    _setup_plot_style()
    plot["display_provider"] = plot["provider"].map(_provider_label)
    plot = plot.sort_values("macro_mrr", ascending=True)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    y = np.arange(len(plot))
    colors = [MORANDI_HEX[i % len(MORANDI_HEX)] for i in range(len(plot))]
    ax.barh(y, plot["macro_mrr"].astype(float), color=colors, edgecolor=EDGE_DARK, linewidth=0.9, alpha=0.86, zorder=1)
    ax.scatter(plot["macro_mrr"].astype(float), y, marker="D", s=46, color=MORANDI_HEX[4], edgecolor=TEXT_DARK, linewidth=0.7, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["display_provider"])
    ax.set_ylabel("Macro MRR")
    ax.set_xlabel("Macro MRR")
    ax.set_ylabel("")
    ax.set_title("Modality-section alignment by model")
    ax.grid(axis="x")
    fig.tight_layout()
    _save_plot(fig, out_base)


def rank_providers(quality: pd.DataFrame, alignment: pd.DataFrame, paths: dict[str, Path]) -> pd.DataFrame:
    if quality.empty:
        ranking = pd.DataFrame()
        _write_csv(ranking, paths["tables"] / "T_api_provider_candidate_ranking.csv", paths["manuscript"])
        return ranking
    q = quality.copy()
    if not alignment.empty and "provider" in alignment.columns:
        q = q.merge(alignment[["provider", "macro_mrr", "macro_recall_at_1", "macro_recall_at_5"]], on="provider", how="left")
    else:
        q["macro_mrr"] = np.nan
    valid = pd.to_numeric(q.get("n_cases", 0), errors="coerce").fillna(0) > 0
    mrr = pd.to_numeric(q["macro_mrr"], errors="coerce")
    if mrr.notna().any() and float(mrr.max()) > float(mrr.min()):
        q["alignment_score_norm"] = (mrr - mrr.min()) / (mrr.max() - mrr.min())
    else:
        q["alignment_score_norm"] = mrr.fillna(0.0)
    q["composite_score"] = np.where(
        valid,
        0.20 * pd.to_numeric(q.get("schema_valid_rate", 0), errors="coerce").fillna(0)
        + 0.20 * pd.to_numeric(q.get("section_completeness", 0), errors="coerce").fillna(0)
        + 0.20 * pd.to_numeric(q.get("mean_modality_support_rate", 0), errors="coerce").fillna(0)
        + 0.20 * pd.to_numeric(q.get("qc_pass_rate", 0), errors="coerce").fillna(0)
        + 0.15 * q["alignment_score_norm"].fillna(0)
        + 0.05 * (1.0 - pd.to_numeric(q.get("max_duplicate_fraction", 1), errors="coerce").fillna(1)),
        np.nan,
    )
    q["is_api_provider"] = q["provider"].isin(API_PROVIDERS).astype(int)
    q = q.sort_values(["is_api_provider", "composite_score"], ascending=[False, False])
    _write_csv(q, paths["tables"] / "T_api_provider_candidate_ranking.csv", paths["manuscript"])
    _plot_ranking(q, paths["figures"] / "Figure_api_provider_candidate_ranking")
    return q


def _plot_ranking(ranking: pd.DataFrame, out_base: Path) -> None:
    if ranking.empty or "composite_score" not in ranking.columns:
        return
    plot = ranking[pd.to_numeric(ranking["composite_score"], errors="coerce").notna()].copy()
    if plot.empty:
        return
    _setup_plot_style()
    plot["display_provider"] = plot["provider"].map(_provider_label)
    plot = plot.sort_values("composite_score", ascending=True)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    y = np.arange(len(plot))
    colors = [MORANDI_HEX[0] if p in API_PROVIDERS else MORANDI_HEX[6] for p in plot["provider"]]
    ax.hlines(y, 0, plot["composite_score"].astype(float), color=GRID_LINE, linewidth=6, alpha=0.8)
    ax.scatter(plot["composite_score"].astype(float), y, color=colors, edgecolor=TEXT_DARK, linewidth=0.8, s=90, marker="o", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["display_provider"])
    ax.set_ylim(-0.6, len(plot) - 0.4)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Composite selection score")
    ax.set_ylabel("")
    ax.set_title("Candidate model ranking for downstream experiment")
    ax.grid(axis="x")
    fig.tight_layout()
    _save_plot(fig, out_base)


def _dense_features(df: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, row in df.iterrows():
        oct_v = _embedding_stats(_safe_str(row.get("oct_embedding_path", "")))
        col_v = _embedding_stats(_safe_str(row.get("colposcopy_embedding_path", "")))
        fus_v = _embedding_stats(_safe_str(row.get("fused_visual_embedding_path", "")))
        ins = instr_vector(row.to_dict())
        rows.append(
            [
                oct_v["norm"],
                oct_v["mean"],
                oct_v["std"],
                col_v["norm"],
                col_v["mean"],
                col_v["std"],
                fus_v["norm"],
                fus_v["mean"],
                fus_v["std"],
                *ins[:8].tolist(),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _fit_surrogate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text_col = "training_report_text"
    train_text = train_df[text_col].fillna("").astype(str).tolist()
    test_text = test_df[text_col].fillna("").astype(str).tolist()
    vec = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2")
    x_train_text = vec.transform(train_text)
    x_test_text = vec.transform(test_text)
    scaler = StandardScaler()
    x_train_dense = scaler.fit_transform(_dense_features(train_df))
    x_test_dense = scaler.transform(_dense_features(test_df))
    x_train = sparse.hstack([sparse.csr_matrix(x_train_dense), x_train_text], format="csr")
    x_test = sparse.hstack([sparse.csr_matrix(x_test_dense), x_test_text], format="csr")
    y_train = train_df["binary_label"].astype(int).to_numpy()
    y_test = test_df["binary_label"].astype(int).to_numpy()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    clf.fit(x_train, y_train)
    p_train = clf.predict_proba(x_train)[:, 1]
    p_test = clf.predict_proba(x_test)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_train, (p_train >= t).astype(int), zero_division=0) for t in thresholds]
    threshold = thresholds[int(np.argmax(f1s))]
    return y_test, p_test, np.asarray([threshold])


def run_downstream_scarcity(
    manifest_index: pd.DataFrame,
    ranking: pd.DataFrame,
    base_manifest: Path,
    paths: dict[str, Path],
    top_k: int,
) -> pd.DataFrame:
    if ranking.empty:
        out = pd.DataFrame([{"status": "skipped_no_provider_ranking"}])
        _write_csv(out, paths["tables"] / "T_api_provider_downstream_scarcity_surrogate.csv", paths["manuscript"])
        return out
    api_rank = ranking[(ranking["provider"].isin(API_PROVIDERS)) & (ranking["composite_score"].notna())].copy()
    if api_rank.empty:
        out = pd.DataFrame([{"status": "skipped_no_api_provider_results"}])
        _write_csv(out, paths["tables"] / "T_api_provider_downstream_scarcity_surrogate.csv", paths["manuscript"])
        return out
    selected = api_rank.sort_values("composite_score", ascending=False).head(top_k)["provider"].tolist()
    manifest_map = {r["provider"]: Path(r["manifest_path"]) for _, r in manifest_index.iterrows()}
    base_df = pd.read_csv(base_manifest)
    test_df = base_df[base_df["split"].astype(str).eq("test")].copy()
    if test_df.empty:
        out = pd.DataFrame([{"status": "skipped_no_test_split"}])
        _write_csv(out, paths["tables"] / "T_api_provider_downstream_scarcity_surrogate.csv", paths["manuscript"])
        return out
    fractions = [1.0, 0.5, 0.25, 0.1]
    seeds = [42, 123, 456, 789, 2026]
    rows = []
    for provider in selected:
        if provider not in manifest_map or not manifest_map[provider].is_file():
            continue
        df = pd.read_csv(manifest_map[provider])
        train_all = df[df["split"].astype(str).eq("train")].copy()
        real_train = train_all[train_all["has_real_report"].fillna(0).astype(int).eq(1)]
        pseudo_train = train_all[
            train_all.get("api_provider_report_source", "").fillna("").astype(str).eq(provider)
            & train_all["pseudo_report_pass_qc"].fillna(0).astype(int).eq(1)
            & train_all["training_report_text"].fillna("").astype(str).str.len().gt(0)
        ]
        for frac in fractions:
            for seed in seeds:
                if real_train.empty or pseudo_train.empty:
                    rows.append({"provider": provider, "real_report_fraction": frac, "seed": seed, "status": "skipped_empty_train"})
                    continue
                n_real = max(1, int(round(len(real_train) * frac)))
                real_sub = real_train.sample(n=min(n_real, len(real_train)), random_state=seed)
                train_df = pd.concat([real_sub, pseudo_train], ignore_index=True)
                try:
                    y_test, p_test, thr_arr = _fit_surrogate(train_df, test_df)
                    if len(np.unique(y_test)) < 2:
                        auc = np.nan
                    else:
                        auc = roc_auc_score(y_test, p_test)
                    threshold = float(thr_arr[0])
                    f1 = f1_score(y_test, (p_test >= threshold).astype(int), zero_division=0)
                    rows.append(
                        {
                            "provider": provider,
                            "real_report_fraction": frac,
                            "seed": seed,
                            "n_train": len(train_df),
                            "n_real_train": len(real_sub),
                            "n_api_pseudo_train": len(pseudo_train),
                            "threshold_train_selected": threshold,
                            "auc": auc,
                            "f1": f1,
                            "status": "ok",
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "provider": provider,
                            "real_report_fraction": frac,
                            "seed": seed,
                            "status": "error",
                            "error": str(exc)[:300],
                        }
                    )
    raw = pd.DataFrame(rows)
    if raw.empty or "auc" not in raw.columns:
        out = raw if not raw.empty else pd.DataFrame([{"status": "skipped_no_rows"}])
    else:
        ok = raw[raw["status"].astype(str).eq("ok")].copy()
        if ok.empty:
            out = raw
        else:
            out = (
                ok.groupby(["provider", "real_report_fraction"])
                .agg(
                    n_runs=("seed", "nunique"),
                    mean_n_train=("n_train", "mean"),
                    mean_n_real_train=("n_real_train", "mean"),
                    mean_n_api_pseudo_train=("n_api_pseudo_train", "mean"),
                    auc_mean=("auc", "mean"),
                    auc_std=("auc", "std"),
                    f1_mean=("f1", "mean"),
                    f1_std=("f1", "std"),
                )
                .reset_index()
                .sort_values(["provider", "real_report_fraction"], ascending=[True, False])
            )
    _write_csv(raw, paths["tables"] / "T_api_provider_downstream_scarcity_surrogate_raw.csv", None)
    _write_csv(out, paths["tables"] / "T_api_provider_downstream_scarcity_surrogate.csv", paths["manuscript"])
    _plot_scarcity(out, paths["figures"] / "Figure_api_provider_downstream_scarcity_surrogate")
    return out


def _plot_scarcity(out: pd.DataFrame, out_base: Path) -> None:
    if out.empty or "auc_mean" not in out.columns:
        return
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    providers = list(out["provider"].drop_duplicates())
    palette = {p: MORANDI_HEX[i % len(MORANDI_HEX)] for i, p in enumerate(providers)}
    offsets = np.linspace(-0.025, 0.025, max(1, len(providers)))
    off_map = {p: offsets[i] for i, p in enumerate(providers)}
    for provider, g in out.groupby("provider"):
        g = g.sort_values("real_report_fraction")
        xs = g["real_report_fraction"].astype(float) + off_map[provider]
        ax.plot(xs, g["auc_mean"], color=palette[provider], linewidth=1.4, alpha=0.82)
        ax.scatter(xs, g["auc_mean"], marker="o", s=78, color=palette[provider], edgecolor=TEXT_DARK, linewidth=0.8, label=_provider_label(provider), zorder=3)
        if "auc_std" in g.columns:
            ax.errorbar(xs, g["auc_mean"], yerr=g["auc_std"].fillna(0), fmt="none", ecolor=TEXT_DARK, elinewidth=1.0, capsize=3, zorder=2)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.set_xlabel("Available real-report supervision fraction")
    ax.set_ylabel("AUROC on locked test set")
    ax.set_title("Report-supervision scarcity surrogate")
    ax.grid(axis="y")
    ax.legend(frameon=False, title="Model")
    fig.tight_layout()
    _save_plot(fig, out_base)


def write_summary(
    paths: dict[str, Path],
    status: pd.DataFrame,
    quality: pd.DataFrame,
    alignment: pd.DataFrame,
    ranking: pd.DataFrame,
    downstream: pd.DataFrame,
    args: argparse.Namespace,
) -> Path:
    summary = paths["out"] / "LLM_API_PROVIDER_COMPARISON_SUMMARY.md"
    lines = [
        "# LLM API Provider Comparison Summary\n\n",
        "Scope: Qwen/GLM/Gemini/GPT/MiniMax structured pseudo-report generation, QC, semantic alignment, and downstream scarcity surrogate.\n\n",
        "## Configuration\n\n",
        f"- sample_size: `{args.sample_size}`\n",
        f"- split: `{args.split}`\n",
        f"- label_mode: `{args.label_mode}`\n",
        f"- providers_requested: `{args.providers}`\n",
        "- privacy: raw images, paths, patient names, hospital names, and internal patient IDs are not sent to APIs.\n\n",
        "## Generated Tables\n\n",
        "- `tables/T_api_provider_generation_status.csv`\n",
        "- `tables/T_api_provider_quality_comparison.csv`\n",
        "- `tables/T_api_provider_alignment_comparison.csv`\n",
        "- `tables/T_api_provider_candidate_ranking.csv`\n",
        "- `tables/T_api_provider_downstream_scarcity_surrogate.csv`\n\n",
    ]
    if not status.empty:
        missing = status[status["status"].astype(str).eq("missing_api_key")]
        if not missing.empty:
            missing_providers = ", ".join(sorted(missing["provider"].dropna().unique()))
            lines.append("## API Execution Status\n\n")
            lines.append(
                f"- External API calls were not completed for missing-key providers: `{missing_providers}`.\n"
            )
            lines.append("- Baseline providers may still be present; do not label missing-key rows as Qwen/GLM/Gemini/GPT results.\n\n")
    if not quality.empty:
        available = quality[pd.to_numeric(quality.get("n_cases", 0), errors="coerce").fillna(0) > 0]
        if not available.empty:
            best_quality = available.sort_values("mean_modality_support_rate", ascending=False).iloc[0]
            lines.append("## Current Quality Finding\n\n")
            lines.append(
                f"- Highest modality-support provider among available outputs: `{best_quality['provider']}` "
                f"(mean support={float(best_quality['mean_modality_support_rate']):.3f}).\n"
            )
    if not alignment.empty and "macro_mrr" in alignment.columns and alignment["macro_mrr"].notna().any():
        top_align = alignment.sort_values("macro_mrr", ascending=False).iloc[0]
        lines.append(
            f"- Highest available modality-section macro MRR: `{top_align['provider']}` "
            f"(MRR={float(top_align['macro_mrr']):.3f}).\n"
        )
    if not ranking.empty and "composite_score" in ranking.columns:
        api_rank = ranking[(ranking["provider"].isin(API_PROVIDERS)) & (ranking["composite_score"].notna())]
        if api_rank.empty:
            lines.append("- No API provider has real generated outputs yet, so no top API model was selected for downstream training.\n")
        else:
            selected = ", ".join(api_rank.sort_values("composite_score", ascending=False).head(args.downstream_top_k)["provider"])
            lines.append(f"- Selected API provider(s) for downstream scarcity surrogate: `{selected}`.\n")
    if not downstream.empty and "auc_mean" in downstream.columns:
        low = downstream[downstream["real_report_fraction"] == downstream["real_report_fraction"].min()]
        for _, row in low.iterrows():
            lines.append(
                f"- Downstream surrogate at {row['real_report_fraction']:.0%} real reports: "
                f"{row['provider']} AUROC={row['auc_mean']:.3f}.\n"
            )
    lines.append("\n## Claim Limits\n\n")
    lines.append("- Missing-key providers are infrastructure status rows, not experimental evidence.\n")
    lines.append("- Scarcity surrogate is not a substitute for full LCAD-RASA retraining with provider-specific pseudo-reports.\n")
    lines.append("- Use provider comparison for pseudo-report quality and alignment robustness, not clinical deployment claims.\n")
    summary.write_text("".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="outputs/publishable/manifests/full_manifest_publishable_with_llm_pseudo.csv")
    p.add_argument("--evidence_dir", default="outputs/publishable/modality_evidence")
    p.add_argument("--output_dir", default="outputs/publishable/llm_api_provider_comparison")
    p.add_argument("--providers", default="qwen,glm,gemini,gpt")
    p.add_argument("--sample_size", type=int, default=100)
    p.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--label_mode", default="label_constrained", choices=["label_constrained", "label_blinded"])
    p.add_argument("--require_embedding", action="store_true", default=True)
    p.add_argument("--include_baselines", action="store_true", default=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--save_raw", action="store_true")
    p.add_argument("--alignment_max_cases", type=int, default=100)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--downstream_top_k", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = ROOT / args.output_dir
    paths = {
        "out": out,
        "tables": out / "tables",
        "figures": out / "figures",
        "reports": out / "reports",
        "raw": out / "raw",
        "manifests": out / "manifests",
        "manuscript": ROOT / "outputs/publishable/tables/manuscript",
    }
    for key in ("tables", "figures", "reports", "manifests"):
        paths[key].mkdir(parents=True, exist_ok=True)

    manifest = ROOT / args.manifest
    df = pd.read_csv(manifest)
    sample = choose_cases(df, args)
    print(f"Selected {len(sample)} pseudo-report candidate cases from split={args.split}.")

    print("[1/6] Generate baseline/API pseudo reports")
    status = generate_reports(df, sample, paths, args)
    print(status["status"].value_counts(dropna=False).to_string() if not status.empty else "No status rows")

    print("[2/6] QC and provider quality comparison")
    per_case_quality, quality = evaluate_quality(sample, paths["reports"], paths, status)
    print(quality.to_string(index=False) if not quality.empty else "No quality rows")

    print("[3/6] Build provider-specific manifests")
    manifest_index = build_provider_manifests(df, sample, paths["reports"], paths)
    print(manifest_index.to_string(index=False) if not manifest_index.empty else "No manifests")

    print("[4/6] Modality-section alignment comparison")
    per_section, alignment = evaluate_alignment(sample, paths["reports"], paths, args.alignment_max_cases, args.device)
    print(alignment.to_string(index=False) if not alignment.empty else "No alignment rows")

    print("[5/6] Rank providers and run top-API downstream scarcity surrogate")
    ranking = rank_providers(quality, alignment, paths)
    print(ranking[["provider", "n_cases", "composite_score"]].to_string(index=False) if not ranking.empty else "No ranking")
    downstream = run_downstream_scarcity(manifest_index, ranking, manifest, paths, args.downstream_top_k)
    print(downstream.to_string(index=False) if not downstream.empty else "No downstream rows")

    print("[6/6] Write summary")
    summary = write_summary(paths, status, quality, alignment, ranking, downstream, args)
    print(f"Wrote summary: {summary}")


if __name__ == "__main__":
    main()
