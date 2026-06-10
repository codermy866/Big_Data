"""Section-wise semantic retrieval bank for KRA-RASA.

The implementation deliberately avoids external API calls. It turns existing
real/pseudo report sections into structured semantic entities, then retrieves
training-bank entities for each case using a blend of clinical text signatures
and reduced visual embeddings.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SECTION_COLUMNS = {
    "diagnostic_summary": "reference_diagnostic_summary",
    "oct_findings": "reference_oct_findings",
    "colposcopy_findings": "reference_colposcopy_findings",
    "clinical_context": "reference_clinical_context",
    "impression": "reference_impression",
    "recommendation": "reference_recommendation",
    "report_anchor": "training_report_text",
}

SECTION_MODALITY = {
    "diagnostic_summary": "multimodal",
    "oct_findings": "oct",
    "colposcopy_findings": "colposcopy",
    "clinical_context": "clinical",
    "impression": "impression",
    "recommendation": "clinical",
    "report_anchor": "multimodal",
}

KEYWORDS = {
    "oct": [
        "oct",
        "b-scan",
        "epithelial",
        "stroma",
        "microstructure",
        "上皮",
        "间质",
        "基底膜",
        "腺体",
        "微结构",
    ],
    "colposcopy": [
        "colposcopy",
        "acetowhite",
        "mosaic",
        "punctation",
        "vessel",
        "阴道镜",
        "醋酸白",
        "镶嵌",
        "点状",
        "血管",
        "转化区",
    ],
    "clinical": [
        "hpv",
        "tct",
        "age",
        "cytology",
        "clinical",
        "感染",
        "年龄",
        "细胞学",
        "临床",
    ],
    "impression": [
        "cin",
        "cin2",
        "cin3",
        "hsil",
        "lsil",
        "cancer",
        "癌",
        "高级别",
        "低级别",
        "上皮内瘤",
        "病变",
        "宫颈炎",
        "湿疣",
    ],
}


@dataclass(frozen=True)
class RetrievalArtifacts:
    manifest: pd.DataFrame
    bank: pd.DataFrame
    retrievals: pd.DataFrame
    summary: dict


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    text = clean_text(text).lower()
    ascii_tokens = re.findall(r"[a-z0-9+.#-]{2,}", text)
    cjk = re.findall(r"[\u4e00-\u9fff]+", text)
    cjk_tokens: list[str] = []
    for span in cjk:
        if len(span) <= 2:
            cjk_tokens.append(span)
        else:
            cjk_tokens.extend(span[i : i + 2] for i in range(len(span) - 1))
            cjk_tokens.extend(span[i : i + 3] for i in range(len(span) - 2))
    return ascii_tokens + cjk_tokens


def stable_hash_vector(text: str, dim: int = 512) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    toks = tokenize(text)
    if not toks:
        return vec
    for tok in toks:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        idx = value % dim
        sign = 1.0 if ((value >> 11) & 1) else -1.0
        vec[idx] += sign
    return l2_normalize(vec)


def l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return arr.astype(np.float32)
    return (arr / norm).astype(np.float32)


def load_numeric_vector(path: object, dim: int) -> np.ndarray:
    text = clean_text(path)
    if not text:
        return np.zeros(dim, dtype=np.float32)
    p = Path(text)
    if not p.is_file():
        return np.zeros(dim, dtype=np.float32)
    arr = np.load(p).astype(np.float32).reshape(-1)
    if arr.size == dim:
        return l2_normalize(arr)
    if arr.size > dim:
        chunks = np.array_split(arr, dim)
        reduced = np.asarray([float(x.mean()) for x in chunks], dtype=np.float32)
    else:
        reduced = np.zeros(dim, dtype=np.float32)
        reduced[: arr.size] = arr
    return l2_normalize(reduced)


def infer_modality(section: str, text: str) -> str:
    base = SECTION_MODALITY.get(section, "multimodal")
    if base != "multimodal":
        return base
    lower = clean_text(text).lower()
    hits = {
        key: sum(1 for kw in words if kw.lower() in lower)
        for key, words in KEYWORDS.items()
    }
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else "multimodal"


def abnormality_attribute(text: str, label: int) -> str:
    lower = clean_text(text).lower()
    positive_terms = ["cin2", "cin3", "hsil", "高级别", "癌", "suspicious", "positive", "cin2+"]
    negative_terms = ["negative", "normal", "慢性子宫颈炎", "no definitive", "未见"]
    if any(t in lower for t in positive_terms):
        return "positive_semantic"
    if any(t in lower for t in negative_terms):
        return "negative_or_benign_semantic"
    return "label_positive_prior" if int(label) == 1 else "label_negative_prior"


def case_query_text(row: pd.Series) -> str:
    fields = [
        row.get("instruction_text", ""),
        row.get("hpv", ""),
        row.get("tct", ""),
        row.get("other_clinical_attributes", ""),
        f"age:{row.get('age', '')}",
        f"missing_oct:{row.get('missing_oct', '')}",
        f"missing_colposcopy:{row.get('missing_colposcopy', '')}",
    ]
    return " ".join(clean_text(v) for v in fields if clean_text(v))


def source_visual_vector(row: pd.Series, modality: str, dim: int = 512) -> np.ndarray:
    if modality == "oct":
        return load_numeric_vector(row.get("oct_embedding_path", ""), dim)
    if modality == "colposcopy":
        return load_numeric_vector(row.get("colposcopy_embedding_path", ""), dim)
    return load_numeric_vector(row.get("fused_visual_embedding_path", ""), dim)


def build_entity_bank(df: pd.DataFrame, *, bank_split: str = "train", dim: int = 512) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    bank_df = df[df["split"].astype(str).eq(bank_split)].copy() if "split" in df.columns else df.copy()
    rows: list[dict] = []
    text_vectors: list[np.ndarray] = []
    visual_vectors: list[np.ndarray] = []
    for _, row in bank_df.iterrows():
        case_id = clean_text(row.get("case_id", ""))
        label = int(row.get("binary_label", 0))
        for section, col in SECTION_COLUMNS.items():
            text = clean_text(row.get(col, ""))
            if len(text) < 8:
                continue
            modality = infer_modality(section, text)
            entity_text = (
                f"section:{section} modality:{modality} "
                f"label_endpoint:{row.get('binary_label_endpoint', 'CIN2+')} "
                f"description:{text}"
            )
            text_vec = stable_hash_vector(entity_text, dim)
            vis_vec = source_visual_vector(row, modality, dim)
            text_vectors.append(text_vec)
            visual_vectors.append(vis_vec)
            rows.append(
                {
                    "entity_id": f"E{len(rows):06d}",
                    "source_case_id": case_id,
                    "source_center_id": row.get("center_id", ""),
                    "section": section,
                    "modality": modality,
                    "description": text,
                    "abnormality_attribute": abnormality_attribute(text, label),
                    "label_prior": label,
                    "report_topic_id": int(row.get("report_topic_id", -1)),
                    "report_topic_confidence": float(row.get("report_topic_confidence", 0.0)),
                    "training_report_type": row.get("training_report_type", ""),
                }
            )
    if not rows:
        return pd.DataFrame(), np.zeros((0, dim), dtype=np.float32), np.zeros((0, dim), dtype=np.float32)
    return (
        pd.DataFrame(rows),
        np.vstack(text_vectors).astype(np.float32),
        np.vstack(visual_vectors).astype(np.float32),
    )


def _section_balanced_topk(scores: np.ndarray, bank: pd.DataFrame, *, top_k: int, query_case_id: str) -> list[int]:
    valid = np.ones(len(bank), dtype=bool)
    if query_case_id:
        valid &= bank["source_case_id"].astype(str).to_numpy() != query_case_id
    if not valid.any():
        return []
    selected: list[int] = []
    for section in ["oct_findings", "colposcopy_findings", "clinical_context", "impression", "diagnostic_summary", "report_anchor"]:
        mask = valid & bank["section"].astype(str).eq(section).to_numpy()
        if not mask.any():
            continue
        idxs = np.flatnonzero(mask)
        order = idxs[np.argsort(scores[idxs])[::-1][: max(1, top_k // 4)]]
        selected.extend(int(i) for i in order)
    idxs = np.flatnonzero(valid)
    selected.extend(int(i) for i in idxs[np.argsort(scores[idxs])[::-1][:top_k]])
    out: list[int] = []
    seen = set()
    for idx in selected:
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
        if len(out) >= top_k:
            break
    return out


def retrieve_for_manifest(
    df: pd.DataFrame,
    bank: pd.DataFrame,
    bank_text_vectors: np.ndarray,
    bank_visual_vectors: np.ndarray,
    *,
    output_dir: Path,
    dim: int = 512,
    top_k: int = 12,
    text_weight: float = 0.45,
    visual_weight: float = 0.55,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_dir = output_dir / "case_semantic_vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    if bank.empty:
        raise ValueError("Semantic bank is empty; cannot retrieve case entities.")
    retrieval_rows: list[dict] = []
    manifest = df.copy()
    semantic_paths = []
    semantic_scores = []
    positive_ratios = []
    section_coverages = []
    entity_summaries = []
    for _, row in manifest.iterrows():
        case_id = clean_text(row.get("case_id", ""))
        q_text = stable_hash_vector(case_query_text(row), dim)
        q_visual = source_visual_vector(row, "multimodal", dim)
        text_scores = bank_text_vectors @ q_text
        visual_scores = bank_visual_vectors @ q_visual
        scores = text_weight * text_scores + visual_weight * visual_scores
        top_idx = _section_balanced_topk(scores, bank, top_k=top_k, query_case_id=case_id)
        if top_idx:
            top_scores = scores[top_idx].astype(np.float32)
            weights = np.exp(top_scores - float(np.max(top_scores)))
            weights = weights / max(float(weights.sum()), 1e-8)
            entity_vecs = 0.75 * bank_text_vectors[top_idx] + 0.25 * bank_visual_vectors[top_idx]
            agg = np.sum(entity_vecs * weights[:, None], axis=0).astype(np.float32)
            label_priors = bank.iloc[top_idx]["label_prior"].astype(float).to_numpy()
            positive_ratio = float(np.sum(label_priors * weights))
            agg[:4] += np.asarray(
                [positive_ratio, float(np.mean(top_scores)), float(np.max(top_scores)), float(len(top_idx) / max(top_k, 1))],
                dtype=np.float32,
            )
            agg = l2_normalize(agg)
            sections = sorted(set(bank.iloc[top_idx]["section"].astype(str)))
            descriptions = [
                f"{r.section}/{r.modality}/{float(scores[i]):.3f}: {clean_text(r.description)[:90]}"
                for i, r in zip(top_idx, bank.iloc[top_idx].itertuples(index=False))
            ]
            mean_score = float(np.mean(top_scores))
        else:
            agg = np.zeros(dim, dtype=np.float32)
            positive_ratio = 0.0
            sections = []
            descriptions = []
            mean_score = 0.0
        vec_path = vector_dir / f"{case_id}.npy"
        np.save(vec_path, agg.astype(np.float32))
        semantic_paths.append(str(vec_path.resolve()))
        semantic_scores.append(mean_score)
        positive_ratios.append(positive_ratio)
        section_coverages.append(len(sections))
        entity_summaries.append(" | ".join(descriptions[:6]))
        retrieval_rows.append(
            {
                "case_id": case_id,
                "split": row.get("split", ""),
                "center_id": row.get("center_id", ""),
                "semantic_retrieval_score": mean_score,
                "semantic_retrieval_positive_ratio": positive_ratio,
                "semantic_retrieval_section_coverage": len(sections),
                "top_entity_ids": json.dumps(bank.iloc[top_idx]["entity_id"].tolist() if top_idx else [], ensure_ascii=False),
                "top_entity_descriptions": " | ".join(descriptions),
            }
        )
    manifest["semantic_retrieval_embedding_path"] = semantic_paths
    manifest["semantic_retrieval_score"] = semantic_scores
    manifest["semantic_retrieval_positive_ratio"] = positive_ratios
    manifest["semantic_retrieval_section_coverage"] = section_coverages
    manifest["semantic_retrieval_top_entities"] = entity_summaries
    return manifest, pd.DataFrame(retrieval_rows)


def build_semantic_retrieval_artifacts(
    manifest_path: Path,
    *,
    output_dir: Path,
    output_manifest_path: Path,
    bank_split: str = "train",
    dim: int = 512,
    top_k: int = 12,
) -> RetrievalArtifacts:
    df = pd.read_csv(manifest_path)
    bank, bank_text_vectors, bank_visual_vectors = build_entity_bank(df, bank_split=bank_split, dim=dim)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = output_dir / "cervical_section_knowledge_bank.csv"
    bank.to_csv(bank_path, index=False)
    np.save(output_dir / "cervical_section_knowledge_bank_text_vectors.npy", bank_text_vectors)
    np.save(output_dir / "cervical_section_knowledge_bank_visual_vectors.npy", bank_visual_vectors)
    manifest, retrievals = retrieve_for_manifest(
        df,
        bank,
        bank_text_vectors,
        bank_visual_vectors,
        output_dir=output_dir,
        dim=dim,
        top_k=top_k,
    )
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest_path, index=False)
    retrieval_path = output_dir / "case_semantic_retrievals.csv"
    retrievals.to_csv(retrieval_path, index=False)
    summary = {
        "source_manifest": str(manifest_path),
        "output_manifest": str(output_manifest_path),
        "bank_split": bank_split,
        "n_manifest_rows": int(len(df)),
        "n_bank_entities": int(len(bank)),
        "vector_dim": int(dim),
        "top_k": int(top_k),
        "mean_retrieval_score": float(retrievals["semantic_retrieval_score"].mean()),
        "mean_positive_ratio": float(retrievals["semantic_retrieval_positive_ratio"].mean()),
        "mean_section_coverage": float(retrievals["semantic_retrieval_section_coverage"].mean()),
    }
    (output_dir / "semantic_retrieval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        "# KRA-RASA Semantic Retrieval Summary",
        "",
        f"- Source manifest: `{manifest_path}`",
        f"- Output manifest: `{output_manifest_path}`",
        f"- Bank entities: {len(bank)} from split `{bank_split}`",
        f"- Case rows retrieved: {len(retrievals)}",
        f"- Vector dimension: {dim}",
        f"- Top-k entities per case: {top_k}",
        f"- Mean retrieval score: {summary['mean_retrieval_score']:.4f}",
        f"- Mean train-bank positive ratio among retrieved entities: {summary['mean_positive_ratio']:.4f}",
        f"- Mean section coverage: {summary['mean_section_coverage']:.2f}",
        "",
        "This artifact adapts STREAM-style regional semantic retrieval to cervical multimodal analytics: report-derived section entities are retrieved from a train-only knowledge bank and packed as semantic evidence tokens for LCAD-RASA.",
    ]
    (output_dir / "SEMANTIC_RETRIEVAL_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return RetrievalArtifacts(manifest=manifest, bank=bank, retrievals=retrievals, summary=summary)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

