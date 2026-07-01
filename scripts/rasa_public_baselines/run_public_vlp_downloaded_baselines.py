#!/usr/bin/env python3
"""Run downloaded public medical VLP image-encoder baselines.

The baselines are intentionally conservative: each downloaded public VLP
checkpoint is used as a frozen image encoder, case-level OCT and colposcopy
embeddings are pooled, and only a lightweight train-split logistic head is fit.
Validation selects the operating threshold; test labels are used only for final
metrics by the common evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "cervix_lcad_rasa" / "outputs" / "publishable" / "manifests" / "full_manifest_publishable.csv"
OUT = ROOT / "outputs" / "rasa_public_baselines" / "downloaded_public_vlp"
DOWNLOADS = ROOT / "outputs" / "rasa_public_baselines" / "downloads"


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
        ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def f1_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> float:
    pred = score >= threshold
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        f1 = f1_at_threshold(y, score, float(t))
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def parse_paths(value: object) -> list[str]:
    text = "" if pd.isna(value) else str(value)
    if not text:
        return []
    try:
        import json as _json

        obj = _json.loads(text)
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return [p.strip() for p in text.split(";") if p.strip()]


def representative_paths(row: pd.Series) -> list[str]:
    paths: list[str] = []
    oct_paths = [p for p in parse_paths(row.get("oct_paths", "")) if Path(p).exists()]
    col_paths = [p for p in parse_paths(row.get("colposcopy_paths", "")) if Path(p).exists() and "report" not in Path(p).name.lower()]
    if oct_paths:
        paths.append(oct_paths[len(oct_paths) // 2])
    if col_paths:
        paths.append(col_paths[0])
    return paths


def load_image(path: str) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def load_state(path: Path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ["state_dict", "model", "module", "net"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def strip_prefix_state(state: dict, prefixes: list[str]) -> dict:
    out = {}
    for k, v in state.items():
        for prefix in prefixes:
            if k.startswith(prefix):
                out[k[len(prefix) :]] = v
                break
    return out


def build_biomedclip_encoder(device: torch.device):
    import timm

    weight = DOWNLOADS / "biomedclip" / "open_clip_pytorch_model.bin"
    if not weight.exists() or weight.stat().st_size < 100_000_000:
        raise FileNotFoundError(f"BiomedCLIP weight missing or incomplete: {weight}")
    state = load_state(weight)
    trunk = strip_prefix_state(state, ["visual.trunk."])
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    missing, unexpected = model.load_state_dict(trunk, strict=False)
    loaded = len(trunk) - len(unexpected)
    if loaded < 50:
        raise RuntimeError(f"BiomedCLIP visual trunk did not load enough parameters; loaded={loaded}")
    model.eval().to(device)
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    return model, tfm, {"loaded_keys": loaded, "missing_keys": len(missing), "unexpected_keys": len(unexpected), "weight": str(weight)}


def build_openclip_vit_encoder(weight: Path, device: torch.device, model_name: str = "ViT-B-16"):
    import open_clip

    if not weight.exists() or weight.stat().st_size < 100_000_000:
        raise FileNotFoundError(f"OpenCLIP-style weight missing or incomplete: {weight}")
    model = open_clip.create_model(model_name, pretrained=None, device=device)
    state = load_state(weight)
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported state object: {type(state)}")
    # Common checkpoint wrappers.
    cleaned = {k.replace("module.", ""): v for k, v in state.items() if torch.is_tensor(v)}
    result = model.load_state_dict(cleaned, strict=False)
    loaded = len(cleaned) - len(result.unexpected_keys)
    if loaded < 50:
        raise RuntimeError(f"OpenCLIP model did not load enough parameters; loaded={loaded}")
    model.eval().to(device)
    _, _, tfm = open_clip.create_model_and_transforms(model_name, pretrained=None)
    return model, tfm, {"loaded_keys": loaded, "missing_keys": len(result.missing_keys), "unexpected_keys": len(result.unexpected_keys), "weight": str(weight)}


def build_medclip_encoder(device: torch.device):
    from medclip import MedCLIPVisionModelViT
    import medclip.constants as medclip_constants

    swin_dir = DOWNLOADS / "medclip_swin_tiny"
    medclip_root = DOWNLOADS / "medclip_vit"
    if not (swin_dir / "config.json").exists() or not (swin_dir / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"Local Swin-Tiny dependency for MedCLIP is missing: {swin_dir}")
    candidates = list(medclip_root.rglob("pytorch_model.bin"))
    if not candidates:
        raise FileNotFoundError(f"Local MedCLIP-ViT checkpoint is missing under: {medclip_root}")
    checkpoint_dir = candidates[0].parent
    old_vit_type = medclip_constants.VIT_TYPE
    medclip_constants.VIT_TYPE = str(swin_dir)
    try:
        vision = MedCLIPVisionModelViT(medclip_checkpoint=str(checkpoint_dir)).eval().to(device)
    finally:
        medclip_constants.VIT_TYPE = old_vit_type
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5862785803043838,) * 3, std=(0.27950088968644304,) * 3),
        ]
    )
    return vision, tfm, {"swin_weight": str(swin_dir), "medclip_weight": str(checkpoint_dir)}


class PooledConvEncoder(nn.Module):
    def __init__(self, features: nn.Module):
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.features(x)), 1)


def build_medklip_encoder(device: torch.device):
    from torchvision import models

    weight = DOWNLOADS / "medklip" / "MedKLIP.pth"
    if not weight.exists() or weight.stat().st_size < 100_000_000:
        raise FileNotFoundError(f"MedKLIP weight missing or incomplete: {weight}")
    obj = torch.load(weight, map_location="cpu")
    if not isinstance(obj, dict) or "model" not in obj:
        raise RuntimeError("Unsupported MedKLIP checkpoint structure")
    state = obj["model"]
    res_state = strip_prefix_state(state, ["module.res_features."])
    base = models.resnet50(weights=None)
    features = nn.Sequential(*list(base.children())[:-2])
    result = features.load_state_dict(res_state, strict=False)
    loaded = len(res_state) - len(result.unexpected_keys)
    if loaded < 200:
        raise RuntimeError(f"MedKLIP ResNet feature trunk did not load enough parameters; loaded={loaded}")
    model = PooledConvEncoder(features).eval().to(device)
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return model, tfm, {"loaded_keys": loaded, "missing_keys": len(result.missing_keys), "unexpected_keys": len(result.unexpected_keys), "weight": str(weight)}


def build_gloria_encoder(device: torch.device):
    from torchvision import models

    weight = DOWNLOADS / "gloria" / "GLoRIA.pth"
    if not weight.exists() or weight.stat().st_size < 100_000_000:
        raise FileNotFoundError(f"GLoRIA weight missing or incomplete: {weight}")
    obj = torch.load(weight, map_location="cpu")
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise RuntimeError("Unsupported GLoRIA checkpoint structure")
    state = obj["state_dict"]
    image_state = strip_prefix_state(state, ["gloria.img_encoder.model."])
    base = models.resnet50(weights=None)
    result = base.load_state_dict(image_state, strict=False)
    loaded = len(image_state) - len(result.unexpected_keys)
    if loaded < 250:
        raise RuntimeError(f"GLoRIA image encoder did not load enough parameters; loaded={loaded}")
    model = PooledConvEncoder(nn.Sequential(*list(base.children())[:-2])).eval().to(device)
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return model, tfm, {"loaded_keys": loaded, "missing_keys": len(result.missing_keys), "unexpected_keys": len(result.unexpected_keys), "weight": str(weight)}


def build_mgca_vit_encoder(device: torch.device):
    import timm

    weight = DOWNLOADS / "mgca_vit" / "MGCA-vit.pth"
    if not weight.exists() or weight.stat().st_size < 100_000_000:
        raise FileNotFoundError(f"MGCA-ViT weight missing or incomplete: {weight}")
    obj = torch.load(weight, map_location="cpu")
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise RuntimeError("Unsupported MGCA checkpoint structure")
    state = obj["state_dict"]
    image_state = strip_prefix_state(state, ["img_encoder_q.model."])
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    result = model.load_state_dict(image_state, strict=False)
    loaded = len(image_state) - len(result.unexpected_keys)
    if loaded < 140:
        raise RuntimeError(f"MGCA ViT encoder did not load enough parameters; loaded={loaded}")
    model.eval().to(device)
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return model, tfm, {"loaded_keys": loaded, "missing_keys": len(result.missing_keys), "unexpected_keys": len(result.unexpected_keys), "weight": str(weight)}


def encode_case(model, tfm, row: pd.Series, device: torch.device, dim_fallback: int = 512) -> np.ndarray:
    vecs = []
    for path in representative_paths(row):
        image = load_image(path)
        if image is None:
            continue
        x = tfm(image).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model, "encode_image"):
                y = model.encode_image(x)
            else:
                out = model(x)
                if isinstance(out, dict):
                    y = out.get("img_embeds", out.get("image_embeds", next(iter(out.values()))))
                elif isinstance(out, (tuple, list)):
                    y = out[0]
                else:
                    y = out
            y = torch.flatten(y, start_dim=1)
            y = torch.nn.functional.normalize(y.float(), dim=1)
            vecs.append(y.squeeze(0).detach().cpu().numpy())
    if not vecs:
        return np.zeros(dim_fallback, dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


def run_frozen_encoder(
    experiment_id: str,
    method_name: str,
    builder: Callable[[torch.device], tuple[torch.nn.Module, Callable, dict]],
    *,
    seed: int,
    force: bool,
) -> dict:
    out_dir = OUT / "predictions" / experiment_id
    feature_path = OUT / "features" / f"{experiment_id}.npz"
    meta_path = OUT / "features" / f"{experiment_id}_meta.json"
    pred_path = out_dir / "all_predictions.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST)
    df["case_id"] = df["case_id"].astype(str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_meta: dict = {}
    if feature_path.exists() and not force:
        data = np.load(feature_path, allow_pickle=True)
        case_ids = data["case_id"].astype(str)
        x = data["x"].astype(np.float32)
    else:
        model, tfm, build_meta = builder(device)
        rows = []
        case_ids = []
        for _, row in df.iterrows():
            rows.append(encode_case(model, tfm, row, device))
            case_ids.append(str(row["case_id"]))
        max_dim = max(v.shape[0] for v in rows)
        x = np.zeros((len(rows), max_dim), dtype=np.float32)
        for i, v in enumerate(rows):
            x[i, : v.shape[0]] = v
        case_ids = np.asarray(case_ids)
        np.savez_compressed(feature_path, case_id=case_ids, x=x)
        meta_path.write_text(json.dumps(build_meta, indent=2), encoding="utf-8")
    feat = pd.DataFrame({"case_id": case_ids})
    work = df.merge(feat, on="case_id", how="inner").copy()
    x = x[[int(np.where(case_ids == cid)[0][0]) for cid in work["case_id"].astype(str)]]
    train = work["split"].astype(str).eq("train").to_numpy()
    val = work["split"].astype(str).eq("val").to_numpy()
    test = work["split"].astype(str).eq("test").to_numpy()
    y = work["binary_label"].astype(int).to_numpy()
    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        ]
    )
    clf.fit(x[train], y[train])
    p = clf.predict_proba(x)[:, 1]
    threshold = best_threshold(y[val], p[val])
    pred = pd.DataFrame(
        {
            "case_id": work["case_id"].astype(str),
            "center_id": work["center_id"].astype(str),
            "split": work["split"].astype(str),
            "y_cin2plus": y,
            "p_cin2plus": p,
            "threshold": threshold,
            "alpha": np.nan,
            "method_name": method_name,
            "experiment_id": experiment_id,
        }
    )
    pred.to_csv(pred_path, index=False)
    summary = {
        "experiment_id": experiment_id,
        "method_name": method_name,
        "prediction_path": str(pred_path),
        "feature_path": str(feature_path),
        "threshold": threshold,
        "val_auroc": auc_rank(y[val], p[val]),
        "test_auroc": auc_rank(y[test], p[test]),
        "n_features": int(x.shape[1]),
        "build_meta": build_meta,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    registry: dict[str, tuple[str, Callable[[torch.device], tuple[torch.nn.Module, Callable, dict]]]] = {
        "B1_gloria_adapted": ("GLoRIA frozen image encoder", build_gloria_encoder),
        "B2_mgca_adapted": ("MGCA-ViT frozen image encoder", build_mgca_vit_encoder),
        "B5_biomedclip_frozen": ("BiomedCLIP frozen image encoder", build_biomedclip_encoder),
        "B6_unimedclip_frozen": (
            "UniMedCLIP frozen image encoder",
            lambda device: build_openclip_vit_encoder(
                DOWNLOADS / "unimedclip" / "unimed-clip-vit-b16.pt",
                device,
                model_name="ViT-B-16-quickgelu",
            ),
        ),
        "B3_medclip_adapted": ("MedCLIP-ViT frozen image encoder", build_medclip_encoder),
        "B4_medklip_kad_adapted": ("MedKLIP/KAD ResNet frozen image encoder", build_medklip_encoder),
    }
    selected = list(registry) if args.all else [args.experiment_id]
    rows = []
    for eid in selected:
        if eid not in registry:
            rows.append({"experiment_id": eid, "status": "NOT_RUN", "reason": "not_in_downloaded_public_vlp_registry"})
            continue
        method, builder = registry[eid]
        try:
            row = run_frozen_encoder(eid, method, builder, seed=args.seed, force=args.force)
            row["status"] = "DONE"
        except Exception as exc:
            row = {"experiment_id": eid, "method_name": method, "status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
    (OUT / "downloaded_public_vlp_run.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
