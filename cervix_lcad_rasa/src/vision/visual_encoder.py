"""ResNet50 / configurable visual encoder with multi-image pooling."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    _TORCH = True
except ImportError:
    _TORCH = False

POOLING = Literal["mean", "max", "attention"]


def _build_encoder(name: str = "resnet50"):
    if not _TORCH:
        raise RuntimeError("torch required for visual embeddings")
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()
        dim = 2048
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        dim = 512
    m.eval()
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return m, transform, dim


def pool_features(feats: np.ndarray, method: str = "mean") -> np.ndarray:
    if feats.size == 0:
        return np.zeros((feats.shape[1] if feats.ndim == 2 else 2048,), dtype=np.float32)
    if method == "max":
        return feats.max(axis=0)
    if method == "attention" and feats.shape[0] > 1:
        s = feats.mean(axis=1)
        e = np.exp(s - s.max())
        w = e / e.sum()
        return (feats * w[:, None]).sum(axis=0)
    return feats.mean(axis=0)


def encode_image_paths(
    paths: list[str],
    encoder_name: str = "resnet50",
    pooling: str = "mean",
    max_images: int = 8,
    device: str = "cuda",
) -> tuple[np.ndarray, dict]:
    meta = {"readable": 0, "flags": []}
    if not _TORCH or not paths:
        meta["flags"].append("no_paths")
        return np.zeros(2048, dtype=np.float32), meta

    model, transform, dim = _build_encoder(encoder_name)
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(dev)
    from PIL import Image

    feats = []
    for p in paths[:max_images]:
        fp = Path(p)
        if not fp.is_file():
            continue
        try:
            with Image.open(fp) as im:
                im = im.convert("RGB")
                x = transform(im).unsqueeze(0).to(dev)
                with torch.no_grad():
                    f = model(x).cpu().numpy().squeeze()
                feats.append(f)
                meta["readable"] += 1
        except Exception:
            meta["flags"].append(f"unreadable:{fp.name}")
    if not feats:
        meta["flags"].append("no_readable_images")
        return np.zeros(dim, dtype=np.float32), meta
    arr = np.stack(feats, axis=0)
    return pool_features(arr, pooling).astype(np.float32), meta
