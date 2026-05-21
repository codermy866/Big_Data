"""LCAD-RASA training with real section alignment (requires torch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.manifest import load_manifest
from src.models.lcad_rasa import build_model
from src.models.section_alignment import SectionAlignmentModule
from src.training.dataset import CervixReportDataset
from src.training.experiment_modes import apply_train_filter, get_experiment_spec
from src.training.losses import compute_total_loss
from src.utils.config import ensure_dir
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader

    _TORCH = True
except ImportError:
    _TORCH = False


def _train_stub(cfg: dict[str, Any]) -> Path:
    manifest_path = Path(cfg["manifest"]["path"])
    df = load_manifest(manifest_path)
    n_train = len(df[df["split"] == "train"]) if "split" in df.columns else len(df)
    ckpt_dir = ensure_dir(Path(cfg["outputs"]["checkpoints"]) / cfg["training"]["experiment_name"])
    best_path = ckpt_dir / "best.pt"
    best_path.write_text(json.dumps({"mock": True, "n_train": n_train}), encoding="utf-8")
    return best_path


def train_lcad_rasa(cfg: dict[str, Any], mock: bool = False) -> Path:
    if not _TORCH:
        return _train_stub(cfg)

    set_seed(int(cfg["training"].get("seed", 42)))
    use_cuda = cfg["training"].get("device", "cuda") == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and not mock else "cpu")

    manifest_path = Path(cfg["manifest"]["path"])
    pseudo_root = Path(cfg["manifest"].get("pseudo_reports_dir", "outputs/pseudo_reports"))
    exp_name = cfg["training"]["experiment_name"]
    spec = cfg.get("experiment_spec") or {}
    if not spec:
        try:
            spec = get_experiment_spec(exp_name)
        except KeyError:
            spec = {}

    df = load_manifest(manifest_path)
    train_df = df[df["split"] == "train"] if "split" in df.columns else df
    if spec.get("train_filter"):
        train_df = apply_train_filter(train_df, spec["train_filter"])

    loss_cfg = {**cfg.get("loss", {}), **spec.get("loss", {})}
    cfg["loss"] = loss_cfg
    model_flags = spec.get("model", {})
    use_align = model_flags.get("use_section_align", True)
    use_risk = model_flags.get("use_risk_head", True)

    min_w = float(cfg.get("training", {}).get("min_pseudo_weight", 0.0))
    dataset = CervixReportDataset(
        train_df,
        pseudo_root=pseudo_root,
        report_source=spec.get("report_source", "mixed"),
        require_qc_pass=bool(spec.get("require_qc_pass", False)),
        min_weight=min_w,
        max_len=int(cfg["training"].get("max_seq_length", 64)),
    )
    logger.info("Experiment %s: n_train_samples=%d", exp_name, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
    )

    model = build_model(cfg).to(device)
    section_align = None
    params = list(model.parameters())
    if use_align:
        section_align = SectionAlignmentModule(hidden_size=model.token_embed.embedding_dim).to(device)
        params += list(section_align.parameters())
    optim = torch.optim.AdamW(
        params,
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    ckpt_dir = ensure_dir(Path(cfg["outputs"]["checkpoints"]) / cfg["training"]["experiment_name"])
    log_dir = ensure_dir(cfg["outputs"]["logs"])
    history = []
    num_epochs = int(cfg["training"]["num_epochs"])
    max_steps = cfg["training"].get("max_steps_per_epoch")
    if max_steps is not None and int(max_steps) <= 0:
        max_steps = None
    if mock and max_steps is None:
        max_steps = int(cfg.get("mock", {}).get("num_steps_per_epoch", 20))

    for epoch in range(num_epochs):
        model.train()
        if section_align is not None:
            section_align.train()
        epoch_loss = 0.0
        n_steps = 0
        for step, batch in enumerate(loader):
            if max_steps is not None and step >= int(max_steps):
                break
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weight"].to(device)

            out = model(input_ids, labels=labels if use_risk else None)
            loss, parts = compute_total_loss(
                model, out, target_ids, labels, weights, loss_cfg, section_align
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_loss += float(loss.item())
            n_steps += 1

        avg = epoch_loss / max(n_steps, 1)
        history.append({"epoch": epoch + 1, "loss": avg, **parts})
        logger.info("Epoch %d/%d loss=%.4f device=%s", epoch + 1, num_epochs, avg, device)

    best_path = ckpt_dir / "best.ckpt"
    payload = {
        "model": model.state_dict(),
        "section_align": section_align.state_dict() if section_align else {},
        "cfg": cfg,
        "experiment_spec": spec,
        "n_train": len(dataset),
    }
    torch.save(payload, best_path)
    torch.save(payload, ckpt_dir / "best.pt")
    with (log_dir / f"{cfg['training']['experiment_name']}_history.json").open("w") as f:
        json.dump(history, f, indent=2)
    curve_path = ensure_dir(ckpt_dir.parent.parent / "tables" / cfg["training"]["experiment_name"]) / "training_curve.csv"
    pd.DataFrame(history).to_csv(curve_path, index=False)
    logger.info("Saved checkpoint to %s", best_path)
    return best_path
