#!/usr/bin/env python3
"""Assemble multi-panel composite figures with A/B/C labels for the JBD manuscript."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
FIG_SRC = PROJECT / "figures"
FIG_OUT_JBD = ROOT / "outputs/publishable/figures/jbd_final"
FIG_OUT_ROOT = ROOT / "outputs/publishable/figures"

sys.path.insert(0, str(ROOT))
from src.supplementary.jbd_figure_typography import (
    FONT_SERIF,
    apply_arial_to_figure,
    panel_label,
    save_figure_arial,
    setup_arial_rcparams,
)


def _resolve(name: str) -> Path:
    for base in (FIG_SRC, FIG_OUT_JBD, FIG_OUT_ROOT, ROOT / "outputs/publishable/external_baselines/figures",
                 ROOT / "outputs/publishable/theme1_alignment/figures",
                 ROOT / "outputs/publishable/llm_api_provider_paper_ready/figures"):
        p = base / f"{name}.png"
        if p.is_file():
            return p
    raise FileNotFoundError(f"Panel image not found: {name}.png")


def _load_rgb(path: Path) -> np.ndarray:
    img = mpimg.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb = img[..., :3].astype(float)
        alpha = img[..., 3:4].astype(float)
        bg = np.ones_like(rgb)
        img = rgb * alpha + bg * (1.0 - alpha)
    return img


def compose_panels(
    panels: list[tuple[str, str]],
    *,
    ncols: int,
    out_stem: str,
    figsize: tuple[float, float] | None = None,
    label_size: float = 17.0,
) -> list[Path]:
    """Compose PNG panels into one figure; panels = [(image_stem, label), ...]."""
    setup_arial_rcparams()
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (7.2 * ncols / 2.0, 4.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat[n:]:
        ax.axis("off")
    for ax, (img_stem, label) in zip(axes_flat, panels):
        ax.imshow(_load_rgb(_resolve(img_stem)))
        ax.axis("off")
        panel_label(ax, label, fontsize=label_size)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    apply_arial_to_figure(fig)
    written: list[Path] = []
    for out_dir in (FIG_OUT_JBD, FIG_OUT_ROOT, FIG_SRC):
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / out_stem
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.06)
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.06)
        written.append(out.with_suffix(".png"))
    plt.close(fig)
    return written


def compose_vertical_stack(
    panels: list[tuple[str, str]],
    *,
    out_stem: str,
    figsize: tuple[float, float] = (7.2, 10.5),
) -> list[Path]:
    setup_arial_rcparams()
    fig = plt.figure(figsize=figsize)
    heights = [1.0] * len(panels)
    gs = fig.add_gridspec(len(panels), 1, height_ratios=heights, hspace=0.05)
    for i, (img_stem, label) in enumerate(panels):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(_load_rgb(_resolve(img_stem)))
        ax.axis("off")
        panel_label(ax, label, fontsize=17.0)
    apply_arial_to_figure(fig)
    written: list[Path] = []
    for out_dir in (FIG_OUT_JBD, FIG_OUT_ROOT, FIG_SRC):
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / out_stem
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.06)
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.06)
        written.append(out.with_suffix(".png"))
    plt.close(fig)
    return written


def main() -> None:
    composites = [
        (
            "Figure_composite_api_stage1_quality",
            2,
            [
                ("P1_stage1_quality_heatmap", "A"),
                ("P2_stage1_quality_risk_bars", "B"),
            ],
        ),
        (
            "Figure_composite_api_reliability",
            2,
            [
                ("P3_stage1_latency_support_scatter", "A"),
                ("P4_stage1_generation_reliability", "B"),
            ],
        ),
        (
            "Figure_composite_api_alignment",
            2,
            [
                ("P5_stage2_macro_mrr", "A"),
                ("P6_stage2_section_mrr", "B"),
            ],
        ),
        (
            "Figure_composite_perturbation_audit",
            2,
            [
                ("Figure3_modality_perturbation_heatmap", "A"),
                ("Figure3_modality_perturbation_lineplot", "B"),
                ("Figure3_risk_delta_stripplot", "C"),
                ("Figure_theme1_perturbation_sensitivity_matrix", "D"),
            ],
        ),
        (
            "Figure_composite_loco",
            2,
            [
                ("fig_loco_heatmap", "A"),
                ("Figure4_loco_forest_catplot", "B"),
            ],
        ),
        (
            "Figure_composite_external_supp",
            2,
            [
                ("Figure_external_baselines_metric_dotplot", "A"),
                ("Figure_external_baselines_paired_delta_auc", "B"),
            ],
        ),
        (
            "Figure_composite_main_harmonisation",
            3,
            [
                ("Figure_main_AUC_pointplot", "A"),
                ("Figure_main_metrics_heatmap", "B"),
                ("Figure_main_auc_f1_scatter", "C"),
            ],
        ),
    ]
    written: list[Path] = []
    for out_stem, ncols, panels in composites:
        try:
            written.extend(compose_panels(panels, ncols=ncols, out_stem=out_stem))
            print(f"Composed {out_stem}")
        except FileNotFoundError as exc:
            print(f"SKIP {out_stem}: {exc}")

    # Modality + RASA ablation (vertical) + QC as separate row in ablation composite
    try:
        compose_vertical_stack(
            [
                ("fig_modality_ablation_stripplot", "A"),
                ("fig_rasa_component_boxenplot", "B"),
            ],
            out_stem="Figure_composite_ablation_modality_rasa",
            figsize=(7.2, 9.6),
        )
        compose_panels(
            [("fig_rasa_lambda_lineplot", "A"), ("fig_lcad_qc_ablation_barplot", "B")],
            ncols=2,
            out_stem="Figure_composite_ablation_lambda_qc",
            figsize=(7.2, 4.2),
        )
        print("Composed ablation composites")
    except FileNotFoundError as exc:
        print(f"SKIP ablation composites: {exc}")

    # Robustness already two-panel; re-export with explicit labels
    try:
        compose_panels(
            [
                ("SupplementaryFigure_S1_masking_validation", "A"),
                ("SupplementaryFigure_S3_multiseed", "B"),
            ],
            ncols=2,
            out_stem="Figure_composite_supplementary_robustness",
            figsize=(7.2, 4.5),
        )
        print("Composed Figure_composite_supplementary_robustness")
    except FileNotFoundError as exc:
        print(f"SKIP robustness composite: {exc}")

    print(f"Total composite outputs: {len(written)}")
    print(f"Font: {FONT_SERIF}")


if __name__ == "__main__":
    main()
