#!/usr/bin/env python3
"""Restyle all publishable experiment figures without rerunning models or APIs."""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supplementary.jbd_ablation_figures import generate_all_ablation_figures
from src.supplementary.jbd_figures_seaborn import (
    C0,
    C1,
    C2,
    C3,
    C4,
    C5,
    C6,
    C7,
    EDGE_DARK,
    JBD_PALETTE_HEX,
    PALETTE_MAIN,
    TEXT_DARK,
    _cmap_diverging,
    _cmap_sequential,
    _save,
    _setup_theme,
    generate_all_seaborn_figures,
)

THEME_DIR = ROOT / "outputs/publishable/theme1_alignment"
API_PAPER = ROOT / "outputs/publishable/llm_api_provider_paper_ready"
SUBMISSION = ROOT / "outputs/publishable_jbd_submission_v2/figures"

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


def _read(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _provider_label(provider: object) -> str:
    label = PROVIDER_LABELS.get(str(provider), str(provider).replace("_", " ").title())
    for token in ("AIHubMix", "aihubmix", "Free", "free", "Preview", "preview"):
        label = label.replace(token, "").strip()
    return re.sub(r"\s+", " ", label)


def _style_axis(ax: plt.Axes) -> None:
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.grid(True, axis="y", color=C7, alpha=0.45)
    sns.despine(ax=ax)


def _save_to_many(fig: plt.Figure, stems: list[Path]) -> None:
    for stem in stems:
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _tile_heatmap(
    matrix: pd.DataFrame,
    stems: list[Path],
    title: str,
    cbar_label: str = "Value",
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".2f",
    cmap=None,
    figsize: tuple[float, float] | None = None,
    max_label_chars: int = 34,
) -> None:
    """Draw a publication-style annotated polar-coordinate heatmap."""
    _setup_theme()
    m = matrix.copy()
    m = m.apply(pd.to_numeric, errors="coerce")
    if m.empty:
        return
    if vmin is None:
        vmin = float(np.nanmin(m.to_numpy()))
    if vmax is None:
        vmax = float(np.nanmax(m.to_numpy()))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
        vmin, vmax = 0.0, 1.0
    cmap = cmap or _cmap_diverging()
    n_rows, n_cols = m.shape
    if figsize is None:
        side = max(8.2, n_cols * 0.95, n_rows * 0.28 + 4.6)
        figsize = (side, side)
    else:
        side = max(max(figsize), 8.2)
        figsize = (side, side)

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Wedge

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    normer = Normalize(vmin=vmin, vmax=vmax)
    values = m.to_numpy(dtype=float)
    theta_start = 90.0
    full_circle_gap = 8.0 if n_cols > 2 else 12.0
    usable_angle = 360.0 - full_circle_gap
    sector = usable_angle / max(n_cols, 1)
    gap = min(7.5, max(3.8, sector * 0.10))
    inner_r = 0.48 if n_rows <= 10 else 0.43
    outer_r = 1.00
    ring_w = (outer_r - inner_r) / max(n_rows, 1)
    group_r = 1.055
    group_w = 0.052
    row_label_font = max(2.9, min(7.2, 64.0 / max(n_rows, 1)))
    col_label_font = max(5.0, min(8.0, 62.0 / max(n_cols, 1)))
    value_font = max(2.8, min(6.8, 58.0 / max(n_rows + 0.65 * n_cols, 1)))
    col_labels = [str(c)[:max_label_chars] for c in m.columns]
    def _compact_row_label(label: object) -> str:
        text = str(label)
        replacements = [
            ("gaussian_noise_colposcopy_", "noise_colpo_"),
            ("gaussian_noise_oct_", "noise_OCT_"),
            ("partial_colposcopy_drop_", "drop_colpo_"),
            ("partial_oct_drop_", "drop_OCT_"),
            ("label_only_inference", "label_only"),
            ("center_shuffle", "center_shuffle"),
            ("hpv_tct_shuffle", "hpv_tct_shuffle"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    row_max_chars = 42 if n_rows <= 12 else 30
    row_labels = [_compact_row_label(i)[:row_max_chars] for i in m.index]
    lim = group_r + group_w + 0.24
    outer_band_colors = ["#6e87bd", "#cf8fa9", "#b8bfd8", "#bc8b7a", "#8ab7c4", "#d9a17c", "#9aa7b8", "#d7c6d6"]
    annotate_values = (n_rows * n_cols) <= 120 and sector >= 18.0

    def _xy(angle_deg: float, radius: float) -> tuple[float, float]:
        angle = np.deg2rad(angle_deg)
        return float(radius * np.cos(angle)), float(radius * np.sin(angle))

    def _label_rotation(angle_deg: float) -> tuple[float, str]:
        rot = angle_deg - 90.0
        ha = "left"
        if rot < -90.0:
            rot += 180.0
            ha = "right"
        if rot > 90.0:
            rot -= 180.0
            ha = "right"
        return rot, ha

    # Outer annotated sector bands: one band per matrix column.
    for j, label in enumerate(col_labels):
        theta2 = theta_start - full_circle_gap / 2.0 - j * sector - gap / 2.0
        theta1 = theta_start - full_circle_gap / 2.0 - (j + 1) * sector + gap / 2.0
        band = Wedge(
            (0, 0),
            group_r + group_w,
            theta1,
            theta2,
            width=group_w,
            facecolor=outer_band_colors[j % len(outer_band_colors)],
            edgecolor="white",
            linewidth=1.4,
            alpha=0.86,
        )
        ax.add_patch(band)
        mid = (theta1 + theta2) / 2.0
        x, y = _xy(mid, group_r + group_w + 0.085)
        rot, ha = _label_rotation(mid)
        ax.text(
            x,
            y,
            label,
            ha=ha,
            va="center",
            fontsize=col_label_font,
            fontweight="bold",
            rotation=rot,
            rotation_mode="anchor",
            color=TEXT_DARK,
        )

    # Annular heatmap tiles. Row order runs from the central hole outward.
    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]
            r0 = inner_r + i * ring_w
            r1 = r0 + ring_w
            theta2 = theta_start - full_circle_gap / 2.0 - j * sector - gap / 2.0
            theta1 = theta_start - full_circle_gap / 2.0 - (j + 1) * sector + gap / 2.0
            face = "#f0f0f0" if not np.isfinite(val) else cmap(normer(val))
            patch = Wedge(
                (0, 0),
                r1,
                theta1,
                theta2,
                width=ring_w,
                facecolor=face,
                edgecolor="white",
                linewidth=1.05 if n_rows <= 12 else 0.72,
            )
            ax.add_patch(patch)
            if not np.isfinite(val) or not annotate_values:
                continue
            norm = float(np.clip((val - vmin) / (vmax - vmin), 0, 1))
            mid = (theta1 + theta2) / 2.0
            x, y = _xy(mid, r0 + ring_w * 0.52)
            ax.text(
                x,
                y,
                format(val, fmt),
                ha="center",
                va="center",
                fontsize=value_font,
                fontweight="bold",
                color="white" if norm < 0.18 or norm > 0.84 else TEXT_DARK,
            )

    ax.text(0, 0.145, cbar_label, ha="center", va="center", fontsize=8.0, fontweight="bold", color=TEXT_DARK)

    # Keep row names in a side label panel instead of over the annular bars.
    left_margin = 0.185 if n_rows <= 12 else 0.255
    fig.subplots_adjust(left=left_margin, right=0.965, top=0.88, bottom=0.07)
    ax_box = ax.get_position()
    side_x = 0.085 if n_rows <= 12 else 0.038
    side_right = left_margin - 0.014
    side_font = max(5.5, min(9.4, 94.0 / max(n_rows, 1)))
    header_font = max(7.2, min(10.2, side_font + 1.0))
    fig.text(side_x, ax_box.y0 + ax_box.height * 0.78, "Rows", ha="left", va="bottom", fontsize=header_font, fontweight="bold", color=TEXT_DARK)
    fig.text(side_x, ax_box.y0 + ax_box.height * 0.755, "inner -> outer", ha="left", va="top", fontsize=max(4.8, side_font - 0.3), color="#6f6f6f")
    if n_rows <= 14:
        ys = np.linspace(ax_box.y0 + ax_box.height * 0.70, ax_box.y0 + ax_box.height * 0.31, n_rows)
        for i, (label, y) in enumerate(zip(row_labels, ys), start=1):
            fig.text(side_x, y, f"{i}. {label}", ha="left", va="center", fontsize=side_font, color=TEXT_DARK)
    else:
        split = int(np.ceil(n_rows / 2))
        columns = [row_labels[:split], row_labels[split:]]
        col_x = [side_x, side_x + (side_right - side_x) * 0.52]
        for col_labels_subset, x0, start_idx in zip(columns, col_x, [1, split + 1]):
            if not col_labels_subset:
                continue
            ys = np.linspace(ax_box.y0 + ax_box.height * 0.70, ax_box.y0 + ax_box.height * 0.27, len(col_labels_subset))
            for offset, (label, y) in enumerate(zip(col_labels_subset, ys)):
                fig.text(x0, y, f"{start_idx + offset}. {label}", ha="left", va="center", fontsize=side_font, color=TEXT_DARK)

    # Compact central color scale; it is bounded by the inner hole.
    sm = ScalarMappable(norm=normer, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([ax_box.x0 + ax_box.width * 0.39, ax_box.y0 + ax_box.height * 0.47, ax_box.width * 0.22, 0.018])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("")
    cbar.outline.set_linewidth(0.8)
    if vmax - vmin <= 1.2:
        cbar.set_ticks([vmin, (vmin + vmax) / 2.0, vmax])
    else:
        cbar.set_ticks([vmin, (vmin + vmax) / 2.0, vmax])
    cbar.ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)

    ax.set_title(title, fontweight="bold", pad=18, fontsize=16)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    _save_to_many(fig, stems)


def _pretty_model(value: object) -> str:
    text = str(value).replace("_", " ").replace("full lcad rasa", "Full LCAD-RASA")
    text = text.title().replace("Lcad", "LCAD").replace("Rasa", "RASA").replace("Oct", "OCT")
    text = text.replace("W/O", "w/o")
    return text


def _restyle_theme1() -> list[Path]:
    _setup_theme()
    out_dir = THEME_DIR / "figures"
    tab_dir = THEME_DIR / "tables"
    written: list[Path] = []

    pseudo = _read(tab_dir / "T_theme1_llm_vs_template_rule_pseudo_report.csv")
    if pseudo is not None and not pseudo.empty:
        p = pseudo.copy()
        source_map = {
            "label_template": "Template",
            "rule_based": "Rule-based",
            "local_llm": "Local embedding LLM",
        }
        p["Source"] = p["pseudo_report_source"].map(source_map).fillna(p["pseudo_report_source"])
        source_order = ["Template", "Rule-based", "Local embedding LLM"]
        source_palette = {"Template": C0, "Rule-based": C2, "Local embedding LLM": C4}
        source_markers = {"Template": "o", "Rule-based": "s", "Local embedding LLM": "D"}

        panels = [
            (
                "Modality grounding",
                [
                    ("OCT support", "oct_supported_rate", "rate"),
                    ("Colposcopy", "colposcopy_supported_rate", "rate"),
                    ("Clinical", "instruction_supported_rate", "rate"),
                    ("Mean support", "mean_modality_support_rate", "rate"),
                ],
                (0.0, 1.05),
                "Support rate",
            ),
            (
                "Text repetition profile",
                [
                    ("Unique text ↑", "unique_text_rate", "rate"),
                    ("Duplicate fraction ↓", "max_duplicate_fraction", "rate"),
                ],
                (0.0, 1.0),
                "Rate",
            ),
            (
                "Label consistency",
                [
                    ("Label consistency ↑", "label_consistency_mean", "rate"),
                ],
                (0.54, 0.66),
                "Rate",
            ),
            (
                "Latent retrieval MRR",
                [
                    ("Alignment MRR", "latent_alignment_mrr_full_model", "mrr"),
                ],
                (0.055, 0.084),
                "MRR",
            ),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.5))
        axes = axes.ravel()
        source_offsets = {"Template": -0.18, "Rule-based": 0.0, "Local embedding LLM": 0.18}

        for ax, (title, specs, xlim, xlabel) in zip(axes, panels):
            metric_labels = [label for label, col, _ in specs if col in p.columns]
            y_map_panel = {label: i for i, label in enumerate(metric_labels)}
            for y, label in enumerate(metric_labels):
                if y % 2 == 1:
                    ax.axhspan(y - 0.46, y + 0.46, color="#f7f7f2", alpha=0.58, zorder=0)
                ax.hlines(y, xlim[0], xlim[1], color=C7, linewidth=0.82, alpha=0.62, zorder=1)
            for source in source_order:
                row = p[p["Source"].eq(source)]
                if row.empty:
                    continue
                row = row.iloc[0]
                color = source_palette[source]
                marker = source_markers[source]
                for label, col, mode in specs:
                    if col not in p.columns:
                        continue
                    value = float(row[col])
                    y0 = y_map_panel[label] + source_offsets[source]
                    ax.plot([xlim[0], value], [y0, y0], color=color, linewidth=1.6, alpha=0.44, zorder=2)
                    ax.scatter(
                        value,
                        y0,
                        s=78,
                        marker=marker,
                        facecolor=color,
                        edgecolor=TEXT_DARK,
                        linewidth=0.85,
                        alpha=0.96,
                        zorder=4,
                    )
                    if title != "Modality grounding":
                        fmt = f"{value:.3f}" if mode in {"mrr", "gap"} else f"{value:.2f}"
                        text_x = min(value + (xlim[1] - xlim[0]) * 0.036, xlim[1] - (xlim[1] - xlim[0]) * 0.018)
                        ha = "left" if text_x > value else "right"
                        ax.text(
                            text_x,
                            y0,
                            fmt,
                            ha=ha,
                            va="center",
                            fontsize=7.8,
                            color=TEXT_DARK,
                        )
            ax.set_title(title, fontsize=11.0, fontweight="bold", pad=7)
            ax.set_xlim(*xlim)
            ax.set_yticks(range(len(metric_labels)))
            ax.set_yticklabels(metric_labels, fontsize=8.8)
            ax.set_ylim(len(metric_labels) - 0.5, -0.5)
            ax.set_xlabel(xlabel, fontsize=8.7, fontweight="bold")
            ax.grid(axis="x", color=C7, alpha=0.38)
            ax.tick_params(axis="x", labelsize=8.2)
            ax.tick_params(axis="y", length=0)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color(EDGE_DARK)
            ax.spines["bottom"].set_linewidth(0.85)

        handles = [
            plt.Line2D(
                [0],
                [0],
                marker=source_markers[source],
                color="none",
                label=source,
                markerfacecolor=source_palette[source],
                markeredgecolor=TEXT_DARK,
                markeredgewidth=0.8,
                markersize=7.5,
            )
            for source in source_order
        ]
        fig.legend(
            handles=handles,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.50, 0.018),
            ncol=3,
            fontsize=9.1,
            handletextpad=0.45,
            columnspacing=1.4,
        )
        fig.suptitle("Pseudo-report source profile", fontsize=15.0, fontweight="bold", y=0.965)
        fig.text(
            0.985,
            0.955,
            "n = 180 pseudo reports per source; all sources were schema-complete.",
            ha="right",
            va="center",
            fontsize=8.1,
            color=TEXT_DARK,
        )
        fig.subplots_adjust(left=0.12, right=0.985, top=0.88, bottom=0.125, wspace=0.34, hspace=0.48)
        _save(fig, out_dir / "Figure_theme1_pseudo_report_source_comparison")
        written.append(out_dir / "Figure_theme1_pseudo_report_source_comparison.png")

    detail = _read(tab_dir / "T_theme1_modality_section_retrieval_alignment.csv")
    align = _read(tab_dir / "T_theme1_rasa_direct_alignment_ablation.csv")
    if detail is not None and not detail.empty and "mrr" in detail.columns:
        d = detail.copy()
        d["Model"] = d["model"].map(_pretty_model)
        section_labels = {
            "oct_findings": "OCT findings",
            "colposcopy_findings": "Colposcopy findings",
            "clinical_context": "Clinical context",
            "impression": "Impression",
        }
        d["Section"] = d["section"].map(section_labels).fillna(d["section"].astype(str).map(_pretty_model))
        if align is not None and not align.empty and "macro_mrr" in align.columns:
            order_raw = align.sort_values("macro_mrr", ascending=False)["model"].tolist()
            macro = align.set_index("model")["macro_mrr"].to_dict()
        else:
            order_raw = d.groupby("model")["mrr"].mean().sort_values(ascending=False).index.tolist()
            macro = d.groupby("model")["mrr"].mean().to_dict()
        model_order = [m for m in order_raw if m in set(d["model"])]
        model_labels = {m: _pretty_model(m) for m in model_order}
        y_map = {m: i for i, m in enumerate(model_order)}
        d["y"] = d["model"].map(y_map).astype(float)
        section_order = ["OCT findings", "Colposcopy findings", "Clinical context", "Impression"]
        section_markers = {"OCT findings": "o", "Colposcopy findings": "s", "Clinical context": "^", "Impression": "D"}
        section_offsets = {"OCT findings": -0.15, "Colposcopy findings": -0.05, "Clinical context": 0.05, "Impression": 0.15}
        section_palette = {
            "OCT findings": C0,
            "Colposcopy findings": C2,
            "Clinical context": C4,
            "Impression": C6,
        }

        max_macro = max(float(macro.get(m, np.nan)) for m in model_order)
        xmax = max(0.086, float(d["mrr"].max()) * 1.13, max_macro * 1.18)
        fig, axes = plt.subplots(
            1,
            5,
            figsize=(11.6, 5.4),
            sharey=True,
            gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.0, 0.92], "wspace": 0.15},
        )

        facet_titles = ["OCT", "Colposcopy", "Clinical", "Impression", "Macro"]
        section_to_axis = dict(zip(section_order, axes[:4]))
        y_positions = np.arange(len(model_order))

        for ax_idx, ax in enumerate(axes):
            for y in y_positions:
                if y % 2 == 1:
                    ax.axhspan(y - 0.5, y + 0.5, color="#f7f7f2", alpha=0.52, zorder=0)
            if "full_lcad_rasa" in y_map:
                y_full = y_map["full_lcad_rasa"]
                ax.axhspan(y_full - 0.5, y_full + 0.5, color=C1, alpha=0.18, zorder=0)
            ax.hlines(y_positions, 0, xmax, color=C7, linewidth=0.72, alpha=0.72, zorder=1)
            ax.set_xlim(0, xmax)
            ax.set_ylim(len(model_order) - 0.5, -0.5)
            ax.set_title(facet_titles[ax_idx], fontsize=10.6, fontweight="bold", pad=8)
            ax.set_xticks([0.00, 0.04, 0.08])
            ax.set_xticklabels(["0.00", "0.04", "0.08"], fontsize=8.8)
            ax.grid(axis="x", color=C7, alpha=0.36, linewidth=0.8)
            ax.tick_params(axis="y", length=0)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color(EDGE_DARK)
            ax.spines["bottom"].set_linewidth(0.9)

        for section, ax in section_to_axis.items():
            sub = d[d["Section"].eq(section)].copy()
            for _, row in sub.iterrows():
                model = row["model"]
                y0 = y_map[model]
                size = 58 + float(row.get("recall_at_5", 0.0)) * 760
                is_full = model == "full_lcad_rasa"
                ax.scatter(
                    row["mrr"],
                    y0,
                    s=size * (1.15 if is_full else 1.0),
                    marker=section_markers.get(section, "o"),
                    facecolor=section_palette.get(section, C0),
                    edgecolor=TEXT_DARK,
                    linewidth=1.15 if is_full else 0.85,
                    alpha=0.96,
                    zorder=4,
                )
        macro_ax = axes[-1]
        for model in model_order:
            y0 = y_map[model]
            macro_mrr = float(macro.get(model, d.loc[d["model"].eq(model), "mrr"].mean()))
            is_full = model == "full_lcad_rasa"
            macro_ax.scatter(
                macro_mrr,
                y0,
                s=112 if not is_full else 148,
                marker="D",
                facecolor=C5 if not is_full else C4,
                edgecolor=TEXT_DARK,
                linewidth=0.95 if not is_full else 1.25,
                alpha=0.96,
                zorder=4,
            )
            macro_ax.text(
                macro_mrr + xmax * 0.062,
                y0,
                f"{macro_mrr:.3f}",
                ha="left",
                va="center",
                fontsize=8.3,
                fontweight="bold" if model in {"full_lcad_rasa", "pseudo_augmented_lcad", "real_report_only"} else "normal",
                color=TEXT_DARK,
            )

        axes[0].set_yticks(y_positions)
        axes[0].set_yticklabels([model_labels[m] for m in model_order], fontsize=9.6)
        for ax in axes[1:]:
            ax.tick_params(labelleft=False)
        macro_ax.set_xlim(0, xmax * 1.22)
        macro_ax.set_xticks([0.00, 0.04, 0.08])
        macro_ax.set_xticklabels(["0.00", "0.04", "0.08"], fontsize=8.8)

        fig.suptitle("Modality-section retrieval alignment", fontsize=15.0, fontweight="bold", y=0.965)
        fig.text(0.57, 0.065, "Mean reciprocal rank", ha="center", va="center", fontsize=10.4, fontweight="bold", color=TEXT_DARK)
        fig.subplots_adjust(left=0.225, right=0.985, top=0.835, bottom=0.175, wspace=0.15)
        _save(fig, out_dir / "Figure_theme1_alignment_retrieval_mrr")
        written.append(out_dir / "Figure_theme1_alignment_retrieval_mrr.png")

    scarcity = _read(tab_dir / "T_theme1_report_supervision_scarcity_curve.csv")
    if scarcity is not None and not scarcity.empty:
        s = scarcity.copy()
        label_map = {"real_report_only_surrogate": "Real-report only", "lcad_augmented_surrogate": "LCAD-augmented"}
        color_map = {"real_report_only_surrogate": C0, "lcad_augmented_surrogate": C4}
        marker_map = {"real_report_only_surrogate": "o", "lcad_augmented_surrogate": "s"}
        fig, ax = plt.subplots(figsize=(8.8, 5.2))
        for setup, g in s.groupby("setup"):
            g = g.sort_values("real_report_fraction")
            color = color_map.get(setup, C0)
            ax.plot(g["real_report_fraction"], g["auc_mean"], color=color, linewidth=1.7, alpha=0.78, zorder=1)
            ax.scatter(g["real_report_fraction"], g["auc_mean"], marker=marker_map.get(setup, "o"), color=color, edgecolor=TEXT_DARK, linewidth=0.8, s=92, label=label_map.get(setup, setup), zorder=3)
            ax.errorbar(g["real_report_fraction"], g["auc_mean"], yerr=g["auc_std"].fillna(0), fmt="none", ecolor=TEXT_DARK, elinewidth=1.0, capsize=3, zorder=2)
        ax.set_xticks([0.1, 0.25, 0.5, 1.0])
        ax.set_xticklabels(["10%", "25%", "50%", "100%"])
        ax.set_xlabel("Available real-report supervision fraction")
        ax.set_ylabel("AUROC on locked test set")
        ax.set_title("Report-supervision scarcity curve")
        ax.legend(frameon=False, loc="lower right")
        _style_axis(ax)
        fig.tight_layout()
        _save(fig, out_dir / "Figure_theme1_report_supervision_scarcity_curve")
        written.append(out_dir / "Figure_theme1_report_supervision_scarcity_curve.png")

    pert = _read(tab_dir / "T_theme1_upgraded_perturbation_sensitivity_matrix.csv")
    if pert is not None and not pert.empty:
        cols = [c for c in ["oct_findings_drop", "colposcopy_findings_drop", "clinical_context_drop", "impression_drop", "report_drop", "risk_abs_delta"] if c in pert.columns]
        if cols:
            matrix = pert.set_index("condition")[cols]
            matrix = matrix.rename(
                index={
                    "normal": "Normal",
                    "mask_oct": "Mask OCT",
                    "mask_colposcopy": "Mask colposcopy",
                    "mask_instruction": "Mask clinical",
                    "shuffle_oct": "Shuffle OCT",
                    "shuffle_colposcopy": "Shuffle colposcopy",
                    "shuffle_instruction": "Shuffle clinical",
                    "mask_visual": "Mask visual",
                    "label_only_inference": "Label-only",
                    "randomize_label": "Random label",
                },
                columns={
                    "oct_findings_drop": "OCT findings",
                    "colposcopy_findings_drop": "Colposcopy findings",
                    "clinical_context_drop": "Clinical context",
                    "impression_drop": "Impression",
                    "report_drop": "Overall report",
                    "risk_abs_delta": "Risk shift",
                },
            )
            _setup_theme()
            metric_colors = {
                "OCT findings": C0,
                "Colposcopy findings": C1,
                "Clinical context": C2,
                "Impression": C6,
                "Overall report": C4,
                "Risk shift": C5,
            }
            metric_markers = {
                "OCT findings": "o",
                "Colposcopy findings": "s",
                "Clinical context": "^",
                "Impression": "v",
                "Overall report": "D",
                "Risk shift": "P",
            }
            y = np.arange(len(matrix))
            fig, axes = plt.subplots(1, len(matrix.columns), figsize=(13.6, 6.1), sharey=True)
            if len(matrix.columns) == 1:
                axes = [axes]
            for ax, metric in zip(axes, matrix.columns):
                vals = pd.to_numeric(matrix[metric], errors="coerce").fillna(0).to_numpy(dtype=float)
                ax.axvspan(0, 0.10, color=C7, alpha=0.40, zorder=0)
                ax.axvline(0, color=TEXT_DARK, lw=1.0, alpha=0.75, zorder=1)
                ax.hlines(y, 0, vals, color=C7, lw=4.0, alpha=0.88, zorder=1)
                ax.scatter(
                    vals,
                    y,
                    s=86,
                    marker=metric_markers.get(metric, "o"),
                    color=metric_colors.get(metric, C0),
                    edgecolor=TEXT_DARK,
                    linewidth=0.75,
                    alpha=0.96,
                    zorder=3,
                )
                for yi, val in zip(y, vals):
                    if val >= 0.30:
                        ax.text(val + 0.035, yi, f"{val:.2f}", ha="left", va="center", fontsize=8.7, fontweight="bold", color=TEXT_DARK)
                ax.set_xlim(0, 1.10)
                ax.set_title(metric, fontsize=11.2, fontweight="bold")
                ax.set_xlabel("Drop / shift")
                ax.grid(axis="x", alpha=0.28)
                ax.grid(axis="y", color=C7, alpha=0.34)
            axes[0].set_yticks(y)
            axes[0].set_yticklabels(matrix.index, fontsize=9.8)
            axes[0].set_ylabel("Perturbation condition")
            for ax in axes[1:]:
                ax.tick_params(axis="y", labelleft=False)
            fig.suptitle("Perturbation sensitivity profiles across report sections", fontsize=16, fontweight="bold", y=1.02)
            fig.tight_layout()
            _save_to_many(fig, [out_dir / "Figure_theme1_perturbation_sensitivity_matrix"])
            written.append(out_dir / "Figure_theme1_perturbation_sensitivity_matrix.png")
    return written


def _restyle_matrix_outputs() -> list[Path]:
    """Refresh matrix-style files as annotated tile heatmaps while preserving paths."""
    written: list[Path] = []
    fig_dir = ROOT / "outputs/publishable/figures"
    jbd_dir = fig_dir / "jbd_final"
    main_dir = fig_dir / "main"
    tables = ROOT / "outputs/publishable/tables/manuscript"

    # Main multi-metric model profile.
    t2 = _read(tables / "T2_main_model_comparison_with_ci.csv")
    if t2 is not None and not t2.empty:
        mcols = [c for c in ["auc", "f1"] if c in t2.columns]
        if {"auc", "f1"}.issubset(t2.columns):
            extra = [c for c in ["sensitivity", "specificity"] if c in t2.columns]
            mcols = ["auc", "f1"] + extra
        matrix = t2.set_index(t2["model"].map(_pretty_model))[mcols].rename(columns={"auc": "AUROC", "f1": "F1", "sensitivity": "Sensitivity", "specificity": "Specificity"})
        _tile_heatmap(matrix, [jbd_dir / "Figure_main_metrics_heatmap"], "Main model metric profile", "Score", 0, 1, figsize=(8.6, 5.8))
        written.append(jbd_dir / "Figure_main_metrics_heatmap.png")

    # Main perturbation matrix.
    s6 = _read(tables / "S6_modality_perturbation_text_decoding.csv")
    if s6 is not None and not s6.empty:
        conds = ["normal", "mask_oct", "mask_colposcopy", "mask_instruction", "mask_visual", "label_only_inference"]
        cols = [
            "oct_findings_similarity_to_normal",
            "colposcopy_findings_similarity_to_normal",
            "clinical_context_similarity_to_normal",
            "impression_similarity_to_normal",
        ]
        use = s6[s6["condition"].isin(conds)].copy()
        if set(cols).issubset(use.columns):
            matrix = use.set_index("condition")[cols].rename(
                index={
                    "normal": "Normal",
                    "mask_oct": "Mask OCT",
                    "mask_colposcopy": "Mask colposcopy",
                    "mask_instruction": "Mask clinical",
                    "mask_visual": "Mask visual",
                    "label_only_inference": "Label-only",
                },
                columns={
                    "oct_findings_similarity_to_normal": "OCT findings",
                    "colposcopy_findings_similarity_to_normal": "Colposcopy findings",
                    "clinical_context_similarity_to_normal": "Clinical context",
                    "impression_similarity_to_normal": "Impression",
                },
            )
            stems = [
                jbd_dir / "Figure3_modality_perturbation_heatmap",
                fig_dir / "Figure3_modality_perturbation_heatmap",
                main_dir / "Figure3_perturbation",
            ]
            _tile_heatmap(matrix, stems, "Perturbation response by report section", "Similarity to normal", 0, 1, figsize=(8.8, 4.8))
            written.extend([p.with_suffix(".png") for p in stems])

    # LOCO AUROC matrix.
    s2 = _read(tables / "S2_loco_strict_retrain.csv")
    if s2 is not None and not s2.empty and "auc" in s2.columns:
        s2 = s2.copy()
        model_labels = {
            "full_lcad_rasa": "Full LCAD-RASA",
            "real_report_only_decoder": "Real-report only",
            "report_generation_without_section_alignment": "No section alignment",
        }
        model_order = ["Real-report only", "No section alignment", "Full LCAD-RASA"]
        model_markers = {"Real-report only": "s", "No section alignment": "^", "Full LCAD-RASA": "D"}
        model_colors = {"Real-report only": C1, "No section alignment": C2, "Full LCAD-RASA": C0}
        s2["model_short"] = s2["model"].map(model_labels).fillna(s2["model"].astype(str).map(_pretty_model))
        full_auc = s2[s2["model"].eq("full_lcad_rasa")][["center_label", "auc"]].rename(columns={"auc": "full_auc"})
        center_summary = (
            s2.groupby("center_label", as_index=False)
            .agg(test_cases=("test_cases", "max"))
            .merge(full_auc, on="center_label", how="left")
            .sort_values(["full_auc", "center_label"], ascending=[True, True])
            .reset_index(drop=True)
        )
        center_to_y = {c: i for i, c in enumerate(center_summary["center_label"])}
        _setup_theme()
        fig, axes = plt.subplots(1, len(model_order), figsize=(11.4, 5.4), sharey=True)
        for ax, model_label in zip(axes, model_order):
            sub = s2[s2["model_short"].eq(model_label)].copy()
            sub["y"] = sub["center_label"].map(center_to_y).astype(float)
            ax.axvspan(0.25, 0.50, color=C4, alpha=0.07, zorder=0)
            ax.axvline(0.50, color=TEXT_DARK, lw=1.0, ls=(0, (2, 2)), alpha=0.72, zorder=1)
            ax.scatter(
                sub["auc"],
                sub["y"],
                s=104 if model_label != "Full LCAD-RASA" else 130,
                marker=model_markers[model_label],
                color=model_colors[model_label],
                edgecolor=TEXT_DARK,
                linewidth=0.85,
                alpha=0.96,
                zorder=3,
            )
            for _, row in sub.iterrows():
                ax.text(float(row["auc"]) + 0.025, float(row["y"]), f"{float(row['auc']):.3f}", ha="left", va="center", fontsize=9.2, fontweight="bold" if model_label == "Full LCAD-RASA" else "normal", color=TEXT_DARK)
            ax.set_title(model_label, fontsize=12.5, fontweight="bold", color=TEXT_DARK)
            ax.set_xlim(0.25, 1.08)
            ax.set_xlabel("AUROC")
            ax.grid(axis="x", alpha=0.30)
            ax.grid(axis="y", color=C7, alpha=0.34)
        y_labels = [f"{row['center_label']}  (n={int(row['test_cases'])})" for _, row in center_summary.iterrows()]
        axes[0].set_yticks(np.arange(len(center_summary)))
        axes[0].set_yticklabels(y_labels, fontsize=10.5)
        axes[0].set_ylabel("Held-out centre")
        for ax in axes[1:]:
            ax.tick_params(axis="y", labelleft=False)
        axes[-1].text(0.505, len(center_summary) - 0.25, "chance", ha="left", va="center", fontsize=9.3, color=TEXT_DARK)
        fig.suptitle("Strict LOCO AUROC by centre and model", fontsize=16, fontweight="bold", y=1.02)
        fig.tight_layout()
        stems = [jbd_dir / "fig_loco_heatmap", fig_dir / "fig_loco_heatmap", fig_dir / "fig_loco_strict_center_heatmap"]
        _save_to_many(fig, stems)
        written.extend([p.with_suffix(".png") for p in stems])

    # Extended perturbation dependency matrix.
    s6b = _read(tables / "S6b_modality_perturbation_extended.csv")
    if s6b is None:
        s6b = _read(ROOT / "outputs/publishable/tables/table_modality_perturbation_extended.csv")
    if s6b is not None and not s6b.empty:
        cols = [c for c in ["eds_oct_findings", "eds_colposcopy_findings", "eds_clinical_context", "risk_delta"] if c in s6b.columns]
        if cols and "condition" in s6b.columns:
            matrix = s6b.head(20).set_index("condition")[cols].abs().rename(
                columns={
                    "eds_oct_findings": "OCT EDS",
                    "eds_colposcopy_findings": "Colposcopy EDS",
                    "eds_clinical_context": "Clinical EDS",
                    "risk_delta": "Risk shift",
                }
            )
            stems = [fig_dir / "fig_perturbation_section_dependency_heatmap", fig_dir / "fig_perturbation_clustermap"]
            _tile_heatmap(matrix, stems, "Extended perturbation sensitivity", "Absolute shift", 0, max(1.0, float(matrix.max().max())), figsize=(9.4, 7.4), max_label_chars=30)
            written.extend([p.with_suffix(".png") for p in stems])

    # Ablation summary matrix.
    frames = []
    for fname, tag in (("S3_modality_ablation.csv", "Modality"), ("S5_rasa_component_ablation.csv", "RASA"), ("S4_lcad_qc_ablation.csv", "QC")):
        d = _read(tables / fname)
        if d is None:
            continue
        d = d.copy()
        d["block_label"] = tag + ": " + d["experiment_id"].map(_pretty_model)
        frames.append(d[["block_label", "auc", "f1"]])
    if frames:
        matrix = pd.concat(frames, ignore_index=True).set_index("block_label").rename(columns={"auc": "AUROC", "f1": "F1"})
        stems = [fig_dir / "ablation/AblationFig_combined_heatmap"]
        _tile_heatmap(matrix, stems, "Ablation summary heatmap", "Score", 0, 1, figsize=(6.8, max(5.0, 0.38 * len(matrix))))
        written.append(stems[0].with_suffix(".png"))

    # API paper-ready quality and reliability matrices.
    q = _read(API_PAPER / "tables/T_api_stage1_quality_for_manuscript.csv")
    if q is not None and not q.empty:
        metrics = {
            "schema_valid_rate": "Schema valid",
            "section_completeness": "Section complete",
            "mean_modality_support_rate": "Modality support",
            "qc_pass_rate": "QC pass",
            "unique_text_rate": "Unique text",
        }
        matrix = q.set_index("provider_label")[[c for c in metrics if c in q.columns]].rename(columns=metrics)
        _tile_heatmap(matrix, [API_PAPER / "figures/P1_stage1_quality_heatmap"], "Structured pseudo-report generation quality", "Rate", 0, 1, figsize=(10.4, 5.8))
        written.append(API_PAPER / "figures/P1_stage1_quality_heatmap.png")

    rel = _read(API_PAPER / "tables/T_api_stage1_generation_reliability.csv")
    if rel is not None and not rel.empty:
        status = {"cached": "Cached", "ok": "Valid", "parse_warning": "Parse warning", "error": "Error"}
        cols = [c for c in status if c in rel.columns]
        if cols:
            matrix = rel.set_index("provider_label")[cols].rename(columns=status)
            _tile_heatmap(matrix, [API_PAPER / "figures/P4_stage1_generation_reliability"], "Generation reliability in the 100-case cohort", "Count", 0, max(1.0, float(matrix.max().max())), fmt=".0f", figsize=(8.8, 4.8))
            written.append(API_PAPER / "figures/P4_stage1_generation_reliability.png")

    provider_wide = _read(API_PAPER / "tables/T_api_llm_provider_comparison_structured_pseudo_report_generation.csv")
    if provider_wide is not None and not provider_wide.empty and "Metric" in provider_wide.columns:
        keep = ["Schema valid rate", "Section completeness", "Modality support", "Contradiction rate", "Hallucination rate", "Duplicate fraction", "Alignment MRR"]
        matrix = provider_wide[provider_wide["Metric"].isin(keep)].set_index("Metric")
        matrix = matrix.apply(pd.to_numeric, errors="coerce")
        _tile_heatmap(matrix, [API_PAPER / "figures/P8_llm_provider_comparison_heatmap"], "LLM provider comparison for structured pseudo-report generation", "Metric value", 0, 1, figsize=(12.0, 5.6))
        written.append(API_PAPER / "figures/P8_llm_provider_comparison_heatmap.png")

    return written


def _load_api38():
    path = ROOT / "scripts/38_run_llm_api_provider_comparison.py"
    spec = importlib.util.spec_from_file_location("api38_restyle", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _restyle_api_stage_dirs() -> list[Path]:
    mod = _load_api38()
    if mod is None:
        return []
    written: list[Path] = []
    for out_dir in sorted((ROOT / "outputs/publishable").glob("llm_api_provider_comparison*")):
        tables = out_dir / "tables"
        figs = out_dir / "figures"
        if not tables.is_dir():
            continue
        quality = _read(tables / "T_api_provider_quality_comparison.csv")
        if quality is not None:
            mod._plot_quality(quality, figs / "Figure_api_provider_quality_comparison")
            written.append(figs / "Figure_api_provider_quality_comparison.png")
        alignment = _read(tables / "T_api_provider_alignment_comparison.csv")
        if alignment is not None:
            mod._plot_alignment(alignment, figs / "Figure_api_provider_alignment_mrr")
            written.append(figs / "Figure_api_provider_alignment_mrr.png")
        ranking = _read(tables / "T_api_provider_candidate_ranking.csv")
        if ranking is not None:
            mod._plot_ranking(ranking, figs / "Figure_api_provider_candidate_ranking")
            written.append(figs / "Figure_api_provider_candidate_ranking.png")
        scarcity = _read(tables / "T_api_provider_downstream_scarcity_surrogate.csv")
        if scarcity is not None:
            mod._plot_scarcity(scarcity, figs / "Figure_api_provider_downstream_scarcity_surrogate")
            written.append(figs / "Figure_api_provider_downstream_scarcity_surrogate.png")
    return written


def _run_script(path: Path) -> None:
    import subprocess

    subprocess.run([sys.executable, str(path)], cwd=str(ROOT), check=True)


def _sync_submission_figures() -> None:
    src = ROOT / "outputs/publishable/figures/jbd_final"
    if not src.is_dir():
        return
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*.*"):
        if p.suffix.lower() in {".png", ".pdf"}:
            shutil.copy2(p, SUBMISSION / p.name)


def main() -> None:
    _setup_theme()
    generated = generate_all_seaborn_figures(ROOT)
    generate_all_ablation_figures(ROOT)
    _run_script(ROOT / "scripts/36_refresh_legacy_figure_styles.py")
    _run_script(ROOT / "scripts/39_generate_llm_api_paper_ready_outputs.py")
    _run_script(ROOT / "scripts/40_generate_llm_provider_comparison_table.py")
    theme_written = _restyle_theme1()
    matrix_written = _restyle_matrix_outputs()
    api_written = _restyle_api_stage_dirs()
    _sync_submission_figures()
    print("Restyled canonical figure groups:", len(generated))
    print("Restyled Theme1 figures:", len(theme_written))
    print("Restyled heatmap replacements:", len(matrix_written))
    print("Restyled staged API figures:", len(api_written))
    print("Palette:", " ".join(JBD_PALETTE_HEX))


if __name__ == "__main__":
    main()
