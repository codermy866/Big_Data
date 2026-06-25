#!/usr/bin/env python3
"""Generate MOSAIC problem-source and explainability schematic figures."""

from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.supplementary.jbd_figure_typography import (  # noqa: E402
    FONT_ARIAL,
    FONT_TIMES,
    apply_arial_to_figure,
    apply_mixed_en_typography,
    setup_arial_rcparams,
)

TABLES = ROOT / "outputs/publishable/tables"
MANUSCRIPT = TABLES / "manuscript"
OUT_JBD = ROOT / "outputs/publishable/figures/jbd_final/Figure_mosaic_explainability_gallery"
FINAL = PROJECT / "final_Fig/Figure_mosaic_explainability_gallery"

TEXT = "#17212B"
BLUE = "#254B6D"
RUST = "#C65A46"
MID = "#557A95"
TEAL = "#436E6F"
GOLD = "#D2AE76"
PURPLE = "#6F5B85"
GRID = "#D9E1EA"
PANEL = "#F7F9FC"
PALE_BLUE = "#E3EEF5"
PALE_RUST = "#F3DDD6"
PALE_GOLD = "#F5E7C6"
PALE_TEAL = "#DDEBE8"
PALE_PURPLE = "#E8E1EE"
PALE_GRAY = "#EEF2F6"


def setup_theme() -> None:
    setup_arial_rcparams(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_ARIAL, "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 14.4,
            "axes.titlesize": 17.8,
            "axes.labelsize": 15.8,
            "xtick.labelsize": 14.3,
            "ytick.labelsize": 14.3,
            "legend.fontsize": 13.2,
            "legend.title_fontsize": 13.6,
            "text.color": TEXT,
            "axes.edgecolor": GRID,
            "mathtext.rm": FONT_TIMES,
            "mathtext.it": f"{FONT_TIMES}:italic",
            "mathtext.bf": f"{FONT_TIMES}:bold",
        }
    )


def apply_style(fig: plt.Figure) -> None:
    fig._jbd_min_font_size_override = 13.0
    fig._jbd_max_font_size_override = 23.5
    apply_arial_to_figure(fig)
    apply_mixed_en_typography(fig)


def save_fig(fig: plt.Figure, stem: str) -> None:
    apply_style(fig)
    for out_dir in [OUT_JBD, FINAL]:
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / stem
        fig.savefig(base.with_suffix(".png"), dpi=330, bbox_inches="tight", facecolor="white", pad_inches=0.08)
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def load_counts() -> dict[str, int]:
    df = pd.read_csv(MANUSCRIPT / "T1a_cohort_summary.csv")
    vals = dict(zip(df["Metric"], df["Value"]))
    return {
        "cases": int(vals.get("Total cases", 1897)),
        "centres": int(vals.get("Centres", 5)),
        "images": int(vals.get("Evaluable images (pipeline)", 137591)),
        "real": int(vals.get("Real reports", 744)),
        "pseudo": int(vals.get("Pseudo-report candidates", 1153)),
        "test": int(vals.get("Test cases", 288)),
    }


def ax_canvas(fig: plt.Figure) -> plt.Axes:
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str = "",
    *,
    fc: str = "white",
    ec: str = GRID,
    lw: float = 1.0,
    fontsize: float = 13.5,
    weight: str = "normal",
    color: str = TEXT,
    ha: str = "center",
    va: str = "center",
    pad: float = 0.012,
    radius: float = 0.012,
) -> Rectangle:
    patch = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, joinstyle="round")
    ax.add_patch(patch)
    if text:
        chars = max(12, int(w * 115))
        ax.text(
            x + (w / 2 if ha == "center" else pad),
            y + (h / 2 if va == "center" else h - pad),
            textwrap.fill(text, width=chars),
            ha=ha,
            va=va,
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            linespacing=1.12,
        )
    return patch


def header(ax: plt.Axes, x: float, y: float, label: str, title: str, w: float) -> None:
    ax.text(
        x,
        y,
        label,
        ha="left",
        va="top",
        fontsize=19.0,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "round,pad=0.18,rounding_size=0.03", "facecolor": BLUE, "edgecolor": "none"},
    )
    ax.text(x + 0.045, y - 0.004, title, ha="left", va="top", fontsize=17.8, fontweight="bold", color=TEXT)
    ax.plot([x + 0.045, x + w], [y - 0.036, y - 0.036], color=GRID, lw=1.2)


def arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float, *, color: str = MID, lw: float = 2.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def mini_image(ax: plt.Axes, x: float, y: float, w: float, h: float, label: str, *, seed: int, tint: str = "blue") -> None:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:1:80j, 0:1:80j]
    base = 0.55 + 0.22 * np.sin(10 * xx) * np.cos(8 * yy) + 0.10 * rng.normal(size=xx.shape)
    if tint == "oct":
        base += 0.26 * np.exp(-((yy - 0.50) ** 2) / 0.008)
    else:
        base += 0.22 * np.exp(-((xx - 0.47) ** 2 + (yy - 0.56) ** 2) / 0.050)
    base = np.clip(base, 0, 1)
    ax.imshow(base, cmap="gray", extent=(x, x + w, y, y + h), origin="lower", zorder=1)
    ax.set_aspect("auto")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=TEXT, linewidth=0.8, zorder=2))
    ax.text(x + w / 2, y - 0.012, label, ha="center", va="top", fontsize=11.4, color=TEXT)


def chip(ax: plt.Axes, x: float, y: float, text: str, *, fc: str, ec: str | None = None, w: float = 0.12) -> None:
    box(ax, x, y, w, 0.040, text, fc=fc, ec=ec or fc, fontsize=12.4, weight="bold")


def bar(ax: plt.Axes, x: float, y: float, w: float, h: float, frac: float, *, left: str, right: str) -> None:
    ax.add_patch(Rectangle((x, y), w * frac, h, facecolor=BLUE, edgecolor="none"))
    ax.add_patch(Rectangle((x + w * frac, y), w * (1 - frac), h, facecolor=RUST, edgecolor="none"))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=GRID, linewidth=1.0))
    ax.text(x, y - 0.016, left, ha="left", va="top", fontsize=11.8, color=TEXT)
    ax.text(x + w, y - 0.016, right, ha="right", va="top", fontsize=11.8, color=TEXT)


def plot_problem_source_workflow() -> None:
    counts = load_counts()
    fig = plt.figure(figsize=(16.8, 10.4))
    ax = ax_canvas(fig)
    fig.suptitle("Problem source and auditable semantic pathway in MOSAIC", fontsize=22.4, fontweight="bold", y=0.988)

    header(ax, 0.035, 0.925, "A", "Problem source: incomplete report supervision", 0.42)
    box(ax, 0.035, 0.545, 0.425, 0.325, fc=PANEL, ec=GRID, lw=1.1)
    mini_image(ax, 0.060, 0.745, 0.105, 0.075, "OCT", seed=1, tint="oct")
    mini_image(ax, 0.185, 0.745, 0.105, 0.075, "Colposcopy", seed=2, tint="colpo")
    box(ax, 0.315, 0.748, 0.115, 0.070, "HPV / TCT\nAge", fc="white", ec=GRID, fontsize=12.6, weight="bold")
    ax.text(0.060, 0.695, f"{counts['cases']:,} cases from {counts['centres']} centres", fontsize=14.2, fontweight="bold")
    ax.text(0.060, 0.655, f"{counts['images']:,} OCT/colposcopy images", fontsize=12.8)
    ax.text(0.060, 0.615, "Report supervision is uneven across centres", fontsize=12.8)
    bar(
        ax,
        0.060,
        0.585,
        0.340,
        0.030,
        counts["real"] / counts["cases"],
        left=f"real reports {counts['real']:,}",
        right=f"pseudo-needed {counts['pseudo']:,}",
    )

    header(ax, 0.535, 0.925, "B", "Failure mode: visual risk without section anchors", 0.42)
    box(ax, 0.535, 0.545, 0.425, 0.325, fc=PANEL, ec=GRID, lw=1.1)
    box(ax, 0.565, 0.775, 0.105, 0.050, "Image-only\nfeatures", fc=PALE_BLUE, ec=GRID, fontsize=12.2, weight="bold")
    box(ax, 0.565, 0.675, 0.105, 0.050, "Structured\nvariables", fc=PALE_TEAL, ec=GRID, fontsize=12.2, weight="bold")
    box(ax, 0.735, 0.720, 0.145, 0.070, "Unanchored\nrisk score", fc=PALE_RUST, ec=GRID, fontsize=13.0, weight="bold")
    arrow(ax, 0.674, 0.800, 0.730, 0.765, color=REF if "REF" in globals() else MID)
    arrow(ax, 0.674, 0.700, 0.730, 0.742, color=MID)
    ax.text(0.555, 0.620, "Ambiguity sources", fontsize=13.8, fontweight="bold")
    chip(ax, 0.555, 0.595, "missing reports", fc=PALE_RUST, w=0.140)
    chip(ax, 0.710, 0.595, "centre wording", fc=PALE_GOLD, w=0.140)
    chip(ax, 0.555, 0.545, "section drift", fc=PALE_PURPLE, w=0.140)
    chip(ax, 0.710, 0.545, "label leakage risk", fc=PALE_GRAY, w=0.155)

    header(ax, 0.035, 0.455, "C", "MOSAIC: controlled semantic scaffold", 0.53)
    box(ax, 0.035, 0.070, 0.545, 0.335, fc=PANEL, ec=GRID, lw=1.1)
    steps = [
        (0.060, "Multimodal\ncase"),
        (0.185, "LCAD\nweak oracle"),
        (0.330, "RASA\nsections"),
        (0.465, "Train-only\nmemory"),
    ]
    colors = [PALE_BLUE, PALE_RUST, PALE_TEAL, PALE_GOLD]
    for i, ((x, txt), fc) in enumerate(zip(steps, colors)):
        box(ax, x, 0.275, 0.105, 0.070, txt, fc=fc, ec=GRID, fontsize=12.4, weight="bold")
        if i < len(steps) - 1:
            arrow(ax, x + 0.108, 0.310, steps[i + 1][0] - 0.006, 0.310, color=MID)
    box(ax, 0.455, 0.155, 0.105, 0.060, "Calibrated\nrisk", fc=PALE_PURPLE, ec=GRID, fontsize=12.4, weight="bold")
    arrow(ax, 0.515, 0.275, 0.515, 0.220, color=MID)
    chip(ax, 0.070, 0.200, "label-constrained", fc="white", ec=GRID, w=0.145)
    chip(ax, 0.225, 0.200, "data-grounded", fc="white", ec=GRID, w=0.130)
    chip(ax, 0.365, 0.200, "QC-gated", fc="white", ec=GRID, w=0.100)
    for j, txt in enumerate(["OCT findings", "Colposcopy", "Clinical", "Impression"]):
        chip(ax, 0.067 + 0.118 * j, 0.105, txt, fc=[PALE_BLUE, PALE_RUST, PALE_TEAL, PALE_PURPLE][j], w=0.110)

    header(ax, 0.635, 0.455, "D", "Audit boundary instead of autonomous diagnosis", 0.325)
    box(ax, 0.635, 0.070, 0.325, 0.335, fc=PANEL, ec=GRID, lw=1.1)
    audit_rows = [
        ("Generated reports", "QC scaffold"),
        ("Physician reports", "reference only"),
        ("Semantic memory", "train only"),
        ("Validation/test", "held out"),
        ("Risk output", "calibrated fusion"),
    ]
    y = 0.345
    for left, right in audit_rows:
        box(ax, 0.660, y, 0.145, 0.040, left, fc="white", ec=GRID, fontsize=11.8, ha="left")
        box(ax, 0.820, y, 0.110, 0.040, right, fc=PALE_BLUE if "train" in right else "white", ec=GRID, fontsize=11.8, weight="bold")
        y -= 0.055
    ax.text(0.660, 0.083, "Audit goal: calibrated disease-risk analytics,\nnot autonomous diagnosis.", fontsize=11.4, fontweight="bold")

    save_fig(fig, "Figure_mosaic_problem_source_workflow")


def cell(ax: plt.Axes, x: float, y: float, w: float, h: float, *, fc: str = "white", ec: str = "#1E2F46", lw: float = 1.0) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw))


def cell_text(ax: plt.Axes, x: float, y: float, w: float, h: float, lines: list[tuple[str, str]], *, fontsize: float = 12.4) -> None:
    ypos = y + h - 0.030
    for text, fc in lines:
        wrapped = textwrap.wrap(text, width=max(18, int(w * 105)))
        for k, line in enumerate(wrapped):
            ax.text(
                x + 0.010,
                ypos,
                line,
                ha="left",
                va="top",
                fontsize=fontsize,
                color=TEXT,
                bbox={"facecolor": fc, "edgecolor": "none", "alpha": 0.86, "pad": 1.4} if fc else None,
            )
            ypos -= 0.034
        ypos -= 0.008


def plot_interpretability_audit_table() -> None:
    fig = plt.figure(figsize=(16.8, 9.2))
    ax = ax_canvas(fig)
    fig.suptitle("Intuitive audit trail for weak-oracle semantic supervision", fontsize=22.4, fontweight="bold", y=0.988)

    xs = [0.035, 0.210, 0.460, 0.720, 0.965]
    top = 0.900
    header_h = 0.070
    row_h = 0.355
    y_rows = [top - header_h - row_h, top - header_h - 2 * row_h]
    titles = ["Input evidence", "Problem source", "MOSAIC semantic scaffold", "Audit readout"]
    for i in range(4):
        cell(ax, xs[i], top - header_h, xs[i + 1] - xs[i], header_h, fc="#F6F8FB", lw=1.2)
        ax.text((xs[i] + xs[i + 1]) / 2, top - header_h / 2, titles[i], ha="center", va="center", fontsize=15.0, fontweight="bold")
    for y in y_rows:
        for i in range(4):
            cell(ax, xs[i], y, xs[i + 1] - xs[i], row_h, fc="white", lw=1.1)

    ax.text(0.040, y_rows[0] + row_h - 0.026, "Missing report supervision", fontsize=13.6, fontweight="bold", color=BLUE)
    mini_image(ax, 0.052, y_rows[0] + 0.185, 0.060, 0.062, "OCT", seed=7, tint="oct")
    mini_image(ax, 0.125, y_rows[0] + 0.185, 0.060, 0.062, "Colpo.", seed=8, tint="colpo")
    cell_text(
        ax,
        xs[0] + 0.012,
        y_rows[0] + 0.020,
        xs[1] - xs[0] - 0.024,
        0.140,
        [("HPV/TCT/age available", PALE_TEAL), ("physician report absent", PALE_RUST)],
        fontsize=12.0,
    )
    cell_text(
        ax,
        xs[1],
        y_rows[0],
        xs[2] - xs[1],
        row_h,
        [
            ("Report-level supervision is incomplete and centre-dependent.", PALE_RUST),
            ("Direct endpoint learning loses section-level clinical semantics.", PALE_GOLD),
        ],
    )
    cell_text(
        ax,
        xs[2],
        y_rows[0],
        xs[3] - xs[2],
        row_h,
        [
            ("LCAD completes missing sections under label constraints.", PALE_BLUE),
            ("RASA anchors OCT, colposcopy, clinical context, and impression separately.", PALE_TEAL),
            ("QC removes unsupported content.", PALE_PURPLE),
        ],
    )
    cell_text(
        ax,
        xs[3],
        y_rows[0],
        xs[4] - xs[3],
        row_h,
        [
            ("Section retrieval tests whether the latent space follows report structure.", PALE_BLUE),
            ("Scarcity curves test utility when real reports are sparse.", PALE_GOLD),
        ],
    )

    ax.text(0.040, y_rows[1] + row_h - 0.026, "Modality perturbation", fontsize=13.6, fontweight="bold", color=BLUE)
    mini_image(ax, 0.052, y_rows[1] + 0.185, 0.060, 0.062, "Normal", seed=11, tint="oct")
    box(ax, 0.125, y_rows[1] + 0.185, 0.060, 0.062, "OCT\nmasked", fc=PALE_RUST, ec=GRID, fontsize=10.8, weight="bold")
    cell_text(
        ax,
        xs[0] + 0.012,
        y_rows[1] + 0.020,
        xs[1] - xs[0] - 0.024,
        0.140,
        [("same case, perturbed evidence", PALE_GOLD), ("risk-score shift recorded", PALE_PURPLE)],
        fontsize=12.0,
    )
    cell_text(
        ax,
        xs[1],
        y_rows[1],
        xs[2] - xs[1],
        row_h,
        [
            ("A grounded model should degrade in the section linked to the perturbed modality.", PALE_RUST),
            ("Unrelated sections should remain relatively stable.", PALE_TEAL),
        ],
    )
    cell_text(
        ax,
        xs[2],
        y_rows[1],
        xs[3] - xs[2],
        row_h,
        [
            ("Normal: OCT microstructure described.", PALE_BLUE),
            ("Perturbed: OCT unavailable; colposcopy and clinical context preserved.", PALE_RUST),
            ("Section change is mapped to risk shift.", PALE_PURPLE),
        ],
    )
    cell_text(
        ax,
        xs[3],
        y_rows[1],
        xs[4] - xs[3],
        row_h,
        [
            ("Expected primary drop is compared with the observed largest degraded section.", PALE_BLUE),
            ("Report similarity and risk shift provide bounded audit evidence.", PALE_GOLD),
        ],
    )

    ax.text(
        0.035,
        0.035,
        "Schematic examples are de-identified and illustrative; they summarize audit logic rather than displaying patient images.",
        fontsize=11.8,
        color="#4B5563",
    )
    save_fig(fig, "Figure_mosaic_interpretability_audit_table")


def write_captions() -> None:
    lines = [
        "# MOSAIC Explainability Gallery",
        "",
        "## Figure_mosaic_problem_source_workflow",
        "",
        "**Caption.** Schematic summary of the problem source and the MOSAIC semantic pathway. The cohort contains multimodal OCT, colposcopy, and structured clinical evidence, but physician-authored report supervision is incomplete and unevenly distributed across centres. MOSAIC treats this setting as weak-oracle prior learning: LCAD completes missing report sections under label-constrained, data-grounded, and QC-gated controls; RASA anchors representations to clinically meaningful report sections; and train-only semantic retrieval plus validation-calibrated fusion produces auditable disease-risk analytics.",
        "",
        "## Figure_mosaic_interpretability_audit_table",
        "",
        "**Caption.** Qualitative schematic of MOSAIC's audit trail for incomplete report supervision and modality perturbation. The first row illustrates how missing reports are converted into section-anchored weak semantic priors, while the second row shows how modality perturbation is expected to induce section-specific report degradation and bounded risk-score displacement. The examples are de-identified schematics and are intended to explain the audit logic rather than display patient-level images.",
        "",
    ]
    for out_dir in [OUT_JBD, FINAL]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "SCI_CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_theme()
    plot_problem_source_workflow()
    plot_interpretability_audit_table()
    write_captions()
    print("Generated MOSAIC explainability figures")


if __name__ == "__main__":
    main()
