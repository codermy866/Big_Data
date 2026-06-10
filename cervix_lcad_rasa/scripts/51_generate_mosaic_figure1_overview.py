#!/usr/bin/env python3
"""Generate MOSAIC Figure 1 overview (four-panel framework schematic)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "outputs/publishable/figures/main",
    ROOT / "outputs/publishable/figures/jbd_final",
    ROOT / "outputs/publishable/figures",
]

PALETTE = {
    "M": "#5b7fa6",
    "O": "#7aa874",
    "S": "#d9a066",
    "A": "#c97b8b",
    "C": "#9e3f3a",
    "bg": "#f7f7f7",
    "edge": "#3a3a3a",
    "muted": "#6a6a6a",
    "accent": "#2f5f8f",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.06, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")


def rounded_box(ax, xy, w, h, text, color, fontsize=8, weight="medium", subtext: str | None = None):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=color,
        edgecolor=PALETTE["edge"],
        linewidth=1.0,
        alpha=0.95,
        zorder=2,
    )
    ax.add_patch(patch)
    if subtext:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color=PALETTE["edge"], zorder=3)
        ax.text(x + w / 2, y + h * 0.28, subtext, ha="center", va="center", fontsize=fontsize - 1, color=PALETTE["muted"], zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color=PALETTE["edge"], zorder=3)


def arrow(ax, start, end, color=PALETTE["edge"], style="-|>", lw=1.4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=lw,
            color=color,
            zorder=1,
        )
    )


def draw_panel_a(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("M + I: Multicentre cohort under report imbalance", fontweight="bold", pad=8)
    rounded_box(ax, (0.05, 0.58), 0.24, 0.28, "5 centres", PALETTE["M"], subtext="n = 1,897 cases")
    rounded_box(ax, (0.38, 0.58), 0.24, 0.28, "137,591 images", PALETTE["M"], subtext="OCT + colposcopy")
    rounded_box(ax, (0.71, 0.58), 0.24, 0.28, "CIN2+ endpoint", PALETTE["M"], subtext="locked split")
    rounded_box(ax, (0.12, 0.12), 0.34, 0.28, "Real reports", "#dce8f4", subtext="744 cases")
    rounded_box(ax, (0.54, 0.12), 0.34, 0.28, "Pseudo candidates", "#f6e2df", subtext="1,153 cases")
    arrow(ax, (0.29, 0.58), (0.29, 0.42))
    arrow(ax, (0.50, 0.58), (0.50, 0.42))
    arrow(ax, (0.71, 0.58), (0.71, 0.42))
    ax.text(0.50, 0.46, "case-level supervision imbalance", ha="center", va="center", fontsize=7.5, color=PALETTE["muted"])


def draw_panel_b(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("O: Offline structured completion (LCAD)", fontweight="bold", pad=8)
    rounded_box(ax, (0.05, 0.56), 0.22, 0.24, "OCT", "#e8f0e8")
    rounded_box(ax, (0.30, 0.56), 0.22, 0.24, "Colposcopy", "#e8f0e8")
    rounded_box(ax, (0.55, 0.56), 0.18, 0.24, "Clinical", "#e8f0e8")
    rounded_box(ax, (0.76, 0.56), 0.18, 0.24, "Label", "#e8f0e8")
    rounded_box(ax, (0.20, 0.18), 0.60, 0.24, "Schema pseudo-report", PALETTE["O"], subtext="oct / colposcopy / clinical / impression")
    arrow(ax, (0.16, 0.56), (0.38, 0.44))
    arrow(ax, (0.41, 0.56), (0.46, 0.44))
    arrow(ax, (0.64, 0.56), (0.52, 0.44))
    arrow(ax, (0.85, 0.56), (0.58, 0.44))
    rounded_box(ax, (0.72, 0.18), 0.22, 0.24, "QC weight", PALETTE["O"], subtext="w = p x q")
    arrow(ax, (0.80, 0.30), (0.80, 0.30))
    ax.text(0.50, 0.08, "real reports never overwritten", ha="center", fontsize=7.5, color=PALETTE["muted"])


def draw_panel_c(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("S: Section-anchored alignment (RASA)", fontweight="bold", pad=8)
    rounded_box(ax, (0.06, 0.58), 0.88, 0.18, "Fusion MLP + Transformer decoder", PALETTE["S"], fontsize=8)
    anchors = [
        ("OCT -> oct_findings", 0.08),
        ("Colposcopy -> colposcopy_findings", 0.30),
        ("Clinical -> clinical_context", 0.58),
        ("Fused -> impression", 0.80),
    ]
    for text, x in anchors:
        rounded_box(ax, (x, 0.24), 0.18, 0.18, text, "#f8ead8", fontsize=6.5)
        arrow(ax, (x + 0.09, 0.58), (x + 0.09, 0.43))
    rounded_box(ax, (0.36, 0.04), 0.28, 0.12, "Risk head", PALETTE["S"], subtext="p_hat^RASA")
    arrow(ax, (0.50, 0.24), (0.50, 0.16))


def draw_panel_d(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("A + I + C: Retrieval-augmented calibrated fusion", fontweight="bold", pad=8)
    rounded_box(ax, (0.05, 0.58), 0.36, 0.22, "Train-only semantic bank", PALETTE["A"], subtext="2,367 section entities")
    rounded_box(ax, (0.48, 0.58), 0.22, 0.22, "Case query", "#f3dfe4", subtext="clinical + visual")
    rounded_box(ax, (0.74, 0.58), 0.21, 0.22, "Retrieval prior", PALETTE["A"], subtext="s^ret")
    arrow(ax, (0.41, 0.69), (0.48, 0.69))
    arrow(ax, (0.70, 0.69), (0.74, 0.69))
    rounded_box(ax, (0.10, 0.18), 0.30, 0.18, "RASA risk", PALETTE["S"], subtext="p_hat^RASA")
    rounded_box(ax, (0.60, 0.18), 0.30, 0.18, "MOSAIC score", PALETTE["C"], subtext="p_hat^MOSAIC")
    arrow(ax, (0.25, 0.36), (0.42, 0.28))
    arrow(ax, (0.84, 0.58), (0.75, 0.36))
    ax.text(
        0.50,
        0.33,
        r"$\hat{p}^{MOSAIC}=\sigma\!\left[(1-\alpha^*)\mathrm{logit}(\hat{p}^{RASA})+\alpha^*\mathrm{logit}(s^{ret})\right]$",
        ha="center",
        va="center",
        fontsize=8,
        color=PALETTE["edge"],
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d0d0"},
    )
    ax.text(0.50, 0.05, "alpha* and threshold selected on validation only", ha="center", fontsize=7.5, color=PALETTE["muted"])


def draw_mosaic_tiles(ax: plt.Axes) -> None:
    """Decorative mosaic motif linking colposcopic metaphor to method acronym."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    tiles = [
        (0.08, 0.72, "#dce8f4"),
        (0.20, 0.72, "#b9cfe6"),
        (0.32, 0.72, "#dce8f4"),
        (0.14, 0.58, "#b9cfe6"),
        (0.26, 0.58, "#8fb0d8"),
        (0.08, 0.44, "#b9cfe6"),
        (0.20, 0.44, "#dce8f4"),
        (0.32, 0.44, "#b9cfe6"),
    ]
    for x, y, c in tiles:
        ax.add_patch(Rectangle((x, y), 0.10, 0.10, facecolor=c, edgecolor=PALETTE["edge"], linewidth=0.8))
    ax.text(0.20, 0.30, "MOSAIC", ha="center", fontsize=16, fontweight="bold", color=PALETTE["accent"])
    ax.text(
        0.20,
        0.16,
        "Multicentre Offline Structured Anchoring\nwith Imbalanced-report Calibration",
        ha="center",
        va="center",
        fontsize=7.2,
        color=PALETTE["muted"],
        linespacing=1.25,
    )


def make_figure() -> plt.Figure:
    setup_style()
    fig = plt.figure(figsize=(12.5, 8.8), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[0.95, 1.05], wspace=0.22, hspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_logo = fig.add_subplot(gs[1, 2])
    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)
    draw_panel_d(ax_d)
    draw_mosaic_tiles(ax_logo)
    panel_label(ax_a, "A")
    panel_label(ax_b, "B")
    panel_label(ax_c, "C")
    panel_label(ax_d, "D")
    fig.suptitle(
        "MOSAIC framework for large-scale cervical analytics under report-supervision imbalance",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    return fig


def save_figure(fig: plt.Figure) -> None:
    names = [
        "Figure1_mosaic_overview",
        "Figure1_study_design",
        "Figure1_pipeline_schematic",
    ]
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            fig.savefig(out_dir / f"{name}.png", dpi=350, bbox_inches="tight", facecolor="white")
            fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", facecolor="white")


def write_design_doc() -> None:
    doc = ROOT / "outputs/publishable/manuscript_latex/MOSAIC_FIGURE1_PANEL_DESIGN.md"
    doc.write_text(
        "\n".join(
            [
                "# MOSAIC Figure 1 — Panel Layout and Caption",
                "",
                "## File outputs",
                "",
                "- `figures/main/Figure1_mosaic_overview.pdf` (primary)",
                "- Aliases for legacy paths: `Figure1_study_design.pdf`, `Figure1_pipeline_schematic.pdf`",
                "",
                "## Panel layout",
                "",
                "| Panel | MOSAIC letter | Content | Legacy module |",
                "|-------|---------------|---------|---------------|",
                "| A | M + I | Five-centre cohort, image scale, real vs pseudo supervision split | Cohort audit |",
                "| B | O | Offline LCAD pseudo-report construction with QC weighting | LCAD |",
                "| C | S | RASA section-anchored alignment and CIN2+ risk head | RASA |",
                "| D | A + I + C | Train-only semantic bank, retrieval prior, validation-calibrated logit fusion | KRA fusion |",
                "| Logo tile | MOSAIC | Acronym expansion and colposcopic mosaic motif | Branding |",
                "",
                "## Recommended LaTeX caption",
                "",
                "```latex",
                "\\begin{figure}[t]",
                "  \\centering",
                "  \\includegraphics[width=\\linewidth]{figures/main/Figure1_mosaic_overview.pdf}",
                "  \\caption{Overview of the MOSAIC framework (Multicentre Offline Structured Anchoring with Imbalanced-report Calibration).",
                "  (A) Five-centre cohort construction and case-level report-supervision imbalance.",
                "  (B) Offline structured completion for report-missing cases through schema-constrained LCAD pseudo reports and QC weighting.",
                "  (C) Section-anchored RASA alignment between OCT, colposcopy, clinical instruction, fused visual evidence, and structured report sections, with a CIN2+ risk head.",
                "  (D) Train-only semantic retrieval bank and validation-calibrated logit fusion that yields the final MOSAIC risk score.",
                "  The name MOSAIC refers both to the colposcopic mosaic pattern and to assembling fragmented multimodal and report-level evidence into a coherent semantic representation.}",
                "  \\label{fig:mosaic_overview}",
                "\\end{figure}",
                "```",
                "",
                "## Regenerate",
                "",
                "```bash",
                "python scripts/51_generate_mosaic_figure1_overview.py",
                "```",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    fig = make_figure()
    save_figure(fig)
    plt.close(fig)
    write_design_doc()
    print("Wrote MOSAIC Figure 1 overview to outputs/publishable/figures/")


if __name__ == "__main__":
    main()
