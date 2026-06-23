#!/usr/bin/env python3
"""Generate MOSAIC neural technical architecture figure (model schematic)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    ROOT / "outputs/publishable/figures/main",
    ROOT / "outputs/publishable/figures/jbd_final",
    ROOT / "outputs/publishable/figures",
]
PROJECT_FIGURES = ROOT.parent / "figures"

PALETTE = {
    "input": "#F7EFE2",
    "proj": "#E1CA9E",
    "fusion": "#1E3A66",
    "report": "#ADB093",
    "align": "#998560",
    "risk": "#4F8FD6",
    "retrieval": "#C5B5E8",
    "output": "#E76B6B",
    "lcad": "#ADB093",
    "qc": "#E1CA9E",
    "loss": "#FBF3E6",
    "edge": "#3E3425",
    "muted": "#998560",
    "train": "#F2D6A6",
    "infer": "#F7F8FB",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.titlesize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def box(ax, xy, w, h, title, color, subtitle=None, fontsize=7, lw=1.0, linestyle="-"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=color,
        edgecolor=PALETTE["edge"],
        linewidth=lw,
        linestyle=linestyle,
        alpha=0.96,
        zorder=2,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=fontsize - 0.5, color=PALETTE["muted"], zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=3)
    return patch


def arrow(ax, start, end, color=PALETTE["edge"], style="-|>", lw=1.1, connectionstyle="arc3,rad=0.0"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
            zorder=1,
        )
    )


def phase_band(ax, y, h, label, color, x0=0.02, x1=0.98):
    ax.add_patch(
        Rectangle(
            (x0, y),
            x1 - x0,
            h,
            facecolor=color,
            edgecolor=PALETTE["edge"],
            linewidth=0.8,
            alpha=0.35,
            zorder=0,
        )
    )
    ax.text(x0 + 0.012, y + h - 0.018, label, fontsize=8, fontweight="bold", va="top", color=PALETTE["edge"], zorder=1)


def make_figure() -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(14.5, 9.2), facecolor="#FBF3E6")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Phase bands ---
    phase_band(ax, 0.78, 0.20, "Phase O — LCAD (offline, pre-training)", PALETTE["lcad"])
    phase_band(ax, 0.40, 0.34, "Phase S — RASA (trainable backbone)", PALETTE["align"])
    phase_band(ax, 0.04, 0.32, "Phase A+I+C — Retrieval-calibrated fusion (inference)", PALETTE["retrieval"])

    # --- Phase O: inputs and LCAD ---
    box(ax, (0.04, 0.86), 0.10, 0.08, "OCT images", PALETTE["input"], "cached emb. 2048-d")
    box(ax, (0.16, 0.86), 0.10, 0.08, "Colposcopy", PALETTE["input"], "cached emb. 2048-d")
    box(ax, (0.28, 0.86), 0.10, 0.08, "Clinical fields", PALETTE["input"], "HPV, age, cytology")
    box(ax, (0.40, 0.86), 0.08, 0.08, "CIN2+ label", PALETTE["input"], "weak endpoint y")

    box(ax, (0.52, 0.84), 0.18, 0.12, "LCAD generator", PALETTE["lcad"], r"$G_{LCAD}$: schema-constrained LLM")
    box(ax, (0.72, 0.84), 0.12, 0.12, "QC gate", PALETTE["qc"], r"$w_i = p_i q_i$")
    box(ax, (0.86, 0.84), 0.10, 0.12, "Report set", PALETTE["report"], r"$r_i^\star$, 6 sections")

    arrow(ax, (0.09, 0.86), (0.55, 0.90))
    arrow(ax, (0.21, 0.86), (0.58, 0.90))
    arrow(ax, (0.33, 0.86), (0.61, 0.88))
    arrow(ax, (0.44, 0.86), (0.61, 0.86))
    arrow(ax, (0.70, 0.90), (0.72, 0.90))
    arrow(ax, (0.84, 0.90), (0.86, 0.90))

    ax.text(0.04, 0.80, "Real reports (744): $w_i=1$, never overwritten", fontsize=6.8, color=PALETTE["muted"])
    ax.text(0.52, 0.80, "Pseudo candidates (1,153): QC-filtered weak supervision", fontsize=6.8, color=PALETTE["muted"])

    # --- Multimodal encoders (shared with RASA) ---
    box(ax, (0.04, 0.62), 0.09, 0.10, r"$P_{oct}$", PALETTE["proj"], r"$W_{oct}v_i^{oct}$")
    box(ax, (0.15, 0.62), 0.09, 0.10, r"$P_{col}$", PALETTE["proj"], r"$W_{col}v_i^{col}$")
    box(ax, (0.26, 0.62), 0.09, 0.10, r"$P_{fus}$", PALETTE["proj"], r"$W_{fus}v_i^{fus}$")
    box(ax, (0.37, 0.62), 0.09, 0.10, r"$P_{clin}$", PALETTE["proj"], r"$W_{clin}u_i$")

    box(ax, (0.50, 0.60), 0.16, 0.14, "Fusion MLP", PALETTE["fusion"], r"$h_i = F_\theta([\cdot])$", fontsize=7.5)

    for x in [0.085, 0.195, 0.305, 0.415]:
        arrow(ax, (x, 0.72), (x + 0.42, 0.67), connectionstyle="arc3,rad=0.08")

    # --- Report-conditioned transformer ---
    box(ax, (0.70, 0.60), 0.12, 0.14, "Token embed", PALETTE["report"], r"$E(t_{i,l}) + h_i$")
    box(ax, (0.84, 0.60), 0.12, 0.14, "Transformer", PALETTE["report"], r"$\mathrm{Transformer}_\phi$")
    arrow(ax, (0.58, 0.67), (0.70, 0.67))
    arrow(ax, (0.82, 0.67), (0.84, 0.67))

    box(ax, (0.70, 0.44), 0.26, 0.10, "Structured report decoder", PALETTE["report"], r"$\hat{r}_i = G_\theta(H_i)$  |  $\mathcal{L}_{rep}$ (weighted by $w_i$)")

    arrow(ax, (0.90, 0.60), (0.83, 0.54))

    # --- Section attractors ---
    sections = [
        (0.04, "OCT", r"$A_{oct}(h_i)$", r"$g_i^{oct}$"),
        (0.20, "Colposcopy", r"$A_{col}(h_i)$", r"$g_i^{col}$"),
        (0.36, "Clinical", r"$A_{clin}(h_i)$", r"$g_i^{clin}$"),
        (0.52, "Impression", r"$A_{imp}(h_i)$", r"$g_i^{imp}$"),
    ]
    for x, name, proj, proxy in sections:
        box(ax, (x, 0.44), 0.13, 0.10, f"{name} align", PALETTE["align"], f"{proj} ↔ {proxy}", fontsize=6.5)
        arrow(ax, (x + 0.065, 0.60), (x + 0.065, 0.54))
    ax.text(0.34, 0.41, r"$\mathcal{L}_{align}$: cosine section attractors ($\mathcal{K}$={oct, col, clin, imp})", fontsize=6.8, color=PALETTE["muted"])

    # --- Risk head ---
    box(ax, (0.70, 0.44), 0.12, 0.10, "Risk head", PALETTE["risk"], r"$\hat{p}_i^{RASA}$")
    arrow(ax, (0.58, 0.64), (0.76, 0.54))
    ax.text(0.70, 0.41, r"$\mathcal{L}_{risk}$: BCE on CIN2+", fontsize=6.8, color=PALETTE["muted"])

    # --- Retrieval bank ---
    box(ax, (0.04, 0.18), 0.22, 0.12, "Train-only semantic bank", PALETTE["retrieval"], "2,367 section entities; train split only")
    box(ax, (0.30, 0.18), 0.16, 0.12, "Case query", PALETTE["infer"], "visual + clinical signatures")
    box(ax, (0.50, 0.18), 0.14, 0.12, "Retrieve", PALETTE["retrieval"], r"$s_i^{ret}$")
    box(ax, (0.68, 0.18), 0.14, 0.12, r"RASA score", PALETTE["risk"], r"$\hat{p}_i^{RASA}$")
    box(ax, (0.86, 0.18), 0.10, 0.12, "MOSAIC", PALETTE["output"], r"$\hat{p}_i^{MOSAIC}$")

    arrow(ax, (0.26, 0.24), (0.30, 0.24))
    arrow(ax, (0.46, 0.24), (0.50, 0.24))
    arrow(ax, (0.64, 0.24), (0.68, 0.24))
    arrow(ax, (0.82, 0.24), (0.86, 0.24))
    arrow(ax, (0.76, 0.44), (0.75, 0.30), connectionstyle="arc3,rad=-0.15")

    ax.text(
        0.50,
        0.10,
        r"$\hat{p}_i^{MOSAIC}=\sigma\!\left[(1-\alpha^*)\mathrm{logit}(\hat{p}_i^{RASA})+\alpha^*\mathrm{logit}(s_i^{ret})\right]$"
        "\n"
        r"$\alpha^*$ and $\tau^*$ selected on validation only; no test-label leakage",
        ha="center",
        va="center",
        fontsize=8,
        color=PALETTE["edge"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F7F8FB", "edgecolor": "#E1CA9E"},
    )

    # --- Training objective strip ---
    ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.05, facecolor=PALETTE["loss"], edgecolor="#E1CA9E", linewidth=0.8))
    ax.text(
        0.50,
        0.045,
        r"Training: $\mathcal{L}_{total}=\lambda_{rep}\mathcal{L}_{rep}+\lambda_{align}\mathcal{L}_{align}+\lambda_{risk}\mathcal{L}_{risk}+\lambda_{cons}\mathcal{L}_{cons}$"
        "   |   Optimizer: AdamW   |   Backbone = MOSAIC--RASA; full MOSAIC adds retrieval fusion at inference",
        ha="center",
        va="center",
        fontsize=7,
        color=PALETTE["muted"],
    )

    # Legend: train vs infer
    box(ax, (0.84, 0.44), 0.12, 0.10, "Audit", PALETTE["train"], "perturbation", fontsize=6.5, linestyle="--")
    ax.text(0.90, 0.38, "modality mask / shuffle", ha="center", fontsize=6, color=PALETTE["muted"])

    fig.suptitle(
        "MOSAIC technical architecture: multimodal encoders, section-anchored RASA, and retrieval-calibrated fusion",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    return fig


def save_figure(fig: plt.Figure) -> None:
    names = ["Figure2_mosaic_architecture", "Figure_mosaic_model_architecture"]
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            fig.savefig(out_dir / f"{name}.png", dpi=350, bbox_inches="tight", facecolor="#FBF3E6")
            fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", facecolor="#FBF3E6")
    PROJECT_FIGURES.mkdir(parents=True, exist_ok=True)
    for name in names:
        pdf_src = OUT_DIRS[0] / f"{name}.pdf"
        link = PROJECT_FIGURES / f"{name}.pdf"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(pdf_src.resolve())


def write_spec() -> None:
    spec = ROOT / "outputs/publishable/manuscript_latex/MOSAIC_ARCHITECTURE_FIGURE_SPEC.md"
    spec.write_text(
        """# MOSAIC Model Architecture Figure — Design Specification

> Purpose: reference schematic for AI/human redrawing into a publication-quality **model architecture figure** (distinct from Figure 1 cohort/pipeline overview).

## Output files

- Primary: `figures/main/Figure2_mosaic_architecture.pdf`
- Alias: `Figure_mosaic_model_architecture.pdf`
- Project symlink: `JBD_2026/figures/Figure2_mosaic_architecture.pdf`

## Figure type

**End-to-end neural + semantic architecture diagram** (horizontal data flow, top-to-bottom phases).

| Layer | What to show | Color hint |
|-------|----------------|------------|
| Phase O (top band) | LCAD offline pseudo-report path | light green |
| Phase S (middle band) | Trainable RASA backbone | amber / blue |
| Phase A+I+C (bottom band) | Train-only retrieval + logit fusion | mauve / red output |

## Module inventory (must appear)

### Inputs (4 streams)
1. **OCT** — cached visual embedding `v_i^oct ∈ R^2048`
2. **Colposcopy** — `v_i^col ∈ R^2048`
3. **Fused visual** — `v_i^fus ∈ R^2048` (derived from available image modalities)
4. **Clinical instruction** — `u_i ∈ R^32` (age, HPV, cytology fields)

### Phase O — LCAD (offline, dashed boundary)
- Evidence extraction from OCT + colposcopy + clinical + weak label `y_i`
- **LCAD generator** `G_LCAD` → schema-constrained pseudo report (6 sections: summary, oct, colposcopy, clinical, impression, recommendation)
- **QC gate**: `w_i = p_i · q_i` if pass; else exclude from report supervision
- **Routing**: real reports (744, `w_i=1`, never overwritten) vs pseudo candidates (1,153)
- Output: unified supervision text `r_i^*` feeding RASA

### Phase S — RASA backbone (trainable)
**Encoder / fusion**
- Projection heads: `P_oct, P_col, P_fus, P_clin` → hidden vectors
- **Fusion MLP** `F_θ` → fused case representation `h_i`

**Report-conditioned decoder**
- Token embedding + add `h_i`: `ē_{i,l} = E(t_{i,l}) + h_i`
- **Transformer encoder** `Transformer_φ` → contextual states `H_i`
- **Report decoder** `G_θ(H_i)` → `r̂_i` with weighted CE loss `L_rep` (weight `w_i`)

**Section-anchored alignment** (4 attractor pairs)
| Modality projection | Report section proxy | Section |
|--------------------|----------------------|---------|
| `A_oct(h_i)` | `g_i^oct = Pool_oct(H_i)` | OCT findings |
| `A_col(h_i)` | `g_i^col` | Colposcopy findings |
| `A_clin(h_i)` | `g_i^clin` | Clinical context |
| `A_imp(h_i)` | `g_i^imp` | Diagnostic impression |

- Loss: `L_align` = mean(1 − cosine similarity)

**Risk head**
- `p̂_i^RASA = σ(W_risk · h_i + b_risk)`
- Loss: `L_risk` BCE on CIN2+

→ This backbone alone = **MOSAIC--RASA backbone**

### Phase A+I+C — Retrieval fusion (inference layer)
- **Train-only semantic bank** B: 2,367 section entities from train-split report sections only
- **Case query**: reduced visual + clinical signatures (no val/test labels)
- **Retrieve** → positive semantic prior `s_i^ret`
- **Logit fusion** (validation-calibrated):
  `p̂_i^MOSAIC = σ[(1−α*)logit(p̂_i^RASA) + α*logit(s_i^ret)]`
- `α* ≈ 0.31`, threshold `τ* = 0.50` (validation-selected)

### Footer strip
- Total loss: `L_total = λ_rep L_rep + λ_align L_align + λ_risk L_risk + λ_cons L_cons`
- Optimizer: AdamW
- Note: retrieval bank is **train-only**; fusion weights locked before test

### Optional side annotation
- **Perturbation audit** (dashed): modality mask / shuffle → Δp, Δsection similarity

## Data-flow arrows (critical paths)

```
[OCT, Colpo, Clinical, Label] --offline--> LCAD --> QC --> r_i*
[OCT, Colpo, Fused, Clinical emb] --> P_* --> F_θ --> h_i
h_i --> Risk head --> p̂^RASA
h_i + r_i* tokens --> Transformer --> H_i --> report decoder + section pools g^k
A_k(h_i) <--> g_i^k  (alignment)
h_i + query --> Semantic bank (train only) --> s^ret
p̂^RASA + s^ret --> logit fusion --> p̂^MOSAIC
```

## Distinction from Figure 1

| Figure 1 (overview) | Figure 2 (architecture) |
|---------------------|-------------------------|
| Cohort + 4 MOSAIC phases at study level | Neural modules, tensor shapes, losses |
| Panels A–D | Single continuous schematic |
| Emphasises data imbalance | Emphasises model wiring |

## Suggested LaTeX caption

```latex
\\begin{figure}[t]
  \\centering
  \\includegraphics[width=\\linewidth]{figures/Figure2_mosaic_architecture.pdf}
  \\caption{Technical architecture of MOSAIC.
  Phase~O (LCAD) constructs QC-weighted structured report supervision for report-missing cases without overwriting real reports.
  Phase~S (RASA) projects OCT, colposcopy, fused visual, and clinical evidence into a shared representation, decodes structured reports with a multimodal-conditioned transformer, aligns modality-specific projections to section-level semantic attractors, and outputs a CIN2+ risk backbone score.
  Phase~A+I+C builds a train-only semantic retrieval bank and fuses the retrieved positive semantic prior with the backbone through validation-calibrated logit fusion to produce the final MOSAIC risk score.}
  \\label{fig:mosaic_architecture}
\\end{figure}
```

## AI redraw prompt (copy-paste)

Redraw a Nature-style medical AI **model architecture figure** for MOSAIC with three horizontal phase bands (green LCAD top, amber RASA middle, purple retrieval bottom). Show four input modalities with 2048-d/32-d embeddings, fusion MLP, transformer report decoder, four section-alignment attractor pairs (OCT/colposcopy/clinical/impression), risk head, train-only semantic bank (2367 entities), logit fusion formula, and loss footer. Use clean vector boxes, left-to-right flow, no photographic elements. Palette: muted blue inputs, green LCAD, amber alignment, rose risk, burgundy MOSAIC output. Include equation for p̂^MOSAIC and note α* selected on validation only.

## Regenerate

```bash
cd cervix_lcad_rasa
python scripts/52_generate_mosaic_architecture_figure.py
```
""",
        encoding="utf-8",
    )


def main() -> None:
    fig = make_figure()
    save_figure(fig)
    plt.close(fig)
    write_spec()
    print("Wrote MOSAIC architecture figure to outputs/publishable/figures/")


if __name__ == "__main__":
    main()
