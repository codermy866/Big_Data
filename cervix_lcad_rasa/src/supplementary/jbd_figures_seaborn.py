"""
Rich Seaborn-style figures for JBD LCAD-RASA results.
Inspired by: https://seaborn.pydata.org/examples/index.html
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from src.supplementary.jbd_figure_typography import FONT_ARIAL, apply_arial_to_figure, setup_arial_rcparams

MANUSCRIPT_REL = "outputs/publishable/tables/manuscript"
PRED_REL = "outputs/publishable/predictions/final_per_case"
TABLES_REL = "outputs/publishable/tables"

# Journal palette requested for the JBD revision.
# Warm categorical colors support framework / mechanism figures, while cool
# continuous scales are reserved for heatmaps and matrix-valued diagnostics.
JBD_WARM_CATEGORICAL = [
    "#3E3425",
    "#998560",
    "#ADB093",
    "#E1CA9E",
]
JBD_COOL_CATEGORICAL = [
    "#1E3A66",
    "#4F8FD6",
    "#C5B5E8",
    "#E76B6B",
]
JBD_VIEWING_SCALE = [
    "#FBF3E6",
    "#E1CA9E",
    "#ADB093",
    "#998560",
    "#3E3425",
]
JBD_PALETTE_HEX = [
    "#3E3425",  # C0 dark brown — primary text / strong category
    "#998560",  # C1 ochre — emphasis category
    "#ADB093",  # C2 sage — auxiliary category
    "#E1CA9E",  # C3 light module / soft category
    "#1E3A66",  # C4 deep blue — primary model highlight
    "#4F8FD6",  # C5 mid blue — secondary model / cool accent
    "#C5B5E8",  # C6 lavender — tertiary accent
    "#E76B6B",  # C7 muted red — negative / contrast
]
PALETTE_MAIN = sns.color_palette(JBD_PALETTE_HEX)
C0, C1, C2, C3, C4, C5, C6, C7 = JBD_PALETTE_HEX
PALETTE_SUPERVISION = [C0, C3]  # real report vs pseudo-report candidate
PALETTE_BINARY = [C5, C7]  # negative vs positive (CIN2+)
TEXT_DARK = "#3E3425"
EDGE_DARK = "#3E3425"
GRID_LINE = "#E1CA9E"
FIG_FACE = "#FBF3E6"
NATURE_HEATMAP_SEQ = [
    "#F7F8FB",
    "#8FA6E3",
    "#4F8FD6",
    "#1E3A66",
]
NATURE_HEATMAP_DIV = [
    "#E76B6B",
    "#F2D6A6",
    "#F7F8FB",
    "#8FA6E3",
    "#1E3A66",
]
PALETTE_MODEL = {
    "Full LCAD-RASA": C4,
    "Pseudo-augmented (LCAD)": C2,
    "LCAD w/o section alignment": C5,
    "Real-report only": C1,
    "Simple concat fusion": C6,
    "Image-only report gen.": C1,
    "Instruction-only report gen.": C7,
    "Fusion w/o report anchor": C3,
    "Simple concat": C6,
    "LCAD w/o section": C5,
    "MOSAIC (full)": C7,
    "MOSAIC--RASA backbone": C4,
    "Semantic retrieval only": C2,
}


def _cmap_sequential() -> object:
    from matplotlib.colors import ListedColormap

    return ListedColormap(NATURE_HEATMAP_SEQ, name="jbd_botanical_seq")


def _cmap_diverging() -> object:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("jbd_botanical_div", NATURE_HEATMAP_DIV, N=256)


def _setup_theme() -> None:
    setup_arial_rcparams(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.size": 8,
            "grid.alpha": 0.35,
            "grid.color": GRID_LINE,
            "axes.edgecolor": EDGE_DARK,
            "axes.labelcolor": TEXT_DARK,
            "text.color": TEXT_DARK,
        }
    )
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font=FONT_ARIAL,
        font_scale=1.0,
        palette=PALETTE_MAIN,
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    apply_arial_to_figure(fig)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    try:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.06)
    except Exception:
        pass
    plt.close(fig)


def _read(project: Path, rel: str) -> pd.DataFrame | None:
    p = project / rel
    return pd.read_csv(p) if p.is_file() else None


def _model_palette(df: pd.DataFrame, col: str = "model") -> dict:
    keys = df[col].astype(str).unique().tolist() if col in df.columns else []
    out = {}
    for i, k in enumerate(keys):
        out[k] = PALETTE_MODEL.get(k, PALETTE_MAIN[i % len(PALETTE_MAIN)])
    return out


def _p_label(p: float | None) -> str:
    from src.supplementary.jbd_figure_stats import format_pvalue

    return format_pvalue(p)


def _add_bracket(ax: plt.Axes, x1: float, x2: float, y: float, text: str, h: float = 0.025) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.25, c=TEXT_DARK, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.12, text, ha="center", va="bottom", fontsize=9, color=TEXT_DARK)


# ---------------------------------------------------------------------------
# Main-text figures (jbd_final/)
# ---------------------------------------------------------------------------


def fig01_pipeline_schematic(out_dir: Path) -> None:
    """Conceptual flow — matplotlib + seaborn colors (no bar chart)."""
    _setup_theme()
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        "Multicentre cohort\nn = 1,897",
        "Phase O: LCAD\npseudo-reports",
        "Phase S: RASA\nsection alignment",
        "Semantic bank\n(train-only)",
        "Phase C: calibrated\nlogit fusion",
        "MOSAIC risk\nscore",
    ]
    colors = [JBD_PALETTE_HEX[i % len(JBD_PALETTE_HEX)] for i in range(len(stages))]
    xs = np.linspace(0.06, 0.94, len(stages))
    for i, (x, s, c) in enumerate(zip(xs, stages, colors)):
        ax.add_patch(FancyBboxPatch((x - 0.075, 0.28), 0.15, 0.44, boxstyle="round,pad=0.02", fc=c, ec=EDGE_DARK, lw=1.2, alpha=0.92))
        ax.text(x, 0.5, s, ha="center", va="center", fontsize=9, color=TEXT_DARK, fontweight="medium")
        if i < len(stages) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.085, 0.5), xytext=(x + 0.085, 0.5), arrowprops=dict(arrowstyle="-|>", color=EDGE_DARK, lw=2))
    ax.set_title("MOSAIC: Multicentre Offline Structured Anchoring with Imbalanced-report Calibration", fontsize=12, pad=12)
    _save(fig, out_dir / "Figure1_pipeline_schematic")
    _save(fig, out_dir / "Figure1_study_design")


def fig02_centre_supervision(project: Path, out_dir: Path) -> None:
    """Centre-level supervision profile — gallery: scatterplot + dot plot."""
    t1b = _read(project, f"{MANUSCRIPT_REL}/T1b_centre_scale_and_supervision.csv")
    if t1b is None:
        return
    _setup_theme()
    data = t1b.copy()
    centre_labels = {
        "enshi": "Enshi",
        "jingzhou": "Jingzhou",
        "shiyan": "Shiyan",
        "wuda": "Wuda",
        "xiangyang": "Xiangyang",
    }
    data["Centre label"] = data["Centre"].map(centre_labels).fillna(data["Centre"].astype(str).str.title())
    data["Real-report coverage"] = data["Real reports"] / data["Cases"]
    data["Total images (k)"] = data["Total images"] / 1000.0
    data = data.sort_values(["Cases", "Real-report coverage"], ascending=[False, True]).reset_index(drop=True)
    centre_colors = {centre: PALETTE_MAIN[i % len(PALETTE_MAIN)] for i, centre in enumerate(data["Centre label"])}

    def _image_size(images: float) -> float:
        return 90 + 660 * (float(images) / float(data["Total images"].max()))

    fig, (ax_scatter, ax_cover) = plt.subplots(
        1,
        2,
        figsize=(12.9, 5.75),
        gridspec_kw={"width_ratios": [1.04, 1.12], "wspace": 0.42},
    )
    fig._jbd_font_scale_override = 1.12
    fig._jbd_min_font_size_override = 7.6
    fig._jbd_max_font_size_override = 18.0

    max_cases = int(np.ceil(data["Cases"].max() / 100) * 100)
    for total in [100, 300, 500]:
        if total <= max_cases:
            ax_scatter.plot(
                [0, total],
                [total, 0],
                color=C7,
                linewidth=0.9,
                linestyle=(0, (2, 2)),
                alpha=0.72,
                zorder=0,
            )
            ax_scatter.text(
                total * 0.62,
                total * 0.38,
                f"{int(total)} cases",
                fontsize=7.5,
                color=TEXT_DARK,
                fontfamily=FONT_ARIAL,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.82),
            )
    label_offsets = {
        "Xiangyang": (35, 38),
        "Shiyan": (-12, -55),
        "Wuda": (14, 12),
        "Jingzhou": (14, 8),
        "Enshi": (14, 10),
    }
    annotate_centres = {"Shiyan", "Xiangyang"}
    for _, row in data.iterrows():
        color = centre_colors[row["Centre label"]]
        ax_scatter.scatter(
            row["Real reports"],
            row["Pseudo-report candidates"],
            s=_image_size(row["Total images"]),
            facecolor=color,
            edgecolor=TEXT_DARK,
            linewidth=0.95,
            alpha=0.94,
            zorder=3,
        )
        centre = row["Centre label"]
        if centre in annotate_centres:
            dx, dy = label_offsets[centre]
            ax_scatter.annotate(
                centre,
                xy=(row["Real reports"], row["Pseudo-report candidates"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if centre == "Xiangyang" else "right",
                va="center",
                fontsize=8.8,
                fontweight="bold",
                color=TEXT_DARK,
                arrowprops=dict(arrowstyle="-", color=TEXT_DARK, lw=0.55, shrinkA=4, shrinkB=2),
                zorder=7,
                clip_on=False,
            )
            continue
        dx, dy = label_offsets.get(centre, (12, 8))
        ha = "left" if dx > 0 else "right"
        ax_scatter.text(
            row["Real reports"] + dx,
            row["Pseudo-report candidates"] + dy,
            centre,
            ha=ha,
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color=TEXT_DARK,
            zorder=7,
            clip_on=False,
        )
    size_handles = [
        ax_scatter.scatter([], [], s=_image_size(k * 1000), facecolor="white", edgecolor=TEXT_DARK, linewidth=0.8, label=f"{k}k images")
        for k in [10, 30, 50]
    ]
    ax_scatter.legend(handles=size_handles, frameon=False, loc="upper right", title="Image volume", fontsize=8.0, title_fontsize=8.4)
    ax_scatter.set_xlim(-25, max_cases + 30)
    ax_scatter.set_ylim(-25, max_cases + 35)
    ax_scatter.set_xlabel("Archived real reports (cases)")
    ax_scatter.set_ylabel("Pseudo-report candidates (cases)")
    ax_scatter.set_title("Supervision coordinates", fontsize=11.2, fontweight="bold")
    ax_scatter.grid(True, color=C7, alpha=0.36)
    sns.despine(fig=fig, ax=ax_scatter)

    y_positions = np.arange(len(data))
    real_count_x = 1.14
    pseudo_count_x = 1.37
    for y, (_, row) in zip(y_positions, data.iterrows()):
        color = centre_colors[row["Centre label"]]
        if y % 2 == 1:
            ax_cover.axhspan(y - 0.5, y + 0.5, color="#f7f7f2", alpha=0.62, zorder=0)
        ax_cover.hlines(y, 0, row["Real-report coverage"], color=color, linewidth=2.4, alpha=0.56, zorder=2)
        ax_cover.scatter(
            row["Real-report coverage"],
            y,
            s=72 + 430 * (row["Cases"] / data["Cases"].max()),
            marker="D",
            facecolor=color,
            edgecolor=TEXT_DARK,
            linewidth=0.9,
            alpha=0.96,
            zorder=6,
            clip_on=False,
        )
        ax_cover.text(
            real_count_x,
            y,
            f"{int(row['Real reports'])}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold" if int(row["Real reports"]) > 0 else "normal",
            color=TEXT_DARK,
            zorder=7,
            clip_on=False,
        )
        ax_cover.text(
            pseudo_count_x,
            y,
            f"{int(row['Pseudo-report candidates'])}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold" if int(row["Pseudo-report candidates"]) > 0 else "normal",
            color=TEXT_DARK,
            zorder=7,
            clip_on=False,
        )
    ax_cover.axvline(0.5, color=EDGE_DARK, linestyle=(0, (2, 2)), linewidth=0.9, alpha=0.65)
    ax_cover.axvline(1.08, color=C7, linestyle=(0, (2, 2)), linewidth=0.9, alpha=0.85)
    ax_cover.text(real_count_x, -0.82, "Real\nreports", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=TEXT_DARK, clip_on=False, linespacing=1.05)
    ax_cover.text(pseudo_count_x, -0.82, "Pseudo\ncases", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=TEXT_DARK, clip_on=False, linespacing=1.05)
    ax_cover.set_yticks(y_positions)
    ax_cover.set_yticklabels(data["Centre label"], fontsize=9.4)
    ax_cover.set_xlim(-0.12, 1.48)
    ax_cover.set_ylim(len(data) - 0.5, -0.5)
    ax_cover.set_xlabel("Real-report coverage", labelpad=10)
    ax_cover.set_title("Coverage and report counts by centre", fontsize=11.2, fontweight="bold")
    ax_cover.set_xticks(np.linspace(0, 1, 6))
    ax_cover.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x * 100))}%"))
    ax_cover.grid(axis="x", color=C7, alpha=0.38)
    ax_cover.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax_cover.spines[spine].set_visible(False)
    ax_cover.spines["bottom"].set_color(EDGE_DARK)

    fig.suptitle("Centre-level cohort scale and report-supervision imbalance", fontsize=14.4, fontweight="bold", y=0.975)
    fig.text(
        0.985,
        0.018,
        "Bubble size encodes total OCT + colposcopy image volume.",
        ha="right",
        va="center",
        fontsize=7.4,
        color=TEXT_DARK,
    )
    fig.subplots_adjust(left=0.085, right=0.975, top=0.85, bottom=0.22)
    _save(fig, out_dir / "Figure2_centre_supervision_catplot")

    long = data.melt(
        id_vars=["Centre", "Centre label", "Cases"],
        value_vars=["Real reports", "Pseudo-report candidates"],
        var_name="Supervision",
        value_name="Count",
    )
    long["Supervision"] = long["Supervision"].str.replace("Pseudo-report candidates", "Pseudo-report candidate")

    # Proportion line (relplot style)
    prop = long.copy()
    prop["fraction"] = prop.groupby("Centre")["Count"].transform(lambda x: x / x.sum())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.lineplot(data=prop, x="Centre", y="fraction", hue="Supervision", marker="o", linewidth=1.5, alpha=0.86, ax=ax, palette=PALETTE_SUPERVISION)
    sns.scatterplot(data=prop, x="Centre", y="fraction", hue="Supervision", marker="s", s=92, edgecolor=TEXT_DARK, linewidth=0.8, ax=ax, palette=PALETTE_SUPERVISION, legend=False)
    ax.set_ylabel("Fraction of cases")
    ax.set_title("Report supervision mix by centre")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Supervision", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=18, ha="right")
    fig.tight_layout()
    _save(fig, out_dir / "Figure2_centre_supervision_fraction_line")
    t1b.to_csv(out_dir / "Figure2_centre_supervision_source.csv", index=False)


def fig03_perturbation(project: Path, out_dir: Path) -> None:
    """Heatmap + lineplot — gallery: heatmap / lineplot."""
    s6 = _read(project, f"{MANUSCRIPT_REL}/S6_modality_perturbation_text_decoding.csv")
    if s6 is None:
        return
    _setup_theme()
    conds = [
        "normal",
        "mask_oct",
        "mask_colposcopy",
        "mask_instruction",
        "mask_visual",
        "label_only_inference",
    ]
    sub = s6[s6["condition"].isin(conds)].copy()
    sec_cols = [
        "oct_findings_similarity_to_normal",
        "colposcopy_findings_similarity_to_normal",
        "clinical_context_similarity_to_normal",
        "impression_similarity_to_normal",
    ]
    melt = sub.melt(id_vars=["condition"], value_vars=sec_cols, var_name="section", value_name="similarity")
    melt["section"] = melt["section"].str.replace("_similarity_to_normal", "").str.replace("_", " ")

    # Heatmap (clustermap-style without cluster)
    piv = melt.pivot(index="condition", columns="section", values="similarity").reindex(conds)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(
        piv,
        annot=True,
        fmt=".2f",
        cmap=_cmap_sequential(),
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor=C7,
        cbar_kws={"label": "Similarity to normal"},
        ax=ax,
    )
    ax.set_title("Modality perturbation: section-specific degradation (n = 128)")
    ax.set_xlabel("")
    fig.tight_layout()
    _save(fig, out_dir / "Figure3_modality_perturbation_heatmap")

    # Faceted section dot plot: one panel per report section, one row per
    # perturbation condition. This avoids a crowded categorical line plot.
    section_order = ["oct findings", "colposcopy findings", "clinical context", "impression"]
    condition_labels = {
        "normal": "Normal",
        "mask_oct": "Mask OCT",
        "mask_colposcopy": "Mask colposcopy",
        "mask_instruction": "Mask clinical",
        "mask_visual": "Mask visual",
        "label_only_inference": "Label-only",
    }
    melt["condition_label"] = melt["condition"].map(condition_labels).fillna(melt["condition"])
    melt["section"] = pd.Categorical(melt["section"], categories=section_order, ordered=True)
    y_order = [condition_labels[c] for c in conds if c in condition_labels]
    y_map = {label: i for i, label in enumerate(y_order)}
    fig, axes = plt.subplots(1, len(section_order), figsize=(12.0, 5.0), sharey=True)
    for ax, section in zip(axes, section_order):
        g = melt[melt["section"].eq(section)].copy()
        g["y"] = g["condition_label"].map(y_map)
        ax.axvspan(0.0, 0.50, color=C4, alpha=0.07, zorder=0)
        ax.axvline(1.0, color=TEXT_DARK, lw=1.0, ls=(0, (2, 2)), alpha=0.55, zorder=1)
        ax.hlines(g["y"], 0, g["similarity"], color=C7, lw=4.2, alpha=0.92, zorder=1)
        ax.scatter(
            g["similarity"],
            g["y"],
            s=88,
            marker="o",
            color=C0 if section != "impression" else C6,
            edgecolor=TEXT_DARK,
            linewidth=0.75,
            alpha=0.96,
            zorder=3,
        )
        for _, row in g.iterrows():
            if float(row["similarity"]) < 0.55:
                ax.text(1.02, float(row["y"]), f"{float(row['similarity']):.2f}", va="center", ha="left", fontsize=8.5, fontweight="bold", color=TEXT_DARK, fontfamily=FONT_ARIAL, clip_on=False)
        ax.set_xlim(0, 1.18)
        ax.set_title(section.title(), fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Similarity")
        ax.grid(axis="x", alpha=0.30)
        ax.grid(axis="y", color=GRID_LINE, alpha=0.35)
    axes[0].set_yticks(np.arange(len(y_order)))
    axes[0].set_yticklabels(y_order, fontsize=9.8)
    axes[0].set_ylabel("Perturbation condition")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Section-wise response to modality perturbations", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "Figure3_modality_perturbation_lineplot")

    # Risk-score displacement plot: paired normal-vs-perturbed means.
    if "risk_score_delta_vs_normal" in sub.columns:
        r = sub.copy()
        r["condition_label"] = r["condition"].map(condition_labels).fillna(r["condition"])
        r = r[~r["condition"].eq("normal")].copy()
        r["abs_delta"] = r["risk_score_delta_vs_normal"].abs()
        r = r.sort_values("abs_delta", ascending=False).reset_index(drop=True)
        normal_mean = float(sub.loc[sub["condition"].eq("normal"), "mean_risk_score"].iloc[0])
        fig, ax = plt.subplots(figsize=(8.5, 4.9))
        y = np.arange(len(r))
        colors = [C4 if v > 0.25 else C2 if v > 0.10 else C0 for v in r["abs_delta"]]
        ax.axvline(normal_mean, color=TEXT_DARK, lw=1.1, ls=(0, (2, 2)), alpha=0.70, zorder=1)
        ax.axvspan(normal_mean - 0.015, normal_mean + 0.015, color=C7, alpha=0.25, zorder=0)
        xmax = float(r["mean_risk_score"].max())
        label_x = min(0.58, xmax + 0.085)
        for yi, (_, row), color in zip(y, r.iterrows(), colors):
            perturbed = float(row["mean_risk_score"])
            delta = float(row["risk_score_delta_vs_normal"])
            ax.hlines(yi, min(normal_mean, perturbed), max(normal_mean, perturbed), color=color, lw=5.0, alpha=0.72, zorder=2)
            ax.scatter(normal_mean, yi, marker="o", s=68, facecolor="white", edgecolor=TEXT_DARK, linewidth=1.1, zorder=4)
            ax.scatter(perturbed, yi, marker="D", s=100, facecolor=color, edgecolor=TEXT_DARK, linewidth=0.85, zorder=5)
            ax.text(
                label_x,
                yi,
                f"{perturbed:.3f} ({delta:+.3f})",
                ha="left",
                va="center",
                fontsize=8.8,
                fontfamily=FONT_ARIAL,
                fontweight="bold" if row["condition"] == "label_only_inference" else "normal",
                color=TEXT_DARK,
                clip_on=False,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(r["condition_label"], fontsize=10.5)
        ax.invert_yaxis()
        ax.set_xlim(max(0.0, normal_mean - 0.04), label_x + 0.10)
        ax.set_xlabel("Mean risk score (open circles mark normal input)")
        ax.set_ylabel("")
        ax.set_title("Risk-score displacement under perturbation", fontsize=16, fontweight="bold")
        ax.grid(axis="x", alpha=0.30)
        ax.grid(axis="y", color=GRID_LINE, alpha=0.35)
        fig.tight_layout()
        _save(fig, out_dir / "Figure3_risk_delta_stripplot")
    sub.to_csv(out_dir / "Figure3_modality_perturbation_source.csv", index=False)


def fig_main_model_comparison(project: Path, out_dir: Path) -> None:
    """Bar+point+CI and heatmap — journal-style summary panels."""
    t2 = _read(project, f"{MANUSCRIPT_REL}/T2_main_model_comparison_with_ci.csv")
    if t2 is None:
        t2 = _read(project, f"{MANUSCRIPT_REL}/T2_main_model_comparison.csv")
    if t2 is None:
        return
    _setup_theme()
    if "auc" not in t2.columns and "auc_all" in t2.columns:
        t2 = t2.rename(columns={"auc_all": "auc", "f1_at_val_threshold": "f1"})

    # Bar + point + CI with paired-bootstrap bracket annotations.
    plot_df = t2.copy()
    ci_low_col = "auc_ci_low" if "auc_ci_low" in plot_df.columns else ("ci_low" if "ci_low" in plot_df.columns else None)
    ci_high_col = "auc_ci_high" if "auc_ci_high" in plot_df.columns else ("ci_high" if "ci_high" in plot_df.columns else None)
    if ci_low_col and ci_high_col:
        fig, ax = plt.subplots(figsize=(12.4, 5.8))
        pal = _model_palette(plot_df)
        x = np.arange(len(plot_df))
        colors = [pal.get(m, PALETTE_MAIN[i % len(PALETTE_MAIN)]) for i, m in enumerate(plot_df["model"])]
        ax.bar(
            x,
            plot_df["auc"],
            color=colors,
            edgecolor=EDGE_DARK,
            linewidth=1.0,
            alpha=0.86,
            width=0.68,
            zorder=1,
        )
        for xi, (_, row) in zip(x, plot_df.iterrows()):
            c = pal.get(row["model"], C6)
            ax.errorbar(
                xi,
                row["auc"],
                yerr=[[row["auc"] - row[ci_low_col]], [row[ci_high_col] - row["auc"]]],
                fmt="o",
                ecolor=EDGE_DARK,
                elinewidth=1.4,
                capsize=4.5,
                markersize=6.5,
                color=c,
                mec=EDGE_DARK,
                mew=0.8,
                zorder=3,
            )
            if "f1" in plot_df.columns:
                ax.scatter(xi + 0.16, row["f1"], marker="^", s=48, color=C4, edgecolor=EDGE_DARK, linewidth=0.7, zorder=4)
        pvals: dict[str, float] = {}
        from src.supplementary.jbd_figure_stats import load_comparator_pvals

        pvals = load_comparator_pvals(project)
        full_idx = plot_df.index[plot_df["model"].eq("Full LCAD-RASA")].tolist()
        if full_idx:
            x_full = list(plot_df.index).index(full_idx[0])
            y0 = min(1.02, float(plot_df["auc"].max()) + 0.04)
            bracket_targets = [
                "Pseudo-augmented (LCAD)",
                "Real-report only",
                "Simple concat fusion",
                "LCAD w/o section alignment",
                "Instruction-only report gen.",
            ]
            for level, target in enumerate(bracket_targets):
                hit = plot_df.index[plot_df["model"].eq(target)].tolist()
                if hit:
                    xt = list(plot_df.index).index(hit[0])
                    _add_bracket(ax, min(x_full, xt), max(x_full, xt), y0 + level * 0.055, _p_label(pvals.get(target)), h=0.018)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["model"], rotation=35, ha="right")
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        ax.set_title("Main model comparison with bootstrap intervals")
        ax.set_ylim(0, 1.32)
        ax.axhline(0.5, color=C6, ls=":", lw=1, alpha=0.75)
        ax.scatter([], [], marker="o", color=C0, edgecolor=EDGE_DARK, label="AUROC")
        ax.scatter([], [], marker="^", color=C4, edgecolor=EDGE_DARK, label="F1")
        ax.legend(frameon=False, loc="upper left", ncol=2)
        fig.tight_layout()
        _save(fig, out_dir / "Figure_main_AUC_pointplot")

    # Metrics profile — grouped horizontal bars (avoid duplicating heatmaps used elsewhere).
    mcols = [c for c in ["auc", "f1", "sensitivity", "specificity"] if c in t2.columns]
    if mcols:
        hm = t2.set_index("model")[mcols]
        long = hm.reset_index().melt(id_vars="model", var_name="metric", value_name="score")
        metric_labels = {"auc": "AUROC", "f1": "F1", "sensitivity": "Sensitivity", "specificity": "Specificity"}
        long["metric"] = long["metric"].map(metric_labels).fillna(long["metric"])
        order = t2["model"].tolist()
        fig, ax = plt.subplots(figsize=(8.2, 4 + 0.28 * len(order)))
        sns.barplot(
            data=long,
            y="model",
            x="score",
            hue="metric",
            order=order,
            hue_order=[metric_labels.get(c, c) for c in mcols],
            palette=PALETTE_MAIN[: len(mcols)],
            orient="h",
            edgecolor=EDGE_DARK,
            linewidth=0.8,
            alpha=0.88,
            ax=ax,
        )
        sns.stripplot(
            data=long,
            y="model",
            x="score",
            hue="metric",
            order=order,
            hue_order=[metric_labels.get(c, c) for c in mcols],
            dodge=True,
            marker="D",
            size=5,
            linewidth=0.6,
            edgecolor=EDGE_DARK,
            palette=PALETTE_MAIN[: len(mcols)],
            ax=ax,
            legend=False,
        )
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Score")
        ax.set_ylabel("")
        ax.set_title("Multi-metric profile (validation-selected threshold)")
        ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        fig.tight_layout()
        _save(fig, out_dir / "Figure_main_metrics_heatmap")

    # F1 vs AUC scatter (joint-style) with pooled p annotation
    if "f1" in t2.columns and "auc" in t2.columns:
        fig, ax = plt.subplots(figsize=(8.2, 6))
        sns.scatterplot(data=t2, x="auc", y="f1", hue="model", s=180, palette=_model_palette(t2), ax=ax, edgecolor=C7, linewidth=0.8, legend="brief")
        from src.supplementary.jbd_figure_stats import format_pvalue, load_comparator_pvals

        pvals = load_comparator_pvals(project)
        ax.text(
            0.03,
            0.97,
            f"vs Full LCAD-RASA (paired bootstrap):\nReal-report only {format_pvalue(pvals.get('Real-report only'))}\nSimple concat {format_pvalue(pvals.get('Simple concat fusion'))}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontfamily=FONT_ARIAL,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#d0d0d0", alpha=0.92),
        )
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, title_fontsize=7, frameon=False)
        ax.set_xlim(0.3, 0.9)
        ax.set_xlabel("AUROC")
        ax.set_ylabel("F1 score")
        ax.set_title("Risk–semantic trade-off (AUC vs F1)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Model", title_fontsize=9, frameon=False)
        fig.tight_layout()
        _save(fig, out_dir / "Figure_main_auc_f1_scatter")
    t2.to_csv(out_dir / "Figure_main_AUC_comparison_source.csv", index=False)


def fig_per_case_distributions(project: Path, out_dir: Path) -> None:
    """Violin + kde + joint — gallery: violinplot / kdeplot / jointplot."""
    pred_dir = project / PRED_REL
    if not pred_dir.is_dir():
        return
    frames = []
    for p in sorted(pred_dir.glob("*_test_predictions.csv")):
        model = p.name.replace("_test_predictions.csv", "")
        d = pd.read_csv(p)
        d["model"] = model.replace("_", " ").replace("full lcad rasa", "Full LCAD-RASA").title()
        if "full_lcad_rasa" in p.name:
            d["model"] = "Full LCAD-RASA"
        elif "real_report" in p.name:
            d["model"] = "Real-report only"
        elif "simple_concat" in p.name:
            d["model"] = "Simple concat"
        elif "report_generation" in p.name:
            d["model"] = "LCAD w/o section"
        frames.append(d)
    if not frames:
        return
    _setup_theme()
    allp = pd.concat(frames, ignore_index=True)
    allp["CIN2+ label"] = allp["y_true_cin2plus"].map({0: "Negative", 1: "Positive"})
    core = allp[allp["model"].isin(["Full LCAD-RASA", "Real-report only", "Simple concat", "LCAD w/o section"])]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.violinplot(data=core, x="model", y="risk_score", hue="CIN2+ label", split=True, inner="quart", palette=PALETTE_BINARY, ax=ax, linewidth=0.8)
    sns.swarmplot(data=core.sample(min(400, len(core)), random_state=42), x="model", y="risk_score", hue="CIN2+ label", dodge=True, size=2, alpha=0.35, ax=ax, legend=False)
    ax.set_title("Predicted risk distributions by model and true label (test set)")
    ax.set_ylabel("Risk score")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _save(fig, out_dir / "Figure_risk_violin_swarm")

    # KDE for full model only
    full = allp[allp["model"] == "Full LCAD-RASA"]
    if len(full):
        g = sns.displot(data=full, x="risk_score", hue="CIN2+ label", kind="kde", fill=True, height=4.5, aspect=1.4, palette=PALETTE_BINARY, linewidth=2)
        g.fig.suptitle("Full LCAD-RASA: risk score density", y=1.02)
        _save(g.fig, out_dir / "Figure_full_model_kdeplot")

        g2 = sns.jointplot(data=full, x="risk_score", y="correct", hue="CIN2+ label", kind="scatter", height=5, palette=PALETTE_BINARY, marginal_kws=dict(fill=True))
        g2.ax_joint.set_ylabel("Prediction correct (0/1)")
        g2.ax_joint.set_xlabel("Risk score")
        g2.fig.suptitle("Joint: risk vs correctness (Full LCAD-RASA)", y=1.02)
        _save(g2.fig, out_dir / "Figure_full_model_jointplot")


# ---------------------------------------------------------------------------
# Supplementary / legacy figure names (outputs/publishable/figures/)
# ---------------------------------------------------------------------------


def _draw_masking_panel(ax: plt.Axes, s10: pd.DataFrame, show_legend: bool = True) -> None:
    setting_order = ["label_only_agent", "modality_only_agent", "modality_plus_label_agent"]
    setting_labels = {
        "label_only_agent": "Label only",
        "modality_only_agent": "Modality only",
        "modality_plus_label_agent": "Modality + label",
    }
    centre_labels = {
        "enshi": "Enshi",
        "jingzhou": "Jingzhou",
        "xiangyang": "Xiangyang",
        "enshi_jingzhou_pooled": "Enshi + Jingzhou",
    }
    centre_order = ["enshi", "jingzhou", "enshi_jingzhou_pooled", "xiangyang"]
    setting_offsets = {"label_only_agent": -0.18, "modality_only_agent": 0.0, "modality_plus_label_agent": 0.18}
    setting_markers = {"label_only_agent": "o", "modality_only_agent": "s", "modality_plus_label_agent": "D"}
    setting_colors = {"label_only_agent": C0, "modality_only_agent": C2, "modality_plus_label_agent": C4}

    data = s10[s10["setting"].isin(setting_order)].copy()
    data = data[data["center_id"].isin(centre_order)].copy()
    data["y_base"] = data["center_id"].map({c: i for i, c in enumerate(centre_order)}).astype(float)
    max_n = max(float(data["n_cases"].max()), 1.0)

    for i, centre in enumerate(centre_order):
        ax.axhline(i, color=C7, linewidth=1.0, alpha=0.55, zorder=0)

    handles = []
    for setting in setting_order:
        g = data[data["setting"].eq(setting)].sort_values("y_base")
        if g.empty:
            continue
        offset = setting_offsets[setting]
        y = g["y_base"] + offset
        x = g["label_consistency_mean"]
        sizes = 42 + 175 * np.sqrt(g["n_cases"] / max_n)
        sc = ax.scatter(
            x,
            y,
            s=sizes,
            marker=setting_markers[setting],
            facecolor=setting_colors[setting],
            edgecolor=TEXT_DARK,
            linewidth=0.8,
            alpha=0.94,
            zorder=3,
            label=setting_labels[setting],
        )
        handles.append(sc)

    ax.set_yticks(np.arange(len(centre_order)))
    ax.set_yticklabels([f"{centre_labels[c]} (n={int(data[data['center_id'].eq(c)]['n_cases'].max())})" for c in centre_order])
    ax.invert_yaxis()
    ax.set_xlim(0.47, 0.78)
    ax.set_xlabel("Label-consistency proxy")
    ax.set_ylabel("")
    ax.set_title("Report masking sensitivity across evidence settings", fontsize=11.8, fontweight="bold", pad=12)
    ax.grid(axis="x", color=C7, alpha=0.45)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    ax.tick_params(axis="y", length=0)
    if show_legend:
        leg = ax.legend(
            handles=handles,
            title="Evidence setting",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            ncol=1,
            fontsize=8.1,
            title_fontsize=8.4,
            handletextpad=0.45,
            borderaxespad=0.0,
        )
        for text in leg.get_texts():
            text.set_color(TEXT_DARK)


def supp_masking(project: Path, out_dir: Path) -> None:
    s10 = _read(project, f"{MANUSCRIPT_REL}/S10_masking_validation.csv")
    if s10 is None:
        return
    _setup_theme()
    fig, ax = plt.subplots(figsize=(8.4, 4.65))
    _draw_masking_panel(ax, s10, show_legend=True)
    fig.subplots_adjust(left=0.29, right=0.76, top=0.84, bottom=0.16)
    _save(fig, out_dir / "fig_masking_pointplot")
    fig, ax = plt.subplots(figsize=(8.4, 4.65))
    _draw_masking_panel(ax, s10, show_legend=True)
    fig.subplots_adjust(left=0.29, right=0.76, top=0.84, bottom=0.16)
    _save(fig, out_dir / "SupplementaryFigure_S1_masking_validation")


def supp_loco(project: Path, out_dir: Path) -> None:
    s2 = _read(project, f"{MANUSCRIPT_REL}/S2_loco_strict_retrain.csv")
    if s2 is None:
        s2 = _read(project, f"{TABLES_REL}/table_loco_main_results.csv")
    if s2 is None:
        return
    _setup_theme()
    s2 = s2.copy()
    loco_model_labels = {
        "full_lcad_rasa": "Full LCAD-RASA",
        "real_report_only_decoder": "Real-report only",
        "report_generation_without_section_alignment": "No section alignment",
    }
    s2["model_short"] = s2["model"].map(loco_model_labels).fillna(s2["model"].astype(str).str.replace("_", " "))
    model_order = ["Real-report only", "No section alignment", "Full LCAD-RASA"]
    model_offsets = {"Real-report only": -0.16, "No section alignment": 0.0, "Full LCAD-RASA": 0.16}
    model_markers = {"Real-report only": "s", "No section alignment": "^", "Full LCAD-RASA": "D"}
    model_colors = {"Real-report only": C1, "No section alignment": C2, "Full LCAD-RASA": C0}
    full_auc = s2[s2["model"].eq("full_lcad_rasa")][["center_label", "auc"]].rename(columns={"auc": "full_auc"})
    center_summary = (
        s2.groupby("center_label", as_index=False)
        .agg(test_cases=("test_cases", "max"), min_auc=("auc", "min"), max_auc=("auc", "max"))
        .merge(full_auc, on="center_label", how="left")
        .sort_values(["full_auc", "center_label"], ascending=[True, True])
        .reset_index(drop=True)
    )
    center_to_y = {c: i for i, c in enumerate(center_summary["center_label"])}

    fig, axes = plt.subplots(1, len(model_order), figsize=(11.4, 5.4), sharey=True)
    label_x = 1.06
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
            ax.text(
                label_x,
                float(row["y"]),
                f"{float(row['auc']):.3f}",
                ha="left",
                va="center",
                fontsize=9.0,
                fontfamily=FONT_ARIAL,
                fontweight="bold" if model_label == "Full LCAD-RASA" else "normal",
                color=TEXT_DARK,
                clip_on=False,
            )
        ax.set_title(model_label, fontsize=12.5, fontweight="bold", color=TEXT_DARK)
        ax.set_xlim(0.25, label_x + 0.08)
        ax.set_xlabel("AUROC")
        ax.grid(axis="x", alpha=0.30)
        ax.grid(axis="y", color=GRID_LINE, alpha=0.45)
    y_labels = [
        f"{row['center_label']}  (n={int(row['test_cases'])})"
        for _, row in center_summary.iterrows()
    ]
    axes[0].set_yticks(np.arange(len(center_summary)))
    axes[0].set_yticklabels(y_labels, fontsize=10.5)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    axes[0].set_ylabel("Held-out centre")
    fig.suptitle("Strict LOCO AUROC by centre and model", fontsize=16, fontweight="bold", y=1.02)
    axes[-1].text(0.505, len(center_summary) - 0.25, "chance", ha="left", va="center", fontsize=9.3, color=TEXT_DARK)
    fig.tight_layout()
    _save(fig, out_dir / "fig_loco_heatmap")

    fig, ax = plt.subplots(figsize=(9.8, 5.9))
    ax.axvspan(0.25, 0.50, color=C4, alpha=0.08, zorder=0)
    ax.axvline(0.50, color=TEXT_DARK, lw=1.2, ls=(0, (2, 2)), alpha=0.78, zorder=1)
    range_x = 1.12
    auc_label_x = 1.02
    ax.text(0.505, len(center_summary) - 0.25, "chance", ha="left", va="center", fontsize=9.5, color=TEXT_DARK, fontfamily=FONT_ARIAL)
    for _, row in center_summary.iterrows():
        y = center_to_y[row["center_label"]]
        ax.hlines(y, row["min_auc"], row["max_auc"], color=C7, lw=9, alpha=0.88, zorder=1)
        ax.hlines(y, row["min_auc"], row["max_auc"], color=EDGE_DARK, lw=1.1, alpha=0.52, zorder=2)
        ax.text(
            range_x,
            y,
            f"range {row['max_auc'] - row['min_auc']:.3f}",
            ha="left",
            va="center",
            fontsize=9.2,
            fontfamily=FONT_ARIAL,
            color=TEXT_DARK,
            fontweight="bold" if row["max_auc"] - row["min_auc"] > 0.20 else "normal",
            clip_on=False,
        )
    for model_label in model_order:
        sub = s2[s2["model_short"].eq(model_label)].copy()
        sub["y"] = sub["center_label"].map(center_to_y).astype(float) + model_offsets[model_label]
        ax.scatter(
            sub["auc"],
            sub["y"],
            s=92 if model_label != "Full LCAD-RASA" else 118,
            marker=model_markers[model_label],
            color=model_colors[model_label],
            edgecolor=TEXT_DARK,
            linewidth=0.85,
            alpha=0.96,
            label=model_label,
            zorder=4,
        )
    for _, row in s2[s2["model_short"].eq("Full LCAD-RASA")].iterrows():
        y = center_to_y[row["center_label"]] + model_offsets["Full LCAD-RASA"]
        ax.text(
            auc_label_x,
            y,
            f"{float(row['auc']):.3f}",
            ha="left",
            va="center",
            fontsize=9.0,
            fontfamily=FONT_ARIAL,
            fontweight="bold",
            color=C0,
            clip_on=False,
        )
    y_labels = [
        f"{row['center_label']}  (n={int(row['test_cases'])})"
        for _, row in center_summary.iterrows()
    ]
    ax.set_yticks(np.arange(len(center_summary)))
    ax.set_yticklabels(y_labels, fontsize=10.8)
    ax.set_xlim(0.25, range_x + 0.10)
    ax.set_ylim(-0.55, len(center_summary) - 0.35)
    ax.set_xlabel("AUROC under strict leave-one-centre-out retraining")
    ax.set_ylabel("")
    ax.set_title("Cross-centre generalisation under strict LOCO")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.50, -0.22), title=None)
    ax.grid(axis="x", alpha=0.30)
    ax.grid(axis="y", visible=False)
    ax.xaxis.label.set_fontweight("bold")
    ax.title.set_fontweight("bold")
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    _save(fig, out_dir / "Figure4_loco_forest_catplot")
    _save(fig, out_dir / "SupplementaryFigure_S2_loco_catplot")


def supp_lambda_sweep(project: Path, out_dir: Path) -> None:
    s1 = _read(project, f"{MANUSCRIPT_REL}/S1_rasa_lambda_align_sweep.csv")
    if s1 is None:
        return
    _setup_theme()
    s1 = s1.sort_values("lambda_align")
    labels = [f"{v:.2f}" if v > 0 else "0" for v in s1["lambda_align"]]
    x = np.arange(len(s1), dtype=float)
    auc = s1["auc"].astype(float).to_numpy()
    baseline = float(s1.loc[s1["lambda_align"].eq(0), "auc"].iloc[0]) if (s1["lambda_align"].eq(0)).any() else float(auc[0])
    delta = auc - baseline
    best = s1.loc[s1["auc"].idxmax()]
    best_idx = int(s1.index.get_loc(best.name))

    fig, (ax, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.3, 1.15], "hspace": 0.08},
    )

    ax.axhline(baseline, color=EDGE_DARK, lw=1.35, ls=(0, (2, 2)), alpha=0.8, zorder=1)
    ax.text(
        len(x) - 0.15,
        baseline + 0.00045,
        rf"$\lambda_{{align}}=0$ baseline: {baseline:.3f}",
        ha="right",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=TEXT_DARK,
    )
    ax.fill_between(x, baseline, auc, where=auc >= baseline, color=C1, alpha=0.34, interpolate=True, zorder=1)
    ax.plot(x, auc, color=C0, lw=2.7, marker="o", markersize=7.5, markerfacecolor="white", markeredgewidth=1.6, markeredgecolor=C0, zorder=3)
    ax.scatter(
        x[best_idx],
        float(best["auc"]),
        s=180,
        marker="D",
        color=C2,
        edgecolor=TEXT_DARK,
        linewidth=1.0,
        zorder=4,
        label="Best observed setting",
    )
    ax.annotate(
        f"best lambda={best['lambda_align']:.2f}\nAUROC={best['auc']:.3f}; delta={float(best['auc'] - baseline):+.3f}",
        xy=(x[best_idx], float(best["auc"])),
        xytext=(x[best_idx] - 1.35, float(best["auc"]) + 0.010),
        ha="right",
        va="bottom",
        fontsize=10.0,
        fontfamily=FONT_ARIAL,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C2, lw=1.0, alpha=0.96),
        arrowprops=dict(arrowstyle="-|>", color=C2, lw=1.2, shrinkA=4, shrinkB=4),
    )
    ax.set_ylabel("Held-out AUROC")
    ax.set_title(r"RASA section-alignment weight sensitivity")
    ax.set_ylim(float(auc.min() - 0.006), float(auc.max() + 0.014))
    ax.legend(frameon=False, loc="lower right")

    ax_delta.axhline(0, color=EDGE_DARK, lw=1.2, alpha=0.8)
    marker_colors = [C4 if d < 0 else C0 for d in delta]
    marker_colors[best_idx] = C2
    ax_delta.vlines(x, 0, delta, color=C7, lw=2.0, alpha=0.85, zorder=1)
    ax_delta.scatter(x, delta, s=82, marker="s", c=marker_colors, edgecolor=TEXT_DARK, linewidth=0.8, zorder=3)
    ax_delta.set_ylabel(r"$\Delta$AUROC")
    ax_delta.set_xlabel(r"Section-alignment weight $\lambda_{\mathrm{align}}$")
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(labels)
    margin = max(0.003, float(np.abs(delta).max()) * 0.35)
    ax_delta.set_ylim(float(delta.min() - margin), float(delta.max() + margin))
    ax_delta.grid(axis="x", alpha=0.18)

    for axis in (ax, ax_delta):
        axis.spines["left"].set_linewidth(1.2)
        axis.spines["bottom"].set_linewidth(1.2)
        axis.tick_params(axis="both", width=1.1, length=4.0)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontsize(10.5)
    ax.yaxis.label.set_fontweight("bold")
    ax_delta.yaxis.label.set_fontweight("bold")
    ax_delta.xaxis.label.set_fontweight("bold")
    fig.tight_layout()
    _save(fig, out_dir / "fig_rasa_lambda_lineplot")


def supp_modality_ablation(project: Path, out_dir: Path) -> None:
    s3 = _read(project, f"{MANUSCRIPT_REL}/S3_modality_ablation.csv")
    if s3 is None:
        return
    _setup_theme()
    s3 = s3.copy()
    s3["modality_set"] = s3["experiment_id"].str.replace("_", " + ")
    s3 = s3.sort_values("auc", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=s3, y="modality_set", x="auc", palette=PALETTE_MAIN[: len(s3)], ax=ax, orient="h", edgecolor=EDGE_DARK, linewidth=0.8, alpha=0.86)
    sns.stripplot(data=s3, x="auc", y="modality_set", color=C4, marker="^", size=8, ax=ax, jitter=False)
    ax.set_xlim(0.55, 0.85)
    ax.set_title("Modality ablation: AUROC by input combination")
    fig.tight_layout()
    _save(fig, out_dir / "fig_modality_ablation_barplot")

    # Dot + strip composite
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.stripplot(data=s3, x="auc", y="modality_set", size=12, palette=PALETTE_MAIN, ax=ax, jitter=False)
    sns.pointplot(data=s3, x="auc", y="modality_set", color=C0, markers="X", linestyles="", ax=ax)
    ax.set_title("Modality subsets (strip + point)")
    fig.tight_layout()
    _save(fig, out_dir / "fig_modality_ablation_stripplot")
    _save(fig, out_dir / "fig_modality_ablation_section_heatmap")  # alias for old name


def supp_rasa_components(project: Path, out_dir: Path) -> None:
    s5 = _read(project, f"{MANUSCRIPT_REL}/S5_rasa_component_ablation.csv")
    if s5 is None:
        return
    _setup_theme()
    ref = s5[s5["experiment_id"] == "full_lcad_rasa"]
    ref_auc = float(ref["auc"].iloc[0]) if len(ref) else 0.8
    s5 = s5.copy()
    s5["delta_auc"] = s5["auc"] - ref_auc
    component_labels = {
        "full_lcad_rasa": "Full LCAD-RASA",
        "no_section_alignment": "No section alignment",
        "no_label_consistency_loss": "No label-consistency loss",
        "risk_head_only_auxiliary": "Risk-head auxiliary only",
        "no_risk_head": "No risk head",
        "report_loss_only": "Report loss only",
        "section_alignment_only_auxiliary": "Section-alignment auxiliary only",
    }
    s5["variant"] = s5["experiment_id"].map(component_labels).fillna(s5["experiment_id"].str.replace("_", " "))
    s5 = s5.sort_values("delta_auc")
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = [C4 if v < 0 else C0 for v in s5["delta_auc"]]
    sns.barplot(data=s5, y="variant", x="delta_auc", palette=colors, ax=ax, orient="h", edgecolor=EDGE_DARK, linewidth=0.9, alpha=0.86, legend=False)
    sns.stripplot(data=s5, y="variant", x="delta_auc", color=C2, marker="D", size=7, ax=ax, jitter=False)
    ax.axvline(0, color=EDGE_DARK, lw=1.0)
    ax.set_title("RASA component ablation: ΔAUROC vs full model")
    ax.set_xlabel("ΔAUROC")
    ax.set_ylabel("")
    fig.tight_layout()
    _save(fig, out_dir / "fig_rasa_component_ablation")

    # Ridgeline-style metric profiles. Each ridge summarises the available
    # scalar metrics for one component variant, avoiding a misleading boxenplot
    # when each variant has only one row of observations.
    if "f1" in s5.columns:
        metric_cols = [c for c in ["auc", "f1", "sensitivity", "specificity", "label_consistency"] if c in s5.columns]
        metric_style = {
            "auc": ("AUROC", "o", C0),
            "f1": ("F1", "s", C4),
            "sensitivity": ("Sensitivity", "^", C2),
            "specificity": ("Specificity", "D", C1),
            "label_consistency": ("Label consistency", "v", C6),
        }
        ridge = s5.sort_values(["auc", "f1"], ascending=[True, True]).reset_index(drop=True)
        x_grid = np.linspace(0.0, 1.02, 420)
        bandwidth = 0.045
        fig, ax = plt.subplots(figsize=(10.0, 6.6))
        for y, (_, row) in enumerate(ridge.iterrows()):
            values = row[metric_cols].astype(float).to_numpy()
            density = np.zeros_like(x_grid)
            for value in values:
                density += np.exp(-0.5 * ((x_grid - value) / bandwidth) ** 2)
            if density.max() > 0:
                density = density / density.max() * 0.72
            if row["experiment_id"] == "full_lcad_rasa":
                fill = C2
                line = C2
                label_weight = "bold"
            elif float(row["auc"]) < 0.65:
                fill = C6
                line = "#777777"
                label_weight = "normal"
            elif float(row["delta_auc"]) >= 0:
                fill = C0
                line = C0
                label_weight = "bold"
            else:
                fill = C4
                line = C4
                label_weight = "normal"
            ax.fill_between(x_grid, y, y + density, color=fill, alpha=0.78, linewidth=0, zorder=1)
            ax.plot(x_grid, y + density, color=line, lw=1.45, alpha=0.95, zorder=2)
            ax.hlines(y, 0, 1.02, color=GRID_LINE, lw=0.8, alpha=0.55, zorder=0)
            for metric in metric_cols:
                metric_label, marker, color = metric_style[metric]
                ax.scatter(
                    float(row[metric]),
                    y + 0.06,
                    s=58 if metric != "auc" else 82,
                    marker=marker,
                    color=color,
                    edgecolor=TEXT_DARK,
                    linewidth=0.65,
                    alpha=0.96,
                    zorder=4,
                )
            ax.text(
                1.035,
                y + 0.34,
                f"AUROC {float(row['auc']):.3f}",
                ha="left",
                va="center",
                fontsize=9.5,
                fontweight=label_weight,
                color=TEXT_DARK,
            )

        ax.axvline(0.5, color=EDGE_DARK, lw=1.0, ls=(0, (2, 2)), alpha=0.72)
        ax.axvline(ref_auc, color=C2, lw=1.2, ls="-.", alpha=0.9)
        ref_text_y = len(ridge) + 0.42
        ax.text(0.505, ref_text_y, "chance", ha="left", va="top", fontsize=9, color=TEXT_DARK)
        ax.text(ref_auc + 0.006, ref_text_y, "full-model AUROC", ha="left", va="top", fontsize=9, fontweight="bold", color=C2)
        ax.set_xlim(0.0, 1.13)
        ax.set_ylim(-0.20, len(ridge) + 0.72)
        ax.set_yticks(np.arange(len(ridge)) + 0.22)
        ax.set_yticklabels(ridge["variant"], fontsize=10.5)
        ax.set_xlabel("Metric value")
        ax.set_ylabel("")
        ax.set_title("RASA component ablation metric profiles")
        handles = [
            plt.Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, markeredgecolor=TEXT_DARK, markersize=8, label=label)
            for metric, (label, marker, color) in metric_style.items()
            if metric in metric_cols
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            ncol=len(handles),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.24),
            title=None,
            columnspacing=1.2,
            handletextpad=0.45,
        )
        ax.grid(axis="x", alpha=0.24)
        ax.grid(axis="y", visible=False)
        for label in ax.get_xticklabels():
            label.set_fontsize(10.5)
        ax.xaxis.label.set_fontweight("bold")
        ax.title.set_fontweight("bold")
        fig.tight_layout(rect=[0, 0.15, 1, 1])
        _save(fig, out_dir / "fig_rasa_component_boxenplot")


def _draw_multiseed_panel(ax: plt.Axes, project: Path, s7: pd.DataFrame) -> None:
    model_labels = {
        "real_report_only_decoder": "Real-report only",
        "report_generation_without_section_alignment": "No section alignment",
        "full_lcad_rasa": "Full LCAD-RASA",
    }
    model_order = ["real_report_only_decoder", "report_generation_without_section_alignment", "full_lcad_rasa"]
    model_colors = {"real_report_only_decoder": C1, "report_generation_without_section_alignment": C2, "full_lcad_rasa": C0}
    y_map = {m: i for i, m in enumerate(model_order)}

    raw = _read(project, f"{TABLES_REL}/table_multiseed_raw.csv")
    if raw is not None and {"model", "auc", "seed"}.issubset(raw.columns):
        raw = raw[raw["model"].isin(model_order)].copy()
        seed_values = sorted(raw["seed"].unique().tolist())
        if seed_values:
            seed_offsets = {seed: offset for seed, offset in zip(seed_values, np.linspace(-0.12, 0.12, len(seed_values)))}
            for _, row in raw.iterrows():
                model = row["model"]
                ax.scatter(
                    row["auc"],
                    y_map[model] + seed_offsets[row["seed"]],
                    s=42,
                    marker="o",
                    facecolor=model_colors[model],
                    edgecolor="white",
                    linewidth=0.75,
                    alpha=0.86,
                    zorder=3,
                )

    auc_summary = s7[(s7["metric"] == "auc") & (s7["model"].isin(model_order))].copy()
    for _, row in auc_summary.iterrows():
        model = row["model"]
        y = y_map[model]
        color = model_colors[model]
        ax.hlines(y, row["ci_low"], row["ci_high"], color=color, linewidth=4.8, alpha=0.36, zorder=1)
        ax.scatter(
            row["mean"],
            y,
            s=92,
            marker="D",
            facecolor=color,
            edgecolor=TEXT_DARK,
            linewidth=0.85,
            alpha=0.98,
            zorder=4,
        )
        ax.text(
            row["ci_high"] + 0.004,
            y,
            f"{row['mean']:.3f}",
            ha="left",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color=TEXT_DARK,
        )

    for y in y_map.values():
        ax.axhline(y, color=C7, linewidth=1.0, alpha=0.50, zorder=0)
    ax.set_yticks([y_map[m] for m in model_order])
    ax.set_yticklabels([model_labels[m] for m in model_order])
    ax.set_xlim(0.63, 0.81)
    ax.set_xlabel("AUROC across random seeds")
    ax.set_ylabel("")
    ax.set_title("Multi-seed stability", fontsize=11.4, fontweight="bold")
    ax.grid(axis="x", color=C7, alpha=0.45)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    ax.tick_params(axis="y", length=0)
    ax.text(0.63, 2.48, "circles: seed runs; diamonds: mean; bars: 95% CI", ha="left", va="center", fontsize=7.4, color=TEXT_DARK)


def _save_combined_robustness(project: Path, out_dir: Path, s7: pd.DataFrame) -> None:
    s10 = _read(project, f"{MANUSCRIPT_REL}/S10_masking_validation.csv")
    if s10 is None:
        return
    fig, (ax_mask, ax_seed) = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [1.03, 1.0], "wspace": 0.35})
    _draw_masking_panel(ax_mask, s10, show_legend=True)
    _draw_multiseed_panel(ax_seed, project, s7)
    fig.suptitle("Supplementary robustness checks: masking sensitivity and random-seed stability", fontsize=14.0, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_dir / "SupplementaryFigure_S1_S3_robustness_combined")


def supp_multiseed(project: Path, out_dir: Path) -> None:
    s7 = _read(project, f"{MANUSCRIPT_REL}/S7_multiseed_stability.csv")
    if s7 is None:
        return
    _setup_theme()
    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    _draw_multiseed_panel(ax, project, s7)
    fig.suptitle("Random-seed stability across model variants", fontsize=13.4, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir / "SupplementaryFigure_S3_multiseed")

    # Faceted metric summary retained for compatibility, but styled as a compact dot grid.
    metric_order = ["auc", "label_consistency", "section_completeness", "hallucination_rate"]
    sub = s7[s7["metric"].isin(metric_order)].copy()
    if len(sub):
        sub["metric"] = pd.Categorical(sub["metric"], metric_order, ordered=True)
        sub["model_label"] = sub["model"].replace(
            {
                "real_report_only_decoder": "Real-report only",
                "report_generation_without_section_alignment": "No section alignment",
                "full_lcad_rasa": "Full LCAD-RASA",
            }
        )
        g = sns.FacetGrid(sub, col="metric", col_wrap=2, height=2.8, aspect=1.2, sharex=False, sharey=True)
        g.map_dataframe(sns.scatterplot, x="mean", y="model_label", hue="model_label", palette=[C1, C2, C0], s=70, edgecolor=TEXT_DARK, linewidth=0.6, legend=False)
        g.set_axis_labels("Mean across seeds", "")
        g.set_titles("{col_name}")
        for ax in g.axes.flat:
            ax.grid(axis="x", color=C7, alpha=0.45)
            ax.grid(axis="y", color=C7, alpha=0.30)
        g.fig.suptitle("Stability across metrics and seeds", y=1.03, fontsize=12.2, fontweight="bold")
        _save(g.fig, out_dir / "fig_multiseed_facetgrid")

    _save_combined_robustness(project, out_dir, s7)


def supp_qc_and_scalability(project: Path, out_dir: Path) -> None:
    s4 = _read(project, f"{MANUSCRIPT_REL}/S4_lcad_qc_ablation.csv")
    if s4 is not None:
        _setup_theme()
        s4 = s4.copy()
        mode_labels = {
            "pseudo_all_no_qc": "No QC reference",
            "pseudo_qc_pass_only": "QC pass only",
            "pseudo_qc_score_only": "QC score only",
            "pseudo_confidence_only": "Confidence only",
            "pseudo_qc_confidence_weighted": "QC confidence weighted",
        }
        s4["mode"] = s4["experiment_id"].map(mode_labels).fillna(s4["experiment_id"].str.replace("pseudo_", "").str.replace("_", " "))
        metric_cols = [c for c in ["auc", "f1", "sensitivity", "specificity", "label_consistency"] if c in s4.columns]
        ref = s4[s4["experiment_id"].eq("pseudo_all_no_qc")]
        ref = ref.iloc[0] if len(ref) else s4.iloc[0]
        metric_style = {
            "auc": ("AUROC", "o", C0),
            "f1": ("F1", "s", C4),
            "sensitivity": ("Sensitivity", "^", C2),
            "specificity": ("Specificity", "D", C1),
            "label_consistency": ("Label consistency", "v", C6),
        }
        rows = []
        for metric in metric_cols:
            label, marker, color = metric_style[metric]
            values = s4[metric].astype(float)
            deltas = values - float(ref[metric])
            rows.append(
                {
                    "metric": metric,
                    "label": label,
                    "marker": marker,
                    "color": color,
                    "values": [float(v) for v in values.tolist()],
                    "value_min": float(values.min()),
                    "value_max": float(values.max()),
                    "value_mean": float(values.mean()),
                    "delta_min": float(deltas.min()),
                    "delta_max": float(deltas.max()),
                    "delta_mean": float(deltas.mean()),
                }
            )
        q = pd.DataFrame(rows)
        q = q.sort_values("value_mean", ascending=True).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8.2, 4.9))
        y = np.arange(len(q))
        for i, row in q.iterrows():
            ax.hlines(i, 0.48, 1.00, color=GRID_LINE, lw=0.9, alpha=0.45, zorder=0)
            values = row["values"]
            offsets = np.linspace(-0.105, 0.105, num=len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(
                values,
                np.full(len(values), i) + offsets,
                s=46,
                marker="o",
                facecolor=row["color"],
                edgecolor=TEXT_DARK,
                linewidth=0.55,
                alpha=0.68,
                zorder=2,
            )
            ax.scatter(
                row["value_mean"],
                i,
                s=175,
                marker=row["marker"],
                facecolor=row["color"],
                edgecolor=TEXT_DARK,
                linewidth=1.1,
                alpha=0.96,
                zorder=4,
            )
            ax.text(
                row["value_mean"] + 0.025,
                i,
                f"{row['value_mean']:.3f}",
                ha="left",
                va="center",
                fontsize=10.0,
                fontweight="bold",
                color=TEXT_DARK,
            )
        max_range = float((q["value_max"] - q["value_min"]).max()) if len(q) else 0.0
        summary_text = f"{len(s4)} QC strategies overlap"
        if max_range > 1e-6:
            summary_text += f"; max across-strategy range = {max_range:.3f}"
        ax.text(
            0.03,
            0.96,
            summary_text,
            ha="left",
            va="top",
            fontsize=10.0,
            fontweight="bold",
            color=TEXT_DARK,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C7, lw=0.9, alpha=0.92),
        )
        ax.set_yticks(y)
        ax.set_yticklabels(q["label"], fontsize=11.2)
        ax.set_xlim(0.48, 1.03)
        ax.set_ylim(-0.5, len(q) - 0.05)
        ax.set_xlabel("Metric value across QC strategies")
        ax.set_ylabel("")
        ax.set_title("LCAD QC ablation: collapsed metric profile", fontsize=16, fontweight="bold")
        ax.grid(axis="x", alpha=0.28)
        ax.grid(axis="y", visible=False)
        ax.xaxis.label.set_fontweight("bold")
        fig.tight_layout()
        _save(fig, out_dir / "fig_lcad_qc_ablation_barplot")

    s11 = _read(project, f"{MANUSCRIPT_REL}/S11_scalability_and_runtime.csv")
    if s11 is not None:
        _setup_theme()
        pipe = s11[s11["section"] == "pipeline_scale"] if "section" in s11.columns else s11
        key = pipe[pipe["metric"].isin(["total_cases", "total_images", "real_report_cases", "pseudo_report_cases"])]
        if len(key):
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=key, x="value", y="metric", palette=PALETTE_MAIN[:4], orient="h", ax=ax, edgecolor=EDGE_DARK, linewidth=0.9, alpha=0.86)
            sns.stripplot(data=key, x="value", y="metric", color=C4, marker="D", size=7, jitter=False, ax=ax)
            ax.set_xscale("log")
            ax.set_xlabel("Count (log scale)")
            ax.set_title("Pipeline scale (S11)")
            fig.tight_layout()
            _save(fig, out_dir / "fig_pipeline_runtime_breakdown")
            _save(fig, out_dir / "SupplementaryFigure_S4_scalability")

    centre = _read(project, f"{TABLES_REL}/table_loco_center_characteristics.csv")
    if centre is None:
        centre = _read(project, f"{MANUSCRIPT_REL}/T1b_centre_scale_and_supervision.csv")
    if centre is not None:
        _setup_theme()
        c = centre.copy()
        if "center" in c.columns:
            c = c.rename(columns={"center": "Centre", "cases": "Cases"})
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if "OCT images" in c.columns:
            melt = c.melt(id_vars=["Centre"], value_vars=["OCT images", "Colposcopy images"], var_name="Modality", value_name="Images")
            sns.barplot(data=melt, x="Centre", y="Images", hue="Modality", palette=PALETTE_SUPERVISION, ax=ax, edgecolor=EDGE_DARK, linewidth=0.8, alpha=0.86)
            sns.stripplot(data=melt, x="Centre", y="Images", hue="Modality", dodge=True, marker="s", color=C4, size=5, ax=ax, legend=False)
            ax.set_yscale("log")
            ax.set_title("Imaging volume by centre (log scale)")
        elif "Cases" in c.columns:
            sns.barplot(data=c, x="Centre", y="Cases", hue="Centre", palette=PALETTE_MAIN, ax=ax, edgecolor=EDGE_DARK, linewidth=0.8, alpha=0.86, legend=False)
            sns.stripplot(data=c, x="Centre", y="Cases", color=C4, marker="s", size=7, ax=ax, jitter=False)
            ax.set_title("Cases per centre")
        elif "cases" in c.columns:
            sns.barplot(data=c, x="center", y="cases", hue="center", palette=PALETTE_MAIN, ax=ax, edgecolor=EDGE_DARK, linewidth=0.8, alpha=0.86, legend=False)
            sns.stripplot(data=c, x="center", y="cases", color=C4, marker="s", size=7, ax=ax, jitter=False)
            ax.set_title("Cases per centre")
        ax.tick_params(axis="x", rotation=18)
        fig.tight_layout()
        _save(fig, out_dir / "fig_centerwise_data_scale")


def supp_perturbation_extended(project: Path, out_dir: Path) -> None:
    pert = _read(project, f"{MANUSCRIPT_REL}/S6b_modality_perturbation_extended.csv")
    if pert is None:
        pert = _read(project, f"{TABLES_REL}/table_modality_perturbation_extended.csv")
    if pert is None:
        return
    _setup_theme()
    sim_cols = [c for c in pert.columns if "similarity" in c.lower() or c.endswith("_to_normal")]
    if not sim_cols:
        return
    sub = pert.head(20)
    melt = sub.melt(id_vars=["condition"] if "condition" in sub.columns else [], value_vars=sim_cols[:6], var_name="metric", value_name="value")
    if "condition" not in melt.columns:
        return
    piv = melt.pivot_table(index="condition", columns="metric", values="value", aggfunc="mean")
    cg = sns.clustermap(piv.fillna(0), cmap=_cmap_sequential(), figsize=(10, 8), linewidths=0.3, annot=True, fmt=".2f", dendrogram_ratio=0.12)
    cg.fig.suptitle("Extended perturbation: clustered similarity", y=1.02)
    _save(cg.fig, out_dir / "fig_perturbation_section_dependency_heatmap")
    _save(cg.fig, out_dir / "fig_perturbation_clustermap")


def supp_pairwise_tests(project: Path, out_dir: Path) -> None:
    pw = _read(project, f"{MANUSCRIPT_REL}/T2_pairwise_statistical_tests.csv")
    if pw is None:
        return
    _setup_theme()
    pw = pw.copy()
    pw["neg_log_p"] = -np.log10(pw["bootstrap_p_auc"].clip(1e-6, 1))
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.scatterplot(data=pw, x="delta_auc", y="neg_log_p", hue="comparator", s=120, palette=PALETTE_MAIN, ax=ax, edgecolor=C7)
    ax.axvline(0, color=C6, ls="--")
    ax.set_xlabel("ΔAUROC (comparator − Full LCAD-RASA)")
    ax.set_ylabel("−log10(bootstrap p)")
    ax.set_title("Paired comparisons vs full model")
    ax.legend(loc="upper left", fontsize=7, title="")
    fig.tight_layout()
    _save(fig, out_dir / "fig_pairwise_comparison_scatter")


MAIN_FIGURE_ALIASES = {
    "Figure1_study_design": "Figure1_pipeline_schematic",
    "Figure2_centre_supervision": "Figure2_centre_supervision_catplot",
    "Figure3_perturbation": "Figure3_modality_perturbation_heatmap",
    "Figure4_loco_strict": "Figure4_loco_forest_catplot",
}


def _sync_main_figures(jbd_final: Path, main_dir: Path) -> None:
    """Copy canonical main-text figure names into figures/main/."""
    main_dir.mkdir(parents=True, exist_ok=True)
    for alias, stem in MAIN_FIGURE_ALIASES.items():
        for ext in (".png", ".pdf"):
            src = jbd_final / f"{stem}{ext}"
            if src.is_file():
                shutil.copy2(src, main_dir / f"{alias}{ext}")


def write_figure_index(out_dir: Path, entries: list[tuple[str, str, str]]) -> None:
    palette_line = ", ".join(JBD_PALETTE_HEX)
    lines = [
        "# JBD Figure Index (JBD palette)\n",
        f"Regenerated: {datetime.now(timezone.utc).isoformat()}\n",
        f"Palette: {', '.join(JBD_PALETTE_HEX)}\n",
        "Font: Arial with bold figure titles and axis labels\n\n",
    ]
    for stem, plot_type, desc in entries:
        lines.append(f"## {stem}\n- **Plot type**: {plot_type}\n- {desc}\n- Files: `{stem}.png`, `{stem}.pdf`\n\n")
    (out_dir / "JBD_FINAL_FIGURE_INDEX.md").write_text("".join(lines), encoding="utf-8")


def generate_all_seaborn_figures(project: Path) -> list[str]:
    """Regenerate jbd_final + publishable/figures with varied Seaborn plot types."""
    _setup_theme()
    jbd_final = project / "outputs/publishable/figures/jbd_final"
    pub_fig = project / "outputs/publishable/figures"
    jbd_final.mkdir(parents=True, exist_ok=True)
    pub_fig.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, str, str]] = []

    fig01_pipeline_schematic(jbd_final)
    entries.append(("Figure1_pipeline_schematic", "flow schematic", "Five-stage pipeline"))

    fig02_centre_supervision(project, jbd_final)
    entries.append(("Figure2_centre_supervision_catplot", "catplot + lineplot", "Centre supervision"))
    entries.append(("Figure2_centre_supervision_fraction_line", "lineplot", "Supervision fractions"))

    fig03_perturbation(project, jbd_final)
    entries.append(("Figure3_modality_perturbation_heatmap", "heatmap", "Section similarity matrix"))
    entries.append(("Figure3_modality_perturbation_lineplot", "lineplot", "Perturbation by section"))
    entries.append(("Figure3_risk_delta_stripplot", "paired displacement plot", "Risk shift under perturbation"))

    fig_main_model_comparison(project, jbd_final)
    entries.append(("Figure_main_AUC_pointplot", "pointplot + errorbar", "Main AUROC with CI"))
    entries.append(("Figure_main_metrics_heatmap", "heatmap", "Multi-metric profile"))
    entries.append(("Figure_main_auc_f1_scatter", "scatterplot", "AUC–F1 trade-off"))

    fig_per_case_distributions(project, jbd_final)
    entries.append(("Figure_risk_violin_swarm", "violin + swarm", "Risk by model and label"))
    entries.append(("Figure_full_model_kdeplot", "kdeplot", "Full model risk density"))
    entries.append(("Figure_full_model_jointplot", "jointplot", "Risk vs correctness"))

    # Mirror key figures to pub_fig with legacy names
    for name in (
        "Figure_main_AUC_pointplot",
        "Figure3_modality_perturbation_heatmap",
        "Figure2_centre_supervision_catplot",
    ):
        src = jbd_final / f"{name}.png"
        if src.is_file():
            shutil.copy2(src, pub_fig / f"{name}.png")

    supp_masking(project, jbd_final)
    supp_masking(project, pub_fig)
    entries.append(("SupplementaryFigure_S1_masking_validation", "pointplot", "Masking validation"))

    supp_loco(project, jbd_final)
    supp_loco(project, pub_fig)
    entries.append(("fig_loco_heatmap", "faceted dot plot", "LOCO AUROC by centre and model"))
    entries.append(("Figure4_loco_forest_catplot", "dumbbell forest plot", "LOCO performance range"))

    supp_lambda_sweep(project, pub_fig)
    entries.append(("fig_rasa_lambda_lineplot", "lineplot", "λ_align sweep"))

    supp_modality_ablation(project, pub_fig)
    entries.append(("fig_modality_ablation_stripplot", "strip + point", "Modality ablation"))

    supp_rasa_components(project, pub_fig)
    entries.append(("fig_rasa_component_boxenplot", "ridgeline metric profile", "RASA components"))

    supp_multiseed(project, jbd_final)
    supp_multiseed(project, pub_fig)

    supp_qc_and_scalability(project, pub_fig)
    entries.append(("fig_lcad_qc_ablation_barplot", "zero-centered deviation plot", "QC ablation"))

    supp_perturbation_extended(project, pub_fig)
    entries.append(("fig_perturbation_clustermap", "clustermap", "Extended perturbation"))

    supp_pairwise_tests(project, jbd_final)
    entries.append(("fig_pairwise_comparison_scatter", "scatterplot", "Paired statistical tests"))

    main_dir = project / "outputs/publishable/figures/main"
    _sync_main_figures(jbd_final, main_dir)

    # Legacy composite name used in manuscript and older submission bundles.
    for ext in (".png", ".pdf"):
        src = jbd_final / f"Figure_main_AUC_pointplot{ext}"
        if src.is_file():
            for dst_dir in (jbd_final, pub_fig):
                shutil.copy2(src, dst_dir / f"Figure_main_AUC_comparison{ext}")

    write_figure_index(jbd_final, entries)
    write_figure_index(pub_fig, entries)
    write_figure_index(main_dir, entries)

    return [e[0] for e in entries]
