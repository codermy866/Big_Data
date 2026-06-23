"""Novel Seaborn-gallery-inspired styles for individual JBD manuscript figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.supplementary.jbd_figure_typography import FONT_ARIAL, apply_arial_to_figure, setup_arial_rcparams
from src.supplementary.jbd_figure_stats import add_significance_bracket, annotate_p_at_xy, format_pvalue
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
    GRID_LINE,
    JBD_PALETTE_HEX,
    PALETTE_MAIN,
    TEXT_DARK,
    _cmap_diverging,
    _cmap_sequential,
)

PALETTE = sns.color_palette(JBD_PALETTE_HEX)


def setup_novel_theme() -> None:
    """Arial + compact typography, Seaborn whitegrid."""
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
    sns.set_theme(style="whitegrid", context="paper", font=FONT_ARIAL, font_scale=1.0, palette=PALETTE)


def polish_ax(ax: plt.Axes, *, legend: bool = False) -> None:
    ax.title.set_fontweight("bold")
    sns.despine(ax=ax)
    if legend and ax.get_legend() is not None:
        ax.legend(frameon=False)


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 300) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    apply_arial_to_figure(fig)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.06)
    plt.close(fig)


def save_many(fig: plt.Figure, stems: list[Path]) -> None:
    apply_arial_to_figure(fig)
    for stem in stems:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.06)
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.06)
    plt.close(fig)


def horizontal_box_strip(
    data: pd.DataFrame,
    *,
    y: str,
    x: str,
    hue: str | None = None,
    order: list | None = None,
    palette: list | None = None,
    title: str,
    xlabel: str,
    figsize: tuple[float, float] = (8.8, 5.6),
    log_x: bool = False,
) -> plt.Figure:
    """Seaborn gallery: horizontal boxplot with strip observations."""
    setup_novel_theme()
    fig, ax = plt.subplots(figsize=figsize)
    kw = {"data": data, "y": y, "x": x, "palette": palette or PALETTE, "linewidth": 0.9, "ax": ax}
    if hue:
        kw["hue"] = hue
        kw["dodge"] = True
    if order:
        kw["order"] = order
    sns.boxplot(**kw, fliersize=0, width=0.55, saturation=0.88)
    strip_kw = {"data": data, "y": y, "x": x, "size": 5.5, "alpha": 0.72, "jitter": 0.18, "ax": ax, "color": TEXT_DARK}
    if hue:
        strip_kw["hue"] = hue
        strip_kw["dodge"] = True
        strip_kw["palette"] = palette or PALETTE
        strip_kw["legend"] = False
    sns.stripplot(**strip_kw)
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    polish_ax(ax, legend=bool(hue))
    fig.tight_layout()
    return fig


def clustermap_figure(
    matrix: pd.DataFrame,
    *,
    title: str,
    cbar_label: str = "Value",
    cmap=None,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (9.2, 7.2),
    row_cluster: bool = True,
    col_cluster: bool = True,
) -> plt.Figure:
    """Seaborn gallery: annotated clustermap with dendrograms."""
    setup_novel_theme()
    m = matrix.apply(pd.to_numeric, errors="coerce")
    cmap = cmap or _cmap_sequential()
    if vmin is None:
        vmin = float(np.nanmin(m.to_numpy()))
    if vmax is None:
        vmax = float(np.nanmax(m.to_numpy()))
    cg = sns.clustermap(
        m,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.6,
        linecolor="white",
        figsize=figsize,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cbar_kws={"label": cbar_label, "shrink": 0.72},
        dendrogram_ratio=0.12,
        colors_ratio=0.03,
        yticklabels=True,
        xticklabels=True,
    )
    cg.ax_heatmap.set_xlabel("")
    cg.ax_heatmap.set_ylabel("")
    cg.fig.suptitle(title, fontsize=10, fontweight="bold", y=1.02)
    cg.fig.subplots_adjust(top=0.92)
    apply_arial_to_figure(cg.fig)
    return cg.fig


def joint_scatter_marginals(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None = None,
    palette: dict | None = None,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (7.4, 6.8),
) -> plt.Figure:
    """Seaborn gallery: JointGrid scatter with marginal histograms."""
    setup_novel_theme()
    g = sns.JointGrid(data=data, x=x, y=y, height=figsize[1], ratio=4, space=0.04, marginal_ticks=True)
    if hue:
        for level, sub in data.groupby(hue):
            color = (palette or {}).get(level, PALETTE_MAIN[0])
            g.ax_joint.scatter(sub[x], sub[y], s=95, color=color, edgecolor=TEXT_DARK, linewidth=0.7, alpha=0.9, label=level)
        g.ax_joint.legend(frameon=False, loc="lower right")
    else:
        g.ax_joint.scatter(data[x], data[y], s=95, color=C0, edgecolor=TEXT_DARK, linewidth=0.7, alpha=0.9)
    sns.histplot(data=data, x=x, ax=g.ax_marg_x, color=C0, bins=12, kde=True, edgecolor="white", linewidth=0.4)
    sns.histplot(data=data, y=y, ax=g.ax_marg_y, color=C4, bins=12, kde=True, edgecolor="white", linewidth=0.4)
    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)
    g.ax_joint.set_title(title, pad=14)
    sns.despine(fig=g.fig)
    apply_arial_to_figure(g.fig)
    return g.fig


def scarcity_heatmap(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str,
    hue_order: list | None = None,
    palette: dict | None = None,
    title: str,
    xlabel: str,
    ylabel: str,
    err_col: str | None = None,
    cbar_label: str | None = None,
    figsize: tuple[float, float] = (8.8, 4.6),
    p_annotation: str | None = None,
) -> plt.Figure:
    """Tile heatmap for supervision-scarcity sweeps (no line chart)."""
    setup_novel_theme()
    plot = data.copy()
    frac_labels = {0.1: "10%", 0.25: "25%", 0.5: "50%", 1.0: "100%"}
    plot["fraction_label"] = plot[x].map(frac_labels).fillna(plot[x].astype(str))
    frac_order = [f for f in ["10%", "25%", "50%", "100%"] if f in plot["fraction_label"].unique()]
    order = hue_order or plot[hue].drop_duplicates().tolist()
    pivot = plot.pivot(index=hue, columns="fraction_label", values=y).reindex(order)
    pivot = pivot[[c for c in frac_order if c in pivot.columns]]
    err_pivot = None
    if err_col and err_col in plot.columns:
        err_pivot = plot.pivot(index=hue, columns="fraction_label", values=err_col).reindex(order)
        err_pivot = err_pivot[[c for c in frac_order if c in err_pivot.columns]]
    annot = pivot.copy().astype(object)
    for r in pivot.index:
        for c in pivot.columns:
            val = pivot.loc[r, c]
            if pd.isna(val):
                annot.loc[r, c] = ""
            elif err_pivot is not None and c in err_pivot.columns and pd.notna(err_pivot.loc[r, c]):
                err = float(err_pivot.loc[r, c])
                annot.loc[r, c] = f"{float(val):.3f}\n±{err:.3f}"
            else:
                annot.loc[r, c] = f"{float(val):.3f}"
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap=_cmap_sequential(),
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": cbar_label or ylabel, "shrink": 0.78},
        ax=ax,
        vmin=float(pivot.min().min()),
        vmax=float(pivot.max().max()),
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if p_annotation:
        ax.text(
            0.99,
            1.04,
            p_annotation,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            fontfamily=FONT_ARIAL,
            color=TEXT_DARK,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C7, alpha=0.92),
        )
    fig.tight_layout()
    return fig


def lambda_sweep_dumbbell(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (8.4, 5.2),
) -> plt.Figure:
    """Alignment-weight sensitivity as horizontal dumbbells (no trend line)."""
    setup_novel_theme()
    plot = data.sort_values(x).reset_index(drop=True)
    baseline = float(plot.loc[plot[x].eq(0), y].iloc[0]) if (plot[x].eq(0)).any() else float(plot[y].iloc[0])
    best_idx = int(plot[y].idxmax())
    labels = [f"{v:.2f}" if float(v) > 0 else "0" for v in plot[x]]
    y_pos = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(baseline, color=C6, ls=(0, (2, 2)), lw=1.1, alpha=0.85, zorder=0)
    for i, row in plot.iterrows():
        val = float(row[y])
        yi = int(i)
        color = C2 if yi == best_idx else (C4 if val < baseline else C0)
        ax.hlines(yi, min(baseline, val), max(baseline, val), color=color, linewidth=3.0, alpha=0.5, zorder=1)
        ax.scatter(val, yi, s=110, color=color, edgecolor=TEXT_DARK, linewidth=0.85, zorder=3)
        ax.text(val + 0.0012, yi, f"{val:.3f}", va="center", ha="left", fontsize=7.5, color=TEXT_DARK)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.text(
        0.98,
        0.04,
        f"Baseline (λ=0): {baseline:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        fontfamily=FONT_ARIAL,
        color=TEXT_DARK,
    )
    polish_ax(ax)
    fig.tight_layout()
    return fig


def horizontal_lollipop_pvals(
    data: pd.DataFrame,
    *,
    y: str,
    x: str,
    p_col: str | None = None,
    pvals: dict[str, float] | None = None,
    xerr_low_col: str | None = None,
    xerr_high_col: str | None = None,
    palette: list | None = None,
    title: str,
    xlabel: str,
    figsize: tuple[float, float] = (9.2, 5.4),
    refline: float | None = 0.0,
) -> plt.Figure:
    """Forest / lollipop with paired-bootstrap p-value column."""
    setup_novel_theme()
    plot = data.sort_values(x, ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(plot))
    colors = (palette or PALETTE)[: len(plot)]
    if refline is not None:
        ax.axvline(refline, color=C6, linestyle=":", linewidth=1.2, alpha=0.75, zorder=0)
    ci_max = float(plot[x].max())
    if xerr_high_col and xerr_high_col in plot.columns:
        ci_max = max(ci_max, float(plot[xerr_high_col].max()))
    label_x = ci_max + abs(ci_max) * 0.06 + 0.012
    for i, row in plot.iterrows():
        val = float(row[x])
        yi = int(i)
        color = colors[yi % len(colors)]
        x0 = refline if refline is not None else 0
        ax.hlines(yi, x0, val, color=color, linewidth=3.0, alpha=0.5, zorder=1)
        if xerr_low_col and xerr_high_col and xerr_low_col in plot.columns:
            lo = float(row[xerr_low_col])
            hi = float(row[xerr_high_col])
            ax.errorbar(val, yi, xerr=[[val - lo], [hi - val]], fmt="o", color=color, ecolor=TEXT_DARK, elinewidth=1.4, capsize=4, markersize=8, markeredgecolor=TEXT_DARK, markeredgewidth=0.8, zorder=3)
        else:
            ax.scatter(val, yi, s=100, color=color, edgecolor=TEXT_DARK, linewidth=0.85, zorder=3)
        p = None
        if p_col and p_col in plot.columns:
            p = float(row[p_col])
        elif pvals is not None:
            p = pvals.get(str(row[y]))
        if p is not None:
            annotate_p_at_xy(ax, label_x, yi, p, ha="left", va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot[y])
    ax.set_xlim(left=(refline - 0.05) if refline is not None else None, right=label_x + 0.08)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    polish_ax(ax)
    fig.tight_layout()
    return fig


def proportion_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    cbar_label: str = "Fraction",
    figsize: tuple[float, float] = (9.0, 4.8),
) -> plt.Figure:
    """Stacked-proportion matrix as annotated heatmap (replaces stacked bars)."""
    setup_novel_theme()
    m = matrix.apply(pd.to_numeric, errors="coerce").fillna(0)
    props = m.div(m.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    annot = props.copy()
    for r in annot.index:
        for c in annot.columns:
            v = float(annot.loc[r, c])
            annot.loc[r, c] = f"{v:.0%}" if v >= 0.06 else ""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        props,
        annot=annot,
        fmt="",
        cmap=_cmap_sequential(),
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": cbar_label, "shrink": 0.72},
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def grouped_violin_strip(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None = None,
    order: list | None = None,
    hue_order: list | None = None,
    palette: list | None = None,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (9.2, 5.0),
) -> plt.Figure:
    """Violin + strip for centre-wise or grouped comparisons (no pointplot)."""
    setup_novel_theme()
    fig, ax = plt.subplots(figsize=figsize)
    kw = {"data": data, "x": x, "y": y, "palette": palette or PALETTE, "linewidth": 0.8, "ax": ax, "cut": 0, "inner": "quart"}
    if hue:
        kw["hue"] = hue
        kw["hue_order"] = hue_order
        kw["split"] = False
        kw["dodge"] = True
    if order:
        kw["order"] = order
    sns.violinplot(**kw, saturation=0.88)
    strip_kw = {"data": data, "x": x, "y": y, "size": 5, "alpha": 0.65, "jitter": 0.15, "ax": ax, "color": TEXT_DARK}
    if hue:
        strip_kw["hue"] = hue
        strip_kw["hue_order"] = hue_order
        strip_kw["dodge"] = True
        strip_kw["palette"] = palette or PALETTE
        strip_kw["legend"] = False
    if order:
        strip_kw["order"] = order
    sns.stripplot(**strip_kw)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    polish_ax(ax, legend=bool(hue))
    fig.tight_layout()
    return fig


def facet_line_band(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str,
    hue_order: list | None = None,
    palette: dict | None = None,
    title: str,
    xlabel: str,
    ylabel: str,
    err_col: str | None = None,
    figsize: tuple[float, float] = (9.4, 5.8),
) -> plt.Figure:
    """Faceted-style line plot with error bars (scarcity / sweep curves)."""
    setup_novel_theme()
    fig, ax = plt.subplots(figsize=figsize)
    order = hue_order or data[hue].drop_duplicates().tolist()
    pal = palette or {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(order)}
    markers = ["o", "s", "D", "^", "v", "P"]
    for i, key in enumerate(order):
        sub = data[data[hue].eq(key)].sort_values(x)
        color = pal.get(key, PALETTE[i % len(PALETTE)])
        marker = markers[i % len(markers)]
        ax.plot(sub[x], sub[y], color=color, linewidth=2.2, alpha=0.85, zorder=1)
        if err_col and err_col in sub.columns:
            ax.errorbar(
                sub[x],
                sub[y],
                yerr=sub[err_col].fillna(0),
                fmt="none",
                ecolor=color,
                elinewidth=1.6,
                capsize=5,
                alpha=0.95,
                zorder=2,
            )
        ax.scatter(sub[x], sub[y], s=120, marker=marker, color=color, edgecolor=TEXT_DARK, linewidth=0.9, label=key, zorder=4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="best")
    polish_ax(ax)
    fig.tight_layout()
    return fig


def horizontal_lollipop(
    data: pd.DataFrame,
    *,
    y: str,
    x: str,
    xerr_low_col: str | None = None,
    xerr_high_col: str | None = None,
    palette: list | None = None,
    title: str,
    xlabel: str,
    figsize: tuple[float, float] = (8.6, 5.4),
    refline: float | None = 0.5,
) -> plt.Figure:
    """Forest / lollipop plot for model comparisons."""
    setup_novel_theme()
    plot = data.sort_values(x, ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(plot))
    colors = (palette or PALETTE)[: len(plot)]
    if refline is not None:
        ax.axvline(refline, color=C6, linestyle=":", linewidth=1.2, alpha=0.75, zorder=0)
    for i, row in plot.iterrows():
        val = float(row[x])
        yi = int(i)
        ax.hlines(yi, refline if refline is not None else 0, val, color=colors[yi % len(colors)], linewidth=3.2, alpha=0.55, zorder=1)
        if xerr_low_col and xerr_high_col and xerr_low_col in plot.columns and xerr_high_col in plot.columns:
            lo = float(row[xerr_low_col])
            hi = float(row[xerr_high_col])
            ax.errorbar(val, yi, xerr=[[val - lo], [hi - val]], fmt="o", color=colors[yi % len(colors)], ecolor=TEXT_DARK, elinewidth=1.5, capsize=5, markersize=9, markeredgecolor=TEXT_DARK, markeredgewidth=0.8, zorder=3)
        else:
            ax.scatter(val, yi, s=110, color=colors[yi % len(colors)], edgecolor=TEXT_DARK, linewidth=0.85, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot[y])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    polish_ax(ax)
    fig.tight_layout()
    return fig


def diagonal_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    cbar_label: str = "Score",
    cmap=None,
    figsize: tuple[float, float] = (8.4, 6.8),
) -> plt.Figure:
    """Masked lower-triangle correlation-style heatmap."""
    setup_novel_theme()
    m = matrix.apply(pd.to_numeric, errors="coerce")
    mask = np.triu(np.ones_like(m, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        m,
        mask=mask,
        cmap=cmap or _cmap_diverging(),
        annot=True,
        fmt=".2f",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": cbar_label, "shrink": 0.78},
        ax=ax,
        vmin=float(np.nanmin(m.to_numpy())),
        vmax=float(np.nanmax(m.to_numpy())),
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig
