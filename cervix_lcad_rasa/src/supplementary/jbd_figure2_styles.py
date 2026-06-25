"""Alternative Seaborn-gallery styles for Figure2 centre-supervision plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from src.supplementary.jbd_figure_typography import FONT_ARIAL, FONT_TIMES, apply_arial_to_figure, save_figure_arial
from src.supplementary.jbd_figures_seaborn import (
    C0,
    C1,
    C2,
    C4,
    C6,
    C7,
    EDGE_DARK,
    JBD_PALETTE_HEX,
    PALETTE_MAIN,
    TEXT_DARK,
    _cmap_sequential,
    _read,
)

MANUSCRIPT_REL = "outputs/publishable/tables/manuscript"

CENTRE_LABELS = {
    "enshi": "Enshi",
    "jingzhou": "Jingzhou",
    "shiyan": "Shiyan",
    "wuda": "Wuda",
    "xiangyang": "Xiangyang",
}

# Readable font profile for Figure 2 (larger than default manuscript compact).
FIG2_RC = {
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "legend.title_fontsize": 9.5,
    "font.size": 10,
}

# Display-only nudges when supervision coordinates nearly coincide.
SCATTER_DISPLAY_OFFSETS = {
    "Xiangyang": (42, -32),
    "Shiyan": (-22, 26),
}

PALETTE_REAL_PSEUDO = {"Real reports": C0, "Pseudo-report candidates": C4}

# Style F earth-tone palette (user-specified).
FIG2_EARTH_PALETTE = ["#E1CA9E", "#ADB093", "#998560", "#3E3425"]
FIG2_EARTH_TEXT = "#3E3425"
FIG2_EARTH_REAL_PSEUDO = {
    "Real reports": "#3E3425",
    "Pseudo-report candidates": "#E1CA9E",
}
FIG2_KDE_FILL_ALPHA = 0.52
FIG2_BAR_DARK_ALPHA = 0.62
FIG2_KDE_EDGE = "#FFFFFF"
FIG2_BAR_HEIGHT = 0.52


def _horizontal_bar_supervision_panel(ax: plt.Axes, data: pd.DataFrame, earth_rp: dict[str, str]) -> pd.DataFrame:
    """Seaborn-style overlapping horizontal bars: total cases (light) + real reports (dark)."""
    plot_data = data.sort_values("Cases", ascending=False).reset_index(drop=True)
    centres = plot_data["Centre label"].astype(str).tolist()
    y = np.arange(len(centres))

    ax.barh(
        y,
        plot_data["Cases"],
        height=FIG2_BAR_HEIGHT,
        color=earth_rp["Pseudo-report candidates"],
        alpha=FIG2_KDE_FILL_ALPHA,
        edgecolor=FIG2_EARTH_TEXT,
        linewidth=0.55,
        label="Total cases",
        zorder=1,
    )
    ax.barh(
        y,
        plot_data["Real reports"],
        height=FIG2_BAR_HEIGHT,
        color=earth_rp["Real reports"],
        alpha=FIG2_BAR_DARK_ALPHA,
        edgecolor=FIG2_EARTH_TEXT,
        linewidth=0.55,
        label="Real reports",
        zorder=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(centres, fontfamily=FONT_ARIAL, color=FIG2_EARTH_TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Number of cases", fontfamily=FONT_ARIAL, color=FIG2_EARTH_TEXT)
    ax.set_title("Supervision case counts by centre", fontweight="bold", pad=8, fontfamily=FONT_ARIAL, color=FIG2_EARTH_TEXT)
    ax.tick_params(axis="x", colors=FIG2_EARTH_TEXT)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#ADB093", alpha=0.22, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    leg = ax.legend(frameon=False, loc="lower right")
    if leg.get_title() is not None:
        leg.get_title().set_fontfamily(FONT_ARIAL)
    for text in leg.get_texts():
        text.set_fontfamily(FONT_ARIAL)
        text.set_color(FIG2_EARTH_TEXT)
    return plot_data


def _palette_map_earth(data: pd.DataFrame) -> dict[str, str]:
    """One distinct earth-palette colour per centre."""
    centres = data["Centre label"].astype(str).tolist()
    order = ["#3E3425", "#998560", "#ADB093", "#E1CA9E", "#998560"]
    return {c: order[i % len(order)] for i, c in enumerate(centres)}


def _prepare_figure2_style() -> None:
    from src.supplementary.jbd_figure_typography import setup_arial_rcparams

    setup_arial_rcparams(
        {
            **FIG2_RC,
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_ARIAL, "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_TIMES,
            "mathtext.it": f"{FONT_TIMES}:italic",
            "mathtext.bf": f"{FONT_TIMES}:bold",
        }
    )
    sns.set_theme(
        style="whitegrid",
        font=FONT_ARIAL,
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_ARIAL, "Helvetica", "DejaVu Sans", "sans-serif"],
        },
    )


def _build_case_long(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ci, (_, row) in enumerate(data.iterrows()):
        rng = np.random.default_rng(2026 + ci)
        base = float(ci)
        centre = str(row["Centre label"])
        for _ in range(int(row["Real reports"])):
            rows.append({"x": base + rng.uniform(-0.18, 0.18), "Supervision": "Real reports", "Centre label": centre})
        for _ in range(int(row["Pseudo-report candidates"])):
            rows.append({"x": base + rng.uniform(-0.18, 0.18), "Supervision": "Pseudo-report candidates", "Centre label": centre})
    return pd.DataFrame(rows)


def _coverage_lollipop_panel(
    ax_r: plt.Axes,
    data: pd.DataFrame,
    pal: dict[str, str],
    *,
    text_color: str = TEXT_DARK,
    grid_color: str = C7,
    refline_color: str = C6,
    stem_alpha: float = 0.55,
) -> None:
    y = np.arange(len(data))
    for i, row in data.iterrows():
        c = pal[str(row["Centre label"])]
        yi = int(i)
        cov = float(row["Real-report coverage"])
        ax_r.hlines(yi, 0, cov, color=c, lw=3.0, alpha=stem_alpha, zorder=1)
        ax_r.scatter(cov, yi, s=140, marker="D", c=c, edgecolors=text_color, linewidths=0.85, zorder=3, alpha=0.92)
        ax_r.text(
            1.04,
            yi,
            f"{int(row['Real reports'])} / {int(row['Pseudo-report candidates'])}",
            va="center",
            ha="left",
            fontsize=10,
            fontfamily=FONT_TIMES,
            color=text_color,
        )
    ax_r.axvline(0.5, color=refline_color, ls=(0, (3, 3)), lw=0.9, alpha=0.55)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels(data["Centre label"].astype(str), fontfamily=FONT_ARIAL)
    ax_r.set_xlim(-0.05, 1.22)
    ax_r.set_ylim(len(data) - 0.55, -0.55)
    ax_r.set_xlabel("Real-report coverage", fontfamily=FONT_ARIAL, color=text_color)
    ax_r.set_title("Coverage and report counts", fontweight="bold", pad=8, fontfamily=FONT_ARIAL, color=text_color)
    ax_r.tick_params(colors=text_color)
    ax_r.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x * 100))}%"))
    ax_r.grid(axis="x", color=grid_color, alpha=0.28)
    for sp in ("top", "right", "left"):
        ax_r.spines[sp].set_visible(False)


def load_centre_data(project: Path) -> pd.DataFrame:
    t1b = _read(project, f"{MANUSCRIPT_REL}/T1b_centre_scale_and_supervision.csv")
    if t1b is None:
        raise FileNotFoundError("T1b_centre_scale_and_supervision.csv not found")
    data = t1b.copy()
    data["Centre label"] = data["Centre"].map(CENTRE_LABELS).fillna(data["Centre"].astype(str).str.title())
    data["Real-report coverage"] = data["Real reports"] / data["Cases"].replace(0, np.nan)
    data["Images (k)"] = data["Total images"] / 1000.0
    data["Supervision type"] = np.where(
        data["Real-report coverage"] >= 0.5,
        "Report-rich",
        np.where(data["Real-report coverage"] > 0, "Report-sparse", "Report-absent"),
    )
    order = ["Xiangyang", "Shiyan", "Jingzhou", "Enshi", "Wuda"]
    data["Centre label"] = pd.Categorical(data["Centre label"], categories=order, ordered=True)
    return data.sort_values("Centre label").reset_index(drop=True)


def _palette_map(data: pd.DataFrame) -> dict[str, str]:
    centres = data["Centre label"].astype(str).tolist()
    return {c: JBD_PALETTE_HEX[i % len(JBD_PALETTE_HEX)] for i, c in enumerate(centres)}


def _bubble_sizes(images, lo: float = 55.0, hi: float = 260.0) -> float | np.ndarray:
    """Sqrt-scaled marker area so large image volumes do not occlude smaller centres."""
    x = np.sqrt(np.asarray(images, dtype=float))
    if x.ndim == 0:
        return (lo + hi) / 2
    if x.max() <= x.min():
        return np.full(len(x), (lo + hi) / 2)
    return lo + (hi - lo) * (x - x.min()) / (x.max() - x.min())


def style_a_bubble_legend_lollipop(data: pd.DataFrame, *, title_suffix: str = "", style_tag: str | None = "Style A · bubble + legend + lollipop  |  Seaborn: varying size/hue scatter") -> plt.Figure:
    """Seaborn ref: Scatterplot with varying point sizes and hues + horizontal lollipop."""
    pal = _palette_map(data)
    fig, (ax_s, ax_r) = plt.subplots(1, 2, figsize=(13.8, 6.2), gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.34})
    fig._jbd_font_scale_override = 1.0
    fig._jbd_max_font_size_override = 12.0

    max_cases = int(np.ceil(data["Cases"].max() / 100) * 100)
    for total in (100, 300, 500):
        if total <= max_cases:
            ax_s.plot([0, total], [total, 0], color=C7, lw=0.9, ls=(0, (3, 3)), alpha=0.55, zorder=0)

    plot_rows = []
    for _, row in data.iterrows():
        plot_rows.append(
            {
                "label": str(row["Centre label"]),
                "x0": float(row["Real reports"]),
                "y0": float(row["Pseudo-report candidates"]),
                "images": float(row["Total images"]),
                "color": pal[str(row["Centre label"])],
            }
        )
    plot_rows.sort(key=lambda r: r["images"])

    for item in plot_rows:
        label = item["label"]
        x0, y0 = item["x0"], item["y0"]
        c = item["color"]
        dx, dy = SCATTER_DISPLAY_OFFSETS.get(label, (0, 0))
        x, y = x0 + dx, y0 + dy
        if (dx, dy) != (0, 0):
            ax_s.plot([x0, x], [y0, y], color=c, lw=0.85, alpha=0.5, zorder=2)
        ax_s.scatter(
            x,
            y,
            s=_bubble_sizes(item["images"]),
            c=c,
            edgecolors=TEXT_DARK,
            linewidths=1.0,
            alpha=0.95,
            label=label,
            zorder=3 + item["images"] / 1e6,
        )
    ax_s.set_xlim(-20, max_cases + 25)
    ax_s.set_ylim(-20, max_cases + 55)
    ax_s.set_xlabel("Archived real reports (cases)", fontfamily=FONT_ARIAL)
    ax_s.set_ylabel("Pseudo-report candidates (cases)", fontfamily=FONT_ARIAL)
    ax_s.set_title("Supervision coordinates", fontweight="bold", pad=8, fontfamily=FONT_ARIAL)
    ax_s.grid(True, color=C7, alpha=0.35)
    leg = ax_s.legend(title="Centre (marker size ~ image volume)", frameon=False, loc="upper right", borderaxespad=0.6)
    if leg.get_title() is not None:
        leg.get_title().set_fontfamily(FONT_ARIAL)

    _coverage_lollipop_panel(ax_r, data, pal)

    supt = "Centre-level cohort scale and report-supervision imbalance"
    if title_suffix:
        supt += f" — {title_suffix}"
    fig.suptitle(supt, fontsize=12, fontweight="bold", y=0.98)
    if style_tag:
        fig.text(0.99, 0.02, style_tag, ha="right", fontsize=8.5, color=C6)
    return fig


def style_b_joint_marginals(data: pd.DataFrame) -> plt.Figure:
    """Seaborn ref: Joint kernel density / marginal histograms."""
    pal = _palette_map(data)
    centres = data["Centre label"].astype(str).tolist()
    idx = np.arange(len(centres))
    fig = plt.figure(figsize=(11.8, 8.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.15], height_ratios=[1.15, 4], wspace=0.08, hspace=0.08)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    for i, row in data.iterrows():
        c = pal[str(row["Centre label"])]
        label = str(row["Centre label"])
        x0 = float(row["Real reports"])
        y0 = float(row["Pseudo-report candidates"])
        dx, dy = SCATTER_DISPLAY_OFFSETS.get(label, (0, 0))
        ax_main.scatter(x0 + dx, y0 + dy, s=_bubble_sizes(row["Total images"], 80, 520), c=c, edgecolors=TEXT_DARK, linewidths=0.85, label=label, zorder=3)
        if (dx, dy) != (0, 0):
            ax_main.plot([x0, x0 + dx], [y0, y0 + dy], color=c, lw=0.75, alpha=0.45, zorder=2)
        ax_top.bar(i, row["Real reports"], color=c, edgecolor=TEXT_DARK, linewidth=0.7, alpha=0.88, width=0.72)
        ax_right.barh(i, row["Pseudo-report candidates"], color=c, edgecolor=TEXT_DARK, linewidth=0.7, alpha=0.88, height=0.72)

    ax_main.set_xlabel("Archived real reports (cases)")
    ax_main.set_ylabel("Pseudo-report candidates (cases)")
    ax_main.set_title("Supervision imbalance across centres", fontweight="bold", pad=10)
    ax_main.grid(True, color=C7, alpha=0.32)
    ax_main.legend(title="Centre", frameon=False, loc="upper right")
    ax_top.set_xticks(idx)
    ax_top.set_xticklabels([])
    ax_top.set_ylabel("Real\nreports", fontsize=10)
    ax_right.set_yticks(idx)
    ax_right.set_yticklabels([])
    ax_right.set_xlabel("Pseudo\nreports", fontsize=10)

    fig.suptitle("Centre-level cohort scale and report-supervision imbalance", fontsize=12, fontweight="bold", y=0.98)
    fig.text(0.99, 0.02, "Style B · joint scatter + marginal bars  |  Seaborn: joint/marginal histograms", ha="right", fontsize=8.5, color=C6)
    return fig


def style_c_annotated_heatmap(data: pd.DataFrame) -> plt.Figure:
    """Seaborn ref: Annotated heatmaps."""
    mat = data.set_index("Centre label")[
        ["Real reports", "Pseudo-report candidates", "Cases", "Real-report coverage", "Images (k)"]
    ].astype(float)
    mat = mat.rename(columns={
        "Real reports": "Real reports",
        "Pseudo-report candidates": "Pseudo cases",
        "Cases": "Total cases",
        "Real-report coverage": "Coverage",
        "Images (k)": "Images (k)",
    })
    norm = mat.copy()
    for col in norm.columns:
        lo, hi = norm[col].min(), norm[col].max()
        norm[col] = 0.5 if hi <= lo else (norm[col] - lo) / (hi - lo)
    annot = mat.copy().astype(object)
    for r in mat.index:
        for c in mat.columns:
            v = mat.loc[r, c]
            annot.loc[r, c] = f"{v:.0%}" if c == "Coverage" else (f"{v:.1f}" if c == "Images (k)" else f"{int(v)}")
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    sns.heatmap(norm, annot=annot, fmt="", cmap=_cmap_sequential(), linewidths=1.0, linecolor="white", cbar_kws={"label": "Relative scale (within column)", "shrink": 0.82}, ax=ax)
    ax.set_title("Centre supervision profile (annotated heatmap)", fontweight="bold", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.suptitle("Centre-level cohort scale and report-supervision imbalance", fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.99, -0.02, "Style C · annotated heatmap  |  Seaborn: Annotated heatmaps", ha="right", fontsize=8.5, color=C6)
    return fig


def style_d_facet_dotplot(data: pd.DataFrame) -> plt.Figure:
    """Seaborn ref: Dot plot with several variables."""
    long = data.melt(
        id_vars=["Centre label", "Images (k)"],
        value_vars=["Real reports", "Pseudo-report candidates"],
        var_name="Supervision",
        value_name="Case count",
    )
    long["Supervision"] = long["Supervision"].map({"Real reports": "Archived real reports", "Pseudo-report candidates": "Pseudo-report candidates"})
    pal = _palette_map(data)
    g = sns.FacetGrid(long, col="Supervision", sharex=True, sharey=False, height=4.8, aspect=0.95)
    g.map_dataframe(
        sns.scatterplot,
        x="Case count",
        y="Centre label",
        hue="Centre label",
        palette=pal,
        s=160,
        edgecolor=TEXT_DARK,
        linewidth=0.85,
        legend=False,
    )
    g.set_axis_labels("Number of cases", "")
    g.set_titles("{col_name}", size=11, weight="bold")
    g.fig.suptitle("Centre-level cohort scale and report-supervision imbalance", fontsize=12, fontweight="bold", y=1.03)
    g.fig.text(0.99, 0.01, "Style D · facet dot plot  |  Seaborn: Dot plot with several variables", ha="right", fontsize=8.5, color=C6)
    return g.fig


def style_e_stacked_proportion(data: pd.DataFrame) -> plt.Figure:
    """Seaborn ref: Paired categorical / 100% composition."""
    pal_real, pal_pseudo = C0, C4
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    y = np.arange(len(data))
    real_frac = data["Real reports"] / data["Cases"]
    pseudo_frac = data["Pseudo-report candidates"] / data["Cases"]
    ax.barh(y, real_frac, color=pal_real, edgecolor=TEXT_DARK, linewidth=0.7, height=0.58, label="Real reports", alpha=0.9)
    ax.barh(y, pseudo_frac, left=real_frac, color=pal_pseudo, edgecolor=TEXT_DARK, linewidth=0.7, height=0.58, label="Pseudo-report candidates", alpha=0.88)
    for yi, (_, row) in enumerate(data.iterrows()):
        label = str(row["Centre label"])
        ax.text(1.02, yi, f"{int(row['Real reports'])} / {int(row['Pseudo-report candidates'])}  ({row['Images (k)']:.0f}k img)", va="center", ha="right", fontsize=9.5, color=TEXT_DARK, transform=ax.get_yaxis_transform())
    ax.set_yticks(y)
    ax.set_yticklabels(data["Centre label"].astype(str), fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x * 100)}%"))
    ax.set_xlabel("Supervision composition within centre cohort")
    ax.set_title("Report vs pseudo-report composition by centre", fontweight="bold", pad=10)
    ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis()
    fig.suptitle("Centre-level cohort scale and report-supervision imbalance", fontsize=12, fontweight="bold", y=0.98)
    fig.text(0.99, 0.02, "Style E · 100% stacked composition  |  Seaborn: Paired categorical plots", ha="right", fontsize=8.5, color=C6)
    return fig


def style_f_conditional_kde(
    data: pd.DataFrame,
    *,
    style_tag: str | None = None,
) -> plt.Figure:
    """Seaborn ref: Horizontal bar plots — overlapping total + real-report bars, with coverage panel."""
    pal = _palette_map_earth(data)
    earth_rp = FIG2_EARTH_REAL_PSEUDO
    fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(13.8, 6.0), gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.34})
    fig._jbd_max_font_size_override = 12.0
    fig._jbd_mixed_en_typography = True

    plot_data = _horizontal_bar_supervision_panel(ax_b, data, earth_rp)

    _coverage_lollipop_panel(
        ax_r,
        plot_data,
        pal,
        text_color=FIG2_EARTH_TEXT,
        grid_color="#ADB093",
        refline_color="#998560",
        stem_alpha=0.78,
    )

    fig.suptitle(
        "Centre-level cohort scale and report-supervision imbalance",
        fontsize=12,
        fontweight="bold",
        y=0.98,
        fontfamily=FONT_ARIAL,
        color=FIG2_EARTH_TEXT,
    )
    if style_tag:
        fig.text(0.99, 0.02, style_tag, ha="right", fontsize=8.5, color=C6, fontfamily=FONT_ARIAL)
    return fig


def style_g_polar_facetgrid(
    data: pd.DataFrame,
    *,
    style_tag: str | None = "Style G · polar FacetGrid  |  Seaborn: FacetGrid with custom projection",
) -> plt.Figure:
    """Seaborn ref: FacetGrid with custom projection — one polar panel per centre."""
    centres = data["Centre label"].astype(str).tolist()
    g = sns.FacetGrid(
        data,
        col="Centre label",
        col_order=centres,
        height=3.35,
        aspect=1.0,
        despine=False,
        subplot_kws={"projection": "polar"},
    )

    def _draw_polar_centre(data, color=None, **kwargs) -> None:
        ax = plt.gca()
        row = data.iloc[0]
        real = int(row["Real reports"])
        pseudo = int(row["Pseudo-report candidates"])
        total = max(real + pseudo, 1)
        theta_start = np.pi / 2
        width_real = 2 * np.pi * real / total
        width_pseudo = 2 * np.pi - width_real
        if real > 0:
            ax.bar(
                theta_start,
                1.0,
                width=width_real,
                bottom=0.12,
                color=PALETTE_REAL_PSEUDO["Real reports"],
                edgecolor=TEXT_DARK,
                linewidth=0.85,
                alpha=0.92,
                align="edge",
            )
        if pseudo > 0:
            ax.bar(
                theta_start + width_real,
                1.0,
                width=width_pseudo,
                bottom=0.12,
                color=PALETTE_REAL_PSEUDO["Pseudo-report candidates"],
                edgecolor=TEXT_DARK,
                linewidth=0.85,
                alpha=0.9,
                align="edge",
            )
        ax.grid(False)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.18)
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.spines["polar"].set_visible(False)
        ax.text(
            0,
            0.0,
            f"{int(row['Cases'])} cases\n{row['Images (k)']:.0f}k images",
            ha="center",
            va="center",
            fontsize=9,
            fontfamily=FONT_ARIAL,
            color=TEXT_DARK,
        )
        ax.text(
            0,
            -0.12,
            f"{real} real / {pseudo} pseudo",
            ha="center",
            va="top",
            fontsize=8.5,
            fontfamily=FONT_ARIAL,
            color=TEXT_DARK,
            transform=ax.transAxes,
        )

    g.map_dataframe(_draw_polar_centre)
    g.set_titles("{col_name}", size=10.5, weight="bold")
    for ax in g.axes.flat:
        ax.title.set_fontfamily(FONT_ARIAL)
    g.fig.set_size_inches(14.2, 4.2)
    g.fig.subplots_adjust(top=0.78, wspace=0.28)
    g.fig.suptitle(
        "Centre-level cohort scale and report-supervision imbalance",
        fontsize=12,
        fontweight="bold",
        y=0.98,
        fontfamily=FONT_ARIAL,
    )
    g.fig.text(
        0.5,
        0.04,
        "Polar arc length = supervision composition (dark brown = real reports, blue = pseudo-report candidates)",
        ha="center",
        fontsize=9.5,
        fontfamily=FONT_ARIAL,
        color=TEXT_DARK,
    )
    if style_tag:
        g.fig.text(0.99, 0.01, style_tag, ha="right", fontsize=8.5, color=C6, fontfamily=FONT_ARIAL)
    return g.fig


STYLE_BUILDERS = {
    "A_bubble_legend_lollipop": style_a_bubble_legend_lollipop,
    "B_joint_marginals": style_b_joint_marginals,
    "C_annotated_heatmap": style_c_annotated_heatmap,
    "D_facet_dotplot": style_d_facet_dotplot,
    "E_stacked_proportion": style_e_stacked_proportion,
    "F_conditional_kde": style_f_conditional_kde,
    "G_polar_facetgrid": style_g_polar_facetgrid,
}


def render_figure2_style(name: str, data: pd.DataFrame, *, for_publication: bool = False) -> plt.Figure:
    if name not in STYLE_BUILDERS:
        raise KeyError(name)
    if for_publication:
        if name == "A_bubble_legend_lollipop":
            return STYLE_BUILDERS[name](data, style_tag=None)
        if name == "F_conditional_kde":
            return STYLE_BUILDERS[name](data, style_tag=None)
        if name == "G_polar_facetgrid":
            return STYLE_BUILDERS[name](data, style_tag=None)
    return STYLE_BUILDERS[name](data)


def save_all_style_comparisons(project: Path, out_dir: Path) -> list[Path]:
    _prepare_figure2_style()
    data = load_centre_data(project)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for key, builder in STYLE_BUILDERS.items():
        fig = builder(data)
        path = out_dir / f"Figure2_style_{key}"
        save_figure_arial(fig, path)
        written.append(path.with_suffix(".png"))
    return written


def render_figure2_final(project: Path, out_dirs: list[Path], *, style: str = "F_conditional_kde") -> None:
    _prepare_figure2_style()
    data = load_centre_data(project)
    fig = render_figure2_style(style, data, for_publication=True)
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        save_figure_arial(fig, d / "Figure2_centre_supervision_catplot")
