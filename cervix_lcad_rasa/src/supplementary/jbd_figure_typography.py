"""Shared Arial typography and layout helpers for JBD manuscript figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

FONT_ARIAL = "Arial"
FONT_TIMES = "Times New Roman"
FONT_SCALE = 1.0
MIN_FONT_SIZE_PT = 7.0
MAX_FONT_SIZE_PT = 11.0


def setup_arial_rcparams(extra: dict | None = None) -> None:
    """Configure matplotlib for compact Arial sans-serif text."""
    rc = {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_ARIAL, "DejaVu Sans", "Helvetica", "sans-serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_ARIAL,
        "mathtext.it": f"{FONT_ARIAL}:italic",
        "mathtext.bf": f"{FONT_ARIAL}:bold",
        "axes.unicode_minus": False,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "font.size": 8,
    }
    if extra:
        rc.update(extra)
    plt.rcParams.update(rc)


def apply_arial_to_figure(fig: plt.Figure) -> None:
    """Force Arial and clamp font sizes to avoid oversized overlapping text."""
    already_boosted = bool(getattr(fig, "_jbd_font_size_boosted", False))
    font_scale = float(getattr(fig, "_jbd_font_scale_override", FONT_SCALE))
    min_font_size = float(getattr(fig, "_jbd_min_font_size_override", MIN_FONT_SIZE_PT))
    max_font_size = float(getattr(fig, "_jbd_max_font_size_override", MAX_FONT_SIZE_PT))
    for artist in fig.findobj(match=lambda x: hasattr(x, "get_text") and x.get_text()):
        try:
            artist.set_fontfamily(FONT_ARIAL)
        except Exception:
            pass
        if not already_boosted and hasattr(artist, "get_fontsize") and hasattr(artist, "set_fontsize"):
            try:
                size = float(artist.get_fontsize())
                artist.set_fontsize(min(max_font_size, max(min_font_size, size * font_scale)))
            except Exception:
                pass
    for ax in fig.get_axes():
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(FONT_ARIAL)
        if ax.get_xlabel():
            ax.xaxis.label.set_fontfamily(FONT_ARIAL)
        if ax.get_ylabel():
            ax.yaxis.label.set_fontfamily(FONT_ARIAL)
        if ax.get_title():
            ax.title.set_fontfamily(FONT_ARIAL)
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontfamily(FONT_ARIAL)
            if leg.get_title() is not None:
                leg.get_title().set_fontfamily(FONT_ARIAL)
    setattr(fig, "_jbd_font_size_boosted", True)


def _text_uses_times(text: str) -> bool:
    """Return True when tick/annotation text is numeric and should use Times New Roman."""
    t = text.strip()
    if not t:
        return False
    if t.endswith("%"):
        core = t[:-1].strip()
        try:
            float(core)
            return True
        except ValueError:
            pass
    if "/" in t and any(ch.isdigit() for ch in t):
        return True
    try:
        float(t)
        return True
    except ValueError:
        return False


def apply_mixed_en_typography(fig: plt.Figure) -> None:
    """English labels in Arial; numeric tick/annotation text in Times New Roman."""
    for ax in fig.get_axes():
        if ax.get_xlabel():
            ax.xaxis.label.set_fontfamily(FONT_ARIAL)
        if ax.get_ylabel():
            ax.yaxis.label.set_fontfamily(FONT_ARIAL)
        if ax.get_title():
            ax.title.set_fontfamily(FONT_ARIAL)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(FONT_TIMES if _text_uses_times(label.get_text()) else FONT_ARIAL)
        leg = ax.get_legend()
        if leg is not None:
            if leg.get_title() is not None:
                leg.get_title().set_fontfamily(FONT_ARIAL)
            for text in leg.get_texts():
                text.set_fontfamily(FONT_ARIAL)
    if fig._suptitle is not None:
        fig._suptitle.set_fontfamily(FONT_ARIAL)
    for artist in fig.findobj(match=lambda x: hasattr(x, "get_text") and x.get_text()):
        text = artist.get_text()
        try:
            if _text_uses_times(text):
                artist.set_fontfamily(FONT_TIMES)
            elif any(ch.isalpha() for ch in text):
                artist.set_fontfamily(FONT_ARIAL)
        except Exception:
            pass


def finalize_figure(fig: plt.Figure, *, rect: list[float] | None = None) -> None:
    """Apply Arial and a tight but padded layout to reduce label overlap."""
    apply_arial_to_figure(fig)
    if getattr(fig, "_jbd_mixed_en_typography", False):
        apply_mixed_en_typography(fig)
    try:
        fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.05, wspace=0.08, hspace=0.10)
    except Exception:
        pass
    try:
        if rect is None:
            fig.tight_layout(pad=0.6)
        else:
            fig.tight_layout(pad=0.6, rect=rect)
    except Exception:
        pass


def panel_label(ax: plt.Axes, label: str, *, fontsize: float = 10.0) -> None:
    ax.text(
        -0.03,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        fontfamily=FONT_ARIAL,
        va="top",
        ha="left",
    )


def save_figure_arial(fig: plt.Figure, path, dpi: int = 300) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    finalize_figure(fig)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    try:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.06)
    except Exception:
        pass
    plt.close(fig)
