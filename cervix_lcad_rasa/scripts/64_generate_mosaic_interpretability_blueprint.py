#!/usr/bin/env python3
"""Generate clear MOSAIC scientific-question and interpretability blueprint figures."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

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

MANUSCRIPT = ROOT / "outputs/publishable/tables/manuscript"
MANIFEST = ROOT / "outputs/manifests/full_manifest.csv"
SCORES = ROOT / "outputs/publishable/kra_semantic_fusion_analysis/kra_semantic_fusion_val_test_scores.csv"
PERT_REPORTS = ROOT / "outputs/publishable/generated_reports/perturbation"
MODALITY_EVIDENCE = ROOT / "outputs/publishable/modality_evidence"
OUT_JBD = ROOT / "outputs/publishable/figures/jbd_final/Figure_mosaic_interpretability_blueprint"
FINAL = PROJECT / "final_Fig/Figure_mosaic_interpretability_blueprint"

TEXT = "#17212B"
BLUE = "#254B6D"
RUST = "#C65A46"
MID = "#557A95"
TEAL = "#436E6F"
GOLD = "#D2AE76"
PURPLE = "#6F5B85"
REF = "#95A1B2"
GRID = "#D9E1EA"
PANEL = "#F7F9FC"
PALE_BLUE = "#E3EEF5"
PALE_RUST = "#F3DDD6"
PALE_GOLD = "#F5E7C6"
PALE_TEAL = "#DDEBE8"
PALE_PURPLE = "#E8E1EE"
PALE_GRAY = "#EEF2F6"
GOOD = "#DDEBE8"
RISK = "#F4D9D2"

ACTUAL_CASE_SPECS = [
    {"case_id": "M20203_2023_P0000534", "perturbation": "mask_oct", "case_tag": "Actual test case 1"},
    {"case_id": "M22102_2023_P0000048", "perturbation": "mask_colposcopy", "case_tag": "Actual test case 2"},
]


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


def load_counts() -> dict[str, int]:
    path = MANUSCRIPT / "T1a_cohort_summary.csv"
    if not path.is_file():
        return {"cases": 1897, "centres": 5, "images": 137591, "real": 744, "pseudo": 1153, "test": 288}
    df = pd.read_csv(path)
    vals = dict(zip(df["Metric"], df["Value"]))
    return {
        "cases": int(vals.get("Total cases", 1897)),
        "centres": int(vals.get("Centres", 5)),
        "images": int(vals.get("Evaluable images (pipeline)", 137591)),
        "real": int(vals.get("Real reports", 744)),
        "pseudo": int(vals.get("Pseudo-report candidates", 1153)),
        "test": int(vals.get("Test cases", 288)),
    }


def apply_style(fig: plt.Figure) -> None:
    fig._jbd_min_font_size_override = 12.6
    fig._jbd_max_font_size_override = 23.5
    apply_arial_to_figure(fig)
    apply_mixed_en_typography(fig)


def save_fig(fig: plt.Figure, stem: str, aliases=None) -> None:
    apply_style(fig)
    stems = [stem] + list(aliases or [])
    for out_dir in [OUT_JBD, FINAL]:
        out_dir.mkdir(parents=True, exist_ok=True)
        for out_stem in stems:
            base = out_dir / out_stem
            fig.savefig(base.with_suffix(".png"), dpi=330, bbox_inches="tight", facecolor="white", pad_inches=0.08)
            fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def canvas(fig: plt.Figure) -> plt.Axes:
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def add_box(
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
    fontsize: float = 13.0,
    weight: str = "normal",
    color: str = TEXT,
    ha: str = "center",
    va: str = "center",
    wrap: int | None = None,
    linespacing: float = 1.12,
) -> Rectangle:
    patch = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, joinstyle="round")
    ax.add_patch(patch)
    if text:
        width = wrap or max(10, int(w * 105))
        ax.text(
            x + (w / 2 if ha == "center" else 0.012),
            y + (h / 2 if va == "center" else h - 0.012),
            textwrap.fill(text, width=width),
            ha=ha,
            va=va,
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            linespacing=linespacing,
        )
    return patch


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


def panel(ax: plt.Axes, x: float, y: float, label: str, title: str, width: float) -> None:
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
    ax.text(x + 0.045, y - 0.002, title, ha="left", va="top", fontsize=17.2, fontweight="bold", color=TEXT)
    ax.plot([x + 0.045, x + width], [y - 0.034, y - 0.034], color=GRID, lw=1.2)


def mini_modality(ax: plt.Axes, x: float, y: float, label: str, *, color: str) -> None:
    add_box(ax, x, y, 0.092, 0.060, label, fc=color, ec=GRID, fontsize=11.8, weight="bold", wrap=10)


def safe_first_colposcopy(paths: list[str]) -> str | None:
    for item in paths:
        lower = str(item).lower()
        if "report" in lower:
            continue
        if "ori_" in lower or "pre_" in lower:
            return str(item)
    return str(paths[0]) if paths else None


def safe_first_oct(paths: list[str]) -> str | None:
    return str(paths[0]) if paths else None


def read_thumbnail(path: str | None, *, grayscale: bool = False) -> np.ndarray | None:
    if not path:
        return None
    try:
        img = Image.open(path)
        if getattr(img, "n_frames", 1) > 1:
            img.seek(0)
        img = ImageOps.exif_transpose(img)
        if grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        w, h = img.size
        crop_margin_x = int(w * 0.04)
        crop_margin_y = int(h * 0.04)
        if w > 2 * crop_margin_x and h > 2 * crop_margin_y:
            img = img.crop((crop_margin_x, crop_margin_y, w - crop_margin_x, h - crop_margin_y))
        img.thumbnail((360, 260), Image.Resampling.LANCZOS)
        return np.asarray(img)
    except Exception:
        return None


def draw_thumbnail(ax: plt.Axes, x: float, y: float, w: float, h: float, path: str | None, label: str, *, grayscale: bool = False) -> None:
    arr = read_thumbnail(path, grayscale=grayscale)
    if arr is None:
        add_box(ax, x, y, w, h, label, fc=PALE_GRAY, ec=GRID, fontsize=9.0, weight="bold", wrap=12)
        return
    ax.imshow(arr, extent=(x, x + w, y, y + h), origin="upper", cmap="gray" if grayscale else None, zorder=2)
    ax.set_aspect("auto")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=TEXT, linewidth=0.65, zorder=3))
    ax.text(x + w / 2, y - 0.010, label, ha="center", va="top", fontsize=8.8, fontweight="bold", color=TEXT)


def clean_value(value, missing: str = "NA") -> str:
    if value is None:
        return missing
    if isinstance(value, float) and np.isnan(value):
        return missing
    text = str(value)
    if text.lower() in {"nan", "none", ""}:
        return missing
    if text.endswith(".0"):
        text = text[:-2]
    return text


def case_label(case_id: str) -> str:
    parts = case_id.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}\n{parts[2]}"
    return case_id


def compact(text: str, max_chars: int = 82) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def clean_report_sentence(text: str) -> str:
    text = " ".join(str(text).split())
    text = text.replace("TCT nan", "TCT unavailable")
    text = text.replace("age 71.0", "age 71")
    text = text.replace("age 49.0", "age 49")
    return text


def section_excerpt(section: str, text: str) -> str:
    text = " ".join(str(text).split())
    lower = text.lower()
    if section == "oct":
        if "unavailable" in lower or "insufficient" in lower:
            return "OCT: unavailable or insufficient"
        if "suspicious epithelial/stromal" in lower:
            return "OCT: suspicious epithelial/stromal signal"
        return "OCT: " + compact(text.replace("OCT microstructural review:", "").strip(), 42)
    if section == "colposcopy":
        if "unavailable" in lower:
            return "Colpo: unavailable"
        if "abnormal vascular" in lower:
            return "Colpo: abnormal vascular appearance"
        return "Colpo: " + compact(text.replace("Colposcopy:", "").strip(), 42)
    if section == "clinical":
        sentence = text.split(".")[0].replace("Clinical context:", "").strip()
        return "Clinical: " + compact(sentence, 42)
    if section == "impression":
        if "suspicious for cin2+" in lower:
            return "Impression: suspicious for CIN2+"
        if "no definitive evidence" in lower:
            return "Impression: no definitive CIN2+ evidence"
        return "Impression: " + compact(text.replace("Impression:", "").strip(), 42)
    return compact(text, 48)


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_modality_evidence(case_id: str, center_id: str) -> dict:
    direct = MODALITY_EVIDENCE / center_id / f"{case_id}.json"
    if direct.is_file():
        return load_json(direct)
    for path in MODALITY_EVIDENCE.glob(f"*/{case_id}.json"):
        return load_json(path)
    return {}


def load_actual_cases() -> list[dict]:
    manifest = pd.read_csv(MANIFEST)
    scores = pd.read_csv(SCORES)
    cases = []
    for spec in ACTUAL_CASE_SPECS:
        case_id = spec["case_id"]
        row = manifest.loc[manifest["case_id"].eq(case_id)].iloc[0]
        score = scores.loc[scores["case_id"].eq(case_id)].iloc[0]
        center = str(row["center_id"])
        evidence = find_modality_evidence(case_id, center)
        normal = load_json(PERT_REPORTS / "normal" / f"{case_id}.json")
        pert = load_json(PERT_REPORTS / spec["perturbation"] / f"{case_id}.json")
        oct_paths = json.loads(row["oct_paths"]) if isinstance(row.get("oct_paths"), str) and row["oct_paths"].startswith("[") else []
        colpo_paths = json.loads(row["colposcopy_paths"]) if isinstance(row.get("colposcopy_paths"), str) and row["colposcopy_paths"].startswith("[") else []
        normal_sections = normal.get("generated_sections", {})
        pert_sections = pert.get("generated_sections", {})
        cases.append(
            {
                "case_id": case_id,
                "case_label": case_label(case_id),
                "case_tag": spec["case_tag"],
                "center_id": center,
                "split": str(row["split"]),
                "archive_tier": str(row["report_archive_tier"]),
                "has_real_report": int(row["has_real_report"]),
                "needs_pseudo_report": int(row["needs_pseudo_report"]),
                "oct_thumb_path": safe_first_oct(oct_paths),
                "colpo_thumb_path": safe_first_colposcopy(colpo_paths),
                "oct_files": len(oct_paths),
                "colposcopy_files": len(colpo_paths),
                "oct_readable": int(evidence.get("oct_evidence", {}).get("readable_images", 0)),
                "colposcopy_readable": int(evidence.get("colposcopy_evidence", {}).get("readable_images", 0)),
                "age": clean_value(row.get("age")),
                "hpv": clean_value(row.get("hpv")),
                "tct": clean_value(row.get("tct"), "unavailable"),
                "endpoint": str(row["binary_label_text"]).replace("=", ""),
                "histology_y": int(score["y_true"]),
                "backbone_score": float(score["risk_score"]),
                "fusion_score": float(score["semantic_fusion_score"]),
                "retrieval_ratio": float(score["semantic_retrieval_positive_ratio"]),
                "section_coverage": float(score["semantic_retrieval_section_coverage"]),
                "normal_risk": float(normal.get("risk_score", np.nan)),
                "perturbation": spec["perturbation"],
                "perturbed_risk": float(pert.get("risk_score", np.nan)),
                "normal_oct_support": float(normal.get("oct_section_supported_score", np.nan)),
                "pert_oct_support": float(pert.get("oct_section_supported_score", np.nan)),
                "normal_colpo_support": float(normal.get("colposcopy_section_supported_score", np.nan)),
                "pert_colpo_support": float(pert.get("colposcopy_section_supported_score", np.nan)),
                "normal_sections": normal_sections,
                "pert_sections": pert_sections,
            }
        )
    return cases


def metric_text(case: dict) -> str:
    return (
        f"{case['case_label']}\n"
        f"{case['center_id']} | {case['split']} split\n"
        f"OCT {case['oct_files']} files ({case['oct_readable']} readable)\n"
        f"Colpo {case['colposcopy_files']} files ({case['colposcopy_readable']} readable)\n"
        f"Age {case['age']}; HPV {case['hpv']}; TCT {case['tct']}\n"
        f"{case['endpoint']}; MOSAIC {case['fusion_score']:.3f}"
    )


def section_lines(case: dict, *, perturbed: bool = False) -> list[tuple[str, str]]:
    sections = case["pert_sections"] if perturbed else case["normal_sections"]
    return [
        (section_excerpt("oct", sections.get("oct_findings", "not available")), PALE_BLUE),
        (section_excerpt("colposcopy", sections.get("colposcopy_findings", "not available")), PALE_TEAL),
        (section_excerpt("clinical", sections.get("clinical_context", "not available")), PALE_GOLD),
        (section_excerpt("impression", sections.get("impression", "not available")), PALE_PURPLE),
    ]


def actual_logic_map(cases: list[dict]) -> None:
    counts = load_counts()
    c1, c2 = cases
    fig = plt.figure(figsize=(16.8, 10.6))
    ax = canvas(fig)
    fig.suptitle("Scientific question logic map of MOSAIC using actual de-identified cases", fontsize=21.8, fontweight="bold", y=0.988)

    panel(ax, 0.030, 0.920, "A", "Clinical data reality from the cohort", 0.485)
    add_box(ax, 0.030, 0.535, 0.470, 0.330, fc=PANEL, ec=BLUE, lw=1.1)
    for x, text, color in [
        (0.052, f"{counts['cases']:,}\ncases", PALE_BLUE),
        (0.155, f"{counts['images']:,}\nimages", PALE_TEAL),
        (0.258, f"{counts['real']:,}\narchived\nreports", PALE_GRAY),
        (0.361, f"{counts['pseudo']:,}\nreport-missing\ncases", PALE_RUST),
    ]:
        add_box(ax, x, 0.775, 0.088, 0.070, text, fc=color, ec=GRID, fontsize=11.4, weight="bold", wrap=12)
    add_box(ax, 0.052, 0.613, 0.200, 0.132, metric_text(c1), fc="white", ec=GRID, fontsize=10.8, weight="bold", wrap=26)
    add_box(ax, 0.276, 0.613, 0.200, 0.132, metric_text(c2), fc="white", ec=GRID, fontsize=10.8, weight="bold", wrap=26)
    add_box(
        ax,
        0.052,
        0.558,
        0.424,
        0.040,
        "Scientific problem: both actual test cases have multimodal evidence but no physician report, so report-level semantics must be weakly completed and audited.",
        fc="white",
        ec=GRID,
        fontsize=11.1,
        weight="bold",
        wrap=78,
    )

    panel(ax, 0.530, 0.920, "B", "Why direct alternatives are insufficient", 0.970)
    add_box(ax, 0.530, 0.535, 0.440, 0.330, fc=PANEL, ec=BLUE, lw=1.1)
    rows = [
        ("Image-only risk", f"backbone {c1['backbone_score']:.3f}", "no section-level reason", PALE_BLUE),
        ("Autonomous report", "fluent section text", "must pass evidence/QC gate", PALE_RUST),
        ("Report-only learning", "clinical semantics", "no real report in either case", PALE_GOLD),
    ]
    y = 0.785
    for name, strength, limitation, fc in rows:
        add_box(ax, 0.558, y, 0.128, 0.058, name, fc=fc, ec=GRID, fontsize=11.0, weight="bold", wrap=16)
        arrow(ax, 0.694, y + 0.030, 0.725, y + 0.030, color=MID, lw=1.8)
        add_box(ax, 0.735, y, 0.100, 0.058, strength, fc="white", ec=GRID, fontsize=10.5, wrap=15)
        add_box(ax, 0.855, y, 0.095, 0.058, limitation, fc=PALE_GRAY, ec=GRID, fontsize=10.3, wrap=14)
        y -= 0.085
    add_box(
        ax,
        0.558,
        0.562,
        0.392,
        0.040,
        "Goal: bounded conversion of incomplete reports into weak semantic priors, not autonomous diagnosis.",
        fc=RISK,
        ec=RUST,
        fontsize=11.0,
        weight="bold",
        color="#8E2F24",
        wrap=72,
    )

    panel(ax, 0.030, 0.460, "C", "MOSAIC answer instantiated on case M20203_2023_P0000534", 0.600)
    add_box(ax, 0.030, 0.070, 0.575, 0.335, fc=PANEL, ec=BLUE, lw=1.1)
    steps = [
        (0.055, "LCAD pseudo-report\ncompletion", PALE_RUST),
        (0.200, "QC and section\ngating", PALE_GOLD),
        (0.345, "RASA section\nanchors", PALE_TEAL),
        (0.490, "Train-only\nretrieval", PALE_BLUE),
    ]
    for i, (x, label, fc) in enumerate(steps):
        add_box(ax, x, 0.310, 0.110, 0.060, label, fc=fc, ec=GRID, fontsize=10.5, weight="bold", wrap=13)
        if i < len(steps) - 1:
            arrow(ax, x + 0.113, 0.340, steps[i + 1][0] - 0.006, 0.340)
    add_highlighted_lines(ax, 0.058, 0.248, section_lines(c1), fontsize=9.4, line_h=0.033)
    add_box(
        ax,
        0.392,
        0.178,
        0.205,
        0.066,
        f"Retrieval positive ratio {c1['retrieval_ratio']:.2f}\nsection coverage {c1['section_coverage']:.0f}\ncalibrated MOSAIC score {c1['fusion_score']:.3f}",
        fc="white",
        ec=GRID,
        fontsize=9.7,
        weight="bold",
        wrap=28,
    )
    add_box(ax, 0.058, 0.095, 0.315, 0.040, "Interpretability unit: named report section + modality evidence + retrieval prior + calibrated risk.", fc="white", ec=GRID, fontsize=9.8, weight="bold", wrap=56)

    panel(ax, 0.645, 0.460, "D", "What becomes auditable in the actual cases", 0.970)
    add_box(ax, 0.645, 0.070, 0.325, 0.335, fc=PANEL, ec=BLUE, lw=1.1)
    audit_rows = [
        ("OCT masking", f"{c1['normal_oct_support']:.2f} -> {c1['pert_oct_support']:.2f} support", f"risk {c1['normal_risk']:.3f} -> {c1['perturbed_risk']:.3f}", PALE_BLUE),
        ("Colpo masking", f"{c2['normal_colpo_support']:.2f} -> {c2['pert_colpo_support']:.2f} support", f"risk {c2['normal_risk']:.3f} -> {c2['perturbed_risk']:.3f}", PALE_TEAL),
        ("Retrieval prior", f"{c1['retrieval_ratio']:.2f} / {c2['retrieval_ratio']:.2f} positive ratio", "train-only memory", PALE_GOLD),
        ("Risk boundary", "test split", "labels held out", PALE_RUST),
    ]
    y = 0.325
    for name, evidence, result, fc in audit_rows:
        add_box(ax, 0.670, y, 0.095, 0.047, name, fc=fc, ec=GRID, fontsize=10.0, weight="bold", wrap=13)
        add_box(ax, 0.785, y, 0.075, 0.047, evidence, fc="white", ec=GRID, fontsize=9.5, wrap=13)
        add_box(ax, 0.875, y, 0.070, 0.047, result, fc="white", ec=GRID, fontsize=9.3, wrap=12)
        y -= 0.063
    add_box(ax, 0.670, 0.095, 0.275, 0.043, "Audit evidence supports risk analytics; it does not replace physician diagnosis.", fc=RISK, ec=RUST, fontsize=10.3, weight="bold", color="#8E2F24", wrap=50)

    save_fig(
        fig,
        "Figure_mosaic_scientific_question_logic_map",
        aliases=["Figure_mosaic_scientific_question_logic_map_actual_cases"],
    )


def actual_audit_table(cases: list[dict]) -> None:
    fig = plt.figure(figsize=(16.8, 10.0))
    ax = canvas(fig)
    fig.suptitle("Case-level interpretability ledger of MOSAIC using actual de-identified cases", fontsize=21.8, fontweight="bold", y=0.985)
    ax.text(0.50, 0.942, "Actual test cases; patient identifiers and raw image paths are not displayed", ha="center", va="top", fontsize=12.4, color="#5C6977")

    left, bottom, width, height = 0.030, 0.110, 0.940, 0.790
    col_w = [0.245, 0.205, 0.285, 0.265]
    row_h = [0.085, 0.455, 0.460]
    headers = ["Observed multimodal evidence", "Weak supervision source", "MOSAIC-generated structured report", "Auditable explanation signal"]
    x_edges = [left]
    for w in col_w:
        x_edges.append(x_edges[-1] + width * w)
    y_edges = [bottom + height]
    for h in row_h:
        y_edges.append(y_edges[-1] - height * h)
    ax.add_patch(Rectangle((left, bottom), width, height, fill=False, edgecolor=BLUE, linewidth=1.35))
    for x in x_edges[1:-1]:
        ax.plot([x, x], [bottom, bottom + height], color=BLUE, lw=0.82, alpha=0.72)
    for y in y_edges[1:-1]:
        ax.plot([left, left + width], [y, y], color=BLUE, lw=0.82, alpha=0.72)
    for i, header_text in enumerate(headers):
        add_box(ax, x_edges[i], y_edges[1], x_edges[i + 1] - x_edges[i], y_edges[0] - y_edges[1], header_text, fc="white", ec="none", fontsize=12.3, weight="bold", color="#0E3975", wrap=25)

    for idx, case in enumerate(cases):
        top, bot = y_edges[idx + 1], y_edges[idx + 2]
        row_y = top - 0.040
        ax.text(x_edges[0] + 0.014, row_y, f"{case['case_tag']}: {case['case_label'].replace(chr(10), '_')}", fontsize=10.7, fontweight="bold", va="top", color="#0E3975")
        add_highlighted_lines(
            ax,
            x_edges[0] + 0.018,
            row_y - 0.060,
            [
                (f"Centre {case['center_id']}; {case['split']} split", PALE_GRAY),
                (f"OCT {case['oct_files']} files; {case['oct_readable']} readable", PALE_BLUE),
                (f"Colposcopy {case['colposcopy_files']} files; {case['colposcopy_readable']} readable", PALE_TEAL),
                (f"Age {case['age']}; HPV {case['hpv']}; TCT {case['tct']}", PALE_GOLD),
                (f"{case['endpoint']}; MOSAIC {case['fusion_score']:.3f}", PALE_RUST if case["histology_y"] else PALE_GRAY),
            ],
            fontsize=9.9,
            line_h=0.039,
        )
        add_highlighted_lines(
            ax,
            x_edges[1] + 0.018,
            row_y - 0.050,
            [
                (f"Real report: {'yes' if case['has_real_report'] else 'no'}", PALE_RUST if not case["has_real_report"] else PALE_TEAL),
                (f"Archive tier: {case['archive_tier']}", PALE_GOLD),
                (f"Needs pseudo-report: {'yes' if case['needs_pseudo_report'] else 'no'}", PALE_RUST if case["needs_pseudo_report"] else PALE_TEAL),
                ("Label held out from free text", PALE_GRAY),
                (f"Source: normal + {case['perturbation']}", PALE_BLUE),
            ],
            fontsize=9.3,
            line_h=0.040,
        )
        add_highlighted_lines(
            ax,
            x_edges[2] + 0.016,
            row_y - 0.040,
            section_lines(case),
            fontsize=8.8,
            line_h=0.041,
        )
        pert_section = "OCT" if case["perturbation"] == "mask_oct" else "Colposcopy"
        support_line = (
            f"{pert_section} support "
            f"{(case['normal_oct_support'] if pert_section == 'OCT' else case['normal_colpo_support']):.2f}"
            f" -> {(case['pert_oct_support'] if pert_section == 'OCT' else case['pert_colpo_support']):.2f}"
        )
        add_highlighted_lines(
            ax,
            x_edges[3] + 0.016,
            row_y - 0.044,
            [
                (f"Retrieval positive ratio {case['retrieval_ratio']:.2f}", PALE_BLUE),
                (f"Section coverage {case['section_coverage']:.0f}", PALE_TEAL),
                (support_line, PALE_GOLD),
                (f"Perturbation risk {case['normal_risk']:.3f} -> {case['perturbed_risk']:.3f}", PALE_RUST),
                ("Held-out labels never build priors", PALE_GRAY),
            ],
            fontsize=9.1,
            line_h=0.039,
        )

    ax.text(
        left,
        0.060,
        "Color key: blue = OCT/retrieval; teal = colposcopy/section coverage; gold = clinical context or perturbation check; purple = diagnostic impression; rust = risk, missingness, or leakage-sensitive boundary.",
        fontsize=10.9,
        color=TEXT,
        ha="left",
        va="top",
    )
    save_fig(
        fig,
        "Figure_mosaic_case_level_interpretability_ledger",
        aliases=["Figure_mosaic_case_level_interpretability_ledger_actual_cases"],
    )


def draw_case_image_card(ax: plt.Axes, case: dict, x: float, y: float, w: float, h: float) -> None:
    add_box(ax, x, y, w, h, fc="white", ec=GRID, lw=1.0)
    ax.text(x + 0.014, y + h - 0.024, case["case_id"], ha="left", va="top", fontsize=8.8, fontweight="bold", color="#0E3975")
    draw_thumbnail(ax, x + 0.016, y + 0.098, w * 0.42, h * 0.38, case.get("oct_thumb_path"), "", grayscale=True)
    draw_thumbnail(ax, x + w * 0.52, y + 0.098, w * 0.42, h * 0.38, case.get("colpo_thumb_path"), "")
    ax.text(x + 0.020, y + 0.098 + h * 0.38 - 0.014, "OCT", ha="left", va="top", fontsize=7.2, fontweight="bold", color="white", bbox={"facecolor": BLUE, "edgecolor": "none", "alpha": 0.82, "pad": 1.0})
    ax.text(x + w * 0.52 + 0.004, y + 0.098 + h * 0.38 - 0.014, "Colpo", ha="left", va="top", fontsize=7.2, fontweight="bold", color="white", bbox={"facecolor": TEAL, "edgecolor": "none", "alpha": 0.82, "pad": 1.0})
    info = (
        f"{case['center_id']} | {case['split']}; report: {'real' if case['has_real_report'] else 'missing'}\n"
        f"OCT {case['oct_files']} files; Colpo {case['colposcopy_files']} files\n"
        f"Age {case['age']}; HPV {case['hpv']}; TCT {case['tct']}\n"
        f"{case['endpoint']}; MOSAIC {case['fusion_score']:.3f}"
    )
    ax.text(x + 0.014, y + 0.014, info, ha="left", va="bottom", fontsize=7.2, linespacing=1.04, color=TEXT)


def draw_simple_node(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, subtitle: str, *, fc: str, icon: str = "") -> None:
    add_box(ax, x, y, w, h, fc=fc, ec=GRID, lw=1.0)
    if icon:
        ax.text(x + 0.018, y + h / 2, icon, ha="center", va="center", fontsize=18, fontweight="bold", color=BLUE)
        tx = x + 0.040
    else:
        tx = x + 0.012
    ax.text(tx, y + h * 0.64, title, ha="left", va="center", fontsize=9.6, fontweight="bold", color=TEXT)
    ax.text(tx, y + h * 0.34, subtitle, ha="left", va="center", fontsize=8.1, color=TEXT)


def report_segments(case: dict, *, perturbed: bool) -> list[tuple[str, str]]:
    sections = case["pert_sections"] if perturbed else case["normal_sections"]
    return [
        ("OCT findings. " + compact(clean_report_sentence(sections.get("oct_findings", "OCT evidence unavailable.")), 190), PALE_BLUE),
        ("Colposcopy findings. " + compact(clean_report_sentence(sections.get("colposcopy_findings", "Colposcopy evidence unavailable.")), 170), PALE_TEAL),
        ("Clinical context. " + compact(clean_report_sentence(sections.get("clinical_context", "Clinical context unavailable.")), 150), PALE_GOLD),
        ("Diagnostic impression. " + compact(clean_report_sentence(sections.get("impression", "Impression unavailable.")), 150), PALE_PURPLE),
    ]


def draw_report_paragraph(
    ax: plt.Axes,
    x: float,
    top: float,
    w: float,
    h: float,
    segments: list[tuple[str, str]],
    *,
    fontsize: float = 7.8,
    wrap: int = 48,
) -> None:
    y = top - 0.012
    line_h = 0.020
    for text, fc in segments:
        for line in textwrap.wrap(text, width=wrap):
            if y < top - h + 0.012:
                return
            ax.text(
                x + 0.010,
                y,
                line,
                ha="left",
                va="top",
                fontsize=fontsize,
                color=TEXT,
                bbox={"facecolor": fc, "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
            )
            y -= line_h
        y -= 0.004


def draw_audit_paragraph(ax: plt.Axes, x: float, top: float, case: dict, *, fontsize: float = 8.0) -> None:
    pert_section = "OCT" if case["perturbation"] == "mask_oct" else "Colposcopy"
    support_from = case["normal_oct_support"] if pert_section == "OCT" else case["normal_colpo_support"]
    support_to = case["pert_oct_support"] if pert_section == "OCT" else case["pert_colpo_support"]
    lines = [
        (f"Train-only retrieval positive ratio {case['retrieval_ratio']:.2f}; section coverage {case['section_coverage']:.0f}.", PALE_BLUE),
        (f"{pert_section} perturbation support changed from {support_from:.2f} to {support_to:.2f}.", PALE_GOLD),
        (f"Report-risk score changed from {case['normal_risk']:.3f} to {case['perturbed_risk']:.3f}.", PALE_RUST),
        ("Held-out labels are not used to build semantic priors.", PALE_GRAY),
    ]
    draw_report_paragraph(ax, x, top, 0.19, 0.28, lines, fontsize=fontsize, wrap=36)


def actual_logic_map(cases: list[dict]) -> None:
    counts = load_counts()
    c1, c2 = cases
    fig = plt.figure(figsize=(16.8, 10.0))
    ax = canvas(fig)
    fig.suptitle("Scientific question logic map of MOSAIC", fontsize=22.0, fontweight="bold", y=0.985)

    panel(ax, 0.030, 0.910, "A", "Actual multimodal cases", 0.490)
    add_box(ax, 0.030, 0.515, 0.470, 0.340, fc=PANEL, ec=BLUE, lw=1.1)
    draw_case_image_card(ax, c1, 0.052, 0.575, 0.198, 0.235)
    draw_case_image_card(ax, c2, 0.278, 0.575, 0.198, 0.235)
    for x, text, color in [
        (0.062, f"{counts['cases']:,}\ncases", PALE_BLUE),
        (0.160, f"{counts['images']:,}\nimages", PALE_TEAL),
        (0.258, f"{counts['real']:,}\nreports", PALE_GRAY),
        (0.356, f"{counts['pseudo']:,}\nmissing", PALE_RUST),
    ]:
        add_box(ax, x, 0.525, 0.078, 0.045, text, fc=color, ec=GRID, fontsize=9.0, weight="bold", wrap=10)

    panel(ax, 0.540, 0.910, "B", "Why a direct model is not enough", 0.970)
    add_box(ax, 0.540, 0.515, 0.430, 0.340, fc=PANEL, ec=BLUE, lw=1.1)
    y = 0.770
    for title, subtitle, limit, color, icon in [
        ("Image-only", "risk score", "no section reason", PALE_BLUE, "I"),
        ("Free report", "fluent text", "must pass QC", PALE_RUST, "R"),
        ("Report-only", "semantics", "missing reports", PALE_GOLD, "T"),
    ]:
        draw_simple_node(ax, 0.565, y, 0.120, 0.060, title, subtitle, fc=color, icon=icon)
        arrow(ax, 0.695, y + 0.030, 0.750, y + 0.030, color=MID, lw=1.8)
        add_box(ax, 0.765, y, 0.145, 0.060, limit, fc="white", ec=GRID, fontsize=9.8, weight="bold", wrap=18)
        y -= 0.095
    add_box(ax, 0.565, 0.535, 0.345, 0.044, "Goal: weak semantic priors with audit evidence, not autonomous diagnosis.", fc=RISK, ec=RUST, fontsize=9.8, weight="bold", color="#8E2F24", wrap=58)

    panel(ax, 0.030, 0.455, "C", "MOSAIC path", 0.620)
    add_box(ax, 0.030, 0.080, 0.600, 0.320, fc=PANEL, ec=BLUE, lw=1.1)
    x0 = 0.060
    steps = [
        ("1\nLCAD\nreport", PALE_RUST),
        ("2\nQC\ngate", PALE_GOLD),
        ("3\nRASA\nanchors", PALE_TEAL),
        ("4\nMemory\ntrain-only", PALE_BLUE),
        ("5\nFusion\nrisk", PALE_PURPLE),
    ]
    for i, (label, color) in enumerate(steps):
        x = x0 + i * 0.106
        add_box(ax, x, 0.265, 0.086, 0.070, label, fc=color, ec=GRID, fontsize=8.8, weight="bold", wrap=11, linespacing=1.02)
        if i < len(steps) - 1:
            arrow(ax, x + 0.089, 0.300, x + 0.102, 0.300, color=MID, lw=1.35)
    draw_thumbnail(ax, 0.065, 0.135, 0.120, 0.085, c1.get("oct_thumb_path"), "OCT", grayscale=True)
    draw_thumbnail(ax, 0.210, 0.135, 0.120, 0.085, c1.get("colpo_thumb_path"), "Colposcopy")
    add_box(ax, 0.360, 0.135, 0.115, 0.085, f"HPV {c1['hpv']}\nTCT {c1['tct']}\nAge {c1['age']}", fc=PALE_GOLD, ec=GRID, fontsize=9.0, weight="bold", wrap=12)
    add_box(ax, 0.492, 0.135, 0.120, 0.085, f"{c1['endpoint']}\nheld-out", fc=PALE_RUST, ec=GRID, fontsize=8.4, weight="bold", wrap=14)

    panel(ax, 0.670, 0.455, "D", "Bounded audit evidence", 0.970)
    add_box(ax, 0.670, 0.080, 0.300, 0.320, fc=PANEL, ec=BLUE, lw=1.1)
    for i, (label, value, color) in enumerate(
        [
            ("OCT support", f"{c1['normal_oct_support']:.2f} -> {c1['pert_oct_support']:.2f}", PALE_BLUE),
            ("Colpo support", f"{c2['normal_colpo_support']:.2f} -> {c2['pert_colpo_support']:.2f}", PALE_TEAL),
            ("Risk shift", f"{c1['normal_risk']:.3f} -> {c1['perturbed_risk']:.3f}", PALE_RUST),
            ("Retrieval", f"{c1['retrieval_ratio']:.2f} / {c2['retrieval_ratio']:.2f}", PALE_GOLD),
        ]
    ):
        y = 0.315 - i * 0.060
        add_box(ax, 0.700, y, 0.115, 0.042, label, fc=color, ec=GRID, fontsize=9.4, weight="bold", wrap=15)
        add_box(ax, 0.835, y, 0.095, 0.042, value, fc="white", ec=GRID, fontsize=8.8, weight="bold", wrap=18)
    add_box(ax, 0.700, 0.105, 0.230, 0.040, "risk analytics only", fc=RISK, ec=RUST, fontsize=9.5, weight="bold", color="#8E2F24", wrap=28)

    save_fig(
        fig,
        "Figure_mosaic_scientific_question_logic_map",
        aliases=["Figure_mosaic_scientific_question_logic_map_actual_cases"],
    )


def actual_audit_table(cases: list[dict]) -> None:
    fig = plt.figure(figsize=(17.2, 9.8))
    ax = canvas(fig)
    fig.suptitle("Case-level report-style interpretability ledger of MOSAIC", fontsize=22.0, fontweight="bold", y=0.985)
    ax.text(0.50, 0.947, "Actual de-identified test cases; raw paths, names, and report screenshots are not displayed", ha="center", va="top", fontsize=11.8, color="#5C6977")

    left, bottom, width, height = 0.025, 0.118, 0.950, 0.770
    col_w = [0.205, 0.285, 0.285, 0.225]
    row_h = [0.075, 0.462, 0.463]
    headers = ["Input data", "MOSAIC normal-input report", "MOSAIC perturbed-input report", "Bounded audit signal"]
    x_edges = [left]
    for w in col_w:
        x_edges.append(x_edges[-1] + width * w)
    y_edges = [bottom + height]
    for h in row_h:
        y_edges.append(y_edges[-1] - height * h)
    ax.add_patch(Rectangle((left, bottom), width, height, fill=False, edgecolor=BLUE, linewidth=1.35))
    for x in x_edges[1:-1]:
        ax.plot([x, x], [bottom, bottom + height], color=BLUE, lw=0.82, alpha=0.72)
    for y in y_edges[1:-1]:
        ax.plot([left, left + width], [y, y], color=BLUE, lw=0.82, alpha=0.72)
    for i, header_text in enumerate(headers):
        add_box(ax, x_edges[i], y_edges[1], x_edges[i + 1] - x_edges[i], y_edges[0] - y_edges[1], header_text, fc="white", ec="none", fontsize=12.0, weight="bold", color="#0E3975", wrap=25)

    for idx, case in enumerate(cases):
        top, bot = y_edges[idx + 1], y_edges[idx + 2]
        row_h_abs = top - bot
        # Input data panel with actual thumbnails.
        short_tag = case["case_tag"].replace("Actual test case", "Case")
        ax.text(x_edges[0] + 0.010, top - 0.026, f"{short_tag}: {case['case_id']}", fontsize=7.9, fontweight="bold", color="#0E3975", va="top")
        draw_thumbnail(ax, x_edges[0] + 0.018, top - 0.138, 0.075, 0.070, case.get("oct_thumb_path"), "OCT", grayscale=True)
        draw_thumbnail(ax, x_edges[0] + 0.108, top - 0.138, 0.075, 0.070, case.get("colpo_thumb_path"), "Colpo")
        input_text = (
            f"{case['center_id']} | {case['split']} split\n"
            f"OCT {case['oct_files']} files; Colpo {case['colposcopy_files']} files\n"
            f"Age {case['age']}; HPV {case['hpv']}; TCT {case['tct']}\n"
            f"{case['endpoint']}; MOSAIC {case['fusion_score']:.3f}\n"
            f"Real report: {'yes' if case['has_real_report'] else 'no'}"
        )
        ax.text(x_edges[0] + 0.018, top - 0.172, input_text, fontsize=8.2, va="top", linespacing=1.18, color=TEXT)

        draw_report_paragraph(ax, x_edges[1] + 0.008, top - 0.018, x_edges[2] - x_edges[1] - 0.016, row_h_abs - 0.045, report_segments(case, perturbed=False), fontsize=7.05, wrap=49)
        draw_report_paragraph(ax, x_edges[2] + 0.008, top - 0.018, x_edges[3] - x_edges[2] - 0.016, row_h_abs - 0.045, report_segments(case, perturbed=True), fontsize=7.05, wrap=49)
        draw_audit_paragraph(ax, x_edges[3] + 0.010, top - 0.026, case, fontsize=7.55)

    ax.text(
        left,
        0.062,
        "Color key: blue = OCT/retrieval; teal = colposcopy; gold = clinical context or perturbation check; purple = diagnostic impression; rust = risk, missingness, or leakage-sensitive boundary.",
        fontsize=10.6,
        color=TEXT,
        ha="left",
        va="top",
    )
    save_fig(
        fig,
        "Figure_mosaic_case_level_interpretability_ledger",
        aliases=["Figure_mosaic_case_level_interpretability_ledger_actual_cases"],
    )


def write_actual_case_sources(cases: list[dict]) -> None:
    rows = []
    for case in cases:
        rows.append(
            {
                "case_id": case["case_id"],
                "center_id": case["center_id"],
                "split": case["split"],
                "has_real_report": case["has_real_report"],
                "needs_pseudo_report": case["needs_pseudo_report"],
                "archive_tier": case["archive_tier"],
                "oct_files": case["oct_files"],
                "oct_readable": case["oct_readable"],
                "colposcopy_files": case["colposcopy_files"],
                "colposcopy_readable": case["colposcopy_readable"],
                "age": case["age"],
                "hpv": case["hpv"],
                "tct": case["tct"],
                "endpoint": case["endpoint"],
                "backbone_score": case["backbone_score"],
                "mosaic_fusion_score": case["fusion_score"],
                "retrieval_positive_ratio": case["retrieval_ratio"],
                "section_coverage": case["section_coverage"],
                "normal_report_risk": case["normal_risk"],
                "perturbation": case["perturbation"],
                "perturbed_report_risk": case["perturbed_risk"],
            }
        )
    for out_dir in [OUT_JBD, FINAL]:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "ACTUAL_CASE_SOURCES.csv", index=False)


def logic_map() -> None:
    counts = load_counts()
    fig = plt.figure(figsize=(16.8, 10.6))
    ax = canvas(fig)
    fig.suptitle("What MOSAIC makes interpretable: from missing reports to bounded semantic evidence", fontsize=22.0, fontweight="bold", y=0.988)

    panel(ax, 0.035, 0.915, "A", "Clinical data reality", 0.47)
    add_box(ax, 0.035, 0.535, 0.455, 0.325, fc=PANEL, ec=GRID, lw=1.1)
    mini_modality(ax, 0.060, 0.770, "OCT\nimages", color=PALE_BLUE)
    mini_modality(ax, 0.168, 0.770, "Colposcopy\nimages", color=PALE_TEAL)
    mini_modality(ax, 0.276, 0.770, "HPV / TCT\nage", color=PALE_GOLD)
    mini_modality(ax, 0.384, 0.770, "Histology\nlabel", color=PALE_RUST)
    ax.text(0.060, 0.708, f"{counts['cases']:,} cases, {counts['centres']} centres, {counts['images']:,} images", fontsize=13.8, fontweight="bold")
    ax.text(0.060, 0.666, f"Only {counts['real']:,} physician reports; {counts['pseudo']:,} cases need report-level completion", fontsize=12.6)
    ax.text(0.060, 0.624, "Reports are incomplete, centre-dependent, and unevenly distributed.", fontsize=12.6)
    add_box(ax, 0.060, 0.565, 0.405, 0.040, "Scientific problem: visual risk can be predicted, but the semantic reason is weakly observed.", fc="white", ec=GRID, fontsize=12.4, weight="bold", wrap=70)

    panel(ax, 0.530, 0.915, "B", "Why direct alternatives are insufficient", 0.96)
    add_box(ax, 0.530, 0.535, 0.430, 0.325, fc=PANEL, ec=GRID, lw=1.1)
    rows = [
        ("Image-only risk model", "high AUROC possible", "no report-section anchor", PALE_BLUE),
        ("Autonomous report generation", "fluent text possible", "hallucination and leakage risk", PALE_RUST),
        ("Report-only supervision", "clinical semantics", "missing and centre-biased", PALE_GOLD),
    ]
    y = 0.782
    for name, strength, limit, fc in rows:
        add_box(ax, 0.555, y, 0.130, 0.055, name, fc=fc, ec=GRID, fontsize=11.2, weight="bold", wrap=17)
        add_box(ax, 0.710, y, 0.095, 0.055, strength, fc="white", ec=GRID, fontsize=10.8, wrap=15)
        add_box(ax, 0.828, y, 0.105, 0.055, limit, fc=PALE_GRAY, ec=GRID, fontsize=10.8, wrap=15)
        y -= 0.086
    ax.text(0.555, 0.575, "Therefore the task is not to let an LLM diagnose, but to convert incomplete reports into controlled semantic priors.", fontsize=12.5, fontweight="bold", wrap=True)

    panel(ax, 0.035, 0.458, "C", "MOSAIC answer: weak-oracle semantic prior", 0.565)
    add_box(ax, 0.035, 0.075, 0.545, 0.330, fc=PANEL, ec=GRID, lw=1.1)
    steps = [
        (0.060, "Label-constrained\npseudo report", PALE_RUST),
        (0.205, "QC gate:\ndata-grounded only", PALE_GOLD),
        (0.350, "RASA section\nanchoring", PALE_TEAL),
        (0.495, "Train-only\nsemantic memory", PALE_BLUE),
    ]
    for i, (x, label, fc) in enumerate(steps):
        add_box(ax, x, 0.290, 0.110, 0.064, label, fc=fc, ec=GRID, fontsize=11.2, weight="bold", wrap=13)
        if i < len(steps) - 1:
            arrow(ax, x + 0.113, 0.322, steps[i + 1][0] - 0.006, 0.322)
    section_y = 0.190
    for x, label, fc in [
        (0.073, "OCT findings", PALE_BLUE),
        (0.198, "Colposcopy findings", PALE_TEAL),
        (0.342, "Clinical context", PALE_GOLD),
        (0.468, "Impression", PALE_PURPLE),
    ]:
        add_box(ax, x, section_y, 0.095, 0.050, label, fc=fc, ec=GRID, fontsize=10.8, weight="bold", wrap=12)
    ax.text(0.060, 0.125, "Interpretability unit: a named report section linked to a modality stream, retrieval prior, and calibrated risk output.", fontsize=12.4, fontweight="bold")

    panel(ax, 0.625, 0.458, "D", "What becomes auditable", 0.96)
    add_box(ax, 0.625, 0.075, 0.335, 0.330, fc=PANEL, ec=GRID, lw=1.1)
    audit_rows = [
        ("Section evidence", "Which section changed?", PALE_TEAL),
        ("Retrieval prior", "Train-only memory?", PALE_BLUE),
        ("Perturbation response", "Expected section drop?", PALE_GOLD),
        ("Risk boundary", "Calibrated risk only", PALE_RUST),
    ]
    y = 0.326
    for name, question, fc in audit_rows:
        add_box(ax, 0.650, y, 0.110, 0.042, name, fc=fc, ec=GRID, fontsize=10.4, weight="bold", wrap=14)
        add_box(ax, 0.775, y, 0.155, 0.042, question, fc="white", ec=GRID, fontsize=10.4, wrap=21)
        y -= 0.058
    add_box(ax, 0.650, 0.092, 0.280, 0.040, "Claim boundary: audit evidence, not autonomous diagnosis.", fc=PALE_GRAY, ec=GRID, fontsize=10.7, weight="bold", wrap=45)

    save_fig(fig, "Figure_mosaic_scientific_question_logic_map")


def add_highlighted_lines(ax: plt.Axes, x: float, y: float, lines: list[tuple[str, str]], *, fontsize: float = 10.7, line_h: float = 0.037) -> None:
    for i, (text, fc) in enumerate(lines):
        ax.text(
            x,
            y - i * line_h,
            text,
            ha="left",
            va="top",
            fontsize=fontsize,
            color=TEXT,
            bbox={"facecolor": fc, "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
        )


def audit_table() -> None:
    fig = plt.figure(figsize=(16.8, 9.8))
    ax = canvas(fig)
    fig.suptitle("Case-level interpretability ledger for MOSAIC semantic priors", fontsize=22.0, fontweight="bold", y=0.986)

    left, bottom, width, height = 0.030, 0.115, 0.940, 0.780
    col_w = [0.205, 0.185, 0.285, 0.265]
    row_h = [0.085, 0.455, 0.460]
    headers = ["Observed multimodal evidence", "Weak supervision source", "MOSAIC semantic scaffold", "Auditable explanation signal"]
    x_edges = [left]
    for w in col_w:
        x_edges.append(x_edges[-1] + width * w)
    y_edges = [bottom + height]
    for h in row_h:
        y_edges.append(y_edges[-1] - height * h)

    ax.add_patch(Rectangle((left, bottom), width, height, fill=False, edgecolor=BLUE, linewidth=1.4))
    for x in x_edges[1:-1]:
        ax.plot([x, x], [bottom, bottom + height], color=TEXT, lw=1.0)
    for y in y_edges[1:-1]:
        ax.plot([left, left + width], [y, y], color=TEXT, lw=1.0)
    for i, header_text in enumerate(headers):
        add_box(ax, x_edges[i], y_edges[1], x_edges[i + 1] - x_edges[i], y_edges[0] - y_edges[1], header_text, fc="white", ec="none", fontsize=12.4, weight="bold", wrap=25)

    row1_top, row1_bot = y_edges[1], y_edges[2]
    row2_top, row2_bot = y_edges[2], y_edges[3]
    row_centers = [(row1_top + row1_bot) / 2, (row2_top + row2_bot) / 2]
    motifs = [
        ("Case motif 1\nimage-rich, report-missing", row1_top, row1_bot),
        ("Case motif 2\ncentre wording shift", row2_top, row2_bot),
    ]
    for label, top, bot in motifs:
        ax.text(x_edges[0] + 0.012, top - 0.035, label, fontsize=11.2, fontweight="bold", va="top")

    add_highlighted_lines(
        ax,
        x_edges[0] + 0.025,
        row1_top - 0.110,
        [
            ("OCT: epithelial-layer disruption", PALE_BLUE),
            ("Colposcopy: acetowhite lesion", PALE_TEAL),
            ("HPV/TCT: high-risk context", PALE_GOLD),
        ],
    )
    add_highlighted_lines(
        ax,
        x_edges[1] + 0.020,
        row1_top - 0.088,
        [
            ("No physician report available", PALE_RUST),
            ("Locked label is not free text", PALE_GRAY),
            ("Completion must pass QC gate", PALE_GOLD),
        ],
        fontsize=10.5,
    )
    add_highlighted_lines(
        ax,
        x_edges[2] + 0.018,
        row1_top - 0.062,
        [
            ("[OCT findings] abnormal microstructure retained", PALE_BLUE),
            ("[Colposcopy findings] lesion morphology retained", PALE_TEAL),
            ("[Clinical context] HPV/TCT summarized, not invented", PALE_GOLD),
            ("[Impression] risk prior separated from final label", PALE_PURPLE),
        ],
        fontsize=10.3,
        line_h=0.034,
    )
    add_highlighted_lines(
        ax,
        x_edges[3] + 0.018,
        row1_top - 0.062,
        [
            ("Section anchors link evidence streams to risk", GOOD),
            ("Train-only retrieval supplies semantic priors", PALE_BLUE),
            ("Fusion is calibrated before held-out testing", PALE_RUST),
            ("Output: disease-risk score with audit trail", PALE_GRAY),
        ],
        fontsize=10.1,
        line_h=0.034,
    )

    add_highlighted_lines(
        ax,
        x_edges[0] + 0.025,
        row2_top - 0.095,
        [
            ("OCT masked or degraded", PALE_RUST),
            ("Colposcopy remains informative", PALE_TEAL),
            ("Structured variables available", PALE_GOLD),
        ],
        fontsize=10.7,
    )
    add_highlighted_lines(
        ax,
        x_edges[1] + 0.020,
        row2_top - 0.075,
        [
            ("Centre wording may shift", PALE_GOLD),
            ("Report section incomplete", PALE_RUST),
            ("Raw text is not a stable target", PALE_GRAY),
        ],
        fontsize=10.5,
    )
    add_highlighted_lines(
        ax,
        x_edges[2] + 0.018,
        row2_top - 0.058,
        [
            ("Mask OCT -> OCT section drops first", PALE_BLUE),
            ("Colposcopy and clinical sections preserved", PALE_TEAL),
            ("Risk shift is measured, not hidden", PALE_RUST),
            ("Mismatch flags a failed semantic explanation", PALE_PURPLE),
        ],
        fontsize=10.3,
        line_h=0.034,
    )
    add_highlighted_lines(
        ax,
        x_edges[3] + 0.018,
        row2_top - 0.058,
        [
            ("Perturbation links modality loss to section change", GOOD),
            ("Retrieval MRR checks semantic alignment", PALE_BLUE),
            ("LOCO tests centre generalization", PALE_GOLD),
            ("Held-out labels never build priors", PALE_RUST),
        ],
        fontsize=10.1,
        line_h=0.034,
    )
    ax.text(
        left,
        0.055,
        "Color key: blue = OCT section; teal = colposcopy section; gold = clinical context; purple = diagnostic impression; rust = risk, missingness, or leakage-sensitive boundary.",
        fontsize=11.6,
        color=TEXT,
        ha="left",
        va="top",
    )
    save_fig(fig, "Figure_mosaic_case_level_interpretability_ledger")


def write_prompts(cases: list[dict]) -> None:
    c1, c2 = cases
    lines = [
        "# MOSAIC Interpretability Figure Blueprint",
        "",
        "## Figure_mosaic_scientific_question_logic_map",
        "",
        "Native-style caption: Scientific-question logic map of MOSAIC instantiated with two actual de-identified held-out cases. Panel A summarizes the cohort-level supervision gap and the two displayed cases, both of which contain multimodal evidence but no physician-authored report. Panel B shows why image-only risk prediction, autonomous report generation, and report-only learning are insufficient for this setting. Panel C traces the MOSAIC weak-oracle pathway for case "
        f"{c1['case_id']}, from label-constrained pseudo-report completion to RASA section anchoring, train-only retrieval support, and calibrated fusion. Panel D shows bounded audit signals from actual perturbation records, including section-support drops and risk-score changes under OCT or colposcopy masking. The figure uses de-identified case IDs only and does not display patient-identifiable images or file paths.",
        "",
        "AI drawing prompt: Create a clean four-panel scientific figure in a Nature/Science style using actual de-identified MOSAIC case values. Panel A should list cohort values (1,897 cases, 137,591 images, 744 archived reports, 1,153 report-missing cases) and two actual cases: "
        f"{c1['case_id']} from {c1['center_id']} with OCT {c1['oct_files']} files, colposcopy {c1['colposcopy_files']} files, age {c1['age']}, HPV {c1['hpv']}, TCT {c1['tct']}, endpoint {c1['endpoint']}, MOSAIC score {c1['fusion_score']:.3f}; and "
        f"{c2['case_id']} from {c2['center_id']} with OCT {c2['oct_files']} files, colposcopy {c2['colposcopy_files']} files, age {c2['age']}, HPV {c2['hpv']}, TCT {c2['tct']}, endpoint {c2['endpoint']}, MOSAIC score {c2['fusion_score']:.3f}. Panel B should compare image-only risk, autonomous report generation, and report-only learning with their limitations. Panel C should show LCAD pseudo-report completion, QC/section gating, RASA section anchors, train-only retrieval, and calibrated MOSAIC score. Panel D should show actual audit values: OCT support drop and risk change for the first case, colposcopy support drop and risk change for the second case, retrieval positive ratio, and held-out-label boundary. Use restrained blue, rust, teal, gold, purple, and gray colors; Arial text; Times New Roman numbers; no patient-identifiable images.",
        "",
        "## Figure_mosaic_case_level_interpretability_ledger",
        "",
        "Native-style caption: Case-level interpretability ledger for two actual de-identified test cases. Each row links observed multimodal evidence, report-supervision status, MOSAIC-generated structured sections, and bounded audit signals. Case "
        f"{c1['case_id']} is a report-missing {c1['center_id']} case with OCT and colposcopy evidence, HPV/TCT clinical context, and a calibrated MOSAIC score of {c1['fusion_score']:.3f}. Case "
        f"{c2['case_id']} is a report-missing {c2['center_id']} case from a sparse-report centre with HPV {c2['hpv']} and a calibrated MOSAIC score of {c2['fusion_score']:.3f}. The final column reports train-only retrieval support, section coverage, modality-perturbation response, and the held-out-label boundary. These examples are audit ledgers for research risk analytics and are not patient-facing diagnostic reports.",
        "",
        "AI drawing prompt: Create a table-like qualitative comparison figure with four columns: observed multimodal evidence, weak supervision source, MOSAIC-generated structured report, and auditable explanation signal. Use two rows corresponding to actual de-identified cases "
        f"{c1['case_id']} and {c2['case_id']}. In the first column, show center, test split, OCT file count/readable image count, colposcopy file count/readable image count, age, HPV, TCT, histopathology-derived endpoint, and MOSAIC score. In the second column, show real report = no, needs pseudo-report = yes, archive tier, source condition, and label boundary. In the third column, show OCT findings, colposcopy findings, clinical context, and diagnostic impression from the MOSAIC-generated structured report. In the fourth column, show retrieval positive ratio, section coverage, perturbation support drop, perturbation risk change, and held-out-label boundary. Use color highlights: blue for OCT/retrieval, teal for colposcopy, gold for clinical context or perturbation check, purple for impression, rust for risk or missingness, gray for audit notes. Do not display raw patient images, patient names, or file paths.",
        "",
        "## Source table",
        "",
        "`ACTUAL_CASE_SOURCES.csv` records the exact case IDs, split, report status, modality counts, clinical variables, endpoint label text, MOSAIC scores, retrieval metrics, and perturbation conditions used in these two figures.",
        "",
    ]
    for out_dir in [OUT_JBD, FINAL]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "AI_REPRODUCTION_PROMPTS.md").write_text("\n".join(lines), encoding="utf-8")
        (out_dir / "SCI_CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_theme()
    cases = load_actual_cases()
    actual_logic_map(cases)
    actual_audit_table(cases)
    write_actual_case_sources(cases)
    write_prompts(cases)
    print("Generated MOSAIC interpretability blueprint figures")


if __name__ == "__main__":
    main()
