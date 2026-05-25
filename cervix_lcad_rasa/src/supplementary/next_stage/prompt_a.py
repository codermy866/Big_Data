"""Prompt A: image count audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.supplementary.io_utils import save_table
from src.supplementary.next_stage.core import count_images_in_paths


def run_image_audit(project: Path, manifest_path: Path, tables_dir: Path) -> None:
    df = pd.read_csv(manifest_path)
    df["n_oct"] = df["oct_paths"].astype(str).map(count_images_in_paths)
    df["n_colpo"] = df["colposcopy_paths"].astype(str).map(count_images_in_paths)
    df["n_total_img"] = df["n_oct"] + df["n_colpo"]

    center_rows = []
    for cid, g in df.groupby("center_id"):
        center_rows.append(
            {
                "center": cid,
                "cases": len(g),
                "oct_images": int(g["n_oct"].sum()),
                "colposcopy_images": int(g["n_colpo"].sum()),
                "total_images": int(g["n_total_img"].sum()),
                "real_report_cases": int(g["has_real_report"].sum()),
                "pseudo_candidates": int(g["needs_pseudo_report"].sum()),
                "missing_oct_rate": float((g.get("missing_oct", 0) == 1).mean()) if "missing_oct" in g else 0,
            }
        )
    audit_df = pd.DataFrame(center_rows)
    total_oct = int(df["n_oct"].sum())
    total_colpo = int(df["n_colpo"].sum())
    total_img = total_oct + total_colpo
    test_n = int((df["split"] == "test").sum()) if "split" in df.columns else len(df)

    final_stats = pd.DataFrame(
        [
            {"metric": "total_cases", "value": len(df), "source": "manifest"},
            {"metric": "total_centers", "value": df["center_id"].nunique(), "source": "manifest"},
            {"metric": "total_images_evaluable", "value": total_img, "source": "path_list_sum"},
            {"metric": "total_oct_images", "value": total_oct, "source": "path_list_sum"},
            {"metric": "total_colposcopy_images", "value": total_colpo, "source": "path_list_sum"},
            {"metric": "legacy_cited_total_images", "value": 137294, "source": "methods_doc_v1"},
            {"metric": "current_pipeline_total_images", "value": total_img, "source": "publishable_manifest"},
            {"metric": "delta_vs_legacy", "value": total_img - 137294, "source": "computed"},
            {"metric": "real_report_cases", "value": int(df["has_real_report"].sum()), "source": "manifest"},
            {"metric": "pseudo_report_candidates", "value": int(df["needs_pseudo_report"].sum()), "source": "manifest"},
            {"metric": "analytic_test_cases", "value": test_n, "source": "split=test"},
        ]
    )

    save_table(audit_df, tables_dir / "table_centerwise_image_count_audit.csv")
    save_table(final_stats, tables_dir / "table_final_dataset_statistics_for_manuscript.csv")

    md = f"""# Image count audit

## Final numbers for manuscript (single source of truth)

| Item | Value |
|------|------:|
| Cases | {len(df)} |
| Centres | {df['center_id'].nunique()} |
| **Total evaluable images** | **{total_img:,}** |
| OCT images | {total_oct:,} |
| Colposcopy images | {total_colpo:,} |
| Real reports | {int(df['has_real_report'].sum())} |
| Pseudo candidates | {int(df['needs_pseudo_report'].sum())} |
| Test cases | {test_n} |

## 137,294 vs {total_img:,}

- Legacy methods text: **137,294** images.
- Current publishable manifest path-sum: **{total_img:,}** (Δ = {total_img - 137294:+d}).
- Likely causes: expanded image discovery in colposcopy/OCT paths, inclusion of additional frames per case, or revised path parsing vs earlier ledger.
- **Recommendation**: Use **{total_img:,}** with wording *evaluable OCT and colposcopy images after current discovery pipeline*.

See `table_centerwise_image_count_audit.csv` and `table_final_dataset_statistics_for_manuscript.csv`.
"""
    (tables_dir / "IMAGE_COUNT_AUDIT.md").write_text(md, encoding="utf-8")
    (tables_dir / "FINAL_DATASET_STATEMENT_FOR_MANUSCRIPT.md").write_text(
        f"We analysed {len(df)} multimodal examinations from five centres, comprising {total_img:,} evaluable "
        f"OCT and colposcopy images ({total_oct:,} OCT; {total_colpo:,} colposcopy). "
        f"Case-level real reports were available for {int(df['has_real_report'].sum())} examinations; "
        f"{int(df['needs_pseudo_report'].sum())} required LCAD pseudo-report weak supervision.\n",
        encoding="utf-8",
    )
