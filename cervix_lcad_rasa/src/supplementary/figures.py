"""Generate supplementary figures — Seaborn gallery styles."""

from __future__ import annotations

from pathlib import Path


def generate_all_figures(tables_dir: Path, figures_dir: Path) -> None:
    """Regenerate all publishable figures from manuscript tables."""
    project = tables_dir
    for _ in range(4):
        if (project / "outputs/publishable/tables").is_dir():
            break
        project = project.parent
    from src.supplementary.jbd_figures_seaborn import generate_all_seaborn_figures

    generate_all_seaborn_figures(project)
