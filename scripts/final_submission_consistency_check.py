#!/usr/bin/env python3
"""Static final-submission checks for the JBD manuscript."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REV_DIR = ROOT / "outputs" / "revision"
TEX = ROOT / "sn-article (5).tex"
BIB = ROOT / "sn-bibliography.bib"

EXPECTED_COUNTS = {
    "1,897": "cohort cases",
    "744": "archived reports",
    "1,153": "pseudo-report candidates",
    "137,591": "image files",
}

BAD_COUNT_PATTERNS = {
    r"(?<![0-9.])742(?![0-9.])": ("742", "obsolete archived-report count"),
    r"(?<![0-9.])137,294(?![0-9.])": ("137,294", "obsolete image-file count"),
}

OVERCLAIM_PATTERNS = [
    r"\bdeployment-ready\b",
    r"\bready for clinical deployment\b",
    r"\bprospectively validated\b",
    r"\bprospective validation was completed\b",
    r"\buniversal superiority\b",
    r"\bautonomous clinical decision[- ]making\b",
]

NEGATION_HINTS = ["not ", "no ", "rather than", "does not", "do not", "without", "nor "]


def uncommented_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            continue
        out = []
        escaped = False
        for ch in line:
            if ch == "%" and not escaped:
                break
            out.append(ch)
            escaped = ch == "\\" and not escaped
            if ch != "\\":
                escaped = False
        lines.append("".join(out))
    return "\n".join(lines)


def extract_citations(text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite[t|p|alp|alt|author|year]*\{([^}]+)\}", text):
        for key in match.group(1).split(","):
            key = key.strip()
            if key:
                keys.add(key)
    return keys


def extract_bib_keys() -> set[str]:
    if not BIB.exists():
        return set()
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", BIB.read_text(encoding="utf-8", errors="ignore")))


def figure_refs(text: str) -> list[str]:
    refs = []
    for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        path = Path(match.group(1))
        if not path.suffix:
            path = path.with_suffix(".pdf")
        refs.append(str(path))
    return refs


def caption_protocol_flags(text: str) -> list[str]:
    flags = []
    for idx, match in enumerate(re.finditer(r"\\caption\{(.+?)\}", text, flags=re.DOTALL), start=1):
        cap = re.sub(r"\s+", " ", match.group(1)).strip()
        lower = cap.lower()
        has_protocol = any(
            token in lower
            for token in [
                "held-out",
                "train-only",
                "validation",
                "same-split",
                "semantic-tag audit",
                "strict leave-one-centre-out",
                "eval-only",
                "supplementary",
                "retrospective",
                "perturbation",
                "qc",
            ]
        )
        if not has_protocol:
            flags.append(f"Caption {idx}: {cap[:160]}")
    return flags


def non_negated_pattern_hits(text: str, pattern: str) -> list[str]:
    hits = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        start = max(0, match.start() - 80)
        context = text[start : match.start()].lower()
        if any(hint in context for hint in NEGATION_HINTS):
            continue
        hits.append(match.group(0))
    return hits


def main() -> None:
    REV_DIR.mkdir(parents=True, exist_ok=True)
    text = TEX.read_text(encoding="utf-8", errors="ignore")
    active = uncommented_text(text)
    report: list[str] = ["# Final Submission Consistency Report", ""]
    unresolved: list[str] = ["# Unresolved Submission Risks", ""]

    report.append("## Count Checks")
    for value, label in EXPECTED_COUNTS.items():
        present = value in active
        report.append(f"- {label}: {'present' if present else 'missing'} (`{value}`).")
        if not present:
            unresolved.append(f"- Expected {label} value `{value}` was not found in active manuscript text.")
    for pat, (value, label) in BAD_COUNT_PATTERNS.items():
        present = bool(re.search(pat, active))
        report.append(f"- obsolete {label}: {'found' if present else 'not found'} (`{value}`).")
        if present:
            unresolved.append(f"- Obsolete {label} value `{value}` remains in active manuscript text.")

    report.append("\n## Claim-Boundary Checks")
    for pat in OVERCLAIM_PATTERNS:
        hits = non_negated_pattern_hits(active, pat)
        report.append(f"- `{pat}`: {len(hits)} active hits.")
        if hits:
            unresolved.append(f"- Potential overclaim pattern remains: `{pat}`.")
    unavailable_positive = bool(non_negated_pattern_hits(active, r"LLM[- ]tag.{0,120}(outperform|superior|improved)"))
    report.append(f"- unavailable LLM-tag rows used as positive evidence: {'possible hit' if unavailable_positive else 'not detected'}.")
    if unavailable_positive:
        unresolved.append("- LLM-tag wording may imply unavailable LLM tags are positive evidence.")

    report.append("\n## Figure Checks")
    refs = figure_refs(active)
    missing = [ref for ref in refs if not (ROOT / ref).exists()]
    report.append(f"- active includegraphics refs: {len(refs)}.")
    report.append(f"- missing active figure files: {len(missing)}.")
    for ref in missing:
        unresolved.append(f"- Missing figure file: `{ref}`.")
    cap_flags = caption_protocol_flags(active)
    report.append(f"- captions without explicit protocol keyword: {len(cap_flags)}.")
    for flag in cap_flags[:12]:
        unresolved.append(f"- Add protocol wording if this caption is main-text evidence: {flag}")

    report.append("\n## Citation Checks")
    cited = extract_citations(active)
    bib = extract_bib_keys()
    undefined = sorted(cited - bib)
    report.append(f"- citation keys used: {len(cited)}.")
    report.append(f"- undefined citation keys by static bib scan: {len(undefined)}.")
    for key in undefined:
        unresolved.append(f"- Undefined citation key by static scan: `{key}`.")

    report.append("\n## AUROC Text Checks")
    required_metric_strings = ["0.908", "0.925", "0.888", "0.856"]
    for value in required_metric_strings:
        report.append(f"- metric `{value}` active mentions: {active.count(value)}.")

    try:
        status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=False)
        files_modified = status.stdout.strip() + "\n"
    except Exception as exc:
        files_modified = f"git status unavailable: {exc}\n"
    (REV_DIR / "files_modified.txt").write_text(files_modified, encoding="utf-8")

    if len(unresolved) == 2:
        unresolved.append("- No blocking static risks detected by this lightweight checker.")
    (REV_DIR / "final_submission_consistency_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (REV_DIR / "unresolved_submission_risks.md").write_text("\n".join(unresolved) + "\n", encoding="utf-8")
    print(f"Wrote {REV_DIR / 'final_submission_consistency_report.md'}")
    print(f"Wrote {REV_DIR / 'unresolved_submission_risks.md'}")
    print(f"Wrote {REV_DIR / 'files_modified.txt'}")


if __name__ == "__main__":
    main()
