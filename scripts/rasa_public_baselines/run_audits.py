#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rasa_public_baselines.core import run_claim_leakage_audit, run_retrieval_memory_audit, write_inventory, write_resolved_registry


def main() -> None:
    write_resolved_registry()
    write_inventory()
    claim = run_claim_leakage_audit()
    retrieval = run_retrieval_memory_audit()
    print(json.dumps({"claim": claim, "retrieval": retrieval}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

