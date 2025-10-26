#!/usr/bin/env python3
"""
Sigma-Lab | Mesh Memory Append Tool
-----------------------------------
Appends structured network reports (Skywire/Fiber) into a persistent
Mesh Memory ledger for nightly validation tracking.

Usage:
    python -m tools.mesh_memory_append \
        --report ./pilots/validation_logs/skywire_validate_only.json \
        --memory ./pilots/validation_logs/mesh_memory.jsonl
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_json(path: Path):
    """Safely load JSON file, returning None if not found or invalid."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def append_to_memory(report_data: dict, memory_path: Path):
    """Append structured report to mesh memory file (JSONL)."""
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "report_summary": report_data.get("summary", {}),
        "metrics": report_data.get("metrics", {}),
        "source": report_data.get("network", "unknown"),
    }

    with open(memory_path, "a", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

    print(f"[MeshMemory] Appended entry for network={entry['source']} → {memory_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Append network validation reports into Mesh Memory (JSONL)."
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to the JSON report to append (from validate-only run).",
    )
    parser.add_argument(
        "--memory",
        default="./pilots/validation_logs/mesh_memory.jsonl",
        help="Target Mesh Memory JSONL file to append to.",
    )

    args = parser.parse_args()
    report_path = Path(args.report)
    memory_path = Path(args.memory)

    report_data = load_json(report_path)
    if not report_data:
        raise FileNotFoundError(f"Invalid or missing report file: {report_path}")

    append_to_memory(report_data, memory_path)


if __name__ == "__main__":
    main()
