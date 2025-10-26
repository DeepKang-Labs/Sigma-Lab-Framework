#!/usr/bin/env python3
"""
Mesh Memory aggregator.

- Appends the latest validate-only report into a persistent mesh_memory.json.
- Robust discovery of --report:
    * use CLI --report if provided
    * else use $REPORT if set
    * else prefer pilots/validation_logs/skywire_validate_only.json
    * else prefer pilots/validation_logs/fiber_validate_only.json
    * else pick the newest *validate_only*.json in pilots/validation_logs/
- Creates the memory file if missing.

Output format (simple & append-only):
{
  "version": "1.0",
  "runs": [
     {"timestamp_utc": "...", "source": "skywire_validate_only.json", "payload": {...}}
  ]
}
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import glob
import sys


DEFAULT_LOG_DIR = Path("pilots/validation_logs")
PREFERRED_REPORTS = [
    DEFAULT_LOG_DIR / "skywire_validate_only.json",
    DEFAULT_LOG_DIR / "fiber_validate_only.json",
]
DEFAULT_MEMORY = DEFAULT_LOG_DIR / "mesh_memory.json"


def discover_report_path(cli_report: Optional[str]) -> Path:
    # 1) CLI
    if cli_report:
        return Path(cli_report)

    # 2) ENV
    env_report = os.environ.get("REPORT")
    if env_report:
        return Path(env_report)

    # 3) Preferred known filenames
    for p in PREFERRED_REPORTS:
        if p.exists():
            return p

    # 4) Newest *validate_only*.json in log dir
    candidates = sorted(
        (Path(p) for p in glob.glob(str(DEFAULT_LOG_DIR / "*validate_only*.json"))),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    # Otherwise, fall back to a clear error
    return DEFAULT_LOG_DIR / "mesh_memory_report.json"  # non-existent sentinel


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_or_init_memory(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            data = load_json(path)
            if isinstance(data, dict) and "runs" in data and isinstance(data["runs"], list):
                return data
        except Exception:
            pass
    return {"version": "1.0", "runs": []}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, help="Path to validate-only JSON report")
    parser.add_argument("--memory", type=str, default=str(DEFAULT_MEMORY),
                        help="Path to mesh memory JSON (will be created if absent)")
    args = parser.parse_args()

    report_path = discover_report_path(args.report)
    if not report_path.exists():
        # Helpful diagnostics
        print(f"[MeshMemory] Expected report not found: {report_path}", file=sys.stderr)
        if DEFAULT_LOG_DIR.exists():
            print("[MeshMemory] Available files in pilots/validation_logs:", file=sys.stderr)
            for p in sorted(DEFAULT_LOG_DIR.glob("*.json")):
                print(f" - {p}", file=sys.stderr)
        raise FileNotFoundError(f"Invalid or missing report file: {report_path}")

    memory_path = Path(args.memory)

    payload = load_json(report_path)
    memory = load_or_init_memory(memory_path)

    # Lightweight envelope
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": report_path.name,
        "payload": payload,
    }
    memory["runs"].append(entry)
    save_json(memory_path, memory)

    print(f"[MeshMemory] Appended {report_path} → {memory_path} (total runs: {len(memory['runs'])})")


if __name__ == "__main__":
    main()
