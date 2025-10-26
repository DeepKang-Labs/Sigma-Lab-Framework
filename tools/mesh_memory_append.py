#!/usr/bin/env python3
# Mesh Memory append tool (JSONL)
# Appends a compact summary line from an integration report.
# Usage:
#   python tools/mesh_memory_append.py --report pilots/validation_logs/skywire_validate_only.json \
#       --memory pilots/mesh_memory/mesh_memory.jsonl

import os
import json
import argparse
from datetime import datetime, timezone

def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to integration report JSON")
    ap.add_argument("--memory", required=False, default="pilots/mesh_memory/mesh_memory.jsonl",
                    help="Path to JSONL memory file")
    args = ap.parse_args()

    report = _read_json(args.report)

    # Extract core fields
    network = _safe_get(report, "integration_report", "network", default="unknown")
    timestamp_report = _safe_get(report, "integration_report", "timestamp", default=None)
    summary = _safe_get(report, "integration_report", "summary", default={}) or {}
    total = summary.get("total_decisions_analyzed", 0)
    high_risk = summary.get("high_risk_decisions", 0)
    avg_stability = summary.get("average_stability_score", None)

    # Try to extract average dimension scores if available
    # (depends on the engine output)
    sigma_results = report.get("sigma_analysis", []) or []
    dims = ["non_harm", "stability", "resilience", "equity"]
    avg_scores = {}
    if sigma_results:
        for d in dims:
            avg_scores[d] = round(sum(r.get("scores", {}).get(d, 0.0) for r in sigma_results) / len(sigma_results), 4)

    # Hashes for traceability
    dig = _safe_get(report, "integration_report", "inputs_digest", default={}) or {}
    discovery_sha = dig.get("discovery_sha256", "")
    mappings_sha = dig.get("mappings_sha256", "")
    sigma_config_sha = dig.get("sigma_config_sha256", "")

    # Build entry
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_timestamp": timestamp_report,
        "network": network,
        "total_decisions": total,
        "high_risk_decisions": high_risk,
        "average_stability": avg_stability,
        "average_scores": avg_scores,
        "inputs_digest": {
            "discovery_sha256": discovery_sha,
            "mappings_sha256": mappings_sha,
            "sigma_config_sha256": sigma_config_sha
        },
        "ci": {
            "github_sha": os.environ.get("GITHUB_SHA", None),
            "github_run_id": os.environ.get("GITHUB_RUN_ID", None)
        },
        "source_report": os.path.relpath(args.report)
    }

    # Append JSON line
    os.makedirs(os.path.dirname(args.memory), exist_ok=True)
    with open(args.memory, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Appended mesh memory entry → {args.memory}")

if __name__ == "__main__":
    main()
