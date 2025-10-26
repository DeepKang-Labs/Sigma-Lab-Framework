#!/usr/bin/env python3
# Priority Matrix from Discovery
# Builds a CSV ranking: score = pain * impact * frequency
# - pain: current_pain_level (0..10) or 5 if missing
# - impact: mean(technical_complexity, economic_impact, user_experience_impact) on 0..10
# - frequency: decision_frequency (0..10) or 5 if missing
# All normalized to 0..1 by /10, then multiplied.
# Usage:
#   python tools/priority_matrix_from_discovery.py \
#       --in discovery/decision_mapper.yaml \
#       --out pilots/validation_logs/governance_priority_matrix.csv

import argparse
import csv
import math
import os
import sys
from typing import Any, Dict, List
import yaml

def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_impact(gd: Dict[str, Any]) -> float:
    vals = []
    for k in ["technical_complexity", "economic_impact", "user_experience_impact"]:
        v = gd.get(k, None)
        if isinstance(v, (int, float)):
            vals.append(v)
    if not vals:
        return 0.5
    return sum(vals) / len(vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Discovery YAML path")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV path")
    args = ap.parse_args()

    data = read_yaml(args.inp)
    dps: List[Dict[str, Any]] = data.get("decision_points", []) or []
    rows = []

    for dp in dps:
        did = dp.get("decision_id", "unknown")
        desc = dp.get("description", "")
        gd = dp.get("governance_dimensions", {}) or {}
        pain = dp.get("current_pain_level", 5)
        freq = dp.get("decision_frequency", 5)

        pain_n = clip01(float(pain) / 10.0)
        impact_n = clip01(float(get_impact(gd)) / 10.0)
        freq_n = clip01(float(freq) / 10.0)

        score = round(pain_n * impact_n * freq_n, 6)

        rows.append({
            "decision_id": did,
            "description": desc,
            "pain": pain,
            "impact_avg": round(get_impact(gd), 3),
            "frequency": freq,
            "score": score
        })

    # Sort by score desc
    rows.sort(key=lambda r: r["score"], reverse=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["decision_id","description","pain","impact_avg","frequency","score"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote priority matrix → {args.out}")
    if rows:
        top = rows[0]
        print(f"Top: {top['decision_id']} (score={top['score']})")

if __name__ == "__main__":
    main()
