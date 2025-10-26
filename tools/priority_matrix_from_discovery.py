#!/usr/bin/env python3
"""
priority_matrix_from_mappings.py
--------------------------------
Reads a YAML mapping file (e.g. network_bridge/mappings_skywire.yaml)
and exports a flat CSV governance priority matrix.

This helps quantify and compare decision points using metrics such as:
- technical_complexity
- economic_impact
- user_experience_impact
- security_implications
- current_pain_level

Usage:
    python -m tools.priority_matrix_from_mappings \
        --mappings ./network_bridge/mappings_skywire.yaml \
        --out ./pilots/validation_logs/priority_matrix.csv
"""

import argparse
import csv
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a governance priority matrix from a mappings YAML file."
    )
    parser.add_argument(
        "--mappings", required=True,
        help="Path to the mappings YAML file (e.g. network_bridge/mappings_skywire.yaml)"
    )
    parser.add_argument(
        "--out", default="./priority_matrix.csv",
        help="Output CSV file path (default: ./priority_matrix.csv)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    yaml_path = Path(args.mappings)
    out_path = Path(args.out)

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not yaml_path.exists():
        print(f"[WARN] Mappings file not found: {yaml_path}")
        return 1

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse YAML: {e}")
        return 2

    # Support both top-level list and discovery-like structure
    entries = data if isinstance(data, list) else data.get("decision_points", [])
    if not entries:
        print(f"[WARN] No decision points found in {yaml_path}")
        return 0

    # Define CSV schema
    fields = [
        "decision_id",
        "description",
        "technical_complexity",
        "economic_impact",
        "user_experience_impact",
        "security_implications",
        "current_pain_level",
    ]

    # Write CSV
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        for entry in entries:
            gd = entry.get("governance_dimensions", {})
            writer.writerow({
                "decision_id": entry.get("decision_id", ""),
                "description": entry.get("description", ""),
                "technical_complexity": gd.get("technical_complexity", ""),
                "economic_impact": gd.get("economic_impact", ""),
                "user_experience_impact": gd.get("user_experience_impact", ""),
                "security_implications": gd.get("security_implications", ""),
                "current_pain_level": gd.get("current_pain_level", ""),
            })

    print(f"[OK] Priority matrix written → {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
