#!/usr/bin/env python3
# tools/sensitivity.py
# Sensitivity analysis for a given case: vary base_risk/irreversibility by ±pct.
# License: MIT — DeepKang Labs

import argparse, pathlib, yaml, sys
from typing import Dict

INDEX = pathlib.Path("corpus/index.yaml")

def load_case(case_id: str) -> Dict:
    idx = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    root = INDEX.parent
    for c in idx.get("cases", []):
        if c["id"] == case_id:
            y = root / c["path"]
            return yaml.safe_load(y.read_text(encoding="utf-8"))
    sys.exit(f"Case '{case_id}' not found in index.yaml")

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def main():
    ap = argparse.ArgumentParser(description="Sensitivity analysis on base_risk/irreversibility.")
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--pct", type=float, default=0.15, help="percent as fraction (0.15 = ±15%)")
    args = ap.parse_args()

    case = load_case(args.case_id)
    p = dict(case["parameters"])
    base, irr = float(p["base_risk"]), float(p["irreversibility"])

    print(f"# Sensitivity for: {case.get('name','')} ({case['id']})")
    print(f"Base params: base_risk={base:.2f}, irreversibility={irr:.2f}")
    print(f"Delta: ±{int(args.pct*100)}%")
    print("\nVariants:")
    for k in ("base_risk", "irreversibility"):
        v = base if k == "base_risk" else irr
        for sgn, tag in ((-1, "-"), (1, "+")):
            newv = clamp01(v * (1 + sgn * args.pct))
            print(f"  {k} {tag}{int(args.pct*100)}% → {newv:.2f}")

    print("\nNote: This tool reports numeric variations only. "
          "Interpretation (e.g., threshold flips) depends on your decision policy.")

if __name__ == "__main__":
    main()
