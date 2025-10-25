#!/usr/bin/env python3
# tools/multi_evaluator.py
# Ethical Divergence Index (IDE) + dataset summary.
# License: MIT — DeepKang Labs

import argparse, statistics, pathlib, yaml, sys
from typing import Dict, List

INDEX = pathlib.Path("corpus/index.yaml")

def load_corpus() -> List[Dict]:
    if not INDEX.exists():
        sys.exit("index.yaml not found at corpus/index.yaml")
    idx = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    cases = []
    root = INDEX.parent
    for c in idx.get("cases", []):
        y = root / c["path"]
        cases.append(yaml.safe_load(y.read_text(encoding="utf-8")))
    return cases

def parse_evals(s: str) -> List[float]:
    """
    Parse --evals "optimist=0.55,neutral=0.70,pessimist=0.80"
    Returns [0.55, 0.70, 0.80]
    """
    out = []
    for kv in s.split(","):
        if "=" in kv:
            _, v = kv.split("=", 1)
            out.append(float(v))
        else:
            out.append(float(kv))
    return out

def ide(values: List[float]) -> Dict[str, float]:
    """
    IDE = max |v - mean|
    Returns dict with mean and ide.
    """
    m = statistics.mean(values)
    return {
        "mean": m,
        "ide": max(abs(v - m) for v in values),
        "count": len(values),
    }

def dataset_summary(cases: List[Dict]) -> Dict[str, float]:
    br = [c["parameters"]["base_risk"] for c in cases]
    irr = [c["parameters"]["irreversibility"] for c in cases]
    gini = [c["parameters"]["equity_gini"] for c in cases]
    return {
        "cases": len(cases),
        "base_risk_mean": statistics.mean(br),
        "base_risk_stdev": statistics.pstdev(br),
        "irreversibility_mean": statistics.mean(irr),
        "irreversibility_stdev": statistics.pstdev(irr),
        "equity_gini_mean": statistics.mean(gini),
        "equity_gini_stdev": statistics.pstdev(gini),
    }

def main():
    ap = argparse.ArgumentParser(description="Ethical Divergence Index or dataset summary.")
    ap.add_argument("--evals", help='Comma list, e.g. "optimist=0.55,neutral=0.70,pessimist=0.80"')
    ap.add_argument("--threshold", type=float, default=0.25, help="Alert threshold for IDE")
    ap.add_argument("--summary", action="store_true", help="Show dataset (corpus) summary")
    args = ap.parse_args()

    if args.summary:
        db = load_corpus()
        s = dataset_summary(db)
        print("# Corpus summary")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
        return

    if not args.evals:
        sys.exit("Provide --evals or use --summary")

    vals = parse_evals(args.evals)
    res = ide(vals)
    status = "⚠️ DIVERGENCE" if res["ide"] > args.threshold else "✓ consensus"
    print(f"Evals={vals} | mean={res['mean']:.2f} | IDE={res['ide']:.2f} (threshold {args.threshold:.2f}) → {status}")

if __name__ == "__main__":
    main()
