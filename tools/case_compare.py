#!/usr/bin/env python3
# tools/case_compare.py
# Compare one historical case to others in the corpus.
# Reads corpus/index.yaml and per-case YAML files.
# License: MIT — DeepKang Labs

import argparse, math, statistics, pathlib, yaml, sys
from typing import Dict, List, Tuple

INDEX = pathlib.Path("corpus/index.yaml")

def load_corpus() -> List[Dict]:
    if not INDEX.exists():
        sys.exit("index.yaml not found at corpus/index.yaml")
    idx = yaml.safe_load(INDEX.read_text(encoding="utf-8"))
    cases = []
    root = INDEX.parent
    for c in idx.get("cases", []):
        y = root / c["path"]
        data = yaml.safe_load(y.read_text(encoding="utf-8"))
        # ensure id present
        data.setdefault("id", c["id"])
        cases.append(data)
    return cases

def vec(params: Dict) -> List[float]:
    """Vector used for similarity: [base_risk, irreversibility, equity_gini]"""
    return [
        float(params.get("base_risk", 0.0)),
        float(params.get("irreversibility", 0.0)),
        float(params.get("equity_gini", 0.0)),
    ]

def euclid(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def cosine(a: List[float], b: List[float]) -> float:
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)

def main():
    ap = argparse.ArgumentParser(description="Compare one case to others in the corpus.")
    ap.add_argument("--case-id", required=True, help="ID from corpus/index.yaml")
    ap.add_argument("--top", type=int, default=3, help="How many similar cases to show")
    ap.add_argument("--metric", choices=["euclid", "cosine"], default="euclid",
                    help="Similarity metric (cosine higher=more similar; euclid lower=more similar)")
    args = ap.parse_args()

    db = load_corpus()
    by_id = {c["id"]: c for c in db}
    if args.case_id not in by_id:
        sys.exit(f"Case '{args.case_id}' not found in index.")

    cur = by_id[args.case_id]
    v_cur = vec(cur["parameters"])

    scored: List[Tuple[str, str, float]] = []
    for c in db:
        if c["id"] == cur["id"]:
            continue
        score = cosine(v_cur, vec(c["parameters"])) if args.metric == "cosine" \
            else euclid(v_cur, vec(c["parameters"]))
        scored.append((c["id"], c.get("name", c["id"]), score))

    reverse = True if args.metric == "cosine" else False
    scored.sort(key=lambda t: t[2], reverse=reverse)
    topk = scored[: args.top]

    # Aggregate check on base_risk for neighbors
    neighbors = [by_id[i]["parameters"]["base_risk"] for i, _, _ in topk]
    mean_risk = statistics.mean(neighbors) if neighbors else float("nan")
    this_risk = cur["parameters"]["base_risk"]
    delta = this_risk - mean_risk
    warn = "OK"
    if abs(delta) >= 0.15:
        warn = "⚠️ deviation ≥ 0.15 vs similar mean"

    print(f"# Case: {cur.get('name','')} ({cur['id']}) — domain: {cur.get('domain','?')}")
    print(f"Metric: {args.metric}")
    print("\nTop similar cases:")
    for cid, name, score in topk:
        if args.metric == "cosine":
            print(f"  • {name} [{cid}] — cos={score:.3f}")
        else:
            print(f"  • {name} [{cid}] — euclid={score:.3f}")

    print("\n# Calibration check on base_risk")
    if neighbors:
        print(f"  mean(similar base_risk)={mean_risk:.2f} | this={this_risk:.2f} | delta={delta:+.2f} → {warn}")
    else:
        print("  (no neighbors)")

if __name__ == "__main__":
    main()
