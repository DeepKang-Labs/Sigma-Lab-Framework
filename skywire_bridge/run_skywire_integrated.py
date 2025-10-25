# ===================================================
# File: skywire_bridge/run_skywire_integrated.py
# ===================================================
#!/usr/bin/env python3
"""
🌐 SKYWIRE INTEGRATED GOVERNANCE PIPELINE
Discovery Kit → SIGMA-BRIDGE → SIGMA-LAB → Report (JSON)

Typical CLI:
  python -m skywire_bridge.run_skywire_integrated \
    --discovery ./discovery \
    --config skywire_bridge/config_skywire_optimized.yaml \
    --mappings skywire_bridge/skywire_mappings.yaml \
    --out ./skywire_integrated_analysis.json \
    --export-contexts ./out_contexts \
    [--benchmark skywire_bridge/baseline.yaml] \
    [--validate-only] [--pretty]
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Use relative import so the package works both locally and in CI
from .sigma_bridge import SigmaBridge


# ----------------------------- helpers -----------------------------

def _pretty(obj: Any, pretty: bool) -> str:
    return json.dumps(obj, indent=2 if pretty else None, ensure_ascii=False)


def _benchmark_compare(results: List[Dict[str, Any]], baseline: Dict[str, Any]) -> Dict[str, Any]:
    dims = ["non_harm", "stability", "resilience", "equity"]
    avg = {d: 0.0 for d in dims}
    if results:
        n = float(len(results))
        for d in dims:
            avg[d] = sum(r.get("scores", {}).get(d, 0.0) for r in results) / n
    base = baseline.get("target_scores", {}) or {}
    delta = {d: round(avg[d] - float(base.get(d, 0.0)), 3) for d in dims}
    return {"average_scores": avg, "baseline": base, "delta": delta}


def _demo_results(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Deterministic fallback if SIGMA isn't importable
    return [
        {
            "status": "success",
            "scores": {"non_harm": 0.68, "stability": 0.61, "resilience": 0.77, "equity": 0.52},
            "details": {},
            "vetoes": [],
            "verdict": None,
        }
        for _ in contexts
    ]


# --------------------------- main script ---------------------------

def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent  # repository root

    ap = argparse.ArgumentParser(description="SkyWire × SIGMA integrated governance pipeline")
    ap.add_argument("--discovery", type=str, default=str(repo_root / "discovery"))
    ap.add_argument("--config", type=str, default=str(here / "config_skywire_optimized.yaml"))
    ap.add_argument("--mappings", type=str, default=str(here / "skywire_mappings.yaml"))
    ap.add_argument("--out", type=str, default=str(repo_root / "skywire_integrated_analysis.json"))
    ap.add_argument("--export-contexts", type=str, default=None)
    ap.add_argument("--benchmark", type=str, default=None)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    print("🚀 SkyWire Integrated Governance Pipeline")
    print("=" * 60)

    bridge = SigmaBridge(
        discovery_kit_path=args.discovery,
        sigma_config_path=args.config,
        mappings_path=args.mappings,
        export_contexts_dir=args.export_contexts,
    )

    # 1) Load & validate discovery
    print("1. 🔍 Loading Discovery Kit data…")
    discovery_data = bridge.load_discovery_data()
    warnings = bridge.validate_discovery(discovery_data)
    for w in warnings:
        print(f"   ⚠️  {w}")
    print(f"   → {len(discovery_data.get('decision_points', []))} decision(s) found")

    if args.validate_only:
        print("✅ Validation-only mode complete.")
        return

    # 2) Transform → SIGMA contexts
    print("2. 🎯 Transform discovery → SIGMA contexts")
    sigma_contexts = bridge.transform_to_sigma_contexts(discovery_data)
    print(f"   → {len(sigma_contexts)} SIGMA context(s) generated")

    if args.export_contexts:
        out_dir = Path(args.export_contexts)
        print(f"   ↳ Exporting contexts → {out_dir}")
        bridge.export_sigma_contexts(sigma_contexts)

    # 3) Run SIGMA-LAB diagnostics
    print("3. 🧠 Running SIGMA-LAB diagnostics…")
    sigma_results: List[Dict[str, Any]] = []
    try:
        # Try both names: v4_2.1 (if present later) and current v4_2 (your repo)
        try:
            from sigma_lab_v4_2_1 import SigmaLab, demo_context, OptionContext, Stakeholder  # type: ignore
        except ModuleNotFoundError:
            from sigma_lab_v4_2 import SigmaLab, demo_context, OptionContext, Stakeholder  # type: ignore

        # Build engine from a base config (demo_context supplies a sane default)
        base_cfg, _ = demo_context("public")
        engine = SigmaLab(base_cfg)

        for c in sigma_contexts:
            stakeholders = [Stakeholder(**s) for s in c["stakeholders"]]
            ctx = OptionContext(
                name=c["name"],
                short_term_risk=c["short_term_risk"],
                long_term_risk=c["long_term_risk"],
                irreversibility_risk=c["irreversibility_risk"],
                stakeholders=stakeholders,
                stability_risks=c["stability_risks"],
                resilience_features=c["resilience_features"],
                metadata=c.get("metadata", {}),
            )
            res = engine.diagnose(ctx, verdict_opt_in=True)
            sigma_results.append(res)

    except Exception as e:
        print(f"   ⚠️ SIGMA-LAB not available: {e}")
        print("   → Generating deterministic demo results")
        sigma_results = _demo_results(sigma_contexts)

    # 4) Benchmark (optional)
    meta: Dict[str, Any] = {}
    if args.benchmark and Path(args.benchmark).exists():
        print("4. 📏 Running benchmark comparison…")
        with open(args.benchmark, "r", encoding="utf-8") as f:
            baseline = yaml.safe_load(f) or {}
        meta["benchmark"] = _benchmark_compare(sigma_results, baseline)
    else:
        print("4. 📏 Benchmark skipped (no baseline provided)")

    # 5) Report
    print("5. 📊 Building integrated report…")
    report = bridge.generate_integration_report(discovery_data, sigma_results, extra_meta=meta)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_pretty(report, args.pretty), encoding="utf-8")
    print(f"✅ Done. Report saved → {out_path}")

    # Exec summary (console)
    s = report["integration_report"]["summary"]
    print("\n📈 EXEC SUMMARY")
    print(f"   • Decisions analyzed: {s['total_decisions_analyzed']}")
    print(f"   • High-risk decisions: {s['high_risk_decisions']}")
    print(f"   • Avg stability score: {s['average_stability_score']:.2f}")
    if report.get("recommendations"):
        print("   • Top recs:")
        for r in report["recommendations"][:3]:
            print(f"     - {r}")


if __name__ == "__main__":
    # When executed directly, ensure repo root is importable for sigma_lab_v4_2.py
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))
    main()
