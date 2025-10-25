#!/usr/bin/env python3
# ============================================================
# Runner — Network Bridge (Generic)
# ============================================================

import os, sys, json, argparse, yaml
from network_bridge.network_bridge import NetworkBridge

def _benchmark_compare(results, baseline):
    dims = ["non_harm", "stability", "resilience", "equity"]
    avg = {d: 0.0 for d in dims}
    if results:
        for d in dims:
            avg[d] = sum(r.get("scores", {}).get(d, 0) for r in results) / len(results)
    base = baseline.get("target_scores", {})
    delta = {d: round(avg[d] - float(base.get(d, 0.0)), 3) for d in dims}
    return {"average_scores": avg, "baseline": base, "delta": delta}

def _demo_results(contexts):
    out = []
    for _ in contexts:
        out.append({
            "scores": {"non_harm": 0.68, "stability": 0.61, "resilience": 0.77, "equity": 0.52},
            "vetoes": [],
            "verdict": None
        })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", choices=["skywire","fiber"], default="skywire")
    ap.add_argument("--discovery", type=str, default="./discovery")
    ap.add_argument("--config", type=str, default="./sigma_config_placeholder.yaml")
    ap.add_argument("--mappings", type=str, required=True)
    ap.add_argument("--out", type=str, default="./pilots/validation_logs/network_validate_only.json")
    ap.add_argument("--export-contexts", type=str, default=None)
    ap.add_argument("--benchmark", type=str, default=None)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--formula-eval", choices=["auto","linear","simple"], default="auto")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    print(f"🚀 Network Bridge ({args.network}) — validate_only={args.validate_only} mode")
    print("=" * 60)

    bridge = NetworkBridge(
        discovery_kit_path=args.discovery,
        sigma_config_path=args.config,
        mappings_path=args.mappings,
        export_contexts_dir=args.export_contexts,
        network_name=args.network,
        formula_eval_mode=args.formula_eval,
    )

    # 1) Load & validate discovery
    print("1) Loading discovery data…")
    discovery_data = bridge.load_discovery_data()
    warnings = bridge.validate_discovery(discovery_data)
    for w in warnings:
        print(f"   ⚠️  {w}")
    print(f"   → {len(discovery_data.get('decision_points', []))} decision(s) found")

    if args.validate_only:
        contexts = bridge.transform_to_sigma_contexts(discovery_data)
        print(f"   → {len(contexts)} SIGMA context(s) generated (validate-only).")
        report = bridge.generate_integration_report(discovery_data, _demo_results(contexts), extra_meta=None)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2 if args.pretty else None, ensure_ascii=False)
        print(f"✅ Validate-only report saved → {args.out}")
        return

    # 2) Transform → SIGMA contexts
    print("2) Transform discovery → SIGMA contexts")
    sigma_contexts = bridge.transform_to_sigma_contexts(discovery_data)
    print(f"   → {len(sigma_contexts)} context(s) generated")

    if args.export_contexts:
        print(f"   ↳ Exporting contexts → {args.export_contexts}")
        bridge.export_sigma_contexts(sigma_contexts)

    # 3) Run Sigma-Lab (try, else demo)
    print("3) Running Sigma-Lab diagnostics…")
    sigma_results = []
    try:
        # Mono-file core (present in repo)
        from sigma_lab_v4_2 import SigmaLab, demo_context, OptionContext, Stakeholder
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
                metadata=c.get("metadata", {})
            )
            res = engine.diagnose(ctx, verdict_opt_in=True)
            sigma_results.append(res)

    except Exception as e:
        print(f"   ⚠️ Sigma-Lab not available: {e}")
        print("   → Generating deterministic demo results")
        sigma_results = _demo_results(sigma_contexts)

    # 4) Benchmark (optional)
    meta = {}
    if args.benchmark and os.path.exists(args.benchmark):
        print("4) Running benchmark comparison…")
        with open(args.benchmark, "r", encoding="utf-8") as f:
            baseline = yaml.safe_load(f) or {}
        meta["benchmark"] = _benchmark_compare(sigma_results, baseline)
    else:
        print("4) Benchmark skipped (no baseline provided)")

    # 5) Report
    print("5) Building integration report…")
    report = bridge.generate_integration_report(discovery_data, sigma_results, extra_meta=meta)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2 if args.pretty else None, ensure_ascii=False)
    print(f"✅ Done. Report saved → {args.out}")

if __name__ == "__main__":
    main()
