#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skywire → Sigma integration runner
- Loads a Skywire vitals JSON (either via --input or by auto-discovery)
- Computes Sigma metrics using engine.SigmaAnalyzer
- Writes:
    - JSON result: reports/integrations/skywire_sigma_analysis_YYYY-MM-DD.json
    - Markdown summary: reports/integrations/skywire_sigma_analysis_YYYY-MM-DD.md (if --also-md)
    - Claude-style Integration Test Report: reports/integrations/Integration_Test_Report_YYYY-MM-DD.md (if --claude-report)
- Prints a short status line to stdout.
"""

from __future__ import annotations
import argparse
import json
import sys
import yaml
from pathlib import Path
from datetime import datetime, timezone

# Local import
try:
    from engine.core import SigmaAnalyzer
except Exception as e:
    print(f"[ERROR] Unable to import SigmaAnalyzer: {e}", file=sys.stderr)
    sys.exit(2)


def _find_latest_vitals(discovery_root: Path) -> Path | None:
    candidates: list[Path] = []
    for p in discovery_root.rglob("skywire_vitals.json"):
        candidates.append(p)
    if not candidates:
        return None

    def date_from_path(p: Path):
        try:
            return datetime.strptime(p.parent.name, "%Y-%m-%d")
        except Exception:
            return None

    dated = [(p, date_from_path(p)) for p in candidates]
    dated = [x for x in dated if x[1] is not None]
    if dated:
        dated.sort(key=lambda x: x[1])
        return dated[-1][0]

    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _write_json(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)


def _write_markdown(md_path: Path, result: dict, source: Path) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    comp = result.get("component_scores", {})
    advice = result.get("advice", [])
    verdict = result.get("verdict", "Unknown")
    score = result.get("overall_score", 0.0)
    timestamp = result.get("timestamp_utc", "")
    lines = []
    lines.append("# Skywire → Sigma Analysis\n")
    lines.append(f"**Timestamp (UTC):** {timestamp}\n")
    lines.append(f"**Source:** `{source.as_posix()}`\n")
    lines.append('**Scope:** local file analysis; live transport is not established.\n')
    lines.append(f"**Verdict:** **{verdict}**\n")
    lines.append(f"**Overall Score:** **{score:.2f} / 100**\n")
    lines.append("\n## Component Scores (0..1)\n")
    for k in ("stability", "latency", "resilience", "equity"):
        if k in comp:
            lines.append(f"- `{k}`: **{comp[k]:.3f}**")
    lines.append("\n## Advice\n")
    if advice:
        for a in advice:
            lines.append(f"- {a}")
    else:
        lines.append("- (no advice)")
    lines.append("")
    md = "\n".join(lines)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md)


def _write_claude_report(md_path: Path, date_str: str, source: Path, duration_s: float,
                         result: dict, metrics_preview: dict) -> None:
    """
    Generate the exact “Integration Test Report” format Claude requested.
    """
    md_path.parent.mkdir(parents=True, exist_ok=True)
    verdict = result.get("verdict", "Unknown")
    overall = result.get("overall_score", 0.0)
    lines = []
    lines.append("## Integration Test Report\n")
    lines.append(f"**Date:** {date_str}\n")
    lines.append("**Test:** Skywire vitals → Sigma-Lab diagnostic\n")
    lines.append("\n### Input")
    lines.append(f"- Source: `{source.as_posix()}`")
    # Show 3–5 key metrics extracted (preview)
    if metrics_preview:
        lines.append(f"- Metrics: {json.dumps(metrics_preview, indent=2)}")
    else:
        lines.append("- Metrics: {}")
    lines.append("\n### Processing")
    lines.append("- Command: `python network_bridge/run_network_integrated.py --network skywire ...`")
    lines.append(f"- Duration: ~{int(duration_s)} seconds")
    lines.append("- Status: SUCCESS for local processing only; live transport was not tested.")
    lines.append("\n### Output")
    out_json_name = f"reports/integrations/skywire_sigma_analysis_{date_str}.json"
    lines.append(f"- File: `{out_json_name}`")
    comp = result.get("component_scores", {})
    lines.append(f"- Scores: {json.dumps(comp, indent=2)}")
    lines.append(f"- Verdict: **{verdict}** (overall **{overall:.2f} / 100**)")
    lines.append("\n### Learnings")
    lines.append("- What worked: Integration pipeline executed end-to-end (data → analyzer → report).")
    lines.append("- What needs improvement: Expand feature extraction; tune weights and latency normalization with real node data.")
    lines.append("")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    import time
    parser = argparse.ArgumentParser(description="Run SigmaAnalyzer on Skywire vitals and emit reports.")
    parser.add_argument("--network", type=str, required=True, help="Network name (use 'skywire').")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Path to a specific vitals JSON file.")
    group.add_argument("--discovery", type=str, help="Folder to search for latest skywire_vitals.json (e.g., data/).")
    parser.add_argument("--out", type=str, default="", help="Optional output JSON path.")
    parser.add_argument("--also-md", action="store_true", help="Also writes a concise Markdown summary.")
    parser.add_argument("--claude-report", action="store_true", help="Writes Integration Test Report in Claude's format.")
    parser.add_argument('--validate-only', action='store_true', help='Validate local mappings/discovery; no live network transport.')
    parser.add_argument('--mappings', help='YAML mapping file for local discovery validation.')
    parser.add_argument('--config', help='Sigma configuration file for local discovery validation.')
    parser.add_argument('--formula-eval', choices=['auto','linear','simple'], default='auto')
    args = parser.parse_args()

    if args.validate_only:
        if not args.discovery or not args.mappings or not args.config or args.input:
            parser.error('--validate-only requires --discovery, --mappings and --config')
        if args.network.lower() not in ['skywire','fiber']:
            parser.error('Local mapping validation supports skywire and fiber')
        try:
            from network_bridge.network_bridge import NetworkBridge
            if not Path(args.config).is_file():
                raise ValueError('Configuration file does not exist: '+args.config)
            from sigma_lab_v4_2 import SigmaLab, OptionContext
            with Path(args.config).open(encoding='utf-8') as config_file:
                config = yaml.safe_load(config_file)
            if not isinstance(config,dict):
                raise ValueError('Configuration must be a YAML object')
            diagnostic_engine = SigmaLab(config)
            bridge = NetworkBridge(args.discovery,args.config,args.mappings,
                network_name=args.network.lower(),formula_eval_mode=args.formula_eval)
            discovery = bridge.load_discovery_data()
            warnings = bridge.validate_discovery(discovery)
            contexts = [] if warnings else bridge.transform_to_sigma_contexts(discovery)
            if not contexts and not warnings:
                raise ValueError('No decision contexts were produced')
            report = {'mode':'local-mapping-validation','network':args.network.lower(),
                'live_transport_established':False,'contexts':contexts,'warnings':warnings,
                'diagnostics':[diagnostic_engine.diagnose(OptionContext(**{key:value for key,value in context.items()
                    if key in OptionContext.__dataclass_fields__})) for context in contexts],
                'discovery_source':'file' if (Path(args.discovery)/'decision_mapper.yaml').is_file() else 'demo',
                'status':'invalid_input' if warnings else 'validated'}
            output = Path(args.out) if args.out else Path('pilots/validation_logs')/(args.network.lower()+'_validate_only.json')
            _write_json(output,report)
            print('[LOCAL] Mapping validation saved to '+str(output)+'; no live transport tested.')
            return 2 if warnings else 0
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
            print('[ERROR] Local mapping validation failed: '+str(error),file=sys.stderr)
            return 2

    if args.network.lower() != "skywire":
        print(f"[ERROR] Unsupported network: {args.network}", file=sys.stderr)
        return 2

    # Resolve source
    if args.input:
        source_path = Path(args.input)
        if not source_path.is_file():
            print(f"[ERROR] Input file not found: {source_path}", file=sys.stderr)
            return 2
    else:
        discovery_root = Path(args.discovery)
        if not discovery_root.exists():
            print(f"[ERROR] Discovery root not found: {discovery_root}", file=sys.stderr)
            return 2
        source_path = _find_latest_vitals(discovery_root)
        if source_path is None:
            print(f"[ERROR] No 'skywire_vitals.json' found under {discovery_root}", file=sys.stderr)
            return 2

    # Analyze
    analyzer = SigmaAnalyzer()
    t0 = time.monotonic()
    try:
        result = analyzer.evaluate_from_file(source_path)
        result.update({'scope':'local-vitals-file-analysis','live_transport_established':False,
                       'source':source_path.as_posix()})
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}", file=sys.stderr)
        return 1
    duration = time.monotonic() - t0

    # Decide outputs
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if args.out:
        out_json = Path(args.out)
    else:
        out_json = Path("reports/integrations") / f"skywire_sigma_analysis_{today}.json"

    _write_json(out_json, result)

    if args.also_md:
        md_path = out_json.with_suffix(".md")
        _write_markdown(md_path, result, source_path)

    if args.claude_report:
        # Minimal preview of metrics used by the analyzer (recomputed from file)
        try:
            with source_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            payloads = raw.get("payloads", [])
            n = len(payloads)
            def get(p, k, d=0.0):
                try:
                    return float(p.get(k, d))
                except Exception:
                    return d
            avg_uptime = sum(get(p, "uptime", 0.0) for p in payloads) / max(n, 1)
            avg_latency = sum(get(p, "latency_ms", 0.0) for p in payloads) / max(n, 1)
            success_ratio = sum(get(p, "success_ratio", 0.0) for p in payloads) / max(n, 1)
            preview = {
                "node_count": n,
                "avg_uptime": round(avg_uptime, 4),
                "avg_latency_ms": round(avg_latency, 2),
                "success_ratio": round(success_ratio, 4),
            }
        except Exception:
            preview = {}
        cr_md = Path("reports/integrations") / f"Integration_Test_Report_{today}.md"
        _write_claude_report(cr_md, today, source_path, duration, result, preview)

    print(
        f"[OK] Local Skywire-data analysis done | verdict={result.get('verdict')} "
        f"| overall={result.get('overall_score')} | json={out_json.as_posix()} "
        f"| took={int(duration)}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
