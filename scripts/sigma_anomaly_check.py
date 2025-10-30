#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sigma Anomaly Check — Skywire → Sigma quick diagnostic
- Lit le JSON des vitals
- Calcule quelques scores (stability / resilience / equity / non_harm)
- Produit : JSON d’analyse, badge SVG, résumé Markdown
- Ne casse pas le job par défaut (peut échouer si --fail-on-critical)
"""

from __future__ import annotations
import argparse, json, os, sys, math, datetime as dt
from pathlib import Path
from typing import Any, Dict, Tuple

def load_json(p: str) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def percentile(values, p):
    if not values:
        return None
    v = sorted(values)
    k = (len(v)-1) * (p/100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c: return v[int(k)]
    return v[f] + (v[c]-v[f])*(k-f)

def clamp(x, lo=0.0, hi=1.0): 
    return max(lo, min(hi, x))

def compute_metrics(vitals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attendu minimal (flexible) :
    {
      "payloads":[
        {"visor":"...", "ok": true, "latency_ms": 42, "uptime_ratio": 0.997, "peers": 12, ...},
        ...
      ]
    }
    """
    payloads = vitals.get("payloads") or vitals.get("vitals") or []
    if isinstance(payloads, dict):  # parfois nommé par clé
        payloads = list(payloads.values())

    latencies = [p.get("latency_ms") for p in payloads if isinstance(p.get("latency_ms"), (int,float))]
    uptimes  = [p.get("uptime_ratio") for p in payloads if isinstance(p.get("uptime_ratio"), (int,float))]
    peers    = [p.get("peers") for p in payloads if isinstance(p.get("peers"), (int,float))]

    n_total = len(payloads)
    n_ok    = sum(1 for p in payloads if p.get("ok") is True)
    success_ratio = (n_ok / n_total) if n_total else 0.0

    p50_lat = percentile(latencies, 50) or None
    p95_lat = percentile(latencies, 95) or None
    avg_upt = sum(uptimes)/len(uptimes) if uptimes else None
    avg_peers = sum(peers)/len(peers) if peers else None

    # Scoring simple et explicite (0..1)
    # - Stability : latence faible et peu de queue p95
    if p95_lat is None: stability = 0.0
    else:
        # 0 ms -> 1.0 ; 500 ms -> ~0 ; au-delà clamp 0
        stability = clamp(1.0 - (p95_lat/500.0))

    # - Resilience : uptime moyen et proportion d’OK
    if avg_upt is None: resilience = success_ratio
    else:
        resilience = clamp(0.5*avg_upt + 0.5*success_ratio)

    # - Equity : dispersion des peering (plus il y a de pairs, mieux c’est)
    if avg_peers is None: equity = 0.5  # neutre
    else:
        equity = clamp((avg_peers/16.0))  # 16 pairs ≈ 1.0

    # - Non-harm : combine latence raisonnable et peu d’échecs
    if p50_lat is None: non_harm = success_ratio
    else:
        non_harm = clamp(0.6*success_ratio + 0.4*clamp(1.0 - p50_lat/400.0))

    return {
        "counts": {"total": n_total, "ok": n_ok},
        "latency_ms": {"p50": p50_lat, "p95": p95_lat},
        "uptime_avg": avg_upt,
        "peers_avg": avg_peers,
        "scores": {
            "stability": round(stability, 3),
            "resilience": round(resilience, 3),
            "equity": round(equity, 3),
            "non_harm": round(non_harm, 3),
        },
        "success_ratio": round(success_ratio, 3),
    }

def verdict_from_scores(scores: Dict[str, float]) -> Tuple[str, str]:
    """
    Renvoie (status, reason)
    status ∈ {OK, WARN, CRITICAL}
    """
    s = scores
    min_score = min(s.values()) if s else 0.0
    if min_score >= 0.8:
        return "OK", "Healthy network vitals."
    if min_score >= 0.55:
        return "WARN", "Degradation detected on at least one dimension."
    return "CRITICAL", "Multiple weak dimensions — attention required."

def badge_svg(status: str) -> str:
    color = {"OK":"#2ea043", "WARN":"#d29922", "CRITICAL":"#d73a49"}.get(status, "#6a737d")
    label = "Sigma"
    text  = status
    # Simple badge GitHub-like
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="130" height="20" role="img" aria-label="{label}: {text}">
  <linearGradient id="b" x2="0" y2="100%"><stop offset="0" stop-color="#fff" stop-opacity=".7"/><stop offset=".1" stop-opacity=".1"/></linearGradient>
  <mask id="a"><rect width="130" height="20" rx="3" fill="#fff"/></mask>
  <g mask="url(#a)">
    <rect width="65" height="20" fill="#555"/>
    <rect x="65" width="65" height="20" fill="{color}"/>
    <rect width="130" height="20" fill="url(#b)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="33" y="14">{label}</text>
    <text x="97" y="14">{text}</text>
  </g>
</svg>"""

def write_text(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input vitals JSON")
    ap.add_argument("--out-json", required=True, help="output analysis JSON")
    ap.add_argument("--badge", required=True, help="output badge SVG")
    ap.add_argument("--summary", required=True, help="output summary MD")
    ap.add_argument("--fail-on-critical", action="store_true", default=False)
    args = ap.parse_args()

    try:
        vitals = load_json(args.inp)
    except Exception as e:
        print(f"[SigmaCheck] Unable to read input: {e}", file=sys.stderr)
        # on produit quand même un CRITICAL “no data”
        analysis = {
            "timestamp": dt.datetime.utcnow().isoformat()+"Z",
            "error": f"read_input_failed: {e}",
            "scores": {"stability":0.0,"resilience":0.0,"equity":0.0,"non_harm":0.0},
            "verdict": {"status":"CRITICAL","reason":"No input data"},
        }
        write_text(Path(args.out_json), json.dumps(analysis, ensure_ascii=False, indent=2))
        write_text(Path(args.badge), badge_svg("CRITICAL"))
        write_text(Path(args.summary), "# Sigma Integration Summary\n\n- Status: **CRITICAL** (no input data)\n")
        return 2 if args.fail_on-critical else 0

    metrics = compute_metrics(vitals)
    status, reason = verdict_from_scores(metrics["scores"])
    analysis = {
        "timestamp": dt.datetime.utcnow().isoformat()+"Z",
        "metrics": metrics,
        "verdict": {"status": status, "reason": reason}
    }

    # Sorties
    write_text(Path(args.out_json), json.dumps(analysis, ensure_ascii=False, indent=2))
    write_text(Path(args.badge), badge_svg(status))

    md = [
        "# Sigma Integration Summary",
        "",
        f"- **Status**: `{status}` — {reason}",
        "- **Scores**:",
        f"  - stability: **{metrics['scores']['stability']}**",
        f"  - resilience: **{metrics['scores']['resilience']}**",
        f"  - equity: **{metrics['scores']['equity']}**",
        f"  - non_harm: **{metrics['scores']['non_harm']}**",
        "",
        "- **Signal**:",
        f"  - success_ratio: **{metrics['success_ratio']}**",
        f"  - p50 latency (ms): **{metrics['latency_ms']['p50']}**",
        f"  - p95 latency (ms): **{metrics['latency_ms']['p95']}**",
        f"  - avg uptime: **{metrics['uptime_avg']}**",
        f"  - avg peers: **{metrics['peers_avg']}**",
        ""
    ]
    write_text(Path(args.summary), "\n".join(md))

    if status == "CRITICAL" and args.fail_on_critical:
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
