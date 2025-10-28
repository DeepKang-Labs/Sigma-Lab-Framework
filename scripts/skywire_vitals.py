#!/usr/bin/env python3
# scripts/skywire_vitals.py

import os, json, sys, datetime, pathlib
import requests
import pandas as pd
try:
    import yaml  # optional local config fallback
except ImportError:
    yaml = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
TODAY = datetime.datetime.utcnow().strftime("%Y-%m-%d")
DATA_DIR = ROOT / "data" / TODAY
DATA_DIR.mkdir(parents=True, exist_ok=True)

def read_csv_env(name: str) -> list[str]:
    v = os.getenv(name, "") or ""
    # split on comma or whitespace; strip quotes/spaces
    parts = [p.strip().strip('"').strip("'") for p in v.replace("\n", ",").split(",")]
    return [p for p in parts if p]

def safe_get(url: str, timeout=15):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}", "__url__": url}

def coalesce(d: dict, keys: list, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default

def flatten_metrics(payloads: list[dict]) -> dict:
    """Heuristics to derive vitals across heterogeneous schemas."""
    metas = []
    errors = 0
    for p in payloads:
        if "__error__" in p:
            errors += 1
            continue
        metas.append(p)

    metrics = {
        "sources_ok": len(payloads) - errors,
        "sources_err": errors,
        "nodes_active_est": None,
        "latency_ms_avg": None,
        "uptime_ratio_avg": None,
        "success_ratio_avg": None,
        "proxy_activity_sum": None,
    }

    # Try common fields found on explorers
    for p in metas:
        # total nodes / peers
        nodes = coalesce(p, ["nodes", "total_nodes", "peer_count", "peers", "ActiveNodes"])
        if isinstance(nodes, (int, float)):
            metrics["nodes_active_est"] = max(metrics["nodes_active_est"] or 0, int(nodes))

        # latency
        lat = coalesce(p, ["latencyMsAvg", "latency", "avg_latency_ms"])
        if isinstance(lat, (int, float)):
            metrics["latency_ms_avg"] = (metrics["latency_ms_avg"] or 0) * 0.5 + float(lat) * 0.5 if metrics["latency_ms_avg"] else float(lat)

        # uptime / success ratios
        up = coalesce(p, ["uptimeRatio", "uptime_ratio", "availability"])
        if isinstance(up, (int, float)):
            metrics["uptime_ratio_avg"] = (metrics["uptime_ratio_avg"] or 0) * 0.5 + float(up) * 0.5 if metrics["uptime_ratio_avg"] else float(up)

        succ = coalesce(p, ["successRatio", "success_ratio", "success"])
        if isinstance(succ, (int, float)):
            metrics["success_ratio_avg"] = (metrics["success_ratio_avg"] or 0) * 0.5 + float(succ) * 0.5 if metrics["success_ratio_avg"] else float(succ)

        # proxy / tx activity
        tx = coalesce(p, ["txCount", "transactions", "proxy_activity", "activity"])
        if isinstance(tx, (int, float)):
            metrics["proxy_activity_sum"] = (metrics["proxy_activity_sum"] or 0) + float(tx)

    return metrics

def collect_group(name: str, urls: list[str]) -> dict:
    payloads = []
    for u in urls:
        payloads.append(safe_get(u))
    vitals = flatten_metrics(payloads)
    return {"name": name, "endpoints": urls, "vitals": vitals, "raw": payloads}

def maybe_load_local_config():
    cfg = ROOT / "scripts" / "skywire.config.yaml"
    if cfg.exists() and yaml:
        with open(cfg, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        return {
            "explorer": [u for u in (y.get("endpoints") or []) if u],
            "nodes": [u for u in (y.get("node_endpoints") or []) if u],
            "fiber": [u for u in (y.get("fiber_endpoints") or []) if u],
        }
    return {"explorer": [], "nodes": [], "fiber": []}

def main():
    explorer = read_csv_env("SKYWIRE_ENDPOINTS")
    nodes = read_csv_env("SKYWIRE_NODE_ENDPOINTS")
    fiber = read_csv_env("FIBER_ENDPOINTS")

    # fallback to local yaml if envs empty
    if not any([explorer, nodes, fiber]):
        local = maybe_load_local_config()
        explorer, nodes, fiber = local["explorer"], local["nodes"], local["fiber"]

    out = {
        "date_utc": TODAY,
        "meta": {
            "repo": "DeepKang-Labs/Sigma-Lab-Framework",
            "agent": "SkywireVitalSigns v2",
        },
        "groups": []
    }

    if explorer:
        out["groups"].append(collect_group("explorer", explorer))
    if nodes:
        out["groups"].append(collect_group("nodes", nodes))
    if fiber:
        out["groups"].append(collect_group("fiber", fiber))

    # derive a top-level merged “best effort”
    merged = {"sources": 0}
    for g in out["groups"]:
        v = g["vitals"]
        merged["sources"] += v.get("sources_ok", 0)
        for k in ["nodes_active_est","latency_ms_avg","uptime_ratio_avg","success_ratio_avg","proxy_activity_sum"]:
            val = v.get(k)
            if val is None: 
                continue
            if k not in merged or merged[k] is None:
                merged[k] = val
            else:
                # average or sum for proxy_activity
                if k == "proxy_activity_sum":
                    merged[k] += val
                else:
                    merged[k] = (merged[k] + val) / 2.0
    out["vitals"] = merged

    # write JSON + short summary
    json_path = DATA_DIR / "skywire_vitals.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    summary = DATA_DIR / "skywire_summary.md"
    vit = out.get("vitals", {})
    with open(summary, "w", encoding="utf-8") as f:
        f.write(f"# Skywire VitalSigns — {TODAY} UTC\n\n")
        f.write(f"- Sources OK: **{vit.get('sources', 0)}**\n")
        f.write(f"- Nodes (est.): **{vit.get('nodes_active_est', 'n/a')}**\n")
        f.write(f"- Latency avg (ms): **{vit.get('latency_ms_avg', 'n/a')}**\n")
        f.write(f"- Uptime ratio avg: **{vit.get('uptime_ratio_avg', 'n/a')}**\n")
        f.write(f"- Success ratio avg: **{vit.get('success_ratio_avg', 'n/a')}**\n")
        f.write(f"- Proxy/tx activity sum: **{vit.get('proxy_activity_sum', 'n/a')}**\n")

    # append datalog
    with open(ROOT / "DATALOG.md", "a", encoding="utf-8") as f:
        f.write(f"{TODAY} : skywire_vitals.json + skywire_summary.md generated\n")

if __name__ == "__main__":
    sys.exit(main())
