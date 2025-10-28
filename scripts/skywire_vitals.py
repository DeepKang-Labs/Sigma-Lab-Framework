#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skywire VitalSigns agent (v3.3.3)
- Lit des variables d'environnement multi-lignes (1 URL par ligne)
- Tolère les lignes mal formées de type 'KEY=https://...'
- Récupère Explorer, Infra publique (SD/TPD/DMSG/AR/RF), Nodes optionnels, Fiber optionnel
- Écrit:
    data/YYYY-MM-DD/skywire_vitals.json
    data/YYYY-MM-DD/skywire_summary.md
"""

import os, sys, json, time
from typing import List, Dict, Any

import requests


# -------------------------- utils --------------------------

def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _split_env_lines(name: str) -> List[str]:
    """
    Read a multiline env var and return clean URL list (no KEY= prefixes).
    """
    raw = os.getenv(name, "")
    lines: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # tolerate 'KEY=https://...' mistakes
        if "://" in line and "=" in line and line.index("://") > line.index("="):
            line = line.split("=", 1)[1].strip()
        lines.append(line)
    return lines

def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def fetch(url: str, timeout: int = 15) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"__url__": url, "__ok__": False}
    if not _is_url(url):
        entry["__error__"] = "Invalid URL"
        return entry
    try:
        r = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
        entry["__status__"] = r.status_code
        entry["__ok__"] = r.ok
        entry["__headers__"] = dict(r.headers)
        if r.ok:
            try:
                entry["data"] = r.json()
            except Exception:
                entry["text"] = r.text
        else:
            entry["error_text"] = r.text[:500]
    except Exception as e:
        entry["__error__"] = f"{type(e).__name__}: {e}"
    return entry


# -------------------------- parsers --------------------------

def parse_explorer(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"height": None, "current_supply": None, "total_supply": None, "coin_hours": None, "notes": []}
    for p in payloads:
        if not p.get("__ok__"):
            continue
        data = p.get("data", {})
        if "current_supply" in data:
            out["current_supply"] = data.get("current_supply")
            out["total_supply"] = data.get("total_supply")
            out["coin_hours"] = data.get("current_coinhour_supply")
        # certains endpoints exposent la hauteur sous d'autres clés
        for k in ("height", "block_height", "seq"):
            if data.get(k) is not None:
                out["height"] = data[k]
                break
    if not any(v for k, v in out.items() if k != "notes"):
        out["notes"].append("Explorer fields were empty — schema mismatch or temporary outage.")
    return out

def parse_public_infra(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    visors = 0
    proxies = 0
    vpn = 0
    transports = 0
    dmsg_entries = 0
    rf_ok = 0
    rf_total = 0

    for p in payloads:
        if not p.get("__ok__"):
            continue
        url = p["__url__"]
        data = p.get("data", {})

        # Service Discovery
        if url.startswith("https://sd.skycoin.com/api/services"):
            items = data if isinstance(data, list) else data.get("services") or data.get("data") or []
            for it in items:
                t = (it.get("type") or "").lower()
                if t == "visor":
                    visors += 1
                elif t in ("socks", "skysocks", "proxy"):
                    proxies += 1
                elif t == "vpn":
                    vpn += 1

        # AR / TPD / DMSG — counts approximatifs
        elif "ar.skywire" in url or "tpd.skywire" in url or "dmsg" in url:
            if isinstance(data, list):
                transports += len(data)
            elif isinstance(data, dict):
                transports += len(data.get("transports", [])) + len(data.get("entries", []))

        # RF status
        elif "rf.skywire" in url:
            rf_total += 1
            if p.get("__status__") == 200 and p.get("__ok__"):
                rf_ok += 1

    return {
        "visors": visors,
        "proxies": proxies,
        "vpn": vpn,
        "transports": transports,
        "dmsg_entries": dmsg_entries,
        "rf_status": f"{p.get('__status__', 'n/a')} (ok={rf_ok})"
    }

def parse_nodes(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    seen = 0
    ok = 0
    latencies = []
    for p in payloads:
        if not p.get("__url__") or "<your-node" in p["__url__"]:
            continue
        seen += 1
        if p.get("__ok__"):
            ok += 1
            d = p.get("data", {})
            for k in ("latency_ms", "latency", "avg_latency_ms"):
                if d.get(k) is not None:
                    try:
                        latencies.append(float(d[k]))
                    except Exception:
                        pass
    avg_latency = None if not latencies else round(sum(latencies) / len(latencies), 2)
    return {
        "nodes_seen_ok": f"{ok}/{seen}",
        "latency_avg_ms": avg_latency,
        "uptime_ratio_avg": None  # sera rempli si on ajoute la collecte UT par PK
    }

def parse_fiber(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    seen = 0
    ok = 0
    height = None
    peers = None
    for p in payloads:
        if not p.get("__ok__"):
            continue
        seen += 1
        ok += 1
        d = p.get("data", {})
        height = d.get("height") or height
        peers = d.get("peers") or peers
    return {
        "fiber_endpoints_seen_ok": f"{ok}/{seen}",
        "fiber_height": height,
        "peers": peers
    }


# -------------------------- main --------------------------

def main() -> int:
    out_dir = os.path.join("data", time.strftime("%Y-%m-%d", time.gmtime()))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "skywire_vitals.json")
    md_path   = os.path.join(out_dir, "skywire_summary.md")

    groups = {
        "explorer": _split_env_lines("SKYWIRE_ENDPOINTS"),
        "public":   _split_env_lines("SKYWIRE_PUBLIC_ENDPOINTS"),
        "nodes":    _split_env_lines("SKYWIRE_NODE_ENDPOINTS"),
        "fiber":    _split_env_lines("FIBER_ENDPOINTS"),
    }

    all_payloads: Dict[str, List[Dict[str, Any]]] = {}
    for gname, urls in groups.items():
        payloads: List[Dict[str, Any]] = []
        for u in urls:
            payloads.append(fetch(u))
        all_payloads[gname] = payloads

    # parse
    explorer = parse_explorer(all_payloads.get("explorer", []))
    public   = parse_public_infra(all_payloads.get("public", []))
    nodes    = parse_nodes(all_payloads.get("nodes", []))
    fiber    = parse_fiber(all_payloads.get("fiber", []))

    # dump JSON
    blob = {
        "date_utc": now_utc_iso(),
        "meta": {"repo": "DeepKang-Labs/Sigma-Lab-Framework", "agent": "SkywireVitalSigns v3.3.3"},
        "groups": [
            {"name": "explorer", "endpoints": groups["explorer"], "payloads": all_payloads["explorer"]},
            {"name": "public",   "endpoints": groups["public"],   "payloads": all_payloads["public"]},
            {"name": "nodes",    "endpoints": groups["nodes"],    "payloads": all_payloads["nodes"]},
            {"name": "fiber",    "endpoints": groups["fiber"],    "payloads": all_payloads["fiber"]},
        ],
        "summary": {
            "explorer": explorer, "public": public, "nodes": nodes, "fiber": fiber
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=2)

    # write MD
    def line(s=""): return s + "\n"
    md: List[str] = []
    md += [line(f"# Skywire VitalSigns — {time.strftime('%Y-%m-%d UTC', time.gmtime())}"), line()]

    md += [line("## Explorer")]
    md += [line(f"- Height: {explorer['height'] or 'None'}")]
    md += [line(f"- Current supply: {explorer['current_supply'] or 'None'}")]
    md += [line(f"- Total supply: {explorer['total_supply'] or 'None'}")]
    md += [line(f"- Coin Hours: {explorer['coin_hours'] or 'None'}")]
    if explorer.get("notes"):
        md += [line(f"> _Note_: {explorer['notes'][0]}")]
    md += [line()]

    md += [line("## Public Infra (Skywire)")]
    md += [line(f"- Visors: {public['visors']}")]
    md += [line(f"- Proxies: {public['proxies']}")]
    md += [line(f"- VPN: {public['vpn']}")]
    md += [line(f"- Transports: {public['transports']}")]
    md += [line(f"- DMSG entries: {public['dmsg_entries']}")]
    md += [line(f"- RF status: {public['rf_status']}")]
    md += [line()]

    md += [line("## Nodes (if provided)")]
    md += [line(f"- Nodes seen/ok: {nodes['nodes_seen_ok']}")]
    md += [line(f"- Latency avg (ms): {nodes['latency_avg_ms'] or 'None'}")]
    md += [line(f"- Uptime ratio avg: {nodes['uptime_ratio_avg'] or 'None'}")]
    md += [line()]

    md += [line("## Uptime Tracker (UT)")]
    md += [line(f"- UT average (%): {'None'} (no PK available)")]
    md += [line()]

    md += [line("## Fiber (if provided)")]
    md += [line(f"- Fiber endpoints seen/ok: {fiber['fiber_endpoints_seen_ok']}")]
    md += [line(f"- Fiber height: {fiber['fiber_height'] or 'None'} — peers: {fiber['peers'] or 'None'}")]
    md += [line()]

    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(md)

    print(f"[ok] wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
