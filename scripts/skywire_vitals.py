#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skywire VitalSigns Agent v3.4.0
DeepKang Labs (AI Kang & Yuri Kang)

Robuste aux ENV manquantes (defaults intégrés) + debug des endpoints réellement utilisés.
Génère:
- data/YYYY-MM-DD/skywire_summary.md
- data/YYYY-MM-DD/skywire_vitals.json   (payloads + urls)
- data/YYYY-MM-DD/debug_env.json         (ce que le runner a vu côté ENV)
"""

import os, sys, json, datetime, requests
from typing import List, Dict, Any

# --------------------------- Defaults de secours ----------------------------
DEFAULT_SKYWIRE_ENDPOINTS = [
    "https://explorer.skycoin.com/api/blockchain/metadata",
    "https://explorer.skycoin.com/api/coinSupply",
]
DEFAULT_SKYWIRE_PUBLIC_ENDPOINTS = [
    "https://sd.skycoin.com/api/services?type=visor",
    "https://sd.skycoin.com/api/services?type=proxy",
    "https://sd.skycoin.com/api/services?type=vpn",
    "https://ar.skywire.skycoin.com/transports",
    "https://tpd.skywire.skycoin.com/all-transports",
    "https://dmsgd.skywire.skycoin.com/dmsg-discovery/entries",
    "https://rf.skywire.skycoin.com/",
]
DEFAULT_NODE_ENDPOINTS = []  # vide par défaut
DEFAULT_FIBER_ENDPOINTS = [
    "https://fiber.skywire.dev/api/status"
]

# --------------------------- Helpers ENV -----------------------------------
def getenv_list(name: str) -> List[str]:
    raw = os.getenv(name, "")
    if not raw:
        return []
    # On accepte séparateur virgule ET retours à la ligne
    parts = []
    for line in raw.splitlines():
        for seg in line.split(","):
            s = seg.strip()
            if s:
                parts.append(s)
    return parts

def pick_or_default(vals: List[str], defaults: List[str]) -> List[str]:
    return vals if vals else defaults

# --------------------------- Lecture ENV -----------------------------------
SKYWIRE_ENDPOINTS = pick_or_default(getenv_list("SKYWIRE_ENDPOINTS"), DEFAULT_SKYWIRE_ENDPOINTS)
SKYWIRE_PUBLIC_ENDPOINTS = pick_or_default(getenv_list("SKYWIRE_PUBLIC_ENDPOINTS"), DEFAULT_SKYWIRE_PUBLIC_ENDPOINTS)
SKYWIRE_NODE_ENDPOINTS = pick_or_default(getenv_list("SKYWIRE_NODE_ENDPOINTS"), DEFAULT_NODE_ENDPOINTS)
FIBER_ENDPOINTS = pick_or_default(getenv_list("FIBER_ENDPOINTS"), DEFAULT_FIBER_ENDPOINTS)

UT_MAX_VISORS = int(os.getenv("UT_MAX_VISORS", "50"))
UT_SAMPLE_MODE = os.getenv("UT_SAMPLE_MODE", "random")

# --------------------------- HTTP util -------------------------------------
def fetch_json(url: str, timeout: int = 15) -> Dict[str, Any]:
    try:
        r = requests.get(url, timeout=timeout)
        ok = (r.status_code == 200)
        try:
            data = r.json() if ok else None
        except Exception:
            data = None
        return {"__url__": url, "__status__": r.status_code, "__ok__": ok, "data": data}
    except Exception as e:
        return {"__url__": url, "__error__": str(e), "__ok__": False}

# --------------------------- Parsers ---------------------------------------
def parse_explorer(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"height": None, "current_supply": None, "total_supply": None, "coin_hours": None}
    for p in payloads:
        if not p.get("__ok__"):
            continue
        url = p.get("__url__", "")
        data = p.get("data") or {}

        # coinSupply endpoint
        if "coinSupply" in url:
            # Normalise les champs
            out["current_supply"] = data.get("current_supply") or data.get("current_suplly")
            out["total_supply"] = data.get("total_supply") or data.get("max_supply")
            out["coin_hours"] = data.get("current_coinhour_supply") or data.get("total_coinhour_supply")
        # blockchain/metadata
        if "blockchain/metadata" in url:
            # différents schémas rencontrés: head_seq / current / height
            out["height"] = data.get("head_seq") or data.get("current") or data.get("height") or out["height"]
    return out

def parse_public_infra(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    visors = proxies = vpn = transports = dmsg_entries = 0
    rf_ok = rf_total = 0

    for p in payloads:
        if not p.get("__ok__"):
            continue
        url = p.get("__url__", "")
        data = p.get("data") or {}

        # Service Discovery lists
        if url.startswith("https://sd.skycoin.com/api/services"):
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("services") or data.get("data") or []
            for it in items:
                if not isinstance(it, dict):
                    continue
                t = (it.get("type") or "").lower()
                if t == "visor":
                    visors += 1
                elif t in ("socks", "skysocks", "proxy"):
                    proxies += 1
                elif t == "vpn":
                    vpn += 1

        # AR / TPD transports & DMSG entries
        elif "ar.skywire" in url or "tpd.skywire" in url:
            if isinstance(data, list):
                transports += len(data)
            elif isinstance(data, dict):
                transports += len(data.get("transports", []))
        elif "dmsg-discovery/entries" in url:
            if isinstance(data, list):
                dmsg_entries += len(data)
            elif isinstance(data, dict):
                dmsg_entries += len(data.get("entries", []))

        # RF health
        elif "rf.skywire" in url:
            rf_total += 1
            if p.get("__status__") == 200:
                rf_ok += 1

    return {
        "visors": visors,
        "proxies": proxies,
        "vpn": vpn,
        "transports": transports,
        "dmsg_entries": dmsg_entries,
        "rf_status": f"{rf_ok} (ok={rf_total})"
    }

def parse_nodes(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    seen = ok = 0
    latencies = []
    for p in payloads:
        seen += 1
        if p.get("__ok__"):
            ok += 1
            data = p.get("data") or {}
            lat = data.get("latency")
            if isinstance(lat, (int, float)):
                latencies.append(lat)
    avg = round(sum(latencies)/len(latencies), 2) if latencies else None
    uptime = round((ok/seen)*100, 2) if seen else None
    return {"nodes_seen": seen, "nodes_ok": ok, "latency_avg": avg, "uptime_avg": uptime}

def parse_fiber(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(payloads)
    ok = sum(1 for p in payloads if p.get("__ok__"))
    return {"fiber_ok": ok, "fiber_total": total}

# --------------------------- Main ------------------------------------------
def main() -> int:
    date_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d")

    # Debug: enregistrer exactement ce que le runner voit côté ENV
    debug_env = {
        "SKYWIRE_ENDPOINTS": SKYWIRE_ENDPOINTS,
        "SKYWIRE_PUBLIC_ENDPOINTS": SKYWIRE_PUBLIC_ENDPOINTS,
        "SKYWIRE_NODE_ENDPOINTS": SKYWIRE_NODE_ENDPOINTS,
        "FIBER_ENDPOINTS": FIBER_ENDPOINTS,
        "UT_MAX_VISORS": UT_MAX_VISORS,
        "UT_SAMPLE_MODE": UT_SAMPLE_MODE
    }

    # Appels HTTP
    explorer_payloads = [fetch_json(u) for u in SKYWIRE_ENDPOINTS]
    public_payloads   = [fetch_json(u) for u in SKYWIRE_PUBLIC_ENDPOINTS]
    node_payloads     = [fetch_json(u) for u in SKYWIRE_NODE_ENDPOINTS]
    fiber_payloads    = [fetch_json(u) for u in FIBER_ENDPOINTS]

    explorer = parse_explorer(explorer_payloads)
    public   = parse_public_infra(public_payloads)
    nodes    = parse_nodes(node_payloads)
    fiber    = parse_fiber(fiber_payloads)

    # Sortie
    outdir = f"data/{date_utc}"
    os.makedirs(outdir, exist_ok=True)

    summary = f"""
# Skywire VitalSigns — {date_utc} UTC

## Explorer
- Height: {explorer.get("height")}
- Current supply: {explorer.get("current_supply")}
- Total supply: {explorer.get("total_supply")}
- Coin Hours: {explorer.get("coin_hours")}
- Note: Explorer fields were empty — schema mismatch or temporary outage.

## Public Infra (Skywire)
- Visors: {public.get("visors")}
- Proxies: {public.get("proxies")}
- VPN: {public.get("vpn")}
- Transports: {public.get("transports")}
- DMSG entries: {public.get("dmsg_entries")}
- RF status: {public.get("rf_status")}

## Nodes (if provided)
- Nodes seen/ok: {nodes.get("nodes_seen")}/{nodes.get("nodes_ok")}
- Latency avg (ms): {nodes.get("latency_avg")}
- Uptime ratio avg: {nodes.get("uptime_avg")}

## Fiber (if provided)
- Fiber endpoints seen/ok: {fiber.get("fiber_ok")}/{fiber.get("fiber_total")}
""".strip() + "\n"

    with open(f"{outdir}/skywire_summary.md", "w", encoding="utf-8") as f:
        f.write(summary)

    with open(f"{outdir}/skywire_vitals.json", "w", encoding="utf-8") as f:
        json.dump({
            "date_utc": date_utc,
            "meta": {"repo": "DeepKang-Labs/Sigma-Lab-Framework", "agent": "SkywireVitalSigns v3.4.0"},
            "urls": {
                "explorer": SKYWIRE_ENDPOINTS,
                "public": SKYWIRE_PUBLIC_ENDPOINTS,
                "nodes": SKYWIRE_NODE_ENDPOINTS,
                "fiber": FIBER_ENDPOINTS
            },
            "payloads": {
                "explorer": explorer_payloads,
                "public": public_payloads,
                "nodes": node_payloads,
                "fiber": fiber_payloads
            }
        }, f, indent=2)

    with open(f"{outdir}/debug_env.json", "w", encoding="utf-8") as f:
        json.dump(debug_env, f, indent=2)

    print("✅ Snapshot generated. See:", outdir)
    print("Endpoints used:", debug_env)
    return 0

if __name__ == "__main__":
    sys.exit(main())
