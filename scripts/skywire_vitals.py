#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skywire VitalSigns Agent v3.3.3
DeepKang Labs (AI Kang & Yuri Kang)
-------------------------------------------------
Collects Skywire public infrastructure and Fiber metrics,
and outputs daily diagnostic summaries for Sigma-Lab.
-------------------------------------------------
"""

import os
import sys
import json
import datetime
import requests
from typing import List, Dict, Any

# === CONFIGURATION ENVIRONNEMENTALE ===

def getenv_list(name: str) -> List[str]:
    val = os.getenv(name, "")
    if not val:
        return []
    return [v.strip() for v in val.split(",") if v.strip()]

SKYWIRE_ENDPOINTS = getenv_list("SKYWIRE_ENDPOINTS")
SKYWIRE_PUBLIC_ENDPOINTS = getenv_list("SKYWIRE_PUBLIC_ENDPOINTS")
SKYWIRE_NODE_ENDPOINTS = getenv_list("SKYWIRE_NODE_ENDPOINTS")
FIBER_ENDPOINTS = getenv_list("FIBER_ENDPOINTS")

UT_MAX_VISORS = int(os.getenv("UT_MAX_VISORS", "50"))
UT_SAMPLE_MODE = os.getenv("UT_SAMPLE_MODE", "random")

# === HTTP FETCH UTILITAIRE ===

def fetch_json(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Exécute une requête HTTP GET sécurisée et renvoie le JSON brut."""
    try:
        r = requests.get(url, timeout=timeout)
        ok = r.status_code == 200
        try:
            data = r.json() if ok else None
        except Exception:
            data = None
        return {
            "__url__": url,
            "__status__": r.status_code,
            "__ok__": ok,
            "data": data
        }
    except Exception as e:
        return {
            "__url__": url,
            "__error__": str(e),
            "__ok__": False
        }

# === PARSEURS DE BLOCS DE DONNÉES ===

def parse_explorer(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyse les données de l'explorer Skycoin."""
    result = {
        "height": None,
        "current_supply": None,
        "total_supply": None,
        "coin_hours": None
    }

    for p in payloads:
        if not p.get("__ok__"):
            continue
        data = p.get("data") or {}
        if "current_suplly" in data or "current_supply" in data:
            result["current_supply"] = data.get("current_supply") or data.get("current_suplly")
            result["total_supply"] = data.get("total_supply")
            result["coin_hours"] = data.get("current_coinhour_supply")
        elif "blockchain" in p["__url__"]:
            result["height"] = data.get("head_seq")
    return result


def parse_public_infra(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyse les endpoints publics de Skywire."""
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
        data = p.get("data") or {}

        # Service Discovery
        if url.startswith("https://sd.skycoin.com/api/services"):
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("services") or data.get("data") or []
            else:
                items = []

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

        # AR / TPD / DMSG
        elif any(x in url for x in ["ar.skywire", "tpd.skywire", "dmsg"]):
            if isinstance(data, list):
                transports += len(data)
            elif isinstance(data, dict):
                transports += len(data.get("transports", [])) + len(data.get("entries", []))

        # RF
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
        "rf_status": f"{rf_ok} (ok={rf_total})"
    }


def parse_nodes(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyse les noeuds individuels, si fournis."""
    seen = 0
    ok = 0
    latencies = []
    for p in payloads:
        seen += 1
        if p.get("__ok__"):
            ok += 1
            data = p.get("data") or {}
            if isinstance(data, dict):
                latency = data.get("latency", None)
                if isinstance(latency, (int, float)):
                    latencies.append(latency)
    avg_lat = sum(latencies) / len(latencies) if latencies else None
    uptime = round((ok / seen) * 100, 2) if seen else None
    return {
        "nodes_seen": seen,
        "nodes_ok": ok,
        "latency_avg": avg_lat,
        "uptime_avg": uptime
    }


def parse_fiber(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyse les endpoints Fiber s'ils sont actifs."""
    ok = 0
    total = 0
    for p in payloads:
        total += 1
        if p.get("__ok__"):
            ok += 1
    return {
        "fiber_ok": ok,
        "fiber_total": total
    }

# === AGRÉGATION GLOBALE ===

def main() -> int:
    date_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    all_payloads = {}

    # 1. Explorer
    explorer_payloads = [fetch_json(u) for u in SKYWIRE_ENDPOINTS]
    all_payloads["explorer"] = explorer_payloads
    explorer = parse_explorer(explorer_payloads)

    # 2. Public infra
    public_payloads = [fetch_json(u) for u in SKYWIRE_PUBLIC_ENDPOINTS]
    all_payloads["public"] = public_payloads
    public = parse_public_infra(public_payloads)

    # 3. Nodes
    node_payloads = [fetch_json(u) for u in SKYWIRE_NODE_ENDPOINTS]
    nodes = parse_nodes(node_payloads)

    # 4. Fiber
    fiber_payloads = [fetch_json(u) for u in FIBER_ENDPOINTS]
    fiber = parse_fiber(fiber_payloads)

    # === GÉNÉRATION RAPPORT ===
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
"""

    output_dir = f"data/{date_utc}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/skywire_summary.md", "w", encoding="utf-8") as f:
        f.write(summary.strip() + "\n")

    with open(f"{output_dir}/skywire_vitals.json", "w", encoding="utf-8") as f:
        json.dump({
            "date_utc": date_utc,
            "meta": {"repo": "DeepKang-Labs/Sigma-Lab-Framework", "agent": "SkywireVitalSigns v3.3.3"},
            "groups": all_payloads
        }, f, indent=2)

    print("✅ Skywire VitalSigns snapshot generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
