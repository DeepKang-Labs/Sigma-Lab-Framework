#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skywire VitalSigns — snapshot agrégé (Explorer + Public Infra + Nodes + Fiber)

- Parse tolérant de l'explorer (coinSupply + metadata/height si dispo)
- Comptage public: visor/proxy/vpn/transports/dmsg + statut RF
- Découverte de visors:
    1) sd.skycoin.com/api/services?type=visor
    2) FALLBACK: extraction des PK depuis les adresses proxy/vpn (03...:port)
- Uptime Tracker agrégé (UT): seen/ok, latence moyenne, uptime moyen
- Sorties:
  - data/YYYY-MM-DD/skywire_vitals.json (données brutes)
  - data/YYYY-MM-DD/skywire_summary.md (résumé lisible)
"""
from __future__ import annotations

import json, os, sys, datetime as dt
from typing import Any, Dict, List, Tuple
import requests
import random

# ---------- util ----------
def now_utc_date() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")

def safe_get(d: Dict[str, Any], *path, default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def fetch(url: str, timeout: int = 15) -> Tuple[bool, Dict[str, Any]]:
    try:
        r = requests.get(url, timeout=timeout)
        ok = 200 <= r.status_code < 300
        try:
            data = r.json()
        except Exception:
            data = r.text
        return ok, {"__url__": url, "__ok__": ok, "__status__": r.status_code, "__headers__": dict(r.headers), "data": data}
    except requests.RequestException as e:
        return False, {"__url__": url, "__ok__": False, "__status__": None, "__error__": repr(e)}

def to_list_from_csv(env_val: str) -> List[str]:
    if not env_val:
        return []
    return [x for x in env_val.split(",") if x]

def write_json(path: str, payload: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def write_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ---------- parsing: explorer ----------
def parse_explorer(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    height = None
    cur_supply = None
    tot_supply = None
    coin_hours = None
    note = None

    # coinSupply
    for p in payloads:
        if not p.get("__ok__"):
            continue
        data = p.get("data")
        if isinstance(data, dict) and "current_supply" in data:
            cur_supply = data.get("current_supply")
            tot_supply = data.get("total_supply") or data.get("max_supply")
            coin_hours = data.get("current_coinhour_supply") or data.get("total_coinhour_supply") or coin_hours

    # height via metadata (schémas multiples possibles)
    for p in payloads:
        if not p.get("__ok__"):
            continue
        data = p.get("data")
        if isinstance(data, dict):
            cand = (
                safe_get(data, "blockchain", "metadata", "head")
                or safe_get(data, "metadata", "head")
                or safe_get(data, "head")
                or safe_get(data, "height")
                or safe_get(data, "block", "seq")
            )
            if isinstance(cand, (int, float, str)):
                height = cand
                break

    if height is None and cur_supply is None and tot_supply is None and coin_hours is None:
        note = "Explorer fields were empty — schema mismatch or temporary outage."

    return {"height": height, "current_supply": cur_supply, "total_supply": tot_supply, "coin_hours": coin_hours, "note": note}

# ---------- parsing: public infra ----------
def _as_services(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("services"), list):
        return [x for x in data["services"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []

def count_services(items: List[Dict[str, Any]], svc_type: str) -> int:
    svc_type = svc_type.lower()
    return sum(1 for it in items if isinstance(it.get("type"), str) and it["type"].lower() == svc_type)

def parse_public(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    services: List[Dict[str, Any]] = []
    dmsg_entries = 0
    transports = 0
    rf_ok = "ok=0"

    for p in payloads:
        url = p.get("__url__", "")
        if not p.get("__ok__"):
            if url.endswith("rf.skywire.skycoin.com/"):
                rf_ok = "404 (ok=0)"
            continue

        data = p.get("data")
        services.extend(_as_services(data))
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            dmsg_entries += len(data["entries"])
        if url.endswith("/all-transports") and isinstance(data, list):
            transports += len(data)

    return {
        "visors": count_services(services, "visor"),
        "proxies": count_services(services, "skysocks"),
        "vpn": count_services(services, "vpn"),
        "transports": transports,
        "dmsg_entries": dmsg_entries,
        "rf_status": rf_ok,
        "samples": {
            "proxy": [s for s in services if (s.get("type") or "").lower() == "skysocks"][:5],
            "vpn":   [s for s in services if (s.get("type") or "").lower() == "vpn"][:5],
            "visor": [s for s in services if (s.get("type") or "").lower() == "visor"][:5],
        },
        "all_services": services,  # pour fallback PK
    }

# ---------- découverte de PKs ----------
def extract_pk_from_addr(addr: str) -> str | None:
    """Adresse format '03abcdef...:PORT' -> '03abcdef...' """
    if not isinstance(addr, str):
        return None
    if ":" in addr:
        addr = addr.split(":", 1)[0]
    # PKs skywire font 66+ chars hex (commencent souvent par 02/03)
    return addr if len(addr) >= 66 else None

def derive_pks_from_services(services: List[Dict[str, Any]], max_n: int, mode: str) -> List[str]:
    pks: List[str] = []
    for it in services:
        addr = it.get("address") or it.get("addr") or it.get("pk")
        pk = extract_pk_from_addr(addr) if isinstance(addr, str) else None
        if pk and pk not in pks:
            pks.append(pk)
    if not pks:
        return []
    if mode == "random":
        random.shuffle(pks)
    return pks[:max_n]

def discover_visors_from_sd(sd_endpoints: List[str], max_n: int, mode: str) -> List[str]:
    pks: List[str] = []
    for ep in sd_endpoints:
        if "type=visor" not in ep:
            continue
        ok, payload = fetch(ep)
        if not ok:
            continue
        services = _as_services(payload.get("data"))
        pks.extend(derive_pks_from_services(services, max_n, mode))
        pks = list(dict.fromkeys(pks))  # dédupe
        if len(pks) >= max_n:
            break
    return pks[:max_n]

# ---------- UT ----------
def query_ut_for_visors(pks: List[str]) -> Dict[str, Any]:
    if not pks:
        return {"nodes_seen_ok": (0, 0), "latency_avg_ms": None, "uptime_ratio_avg": None, "ut_note": "No PK provided."}

    base = "https://ut.skywire.skycoin.com/uptimes?v=v2&visors="
    ok_cnt = 0
    total = 0
    lat_sum = 0.0
    lat_n = 0
    up_sum = 0.0
    up_n = 0

    for i in range(0, len(pks), 25):
        chunk = pks[i:i+25]
        ok, payload = fetch(base + ";".join(chunk))
        total += len(chunk)
        data = payload.get("data")
        if not ok or not isinstance(data, dict):
            continue
        for pk, stats in data.items():
            if not isinstance(stats, dict):
                continue
            up = stats.get("uptime_ratio")
            lat = stats.get("latency_ms")
            if isinstance(up, (int, float)):
                up_sum += float(up)
                up_n += 1
                if up > 0:
                    ok_cnt += 1
            if isinstance(lat, (int, float)):
                lat_sum += float(lat)
                lat_n += 1

    lat_avg = (lat_sum / lat_n) if lat_n else None
    up_avg  = (up_sum / up_n) if up_n else None
    return {"nodes_seen_ok": (ok_cnt, total), "latency_avg_ms": lat_avg, "uptime_ratio_avg": up_avg, "ut_note": None}

# ---------- main ----------
def main() -> int:
    date_utc = now_utc_date()
    out_dir = os.path.join("data", date_utc)
    os.makedirs(out_dir, exist_ok=True)

    env_explorer = to_list_from_csv(os.getenv("SKYWIRE_ENDPOINTS", ""))
    env_public   = to_list_from_csv(os.getenv("SKYWIRE_PUBLIC_ENDPOINTS", ""))
    env_nodes    = to_list_from_csv(os.getenv("SKYWIRE_NODE_ENDPOINTS", ""))
    env_fiber    = to_list_from_csv(os.getenv("FIBER_ENDPOINTS", ""))
    visors_pks   = to_list_from_csv(os.getenv("VISORS_PKS", ""))
    visors_auto  = os.getenv("VISORS_PKS_AUTO", "false").lower() == "true"
    ut_max       = int(os.getenv("UT_MAX_VISORS", "100") or "100")  # on élargit à 100
    ut_mode      = os.getenv("UT_SAMPLE_MODE", "random")

    # Explorer
    explorer_payloads = [fetch(ep)[1] for ep in env_explorer]
    explorer_summary  = parse_explorer(explorer_payloads)

    # Public
    public_payloads = [fetch(ep)[1] for ep in env_public]
    public_summary  = parse_public(public_payloads)

    # Nodes (PKs)
    if visors_auto and not visors_pks:
        # 1) essayer sd?type=visor
        visors_pks = discover_visors_from_sd(env_public, ut_max, ut_mode)

    if visors_auto and not visors_pks:
        # 2) FALLBACK: extraire depuis proxy/vpn
        services_all = public_summary.get("all_services", [])
        visors_pks = derive_pks_from_services(services_all, ut_max, ut_mode)

    nodes_summary = query_ut_for_visors(visors_pks)

    # Fiber
    fiber_payloads = [fetch(ep)[1] for ep in env_fiber]
    fiber_seen = sum(1 for p in fiber_payloads if p.get("__ok__"))
    fiber_summary = {"endpoints_seen_ok": (len(fiber_payloads), fiber_seen), "height": None, "peers": None}

    # Assemble
    bundle = {
        "date_utc": date_utc,
        "meta": {"repo": "DeepKang-Labs/Sigma-Lab-Framework", "agent": "SkywireVitalSigns v3.5.0"},
        "groups": [
            {"name": "explorer", "endpoints": env_explorer, "payloads": explorer_payloads, "summary": explorer_summary},
            {"name": "public",   "endpoints": env_public,   "payloads": public_payloads,   "summary": public_summary},
            {"name": "nodes",    "endpoints": env_nodes,    "payloads": [],                 "summary": nodes_summary, "visors_pks": visors_pks},
            {"name": "fiber",    "endpoints": env_fiber,    "payloads": fiber_payloads,     "summary": fiber_summary},
        ],
    }

    # Write
    json_path = os.path.join(out_dir, "skywire_vitals.json")
    write_json(json_path, bundle)

    md = []
    md += [f"# Skywire VitalSigns — {date_utc} UTC\n",
           "## Explorer\n",
           f"- Height: {explorer_summary['height']}\n",
           f"- Current supply: {explorer_summary['current_supply']}\n",
           f"- Total supply: {explorer_summary['total_supply']}\n",
           f"- Coin Hours: {explorer_summary['coin_hours']}\n",]
    if explorer_summary.get("note"):
        md.append(f"- Note: {explorer_summary['note']}\n")

    md += ["\n## Public Infra (Skywire)\n",
           f"- Visors: {public_summary['visors']}\n",
           f"- Proxies: {public_summary['proxies']}\n",
           f"- VPN: {public_summary['vpn']}\n",
           f"- Transports: {public_summary['transports']}\n",
           f"- DMSG entries: {public_summary['dmsg_entries']}\n",
           f"- RF status: {public_summary['rf_status']}\n"]

    md += ["\n## Nodes (if provided)\n"]
    ok_seen = nodes_summary["nodes_seen_ok"]
    md += [f"- Nodes seen/ok: {ok_seen[0]}/{ok_seen[1]}\n",
           f"- Latency avg (ms): {nodes_summary['latency_avg_ms']}\n",
           f"- Uptime ratio avg: {nodes_summary['uptime_ratio_avg']}\n"]
    if nodes_summary.get("ut_note"):
        md.append(f"- UT note: {nodes_summary['ut_note']}\n")

    md += ["\n## Fiber (if provided)\n",
           f"- Fiber endpoints seen/ok: {fiber_summary['endpoints_seen_ok'][0]}/{fiber_summary['endpoints_seen_ok'][1]}\n",
           f"- Fiber height: {fiber_summary['height']} — peers: {fiber_summary['peers']}\n"]

    md_path = os.path.join(out_dir, "skywire_summary.md")
    write_text(md_path, "".join(md))
    print(f"Wrote: {json_path} & {md_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
