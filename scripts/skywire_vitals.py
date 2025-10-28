#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SkywireVitals — Daily Skywire/Skycoin VitalSigns agent
------------------------------------------------------
Collects metrics from:
  • Explorer endpoints (coin supply, blockchain metadata)
  • Skywire public infra (visor/proxy/vpn/discovery + RF probe)
  • Optional visors list (PKs) from .tmp/visors_pks.txt
  • Fiber endpoints (if provided)

Outputs:
  data/{TODAY}/skywire_vitals.json   — full JSON (private artefact)
  data/{TODAY}/skywire_summary.md    — Markdown summary (public commit)
"""

from __future__ import annotations
import os, sys, json, textwrap, random
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import requests  # pip install requests

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def split_env_list(s: str | None) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def http_json(url: str, timeout: float = 15.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta = {"__url__": url, "__ok__": False}
    try:
        r = requests.get(url, timeout=timeout)
        meta["__status__"] = r.status_code
        r.raise_for_status()
        meta["__ok__"] = True
        return meta, r.json()
    except Exception as e:
        meta["__error__"] = str(e)
        return meta, {}

def read_visors_file(path: str, cap: int = 200) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        pks = [ln.strip() for ln in f if ln.strip()]
    if len(pks) > cap:
        random.seed(0xC0FFEE)
        pks = random.sample(pks, cap)
    return pks

# -------------------------------------------------------------
# Collectors
# -------------------------------------------------------------

def collect_explorer(endpoints: List[str]) -> Dict[str, Any]:
    height, cur, total, hours = None, None, None, None
    payloads = []
    for u in endpoints:
        meta, data = http_json(u)
        payloads.append(meta)
        if "coinSupply" in u and meta["__ok__"]:
            cur = data.get("current_supply") or data.get("currentSupply")
            total = data.get("total_supply") or data.get("totalSupply")
            hours = data.get("current_coinhour_supply") or data.get("coinHours")
        if "metadata" in u and meta["__ok__"]:
            chain = data.get("chain") or {}
            height = chain.get("head_seq") or chain.get("height")
    return {
        "name": "explorer",
        "endpoints": endpoints,
        "payloads": payloads,
        "summary": {
            "height": height,
            "current_supply": cur,
            "total_supply": total,
            "coin_hours": hours
        }
    }

def collect_public(endpoints: List[str], limit: int = 5) -> Dict[str, Any]:
    counters = {"visor": 0, "proxy": 0, "vpn": 0, "transports": 0, "dmsg": 0}
    payloads: List[Dict[str, Any]] = []
    samples: Dict[str, Any] = {}
    rf_status = "n/a"

    def sample_list(kind: str, data: Any) -> List[Dict[str, Any]]:
        lst = data if isinstance(data, list) else data.get("data", [])
        out = []
        for it in lst[:limit]:
            pk = it.get("address") or it.get("pk")
            if pk:
                out.append({"address": pk, "type": kind})
        return out

    for u in endpoints:
        meta, data = http_json(u)
        payloads.append(meta)
        if "services?type=visor" in u:
            lst = data if isinstance(data, list) else data.get("data", [])
            counters["visor"] = len(lst)
            samples["visor"] = sample_list("visor", data)
        elif "services?type=proxy" in u:
            lst = data if isinstance(data, list) else data.get("data", [])
            counters["proxy"] = len(lst)
            samples["proxy"] = sample_list("proxy", data)
        elif "services?type=vpn" in u:
            lst = data if isinstance(data, list) else data.get("data", [])
            counters["vpn"] = len(lst)
            samples["vpn"] = sample_list("vpn", data)
        elif "all-transports" in u:
            lst = data if isinstance(data, list) else data.get("data", [])
            counters["transports"] = len(lst)
        elif "dmsg-discovery/entries" in u:
            lst = data if isinstance(data, list) else data.get("entries", [])
            counters["dmsg"] = len(lst)
        elif "rf.skywire" in u:
            rf_status = f"{meta.get('__status__', 0)} (ok={'1' if meta['__ok__'] else '0'})"

    return {
        "name": "public",
        "endpoints": endpoints,
        "payloads": payloads,
        "summary": {
            "visors": counters["visor"],
            "proxies": counters["proxy"],
            "vpn": counters["vpn"],
            "transports": counters["transports"],
            "dmsg_entries": counters["dmsg"],
            "rf_status": rf_status,
            "samples": samples
        }
    }

def collect_nodes(pks: List[str], cap: int) -> Dict[str, Any]:
    total = len(pks)
    note = "No PK provided." if not total else f"{total} PK(s) loaded (passive only)"
    return {
        "name": "nodes",
        "endpoints": [],
        "payloads": [],
        "summary": {"nodes_seen_ok": [0, total], "ut_note": note},
        "visors_pks": pks
    }

def collect_fiber(endpoints: List[str]) -> Dict[str, Any]:
    payloads: List[Dict[str, Any]] = []
    seen = [0, len(endpoints)]
    height, peers = None, None
    for u in endpoints:
        meta, data = http_json(u)
        payloads.append(meta)
        if meta["__ok__"]:
            seen[0] += 1
        if isinstance(data, dict):
            height = data.get("height") or height
            peers = data.get("peers") or peers
    return {
        "name": "fiber",
        "endpoints": endpoints,
        "payloads": payloads,
        "summary": {"endpoints_seen_ok": seen, "height": height, "peers": peers}
    }

# -------------------------------------------------------------
# Writers
# -------------------------------------------------------------

def write_outputs(today: str, doc: Dict[str, Any]) -> None:
    outdir = os.path.join("data", today)
    ensure_dir(outdir)
    raw_path = os.path.join(outdir, "skywire_vitals.json")
    md_path = os.path.join(outdir, "skywire_summary.md")

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    e = next((g for g in doc["groups"] if g["name"] == "explorer"), {}).get("summary", {})
    p = next((g for g in doc["groups"] if g["name"] == "public"), {}).get("summary", {})
    n = next((g for g in doc["groups"] if g["name"] == "nodes"), {}).get("summary", {})
    fsum = next((g for g in doc["groups"] if g["name"] == "fiber"), {}).get("summary", {})

    md = textwrap.dedent(f"""
    # Skywire VitalSigns — {today} UTC

    ## Explorer
    - Height: {e.get('height')}
    - Current supply: {e.get('current_supply')}
    - Total supply: {e.get('total_supply')}
    - Coin Hours: {e.get('coin_hours')}

    ## Public Infra (Skywire)
    - Visors: {p.get('visors')}
    - Proxies: {p.get('proxies')}
    - VPN: {p.get('vpn')}
    - Transports: {p.get('transports')}
    - DMSG entries: {p.get('dmsg_entries')}
    - RF status: {p.get('rf_status')}

    ## Nodes (if provided)
    - Nodes seen/ok: {n.get('nodes_seen_ok')}
    - Note: {n.get('ut_note')}

    ## Fiber (if provided)
    - Endpoints seen/ok: {fsum.get('endpoints_seen_ok')}
    - Height: {fsum.get('height')}
    - Peers: {fsum.get('peers')}
    """).strip()

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"✅ Wrote {raw_path}")
    print(f"✅ Wrote {md_path}")

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main() -> int:
    today = os.environ.get("TODAY", datetime.utcnow().strftime("%Y-%m-%d"))
    ensure_dir(os.path.join("data", today))

    exp = split_env_list(os.environ.get("SKYWIRE_ENDPOINTS"))
    pub = split_env_list(os.environ.get("SKYWIRE_PUBLIC_ENDPOINTS"))
    fib = split_env_list(os.environ.get("FIBER_ENDPOINTS"))

    visors_file = os.environ.get("VISORS_FILE", ".tmp/visors_pks.txt")
    maxv = int(os.environ.get("UT_MAX_VISORS", "200"))

    pks = read_visors_file(visors_file, maxv)

    groups = [
        collect_explorer(exp),
        collect_public(pub),
        collect_nodes(pks, maxv),
        collect_fiber(fib),
    ]

    doc = {
        "date_utc": today,
        "generated_at": utc_now(),
        "groups": groups,
    }

    write_outputs(today, doc)
    return 0

if __name__ == "__main__":
    sys.exit(main())
