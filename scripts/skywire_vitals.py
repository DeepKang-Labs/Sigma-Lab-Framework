#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skywire VitalSigns agent
- Aggregates Explorer + Public Skywire infra + optional Node/Fiber endpoints
- Tolerant to missing/blank env vars and transient HTTP errors
- Produces JSON payload and Markdown summary in data/YYYY-MM-DD/

Env variables (multi-line or comma-separated supported):
  SKYWIRE_ENDPOINTS
  SKYWIRE_PUBLIC_ENDPOINTS
  SKYWIRE_NODE_ENDPOINTS
  FIBER_ENDPOINTS
  VISORS_PKS                  (optional; semicolon/comma/newline separated)
  UT_MAX_VISORS               (optional int; default 50)
  UT_SAMPLE_MODE              (optional; "random" or "first")
"""

from __future__ import annotations

import os
import json
import sys
import time
import random
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import requests

# ---------------------------
# Helpers
# ---------------------------

UA = "SigmaLab-SkywireVitalSigns/3.3.3 (+github.com/DeepKang-Labs)"
TIMEOUT = 10


def log(msg: str) -> None:
    print(f"[vitals] {msg}", flush=True)


def safe_int_env(name: str, default: int) -> int:
    """Parse int from env with strong fallback on empty/invalid values."""
    val = os.getenv(name, "")
    if val is None:
        return default
    val = val.strip()
    if not val:
        return default
    try:
        return int(val)
    except Exception:
        return default


def split_env_urls(name: str) -> List[str]:
    """Accept comma and/or newline separated env; strip and keep only http(s) URLs."""
    raw = os.getenv(name, "") or ""
    # Also tolerate accidental inclusion like NAME=https://a, https://b in a single token
    parts = []
    for chunk in raw.replace(";", "\n").splitlines():
        for token in chunk.split(","):
            url = token.strip().strip('"').strip("'")
            if url.startswith("http://") or url.startswith("https://"):
                parts.append(url)
    return parts


def split_env_list(name: str) -> List[str]:
    """Generic list split (for VISORS_PKS); splits on newline/semicolon/comma."""
    raw = os.getenv(name, "") or ""
    out = []
    for chunk in raw.replace(";", "\n").splitlines():
        for token in chunk.split(","):
            val = token.strip()
            if val:
                out.append(val)
    return out


def http_get_json(url: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    """GET JSON with basic error capture; returns (data, meta)."""
    meta = {"__url__": url, "__ok__": False}
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        meta["__status__"] = resp.status_code
        resp.raise_for_status()
        data = resp.json()
        meta["__ok__"] = True
        return data, meta
    except Exception as e:
        meta["__error__"] = f"{type(e).__name__}: {e}"
        return None, meta


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Data collectors
# ---------------------------

def collect_explorer(endpoints: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Collect blockchain/coinSupply-like metrics from Explorer endpoints."""
    summary = {
        "height": None,
        "current_supply": None,
        "total_supply": None,
        "coin_hours": None,
        "note": None,
    }
    payloads: List[Dict[str, Any]] = []

    # Try to find coinSupply-like endpoint
    best = None
    for url in endpoints:
        data, meta = http_get_json(url)
        payloads.append(meta)
        if not meta.get("__ok__"):
            continue

        # explorer.skycoin.com/api/coinSupply style
        if isinstance(data, dict):
            # Height is not in coinSupply; leave None if absent
            cs = data
            # normalize keys
            summary["current_supply"] = cs.get("current_supply") or cs.get("currentSuply") or cs.get("current_suply")
            summary["total_supply"] = cs.get("total_supply") or cs.get("max_supply")
            summary["coin_hours"] = cs.get("current_coinhour_supply") or cs.get("coinhour_supply")
            best = data
            break

        # If it is metadata-like, try to extract a height
        if isinstance(data, list) or isinstance(data, tuple):
            # Unknown schema; ignore silently
            pass

    if best is None and any(m.get("__ok__") for m in payloads):
        summary["note"] = "Explorer fields were empty — schema mismatch or temporary outage."

    return summary, payloads


def parse_sd_services(items: Any, expected_type: Optional[str] = None) -> Tuple[int, List[Dict[str, Any]]]:
    """Parse Service Directory entries; return count and small sample."""
    if not isinstance(items, list):
        return 0, []
    filtered = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if expected_type and it.get("type") != expected_type:
            continue
        filtered.append({
            "address": it.get("address"),
            "type": it.get("type"),
            "version": it.get("version"),
            "geo": (it.get("geo") or {}),
        })
    return len(filtered), filtered[:5]  # include a small sample


def collect_public_infra(public_urls: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Collect counts for proxies, VPN, (optional) visors from SD;
    transports from TPD; DMSG entries; RF status ping.
    """
    result = {
        "visors": 0,
        "proxies": 0,
        "vpn": 0,
        "transports": 0,
        "dmsg_entries": 0,
        "rf_status": None,  # e.g. "404 (ok=0)"
        "samples": {
            "proxy": [],
            "vpn": [],
            "visor": [],
        },
    }
    payloads: List[Dict[str, Any]] = []

    for url in public_urls:
        data, meta = http_get_json(url)
        payloads.append(meta)
        if not meta.get("__ok__"):
            # if it's clearly an RF or non-json endpoint, we accept status code in rf_status
            if "rf." in url or "/rf" in url:
                status = meta.get("__status__")
                ok_flag = 1 if status and 200 <= status < 300 else 0
                result["rf_status"] = f"{status} (ok={ok_flag})"
            continue

        # Service Directory endpoints
        if "sd.skycoin.com/api/services" in url:
            # Try proxies
            if "type=proxy" in url:
                count, sample = parse_sd_services(data)
                result["proxies"] += count
                result["samples"]["proxy"].extend(sample)
            # Try VPN
            elif "type=vpn" in url:
                count, sample = parse_sd_services(data)
                result["vpn"] += count
                result["samples"]["vpn"].extend(sample)
            # Try visor listings (if available)
            elif "type=visor" in url:
                count, sample = parse_sd_services(data)
                result["visors"] += count
                result["samples"]["visor"].extend(sample)

        # Transport Discovery (TPD)
        elif "tpd.skywire.skycoin.com" in url and "all-transports" in url:
            if isinstance(data, list):
                result["transports"] = len(data)

        # DMSG discovery
        elif "dmsgd.skywire.skycoin.com" in url and "dmsg-discovery/entries" in url:
            if isinstance(data, list):
                result["dmsg_entries"] = len(data)

        # RF status (raw) — when JSON exists
        elif "rf.skywire.skycoin.com" in url:
            status = meta.get("__status__")
            ok_flag = 1 if status and 200 <= status < 300 else 0
            result["rf_status"] = f"{status} (ok={ok_flag})"

    # Ensure rf_status isn't None
    if result["rf_status"] is None:
        result["rf_status"] = "n/a (no RF endpoint)"

    return result, payloads


def collect_nodes(node_urls: List[str], visors_pks: List[str], ut_max: int, ut_mode: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Optional health/metrics for explicit nodes (private visors you know),
    plus optional UT sampling if PKs provided (we aggregate only basic counters).
    """
    result = {
        "nodes_seen_ok": [0, 0],    # seen/ok
        "latency_avg_ms": None,
        "uptime_ratio_avg": None,
        "ut_note": None,
    }
    payloads: List[Dict[str, Any]] = []

    # Simple health checks against provided node endpoints (if any)
    seen = 0
    oks = 0
    for url in node_urls:
        data, meta = http_get_json(url)
        payloads.append(meta)
        if meta.get("__ok__"):
            oks += 1
        seen += 1
    result["nodes_seen_ok"] = [seen, oks]

    # Minimal UT sampling (if we have PKs)
    # We don't have a universal UT endpoint per PK available publicly; mark a note.
    if visors_pks:
        # Respect maximum and mode
        items = visors_pks[:]
        if ut_mode == "random":
            random.shuffle(items)
        sample = items[:max(1, ut_max)]
        result["ut_note"] = f"Sampled {len(sample)}/{len(visors_pks)} visors for UT (no public PK UT endpoint available)."
        # (If a public UT per PK endpoint is published later, aggregate here.)
    else:
        result["ut_note"] = "No PK provided."

    return result, payloads


def collect_fiber(fiber_urls: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Probe Fiber endpoints (status/metrics). We only count reachable ones to avoid schema coupling.
    """
    result = {
        "endpoints_seen_ok": [0, 0],
        "height": None,
        "peers": None,
    }
    payloads: List[Dict[str, Any]] = []

    seen = 0
    oks = 0
    for url in fiber_urls:
        data, meta = http_get_json(url)
        payloads.append(meta)
        seen += 1
        if meta.get("__ok__"):
            oks += 1
            # Try to guess some fields if present
            if isinstance(data, dict):
                result["height"] = result["height"] or data.get("height")
                # peers could be integer or list length
                peers = data.get("peers")
                if isinstance(peers, list):
                    result["peers"] = len(peers)
                elif isinstance(peers, int):
                    result["peers"] = peers

    result["endpoints_seen_ok"] = [seen, oks]
    return result, payloads


# ---------------------------
# Main
# ---------------------------

def build_markdown(date_utc: str,
                   explorer: Dict[str, Any],
                   public: Dict[str, Any],
                   nodes: Dict[str, Any],
                   fiber: Dict[str, Any]) -> str:
    md = []
    md.append(f"# Skywire VitalSigns — {date_utc} UTC\n")

    md.append("## Explorer\n")
    md.append(f"- Height: {explorer.get('height')}\n")
    md.append(f"- Current supply: {explorer.get('current_supply')}\n")
    md.append(f"- Total supply: {explorer.get('total_supply')}\n")
    md.append(f"- Coin Hours: {explorer.get('coin_hours')}\n")
    if explorer.get("note"):
        md.append(f"- Note: {explorer.get('note')}\n")

    md.append("\n## Public Infra (Skywire)\n")
    md.append(f"- Visors: {public.get('visors')}\n")
    md.append(f"- Proxies: {public.get('proxies')}\n")
    md.append(f"- VPN: {public.get('vpn')}\n")
    md.append(f"- Transports: {public.get('transports')}\n")
    md.append(f"- DMSG entries: {public.get('dmsg_entries')}\n")
    md.append(f"- RF status: {public.get('rf_status')}\n")

    md.append("\n## Nodes (if provided)\n")
    ns, ok = public.get("noop", 0), public.get("noop", 0)  # placeholder if needed
    seen, oks = nodes.get("nodes_seen_ok", [0, 0])
    md.append(f"- Nodes seen/ok: {seen}/{oks}\n")
    md.append(f"- Latency avg (ms): {nodes.get('latency_avg_ms')}\n")
    md.append(f"- Uptime ratio avg: {nodes.get('uptime_ratio_avg')}\n")

    md.append("\n## Uptime Tracker (UT)\n")
    md.append(f"- UT note: {nodes.get('ut_note')}\n")

    md.append("\n## Fiber (if provided)\n")
    f_seen, f_ok = fiber.get("endpoints_seen_ok", [0, 0])
    md.append(f"- Fiber endpoints seen/ok: {f_seen}/{f_ok}\n")
    md.append(f"- Fiber height: {fiber.get('height')} — peers: {fiber.get('peers')}\n")

    return "".join(md)


def main() -> int:
    # Resolve env
    skywire_exp = split_env_urls("SKYWIRE_ENDPOINTS")
    public_urls = split_env_urls("SKYWIRE_PUBLIC_ENDPOINTS")
    node_urls = split_env_urls("SKYWIRE_NODE_ENDPOINTS")
    fiber_urls = split_env_urls("FIBER_ENDPOINTS")
    visors_pks = split_env_list("VISORS_PKS")

    ut_max = safe_int_env("UT_MAX_VISORS", 50)
    ut_mode = os.getenv("UT_SAMPLE_MODE", "random").strip() or "random"

    log(f"Explorer endpoints: {len(skywire_exp)} | Public: {len(public_urls)} | Nodes: {len(node_urls)} | Fiber: {len(fiber_urls)}")
    if visors_pks:
        log(f"VISORS_PKS provided: {len(visors_pks)} (sampling {min(ut_max, len(visors_pks))} / mode={ut_mode})")
    else:
        log("VISORS_PKS not provided.")

    # Collect
    explorer, pl_explorer = collect_explorer(skywire_exp)
    public, pl_public = collect_public_infra(public_urls)
    nodes, pl_nodes = collect_nodes(node_urls, visors_pks, ut_max, ut_mode)
    fiber, pl_fiber = collect_fiber(fiber_urls)

    # Prepare outputs
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path("data") / now_utc
    ensure_dir(out_dir)

    # JSON (raw)
    payload = {
        "date_utc": now_utc,
        "meta": {
            "repo": "DeepKang-Labs/Sigma-Lab-Framework",
            "agent": "SkywireVitalSigns v3.3.3",
        },
        "groups": [
            {"name": "explorer", "endpoints": skywire_exp, "payloads": pl_explorer, "summary": explorer},
            {"name": "public", "endpoints": public_urls, "payloads": pl_public, "summary": public},
            {"name": "nodes", "endpoints": node_urls, "payloads": pl_nodes, "summary": nodes, "visors_pks": visors_pks[:10]},
            {"name": "fiber", "endpoints": fiber_urls, "payloads": pl_fiber, "summary": fiber},
        ],
    }
    (out_dir / "skywire_vitals.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown (summary)
    md = build_markdown(now_utc, explorer, public, nodes, fiber)
    (out_dir / "skywire_summary.md").write_text(md, encoding="utf-8")

    log(f"Wrote: {out_dir/'skywire_vitals.json'}")
    log(f"Wrote: {out_dir/'skywire_summary.md'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # Never fail hard on runner — emit trace in logs and exit 0 to keep pipeline green if desired.
        # If you prefer hard fail, change return code to 1.
        traceback.print_exc()
        sys.exit(1)
