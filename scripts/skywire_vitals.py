#!/usr/bin/env python3
# scripts/skywire_vitals.py
# DeepKang Labs — Sigma: Skywire VitalSigns
# v3.3.1 — Fix type hint, SD auto-PK discovery, UT sampling, explorer/public/nodes/fiber aggregation.

import os, json, sys, datetime, pathlib, time, re, random
from typing import Any, Dict, List, Optional
import requests
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

ROOT = pathlib.Path(__file__).resolve().parents[1]
TODAY = datetime.datetime.utcnow().strftime("%Y-%m-%d")
DATA_DIR = ROOT / "data" / TODAY
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ----------- Public endpoints -----------
DEFAULT_SKY_EXPLORER_ENDPOINTS = [
    "https://explorer.skycoin.com/api/blockchain/metadata",
    "https://explorer.skycoin.com/api/coinSupply",
]

DEFAULT_SKYWIRE_PUBLIC_ENDPOINTS = [
    # Service Discovery
    "https://sd.skycoin.com/api/services?type=visor",
    "https://sd.skycoin.com/api/services?type=proxy",
    "https://sd.skycoin.com/api/services?type=vpn",
    # Address Resolver / Transport Discovery / DMSG Discovery
    "https://ar.skywire.skycoin.com/transports",
    "https://tpd.skywire.skycoin.com/all-transports",
    "https://dmsgd.skywire.skycoin.com/dmsg-discovery/entries",
    # Reward Framework health (statut HTTP)
    "https://rf.skywire.skycoin.com/",
]

UT_BASE = "https://ut.skywire.skycoin.com/uptimes?v=v2&visors="  # + pk1;pk2;...

# ----------- Helpers -----------
def read_csv_env(name: str) -> List[str]:
    v = os.getenv(name, "") or ""
    if not v.strip():
        return []
    parts = [p.strip().strip('"').strip("'") for p in v.replace("\n", ",").split(",")]
    return [p for p in parts if p]

def load_yaml(path: pathlib.Path) -> Optional[dict]:
    if not path.exists() or yaml is None:
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def maybe_load_local_config() -> Dict[str, List[str]]:
    cfg = load_yaml(ROOT / "scripts" / "skywire.config.yaml") or {}
    return {
        "explorer": [u for u in (cfg.get("endpoints") or []) if u],
        "public": [u for u in (cfg.get("public_endpoints") or []) if u],
        "nodes": [u for u in (cfg.get("node_endpoints") or []) if u],
        "fiber": [u for u in (cfg.get("fiber_endpoints") or []) if u],
    }

def load_nodes_yaml_endpoints_and_pks():
    inv = load_yaml(ROOT / "scripts" / "nodes.yaml")
    urls: List[str] = []
    pks: List[str] = []
    if not inv or "nodes" not in inv:
        return urls, pks
    for n in inv.get("nodes", []):
        host = (n or {}).get("host")
        if host:
            scheme = (n.get("scheme") or "http").strip()
            port = int(n.get("port") or (443 if scheme == "https" else 80))
            hp = (n.get("health_path") or "/api/health").strip()
            mp = (n.get("metrics_path") or "/api/metrics").strip()
            base = f"{scheme}://{host}:{port}"
            urls.append(f"{base}{hp}")
            urls.append(f"{base}{mp}")
        pk = (n or {}).get("public_key") or ""
        if pk.strip():
            pks.append(pk.strip())
    return urls, pks

def safe_get(url: str, timeout: int = 20) -> Dict[str, Any]:
    try:
        r = requests.get(url, timeout=timeout)
        info = {"__url__": url, "__status__": r.status_code, "__ok__": r.ok, "__headers__": dict(r.headers)}
        ctype = r.headers.get("Content-Type", "")
        if "application/json" in (ctype or "").lower():
            try:
                info["data"] = r.json()
            except Exception as e:
                info["__error__"] = f"JSONDecodeError: {e}"
                info["data"] = None
        else:
            info["data"] = {"__text__": (r.text or "")[:4000]}
        return info
    except Exception as e:
        return {"__url__": url, "__error__": f"{type(e).__name__}: {e}", "__ok__": False}

def pick(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default

# ----------- Parsers -----------
def parse_explorer_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    m = {"height": None, "current_supply": None, "total_supply": None, "coin_hours": None}
    for p in payloads:
        data = p.get("data")
        if not isinstance(data, dict):
            continue
        h = pick(data, ["head", "height", "seq"])
        if isinstance(h, (int, float)):
            m["height"] = max(m["height"] or 0, int(h))
        cs = pick(data, ["currentSupply", "current_supply"])
        if isinstance(cs, (int, float)):
            m["current_supply"] = float(cs)
        ts = pick(data, ["totalSupply", "total_supply"])
        if isinstance(ts, (int, float)):
            m["total_supply"] = float(ts)
        ch = pick(data, ["coinHours", "coin_hours"])
        if isinstance(ch, (int, float)):
            m["coin_hours"] = float(ch)
    return m

def parse_public_infra_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    c = {"visors": 0, "proxies": 0, "vpn": 0, "transports": 0, "dmsg_entries": 0, "rf_ok": 0, "rf_last_status": None}
    for p in payloads:
        url, ok, status, data = p.get("__url__", ""), bool(p.get("__ok__", False)), p.get("__status__"), p.get("data")
        if "sd.skycoin.com" in url and "api/services" in url:
            if ok and isinstance(data, (list, dict)):
                n = len(data) if isinstance(data, list) else len(data.keys())
                if "type=visor" in url:
                    c["visors"] = n
                if "type=proxy" in url:
                    c["proxies"] = n
                if "type=vpn" in url:
                    c["vpn"] = n
        elif "ar.skywire.skycoin.com" in url or "tpd.skywire.skycoin.com" in url:
            if ok and isinstance(data, (list, dict)):
                n = len(data) if isinstance(data, list) else len(data.keys())
                c["transports"] = max(c["transports"], n)
        elif "dmsgd.skywire.skycoin.com" in url:
            if ok and isinstance(data, (list, dict)):
                n = len(data) if isinstance(data, list) else len(data.keys())
                c["dmsg_entries"] = n
        elif "rf.skywire.skycoin.com" in url:
            c["rf_last_status"] = status
            if ok and status == 200:
                c["rf_ok"] = 1
    return c

def extract_public_pks_from_sd(payloads: List[Dict[str, Any]]) -> List[str]:
    """Find plausible PK fields in SD output."""
    pks = []
    for p in payloads:
        url, ok, data = p.get("__url__", ""), bool(p.get("__ok__", False)), p.get("data")
        if not ok or "sd.skycoin.com" not in url or "type=visor" not in url:
            continue
        items = data if isinstance(data, list) else []
        for it in items:
            if not isinstance(it, dict):
                continue
            cand = (
                it.get("pk")
                or it.get("public_key")
                or (it.get("attrs") or {}).get("pk")
                or (it.get("metadata") or {}).get("public_key")
            )
            if isinstance(cand, str) and len(cand) >= 64:
                pks.append(cand.strip())
    # deduplicate
    seen, out = set(), []
    for k in pks:
        if k not in seen:
            out.append(k)
            seen.add(k)
    return out

def parse_nodes_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    seen = ok = 0
    lat_avg = up_avg = None
    for p in payloads:
        seen += 1
        if p.get("__ok__"):
            ok += 1
        data = p.get("data")
        if isinstance(data, dict):
            lat = pick(data, ["latency_ms", "avg_latency_ms", "latency"])
            if isinstance(lat, (int, float)):
                lat_avg = float(lat) if lat_avg is None else (lat_avg + float(lat)) / 2.0
            up = pick(data, ["uptime_ratio", "uptime", "availability"])
            if isinstance(up, (int, float)):
                up_avg = float(up) if up_avg is None else (up_avg + float(up)) / 2.0
    return {"nodes_seen": seen, "nodes_ok": ok, "latency_ms_avg": lat_avg, "uptime_ratio_avg": up_avg}

def parse_fiber_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    seen = ok = 0
    height = peers = None
    for p in payloads:
        seen += 1
        if p.get("__ok__"):
            ok += 1
        data = p.get("data")
        if isinstance(data, dict):
            h = pick(data, ["height", "block", "lastBlock", "bestHeight"])
            if isinstance(h, (int, float)):
                height = int(h) if height is None else max(height, int(h))
            pr = pick(data, ["peers", "peerCount", "connections"])
            if isinstance(pr, (int, float)):
                peers = int(pr) if peers is None else max(peers, int(pr))
        elif isinstance(data, dict) and "__text__" in data:
            txt = data["__text__"]
            mh = re.search(r"height[^\d]*(\d+)", txt, re.I)
            if mh and height is None:
                height = int(mh.group(1))
            mp = re.search(r"peers?[^\d]*(\d+)", txt, re.I)
            if mp and peers is None:
                peers = int(mp.group(1))
    return {"fiber_seen": seen, "fiber_ok": ok, "fiber_height": height, "fiber_peers": peers}

def parse_ut_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    avg = None
    count = 0
    data = payload.get("data")
    if isinstance(data, dict):
        visors = data.get("visors") or data.get("data") or []
        if isinstance(visors, dict):
            visors = list(visors.values())
        if isinstance(visors, list):
            vals = []
            for v in visors:
                if isinstance(v, dict):
                    u = pick(v, ["uptime", "uptime_ratio", "ut", "percent"])
                    if isinstance(u, (int, float)):
                        vals.append(float(u))
            if vals:
                avg = sum(vals) / len(vals)
                count = len(vals)
    return {"ut_avg": avg, "ut_count": count}

# ----------- Collect -----------
def collect_group(name: str, urls: List[str]) -> Dict[str, Any]:
    payloads = []
    for u in urls:
        if not u:
            continue
        payloads.append(safe_get(u))
        time.sleep(0.2)
    return {"name": name, "endpoints": urls, "payloads": payloads}

def main() -> int:
    explorer_env = read_csv_env("SKYWIRE_ENDPOINTS")
    public_env = read_csv_env("SKYWIRE_PUBLIC_ENDPOINTS")
    nodes_env = read_csv_env("SKYWIRE_NODE_ENDPOINTS")
    fiber_env = read_csv_env("FIBER_ENDPOINTS")
    visors_pks = os.getenv("VISORS_PKS", "").strip()

    ut_max = int(os.getenv("UT_MAX_VISORS", "20"))
    ut_mode = (os.getenv("UT_SAMPLE_MODE", "top") or "top").lower()  # "top" or "random"

    if not any([explorer_env, public_env, nodes_env, fiber_env]):
        local = maybe_load_local_config()
        explorer_env, public_env, nodes_env, fiber_env = (
            local["explorer"],
            local["public"],
            local["nodes"],
            local["fiber"],
        )

    nodes_urls, nodes_pks_yaml = load_nodes_yaml_endpoints_and_pks()

    explorer = explorer_env or DEFAULT_SKY_EXPLORER_ENDPOINTS
    public = public_env or DEFAULT_SKYWIRE_PUBLIC_ENDPOINTS
    nodes = nodes_env if nodes_env else nodes_urls
    fiber = fiber_env or []

    # Collecte publique (inclut SD:visor)
    g_public = collect_group("skywire_public", public)
    g_explorer = collect_group("explorer", explorer)

    # Auto-découverte de PKs si rien n’est fourni
    pk_list: List[str] = []
    if visors_pks:
        pk_list += [p for p in visors_pks.split(";") if p.strip()]
    if not pk_list and nodes_pks_yaml:
        pk_list += nodes_pks_yaml
    if not pk_list:
        sd_pks = extract_public_pks_from_sd(g_public["payloads"])
        if sd_pks:
            if ut_mode == "random" and len(sd_pks) > ut_max:
                pk_list = random.sample(sd_pks, ut_max)
            else:
                pk_list = sd_pks[:ut_max]

    ut_payload = None
    if pk_list:
        ut_url = UT_BASE + ";".join(pk_list[:50])  # limite sécurité
        ut_payload = safe_get(ut_url)

    # Groupes restants
    groups = [g_explorer, g_public]
    if nodes:
        groups.append(collect_group("nodes", nodes))
    if fiber:
        groups.append(collect_group("fiber", fiber))

    explorer_m = parse_explorer_payloads(g_explorer["payloads"])
    public_c = parse_public_infra_payloads(g_public["payloads"])
    nodes_m = (
        parse_nodes_payloads(groups[2]["payloads"])
        if len(groups) >= 3 and groups[2]["name"] == "nodes"
        else {}
    )
    fiber_m = (
        parse_fiber_payloads(groups[-1]["payloads"]) if groups and groups[-1]["name"] == "fiber" else {}
    )
    ut_m = parse_ut_payload(ut_payload) if ut_payload else {}

    out = {
        "date_utc": TODAY,
        "meta": {"repo": "DeepKang-Labs/Sigma-Lab-Framework", "agent": "SkywireVitalSigns v3.3.1"},
        "groups": groups,
        "pk_sampled": pk_list,
        "vitals": {
            "height": explorer_m.get("height"),
            "current_supply": explorer_m.get("current_supply"),
            "total_supply": explorer_m.get("total_supply"),
            "coin_hours": explorer_m.get("coin_hours"),
            "public_visors": public_c.get("visors"),
            "public_proxies": public_c.get("proxies"),
            "public_vpn": public_c.get("vpn"),
            "public_transports": public_c.get("transports"),
            "public_dmsg_entries": public_c.get("dmsg_entries"),
            "rf_ok": public_c.get("rf_ok"),
            "rf_last_status": public_c.get("rf_last_status"),
            "nodes_seen": nodes_m.get("nodes_seen"),
            "nodes_ok": nodes_m.get("nodes_ok"),
            "latency_ms_avg": nodes_m.get("latency_ms_avg"),
            "uptime_ratio_avg": nodes_m.get("uptime_ratio_avg"),
            "fiber_seen": fiber_m.get("fiber_seen"),
            "fiber_ok": fiber_m.get("fiber_ok"),
            "fiber_height": fiber_m.get("fiber_height"),
            "fiber_peers": fiber_m.get("fiber_peers"),
            "ut_avg": ut_m.get("ut_avg"),
            "ut_count": ut_m.get("ut_count"),
        },
    }

    (DATA_DIR / "skywire_vitals.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = []
    md.append(f"# Skywire VitalSigns — {TODAY} UTC\n")
    md.append("## Explorer")
    md.append(f"- Height: **{out['vitals'].get('height','None')}**")
    md.append(f"- Current supply: **{out['vitals'].get('current_supply','None')}**")
    md.append(f"- Total supply: **{out['vitals'].get('total_supply','None')}**")
    md.append(f"- Coin Hours: **{out['vitals'].get('coin_hours','None')}**\n")
    md.append("## Public Infra (Skywire)")
    md.append(f"- Visors: **{out['vitals'].get('public_visors','0')}**")
    md.append(f"- Proxies: **{out['vitals'].get('public_proxies','0')}**")
    md.append(f"- VPN: **{out['vitals'].get('public_vpn','0')}**")
    md.append(f"- Transports: **{out['vitals'].get('public_transports','0')}**")
    md.append(f"- DMSG entries: **{out['vitals'].get('public_dmsg_entries','0')}**")
    md.append(f"- RF status: **{out['vitals'].get('rf_last_status','n/a')}** (ok={out['vitals'].get('rf_ok','0')})\n")
    md.append("## Nodes (if provided)")
    md.append(f"- Nodes seen/ok: **{out['vitals'].get('nodes_seen','None')}/{out['vitals'].get('nodes_ok','None')}**")
    md.append(f"- Latency avg (ms): **{out['vitals'].get('latency_ms_avg','None')}**")
    md.append(f"- Uptime ratio avg: **{out['vitals'].get('uptime_ratio_avg','None')}**\n")
    md.append("## Uptime Tracker (UT)")
    if out["vitals"].get("ut_avg") is not None:
        md.append(f"- UT average (%): **{round(out['vitals']['ut_avg'], 2)}** over **{out['vitals'].get('ut_count',0)}** visors")
        md.append(f"- PKs sampled ({len(out['pk_sampled'])}): `{';'.join(out['pk_sampled'])}`\n")
    else:
        md.append("- UT average (%): **None** (no PK available)\n")
    md.append("## Fiber (if provided)")
    md.append(f"- Fiber endpoints seen/ok: **{out['vitals'].get('fiber_seen','None')}/{out['vitals'].get('fiber_ok','None')}**")
    md.append(f"- Fiber height: **{out['vitals'].get('fiber_height','None')}** — peers: **{out['vitals'].get('fiber_peers','None')}**\n")
    (DATA_DIR / "skywire_summary.md").write_text("\n".join(md), encoding="utf-8")

    with open(ROOT / "DATALOG.md", "a", encoding="utf-8") as f:
        f.write(f"{TODAY} : vitals+summary (Explorer+Public+Nodes+Fiber+UT, auto-PKs={len(out['pk_sampled'])})\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
