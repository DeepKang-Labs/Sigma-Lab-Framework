#!/usr/bin/env python3
# scripts/skywire_vitals.py
# DeepKang Labs — Sigma: Skywire VitalSigns (Explorer + Public Infra + Nodes + Fiber)
# - Lit des endpoints depuis les variables d'environnement (GitHub Actions) ou un YAML local
# - Utilise des endpoints par défaut connus et publics si rien n'est fourni
# - Normalise les métriques clés + garde les payloads bruts par groupe
# - Produit:
#     data/YYYY-MM-DD/skywire_vitals.json
#     data/YYYY-MM-DD/skywire_summary.md
#     DATALOG.md (append)

import os, json, sys, datetime, pathlib, time
from typing import Any, Dict, List, Tuple, Optional
import requests
import pandas as pd

try:
    import yaml  # optionnel (fallback config locale)
except ImportError:
    yaml = None

# ---------- Chemins ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]
TODAY = datetime.datetime.utcnow().strftime("%Y-%m-%d")
DATA_DIR = ROOT / "data" / TODAY
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Defaults officiels ----------
DEFAULT_SKY_EXPLORER_ENDPOINTS = [
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

# ---------- Utilitaires ----------
def read_csv_env(name: str) -> List[str]:
    v = os.getenv(name, "") or ""
    if not v.strip():
        return []
    parts = [p.strip().strip('"').strip("'") for p in v.replace("\n", ",").split(",")]
    return [p for p in parts if p]

def maybe_load_local_config() -> Dict[str, List[str]]:
    """Charge scripts/skywire.config.yaml si présent (facultatif)."""
    cfg = ROOT / "scripts" / "skywire.config.yaml"
    if cfg.exists() and yaml:
        try:
            y = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception:
            y = {}
        return {
            "explorer": [u for u in (y.get("endpoints") or []) if u],
            "public": [u for u in (y.get("public_endpoints") or []) if u],
            "nodes": [u for u in (y.get("node_endpoints") or []) if u],
            "fiber": [u for u in (y.get("fiber_endpoints") or []) if u],
        }
    return {"explorer": [], "public": [], "nodes": [], "fiber": []}

def safe_get(url: str, timeout: int = 15) -> Dict[str, Any]:
    """GET tolérant: capture JSON si possible, sinon statut/texte."""
    try:
        r = requests.get(url, timeout=timeout)
        info = {
            "__url__": url,
            "__status__": r.status_code,
            "__ok__": r.ok,
            "__headers__": dict(r.headers),
        }
        ctype = r.headers.get("Content-Type", "")
        if "application/json" in ctype.lower():
            try:
                info["data"] = r.json()
            except Exception as e:
                info["__error__"] = f"JSONDecodeError: {e}"
                info["data"] = None
        else:
            # RF peut renvoyer HTML/texte — on garde un extrait court
            txt = r.text
            info["data"] = {"__text__": txt[:2000]}  # tronqué
        return info
    except Exception as e:
        return {"__url__": url, "__error__": f"{type(e).__name__}: {e}", "__ok__": False}

def coalesce(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default

# ---------- Parseurs par famille ----------
def parse_explorer_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Essaye d'extraire des métriques réseau/économie depuis l'explorer SKY."""
    metrics = {
        "height": None,
        "current_supply": None,
        "total_supply": None,
        "coin_hours": None,
    }
    for p in payloads:
        data = p.get("data")
        if not isinstance(data, dict):
            continue
        # /api/blockchain/metadata
        h = coalesce(data, ["head", "height", "seq"])
        if isinstance(h, (int, float)):
            metrics["height"] = max(metrics["height"] or 0, int(h))
        # /api/coinSupply
        cs = coalesce(data, ["currentSupply", "current_supply"])
        if isinstance(cs, (int, float)):
            metrics["current_supply"] = float(cs)
        ts = coalesce(data, ["totalSupply", "total_supply"])
        if isinstance(ts, (int, float)):
            metrics["total_supply"] = float(ts)
        ch = coalesce(data, ["coinHours", "coin_hours"])
        if isinstance(ch, (int, float)):
            metrics["coin_hours"] = float(ch)
    return metrics

def parse_public_infra_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Résumé des services publics Skywire: compte des visors/proxies/vpn/transports/dmsg/health RF."""
    counts = {
        "visors": 0,
        "proxies": 0,
        "vpn": 0,
        "transports": 0,
        "dmsg_entries": 0,
        "rf_ok": 0,          # RF accessible (HTTP 200)
        "rf_last_status": None,
    }
    for p in payloads:
        url = p.get("__url__", "")
        ok = bool(p.get("__ok__", False))
        status = p.get("__status__")
        data = p.get("data")
        # SD (services?type=...)
        if "sd.skycoin.com" in url and "api/services" in url:
            if ok and isinstance(data, (list, dict)):
                if "type=visor" in url:
                    counts["visors"] = len(data) if isinstance(data, list) else len(data.keys())
                elif "type=proxy" in url:
                    counts["proxies"] = len(data) if isinstance(data, list) else len(data.keys())
                elif "type=vpn" in url:
                    counts["vpn"] = len(data) if isinstance(data, list) else len(data.keys())
        # AR
        elif "ar.skywire.skycoin.com" in url:
            if ok and data is not None:
                # La forme peut varier, on compte grossièrement les entrées
                if isinstance(data, list):
                    counts["transports"] = max(counts["transports"], len(data))
                elif isinstance(data, dict):
                    counts["transports"] = max(counts["transports"], len(data.keys()))
        # TPD
        elif "tpd.skywire.skycoin.com" in url:
            if ok and data is not None:
                if isinstance(data, list):
                    counts["transports"] = max(counts["transports"], len(data))
                elif isinstance(data, dict):
                    counts["transports"] = max(counts["transports"], len(data.keys()))
        # DMSGD
        elif "dmsgd.skywire.skycoin.com" in url:
            if ok and data is not None:
                if isinstance(data, list):
                    counts["dmsg_entries"] = len(data)
                elif isinstance(data, dict):
                    counts["dmsg_entries"] = len(data.keys())
        # RF (peut renvoyer autre chose que JSON)
        elif "rf.skywire.skycoin.com" in url:
            counts["rf_last_status"] = status
            if ok and status == 200:
                counts["rf_ok"] = 1
    return counts

def parse_nodes_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Heuristiques légères pour endpoints de nœuds:
     - /api/health, /api/metrics, /healthz, etc.
    On calcule disponibilité + on agrège quelques champs classiques si présents.
    """
    nodes_seen = 0
    nodes_ok = 0
    latency_ms_avg = None
    uptime_ratio_avg = None
    for p in payloads:
        nodes_seen += 1
        if p.get("__ok__"):
            nodes_ok += 1
        data = p.get("data")
        if isinstance(data, dict):
            lat = coalesce(data, ["latency_ms", "avg_latency_ms", "latency"])
            if isinstance(lat, (int, float)):
                latency_ms_avg = float(lat) if latency_ms_avg is None else (latency_ms_avg + float(lat)) / 2.0
            up = coalesce(data, ["uptime_ratio", "uptime", "availability"])
            if isinstance(up, (int, float)):
                uptime_ratio_avg = float(up) if uptime_ratio_avg is None else (uptime_ratio_avg + float(up)) / 2.0
    return {
        "nodes_seen": nodes_seen,
        "nodes_ok": nodes_ok,
        "latency_ms_avg": latency_ms_avg,
        "uptime_ratio_avg": uptime_ratio_avg,
    }

def parse_fiber_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fiber: status/metrics publics si exposés. On se contente de:
     - ressources OK/KO
     - quelques compteurs 'height'/'peers' si trouvés
    """
    seen = 0; ok = 0
    height = None; peers = None
    for p in payloads:
        seen += 1
        if p.get("__ok__"): ok += 1
        data = p.get("data")
        if isinstance(data, dict):
            h = coalesce(data, ["height", "block", "lastBlock", "bestHeight"])
            if isinstance(h, (int, float)):
                height = int(h) if height is None else max(height, int(h))
            pr = coalesce(data, ["peers", "peerCount", "connections"])
            if isinstance(pr, (int, float)):
                peers = int(pr) if peers is None else max(peers, int(pr))
    return {"fiber_seen": seen, "fiber_ok": ok, "fiber_height": height, "fiber_peers": peers}

# ---------- Collecteurs ----------
def collect_group(name: str, urls: List[str]) -> Dict[str, Any]:
    payloads = []
    for u in urls:
        if not u: 
            continue
        payloads.append(safe_get(u))
        time.sleep(0.2)  # politesse
    return {"name": name, "endpoints": urls, "payloads": payloads}

# ---------- Main ----------
def main() -> int:
    # 1) Lire ENV
    explorer_env = read_csv_env("SKYWIRE_ENDPOINTS")            # explorer SKY
    public_env   = read_csv_env("SKYWIRE_PUBLIC_ENDPOINTS")     # infra publique Skywire
    nodes_env    = read_csv_env("SKYWIRE_NODE_ENDPOINTS")       # endpoints de nœuds (si tu en as)
    fiber_env    = read_csv_env("FIBER_ENDPOINTS")              # endpoints Fiber (si tu en as)

    # 2) Fallback config locale
    if not any([explorer_env, public_env, nodes_env, fiber_env]):
        local = maybe_load_local_config()
        explorer_env, public_env, nodes_env, fiber_env = local["explorer"], local["public"], local["nodes"], local["fiber"]

    # 3) Defaults si toujours vides
    explorer = explorer_env or DEFAULT_SKY_EXPLORER_ENDPOINTS
    public   = public_env   or DEFAULT_SKYWIRE_PUBLIC_ENDPOINTS
    nodes    = nodes_env    or []   # tu peux remplir plus tard
    fiber    = fiber_env    or []   # idem

    out = {
        "date_utc": TODAY,
        "meta": {
            "repo": "DeepKang-Labs/Sigma-Lab-Framework",
            "agent": "SkywireVitalSigns v3",
        },
        "groups": []
    }

    # Collecte par groupes
    g_explorer = collect_group("explorer", explorer)
    g_public   = collect_group("skywire_public", public)
    out["groups"].append(g_explorer)
    out["groups"].append(g_public)

    if nodes:
        g_nodes = collect_group("nodes", nodes)
        out["groups"].append(g_nodes)
    if fiber:
        g_fiber = collect_group("fiber", fiber)
        out["groups"].append(g_fiber)

    # ---- Normalisation / agrégats top-level ----
    # Explorer
    explorer_metrics = parse_explorer_payloads(g_explorer["payloads"])
    # Public infra
    public_counts = parse_public_infra_payloads(g_public["payloads"])
    # Nodes
    nodes_metrics = parse_nodes_payloads(g_nodes["payloads"]) if "g_nodes" in locals() else {}
    # Fiber
    fiber_metrics = parse_fiber_payloads(g_fiber["payloads"]) if "g_fiber" in locals() else {}

    # Vitals global (best effort)
    vitals = {
        "height": explorer_metrics.get("height"),
        "current_supply": explorer_metrics.get("current_supply"),
        "total_supply": explorer_metrics.get("total_supply"),
        "coin_hours": explorer_metrics.get("coin_hours"),

        "public_visors": public_counts.get("visors"),
        "public_proxies": public_counts.get("proxies"),
        "public_vpn": public_counts.get("vpn"),
        "public_transports": public_counts.get("transports"),
        "public_dmsg_entries": public_counts.get("dmsg_entries"),
        "rf_ok": public_counts.get("rf_ok"),
        "rf_last_status": public_counts.get("rf_last_status"),

        "nodes_seen": nodes_metrics.get("nodes_seen"),
        "nodes_ok": nodes_metrics.get("nodes_ok"),
        "latency_ms_avg": nodes_metrics.get("latency_ms_avg"),
        "uptime_ratio_avg": nodes_metrics.get("uptime_ratio_avg"),

        "fiber_seen": fiber_metrics.get("fiber_seen"),
        "fiber_ok": fiber_metrics.get("fiber_ok"),
        "fiber_height": fiber_metrics.get("fiber_height"),
        "fiber_peers": fiber_metrics.get("fiber_peers"),
    }

    out["vitals"] = vitals

    # Sauvegarde JSON + résumé MD
    json_path = DATA_DIR / "skywire_vitals.json"
    json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = DATA_DIR / "skywire_summary.md"
    lines = []
    lines.append(f"# Skywire VitalSigns — {TODAY} UTC\n")
    lines.append("## Explorer")
    lines.append(f"- Height: **{vitals.get('height','n/a')}**")
    lines.append(f"- Current supply: **{vitals.get('current_supply','n/a')}**")
    lines.append(f"- Total supply: **{vitals.get('total_supply','n/a')}**")
    lines.append(f"- Coin Hours: **{vitals.get('coin_hours','n/a')}**\n")

    lines.append("## Public Infra (Skywire)")
    lines.append(f"- Visors: **{vitals.get('public_visors','n/a')}**")
    lines.append(f"- Proxies: **{vitals.get('public_proxies','n/a')}**")
    lines.append(f"- VPN: **{vitals.get('public_vpn','n/a')}**")
    lines.append(f"- Transports: **{vitals.get('public_transports','n/a')}**")
    lines.append(f"- DMSG entries: **{vitals.get('public_dmsg_entries','n/a')}**")
    lines.append(f"- RF status: **{vitals.get('rf_last_status','n/a')}** (ok={vitals.get('rf_ok','0')})\n")

    lines.append("## Nodes (if provided)")
    lines.append(f"- Nodes seen/ok: **{vitals.get('nodes_seen','0')}/{vitals.get('nodes_ok','0')}**")
    lines.append(f"- Latency avg (ms): **{vitals.get('latency_ms_avg','n/a')}**")
    lines.append(f"- Uptime ratio avg: **{vitals.get('uptime_ratio_avg','n/a')}**\n")

    lines.append("## Fiber (if provided)")
    lines.append(f"- Fiber endpoints seen/ok: **{vitals.get('fiber_seen','0')}/{vitals.get('fiber_ok','0')}**")
    lines.append(f"- Fiber height: **{vitals.get('fiber_height','n/a')}** — peers: **{vitals.get('fiber_peers','n/a')}**\n")

    summary.write_text("\n".join(lines), encoding="utf-8")

    # DATALOG
    with open(ROOT / "DATALOG.md", "a", encoding="utf-8") as f:
        f.write(f"{TODAY} : skywire_vitals.json + skywire_summary.md generated (Explorer+Public+Nodes+Fiber)\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
