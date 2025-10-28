#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skywire VitalSigns — snapshot (Explorer + Public Infra + Nodes + Fiber)
- Produit: data/<YYYY-MM-DD>/skywire_vitals.json
- Produit: data/<YYYY-MM-DD>/skywire_summary.md
Compatibilité: Python 3.10+

Variables d'env (toutes optionnelles, valeurs par défaut intégrées) :
  TODAY=2025-10-28 (sinon date UTC du jour)
  EXPLORER_ENDPOINTS="https://explorer.skycoin.com/api/blockchain/metadata,https://explorer.skycoin.com/api/coinSupply"
  PUBLIC_ENDPOINTS="https://sd.skycoin.com/api/services?type=proxy,https://sd.skycoin.com/api/services?type=vpn,https://tpd.skywire.skycoin.com/all-transports,https://dmsgd.skywire.skycoin.com/dmsg-discovery/entries,https://rf.skywire.skycoin.com/"
  FIBER_ENDPOINTS="https://fiber.skywire.dev/api/status,https://fiber.skywire.dev/api/metrics"
  VISORS_PKS_FILE=".tmp/visors_pks.txt"  (une PK par ligne ; optionnel)

Sécurité: Aucune donnée sensible n’est publiée. Les PKs (si fournies) ne sont JAMAIS écrites dans le JSON (seulement comptées).
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import gzip
import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

# Dépendance standard côté runner GitHub (installée via requirements)
try:
    import requests
except Exception as e:
    print("FATAL: `requests` is required. Install with `pip install requests`.", file=sys.stderr)
    raise

# ---------- utilitaires ----------

def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def getenv_list(name: str, default_csv: str) -> List[str]:
    val = os.getenv(name, default_csv)
    # on autorise les sauts de ligne ou virgules
    parts = []
    for chunk in val.replace("\n", ",").split(","):
        c = chunk.strip()
        if c:
            parts.append(c)
    return parts

def fetch_url(url: str, timeout: float = 12.0) -> Tuple[bool, int, Dict[str, str], Any, Optional[str]]:
    """
    Retourne: (ok, status, headers, data, error)
    - Décode JSON si possible
    - Supporte gzip
    - Ne lève pas d'exception ; encapsule tout dans (ok,err)
    """
    try:
        resp = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
        status = resp.status_code
        headers = {k.lower(): v for k, v in resp.headers.items()}
        raw = resp.content

        # gérer gzip éventuel
        if headers.get("content-encoding", "").lower() == "gzip":
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gzf:
                    raw = gzf.read()
            except Exception:
                pass

        data: Any = None
        # essaie JSON
        try:
            data = resp.json()
        except Exception:
            # si pas JSON, on tente texte
            try:
                txt = raw.decode("utf-8", errors="replace")
                data = txt
            except Exception:
                data = None

        ok = 200 <= status < 300
        return ok, status, headers, data, None
    except requests.RequestException as e:
        return False, 0, {}, None, f"{type(e).__name__}: {e}"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ---------- collecte Explorer ----------

def collect_explorer(endpoints: List[str]) -> Dict[str, Any]:
    payloads = []
    height = None
    curr_supply = None
    total_supply = None
    coin_hours = None
    note = None

    for url in endpoints:
        ok, status, headers, data, err = fetch_url(url)
        payloads.append({
            "__url__": url,
            "__ok__": ok,
            "__status__": status,
            "__error__": err if err else None
        })
        # on tente d'extraire ces champs si la structure est connue
        if ok and isinstance(data, dict):
            # coinSupply
            if "current_supply" in data or "max_supply" in data or "current_coinhour_supply" in data:
                curr_supply = data.get("current_supply", curr_supply)
                total_supply = data.get("total_supply", total_supply) or data.get("max_supply", total_supply)
                coin_hours = data.get("current_coinhour_supply", coin_hours)
            # blockchain/metadata
            if "lastBlocks" in data or "last_block" in data or "height" in data:
                height = data.get("height", height)

    # champ note si aucun champ extrait — utile pour le rendu MD
    if all(v is None for v in [height, curr_supply, total_supply, coin_hours]):
        note = "Explorer fields were empty — schema mismatch or temporary outage."

    return {
        "name": "explorer",
        "endpoints": endpoints,
        "payloads": payloads,
        "summary": {
            "height": height,
            "current_supply": curr_supply,
            "total_supply": total_supply,
            "coin_hours": coin_hours,
            "note": note
        }
    }

# ---------- collecte Public Infra ----------

def _geo_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    # Structure attendue (proxy/vpn samples)
    out = {
        "address": item.get("address"),
        "type": item.get("type"),
        "version": item.get("version"),
    }
    g = item.get("geo") or {}
    if isinstance(g, dict):
        out["geo"] = {
            "lat": g.get("lat"),
            "lon": g.get("lon"),
            "country": g.get("country"),
            "region": g.get("region")
        }
    return out

def collect_public(endpoints: List[str]) -> Dict[str, Any]:
    payloads = []
    proxies = None
    vpn = None
    transports = None
    dmsg_entries = None
    rf_ok = 0
    rf_status = None

    # on stocke quelques échantillons (limités) pour le JSON
    samples = {"proxy": [], "vpn": [], "visor": []}

    for url in endpoints:
        ok, status, headers, data, err = fetch_url(url)
        payloads.append({
            "__url__": url,
            "__ok__": ok,
            "__status__": status,
            "__error__": err if err else None
        })

        # --- bloc robuste: data peut être None, list, dict, str ---
        if data is None:
            print(f"⚠️ Aucun retour valide depuis {url}")
            lst: List[Any] = []
        elif isinstance(data, list):
            lst = data
        elif isinstance(data, dict):
            lst = data.get("data", [])
            if not isinstance(lst, list):
                lst = []
        else:
            lst = []

        # routing: type d’endpoint
        if "services?type=proxy" in url:
            proxies = len(lst)
            # collecter 5 échantillons max
            for it in lst[:5]:
                # API sd.skycoin renvoie déjà address/type/version/geo
                if isinstance(it, dict):
                    samples["proxy"].append(_geo_sample(it))

        elif "services?type=vpn" in url:
            vpn = len(lst)
            for it in lst[:5]:
                if isinstance(it, dict):
                    samples["vpn"].append(_geo_sample(it))

        elif "all-transports" in url:
            transports = len(lst)

        elif "dmsg-discovery/entries" in url:
            dmsg_entries = len(lst)

        elif "rf.skywire" in url:
            rf_status = f"{status} (ok={1 if ok else 0})"
            if ok:
                rf_ok += 1

    visors = 0  # inconnu via endpoints publics (comptage “visors” pas fiable ici)
    return {
        "name": "public",
        "endpoints": endpoints,
        "payloads": payloads,
        "summary": {
            "visors": visors,
            "proxies": proxies or 0,
            "vpn": vpn or 0,
            "transports": transports or 0,
            "dmsg_entries": dmsg_entries or 0,
            "rf_status": rf_status or "n/a",
            "samples": samples,
        }
    }

# ---------- collecte Nodes (optionnelle, à partir d'une liste de PKs locale) ----------

def load_visors_pks(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pk = line.strip()
            if pk and not pk.startswith("#"):
                out.append(pk)
    return out

def collect_nodes(visors_file: str) -> Dict[str, Any]:
    """
    Par design: aucune requête réseau vers des visors individuels ici (respect de la vie privée).
    On compte seulement le nombre de PKs et on renvoie des métriques 'None' si aucune info privée fournie.
    """
    pks = load_visors_pks(visors_file)
    nodes_seen_ok = [0, 0]
    latency_avg_ms = None
    uptime_ratio_avg = None
    ut_note = None

    if not pks:
        ut_note = "No PK provided."
    else:
        # Ici, on pourrait brancher un Uptime Tracker privé si disponible.
        nodes_seen_ok = [0, 0]  # placeholder
        ut_note = "PK list loaded (private), metrics omitted by design."

    return {
        "name": "nodes",
        "endpoints": [],  # on n’expose rien
        "payloads": [],
        "summary": {
            "nodes_seen_ok": nodes_seen_ok,
            "latency_avg_ms": latency_avg_ms,
            "uptime_ratio_avg": uptime_ratio_avg,
            "ut_note": ut_note
        },
        "visors_pks": []  # ne jamais publier la liste des PKs
    }

# ---------- collecte Fiber (publique) ----------

def collect_fiber(endpoints: List[str]) -> Dict[str, Any]:
    payloads = []
    height = None
    peers = None
    seen = 0
    okc = 0
    for url in endpoints:
        ok, status, headers, data, err = fetch_url(url)
        payloads.append({
            "__url__": url,
            "__ok__": ok,
            "__status__": status,
            "__error__": err if err else None
        })
        seen += 1
        okc += 1 if ok else 0

        if ok and isinstance(data, dict):
            if "height" in data and height is None:
                height = data.get("height")
            if "peers" in data and peers is None:
                try:
                    peers = len(data.get("peers") or [])
                except Exception:
                    peers = None

    return {
        "name": "fiber",
        "endpoints": endpoints,
        "payloads": payloads,
        "summary": {
            "endpoints_seen_ok": [seen, okc],
            "height": height,
            "peers": peers
        }
    }

# ---------- rendu Markdown ----------

def fmt_none(x: Any) -> str:
    return "None" if x in (None, "", [], {}) else str(x)

def md_section_explorer(ex: Dict[str, Any]) -> str:
    s = ex["summary"]
    lines = [
        "## Explorer",
        f"- Height: {fmt_none(s.get('height'))}",
        f"- Current supply: {fmt_none(s.get('current_supply'))}",
        f"- Total supply: {fmt_none(s.get('total_supply'))}",
        f"- Coin Hours: {fmt_none(s.get('coin_hours'))}",
    ]
    if s.get("note"):
        lines.append(f"- Note: {s.get('note')}")
    return "\n".join(lines)

def md_section_public(pub: Dict[str, Any]) -> str:
    s = pub["summary"]
    lines = [
        "## Public Infra (Skywire)",
        f"- Visors: {fmt_none(s.get('visors'))}",
        f"- Proxies: {fmt_none(s.get('proxies'))}",
        f"- VPN: {fmt_none(s.get('vpn'))}",
        f"- Transports: {fmt_none(s.get('transports'))}",
        f"- DMSG entries: {fmt_none(s.get('dmsg_entries'))}",
        f"- RF status: {fmt_none(s.get('rf_status'))}",
    ]
    return "\n".join(lines)

def md_section_nodes(nodes: Dict[str, Any]) -> str:
    s = nodes["summary"]
    lines = [
        "## Nodes (if provided)",
        f"- Nodes seen/ok: {s.get('nodes_seen_ok')[0]}/{s.get('nodes_seen_ok')[1]}",
        f"- Latency avg (ms): {fmt_none(s.get('latency_avg_ms'))}",
        f"- Uptime ratio avg: {fmt_none(s.get('uptime_ratio_avg'))}",
    ]
    if s.get("ut_note"):
        lines.append(f"- UT note: {s.get('ut_note')}")
    return "\n".join(lines)

def md_section_fiber(fb: Dict[str, Any]) -> str:
    s = fb["summary"]
    lines = [
        "## Fiber (if provided)",
        f"- Fiber endpoints seen/ok: {s.get('endpoints_seen_ok')[0]}/{s.get('endpoints_seen_ok')[1]}",
        f"- Fiber height: {fmt_none(s.get('height'))} — peers: {fmt_none(s.get('peers'))}",
    ]
    return "\n".join(lines)

def make_markdown(date_utc: str, groups: List[Dict[str, Any]]) -> str:
    # récupérer par nom
    g = {grp["name"]: grp for grp in groups}
    title = f"# Skywire VitalSigns — {date_utc} UTC"

    parts = [title]
    if "explorer" in g:
        parts.append(md_section_explorer(g["explorer"]))
    if "public" in g:
        parts.append(md_section_public(g["public"]))
    if "nodes" in g:
        parts.append(md_section_nodes(g["nodes"]))
    if "fiber" in g:
        parts.append(md_section_fiber(g["fiber"]))
    return "\n\n".join(parts) + "\n"

# ---------- main ----------

def main() -> int:
    today = os.getenv("TODAY", utc_date())

    explorer_eps = getenv_list(
        "EXPLORER_ENDPOINTS",
        "https://explorer.skycoin.com/api/blockchain/metadata,https://explorer.skycoin.com/api/coinSupply"
    )
    public_eps = getenv_list(
        "PUBLIC_ENDPOINTS",
        "https://sd.skycoin.com/api/services?type=proxy,https://sd.skycoin.com/api/services?type=vpn,https://tpd.skywire.skycoin.com/all-transports,https://dmsgd.skywire.skycoin.com/dmsg-discovery/entries,https://rf.skywire.skycoin.com/"
    )
    fiber_eps = getenv_list(
        "FIBER_ENDPOINTS",
        "https://fiber.skywire.dev/api/status,https://fiber.skywire.dev/api/metrics"
    )
    visors_file = os.getenv("VISORS_PKS_FILE", ".tmp/visors_pks.txt")

    # Collecte
    explorer = collect_explorer(explorer_eps)
    public = collect_public(public_eps)
    nodes = collect_nodes(visors_file)
    fiber = collect_fiber(fiber_eps)

    doc = {
        "date_utc": today,
        "meta": {
            "repo": os.getenv("GITHUB_REPOSITORY", ""),
            "agent": "SkywireVitalSigns v3.4.0",
        },
        "groups": [explorer, public, nodes, fiber],
    }

    # Écrit fichiers
    out_dir = os.path.join("data", today)
    json_path = os.path.join(out_dir, "skywire_vitals.json")
    md_path = os.path.join(out_dir, "skywire_summary.md")

    write_json(json_path, doc)
    md = make_markdown(today, doc["groups"])
    write_text(md_path, md)

    print(f"Wrote: {json_path} & {md_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
