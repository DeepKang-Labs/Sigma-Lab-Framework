#!/usr/bin/env python3
# Skywire VitalSigns – collecte quotidienne de "signes vitaux" réseau
# - Compatible endpoints hétérogènes (Explorer, nodes…)
# - Normalise ce qu'il peut, ignore le reste, loggue les erreurs
#
# Sorties:
#   data/YYYY-MM-DD/skywire_vitals.json
#   data/YYYY-MM-DD/skywire_summary.md
#   DATALOG.md (append)

import os, json, pathlib, datetime as dt, statistics as stats
import requests, yaml

TODAY = dt.date.today().isoformat()
OUT_DIR = pathlib.Path("data")/TODAY
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_MD = OUT_DIR/"skywire_summary.md"
VITALS_JSON = OUT_DIR/"skywire_vitals.json"
DATALOG = pathlib.Path("DATALOG.md")

TIMEOUT = 20

def load_config():
    cfg_path = os.getenv("SKYWIRE_CONFIG")
    if cfg_path and pathlib.Path(cfg_path).exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        endpoints = cfg.get("endpoints", [])
    else:
        csv = os.getenv("SKYWIRE_ENDPOINTS", "")
        endpoints = [u.strip() for u in csv.split(",") if u.strip()]
    return endpoints

def fetch_json(url):
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def normalize_from_explorer(url, payload):
    """
    Heuristiques pour l'explorer Skycoin:
      - /api/blockchain/metadata : peut contenir hauteur, version…
      - /api/coinSupply : supply total / coin hours
      - /api/transactions : liste de tx (activité/proxy)
    On dérive des 'vital signs' approximatifs :
      - nodes_active_est -> None (pas dispo ici)
      - latency_ms_avg   -> None (pas dispo)
      - success_ratio    -> 1.0 si réponse OK
      - proxy_activity   -> taille liste tx
    """
    base = {
        "nodes_active": None,
        "latency_ms_avg": None,
        "uptime_ratio_avg": None,
        "success_ratio": 1.0,   # si endpoint répond OK
        "proxy_activity": None
    }
    if "transactions" in url and isinstance(payload, list):
        base["proxy_activity"] = len(payload)
    return base

def normalize_generic(payload):
    """
    Essaie d'extraire des métriques si disponibles (schéma node API).
    """
    nodes_active = None
    latency_avg = None
    uptime_avg = None
    success_ratio = None

    for k in ("nodes_active", "active_nodes", "num_nodes", "node_count"):
        if k in payload and isinstance(payload[k], (int, float)):
            nodes_active = int(payload[k]); break

    for k in ("latency_ms_avg","avg_latency_ms","latency_avg_ms"):
        if k in payload and isinstance(payload[k], (int, float)):
            latency_avg = float(payload[k]); break

    for k in ("uptime_ratio_avg","avg_uptime","uptime_avg"):
        if k in payload and isinstance(payload[k], (int, float)):
            uptime_avg = float(payload[k]); break

    for k in ("success_ratio","connect_success_ratio","ok_ratio"):
        if k in payload and isinstance(payload[k], (int, float)):
            success_ratio = float(payload[k]); break

    # liste de nœuds possible
    if "nodes" in payload and isinstance(payload["nodes"], list) and payload["nodes"]:
        nodes = payload["nodes"]

        if nodes_active is None:
            ups = [n for n in nodes if str(n.get("status","")).lower() in ("up","ok","alive","active")]
            nodes_active = len(ups) if ups else len(nodes)

        if latency_avg is None:
            lats = [float(n.get("latency_ms")) for n in nodes if n.get("latency_ms") is not None]
            if lats:
                latency_avg = float(stats.mean(lats))

        if uptime_avg is None:
            upt = [float(n.get("uptime_ratio")) for n in nodes if n.get("uptime_ratio") is not None]
            if upt:
                uptime_avg = float(stats.mean(upt))

        if success_ratio is None:
            oks = [n for n in nodes if n.get("last_check_ok") in (True, "true", 1)]
            success_ratio = len(oks)/len(nodes) if nodes else None

    return {
        "nodes_active": nodes_active,
        "latency_ms_avg": latency_avg,
        "uptime_ratio_avg": uptime_avg,
        "success_ratio": success_ratio
    }

def normalize(url, payload):
    try:
        if "explorer.skycoin.com/api" in url:
            return normalize_from_explorer(url, payload)
    except Exception:
        pass
    return normalize_generic(payload)

def merge_metrics(collected):
    vals = {"nodes_active": [], "latency_ms_avg": [], "uptime_ratio_avg": [], "success_ratio": [], "proxy_activity": []}
    for item in collected:
        for k in vals:
            v = item.get(k)
            if isinstance(v, (int, float)):
                vals[k].append(v)

    def mean_or_none(a):
        return float(stats.mean(a)) if a else None

    agg = {
        "nodes_active_est": int(stats.mean(vals["nodes_active"])) if vals["nodes_active"] else None,
        "latency_ms_avg": mean_or_none(vals["latency_ms_avg"]),
        "uptime_ratio_avg": mean_or_none(vals["uptime_ratio_avg"]),
        "success_ratio_avg": mean_or_none(vals["success_ratio"]),
        "proxy_activity_sum": sum(vals["proxy_activity"]) if vals["proxy_activity"] else None
    }
    return agg

def write_summary(agg, sources_count):
    lines = [
        f"# Skywire VitalSigns – {TODAY}",
        "",
        f"- **Sources agrégées** : {sources_count}",
        f"- **Nœuds actifs (estim.)** : {agg.get('nodes_active_est','?')}",
        f"- **Latence moyenne** : {round(agg['latency_ms_avg'],2) if agg.get('latency_ms_avg') is not None else '–'} ms",
        f"- **Uptime moyen** : {round(agg['uptime_ratio_avg']*100,2)} %" if agg.get('uptime_ratio_avg') is not None else "–",
        f"- **Taux de succès** : {round(agg['success_ratio_avg']*100,2)} %" if agg.get('success_ratio_avg') is not None else "–",
        f"- **Activité proxy (tx)** : {int(agg['proxy_activity_sum']) if agg.get('proxy_activity_sum') is not None else '–'}",
        "",
        "> Rapport généré automatiquement par l’agent **Skywire VitalSigns**."
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")

def append_datalog(success=True, msg="ok"):
    line = f"- {TODAY}: skywire-vitals {msg}"
    if DATALOG.exists():
        txt = DATALOG.read_text(encoding="utf-8").strip()
        if txt: txt += "\n"
        DATALOG.write_text(txt + line + "\n", encoding="utf-8")
    else:
        DATALOG.write_text(line + "\n", encoding="utf-8")

def tg_alert(text):
    bot = os.getenv("TG_BOT_TOKEN")
    chat = os.getenv("TG_CHAT_ID")
    if not (bot and chat): return
    try:
        requests.get(
            f"https://api.telegram.org/bot{bot}/sendMessage",
            params={"chat_id": chat, "text": text}, timeout=10
        )
    except Exception:
        pass

def main():
    endpoints = load_config()
    if not endpoints:
        append_datalog(False, "no-endpoints")
        raise SystemExit("No endpoints provided. Set SKYWIRE_CONFIG or SKYWIRE_ENDPOINTS.")

    collected, errors = [], {}
    for url in endpoints:
        try:
            data = fetch_json(url)
            collected.append(normalize(url, data))
        except Exception as e:
            errors[url] = str(e)

    agg = merge_metrics(collected)
    payload = {
        "date": TODAY,
        "sources": endpoints,
        "metrics_per_source": collected,
        "aggregate": agg,
        "errors": errors
    }

    VITALS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(agg, len(endpoints))
    append_datalog(True, f"sources={len(endpoints)} collected={len(collected)} errors={len(errors)}")

    # Alerte simple si dégradation
    if agg.get("success_ratio_avg") is not None and agg["success_ratio_avg"] < 0.7:
        tg_alert(f"⚠️ Skywire VitalSigns: success_ratio_avg < 70% ({round(agg['success_ratio_avg']*100,1)}%)")
    if agg.get("latency_ms_avg") is not None and agg["latency_ms_avg"] > 300:
        tg_alert(f"⚠️ Skywire VitalSigns: latence élevée ({round(agg['latency_ms_avg'],0)} ms)")

if __name__ == "__main__":
    main()
