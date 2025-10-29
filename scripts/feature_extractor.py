# scripts/feature_extractor.py
import os
import json
from datetime import datetime

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json(p: str, default):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default

def dump_json(p: str, obj):
    safe_mkdir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def extract_scalar(v, default=0.0):
    try:
        if v is None:
            return float(default)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and v.strip().isdigit():
            return float(v)
        # ex: "200", "404"
        return float(v)
    except Exception:
        return float(default)

def main():
    today = os.getenv("TODAY") or datetime.utcnow().strftime("%Y-%m-%d")
    day_dir = os.path.join("data", today)

    heartbeat_path = os.path.join(day_dir, "sigma_heartbeat.json")
    features_path  = os.path.join(day_dir, "features.json")
    history_path   = os.path.join(".state", "features_history.json")

    hb = load_json(heartbeat_path, {})
    now_iso = datetime.utcnow().isoformat() + "Z"

    # Champs attendus dans le heartbeat (préparé par sigma_node.py)
    # {
    #   "ts_utc": "...",
    #   "http_ok_ratio": float,
    #   "sample_size": int,
    #   "status_counters": {"2xx": n, "3xx": n, "4xx": n, "5xx": n},
    #   "latency_avg_ms": float,
    #   ...
    # }
    ok_ratio       = extract_scalar(hb.get("http_ok_ratio"), 0.0)
    sample_size    = extract_scalar(hb.get("sample_size"), 0.0)
    latency_avg_ms = extract_scalar(hb.get("latency_avg_ms"), 0.0)

    sc = hb.get("status_counters", {}) or {}
    s2 = extract_scalar(sc.get("2xx"), 0.0)
    s3 = extract_scalar(sc.get("3xx"), 0.0)
    s4 = extract_scalar(sc.get("4xx"), 0.0)
    s5 = extract_scalar(sc.get("5xx"), 0.0)

    # Ratios normalisés (évite division par zéro)
    total = max(1.0, s2 + s3 + s4 + s5)
    r2 = s2 / total
    r3 = s3 / total
    r4 = s4 / total
    r5 = s5 / total

    features = {
        "ts_utc": now_iso,
        "ok_ratio": ok_ratio,
        "latency_avg_ms": latency_avg_ms,
        "sample_size": sample_size,
        "ratio_2xx": r2,
        "ratio_3xx": r3,
        "ratio_4xx": r4,
        "ratio_5xx": r5
    }

    # Écrit les features du jour
    dump_json(features_path, features)

    # Met à jour l’historique borné (fenêtre 200)
    history = load_json(history_path, [])
    history.append({"date": today, **features})
    if len(history) > 200:
        history = history[-200:]
    dump_json(history_path, history)

    print(f"✅ features written: {features_path}")
    print(f"🧳 history updated: {history_path} (len={len(history)})")

if __name__ == "__main__":
    main()
