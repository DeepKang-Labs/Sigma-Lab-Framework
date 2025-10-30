# scripts/fallback_vitals.py
import os, json, datetime, pathlib, sys

date = os.environ.get("DATE", "unknown")
out_json = f"data/{date}/skywire_vitals.json"
out_md   = f"data/{date}/skywire_summary.md"

payload = {
    "date_utc": date,
    "meta": {"source": "fallback", "repo": "Sigma-Lab-Framework"},
    "groups": ["explorer", "public"],
    "payloads": [{"visor": "demo", "ok": True, "latency_ms": 42.0}],
}

pathlib.Path(os.path.dirname(out_json)).mkdir(parents=True, exist_ok=True)

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

with open(out_md, "w", encoding="utf-8") as f:
    f.write(f"# Skywire VitalSigns (fallback) — {date}\n\n")
    f.write(f"- generated: {datetime.datetime.utcnow().isoformat()}Z\n")
    f.write(f"- items: {len(payload.get('payloads', []))}\n")
