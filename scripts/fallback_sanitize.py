# scripts/fallback_sanitize.py
import os, json, datetime, pathlib

date = os.environ.get("DATE", "unknown")
inp      = f"data/{date}/skywire_vitals.json"
out_json = f"reports/{date}/skywire_vitals_sanitized.json"
out_md   = f"reports/{date}/skywire_vital_report.md"

pathlib.Path(os.path.dirname(out_json)).mkdir(parents=True, exist_ok=True)

with open(inp, "r", encoding="utf-8") as f:
    data = json.load(f)

# light sanitize top-level keys if present
if isinstance(data, dict):
    for k in ("ip", "public_key", "debug"):
        data.pop(k, None)

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

md = [
  f"# Skywire Vital Report ({date})",
  "",
  f"- Generated: {datetime.datetime.utcnow().isoformat()}Z",
  f"- Fields: {len(data) if hasattr(data, '__len__') else 'n/a'}",
  "",
  "This is an auto-generated minimal report.",
]
with open(out_md, "w", encoding="utf-8") as f:
  f.write("\n".join(md))
