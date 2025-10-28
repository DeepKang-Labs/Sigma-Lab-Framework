import os, json, sys

today = os.getenv("TODAY")
p = f"data/{today}/skywire_vitals.json"
q = f"data/{today}/skywire_vitals_sanitized.json"

try:
    with open(p) as f:
        doc = json.load(f)
except FileNotFoundError:
    print("⚠️ No raw JSON found.")
    sys.exit(0)

for g in doc.get("groups", []):
    if g.get("name") == "nodes":
        # ne jamais publier la liste de PK
        g["visors_pks"] = []
        s = g.get("summary", {}) or {}
        if "ut_note" in s:
            s["ut_note"] = "(redacted)"
        g["summary"] = s

with open(q, "w") as f:
    json.dump(doc, f, indent=2, ensure_ascii=False)

print(f"✅ Sanitized written: {q}")
