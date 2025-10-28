# scripts/sanitize_vitals.py
import os, json, sys

def main():
    today = os.getenv("TODAY")
    if not today:
        print("TODAY env missing"); sys.exit(1)

    p = f"data/{today}/skywire_vitals.json"
    q = f"data/{today}/skywire_vitals_sanitized.json"

    try:
        with open(p, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except FileNotFoundError:
        print("No raw JSON found."); return 0

    for g in doc.get("groups", []):
        if g.get("name") == "nodes":
            # ne jamais publier les PK
            g["visors_pks"] = []
            s = g.get("summary", {}) or {}
            if "ut_note" in s:
                s["ut_note"] = "(redacted)"
            g["summary"] = s

    os.makedirs(f"data/{today}", exist_ok=True)
    with open(q, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"Sanitized written: {q}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
