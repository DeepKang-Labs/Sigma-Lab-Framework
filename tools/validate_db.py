import yaml, pathlib, sys

SCHEMA = yaml.safe_load(open("tools/schema_case.yaml", encoding="utf-8"))
ROOT = pathlib.Path("corpus/historical")

def fail(msg, errors): errors.append(msg)

errors = []
for y in ROOT.rglob("*.yaml"):
    d = yaml.safe_load(open(y, encoding="utf-8"))

    # status allowed
    if d.get("status") not in SCHEMA["validation"]["allowed_status"]:
        fail(f"{y}: invalid status '{d.get('status')}'", errors)

    # required fields
    for f in SCHEMA["validation"]["required_fields"]:
        if f not in d:
            fail(f"{y}: missing field '{f}'", errors)

    # sources count
    if not isinstance(d.get("sources"), list) or len(d["sources"]) < SCHEMA["validation"]["min_sources"]:
        fail(f"{y}: need >= {SCHEMA['validation']['min_sources']} sources", errors)

    # parameters completeness
    p = d.get("parameters", {})
    needed = ("base_risk","irreversibility","equity_distribution","equity_gini")
    for k in needed:
        if k not in p:
            fail(f"{y}: missing parameters.{k}", errors)

    # equity_distribution sums ~ 1
    dist = p.get("equity_distribution",[0,0])
    if not (isinstance(dist, list) and len(dist)==2):
        fail(f"{y}: equity_distribution must be a 2-item list", errors)
    s = sum(dist) if isinstance(dist, list) else 0
    if not (0.95 <= s <= 1.05):
        fail(f"{y}: equity_distribution should sum ≈1 (got {s})", errors)

if errors:
    print("❌ Validation failed:\n" + "\n".join(errors))
    sys.exit(1)

print("✅ All historical cases validated.")
