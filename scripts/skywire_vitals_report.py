#!/usr/bin/env python3
# Builds a time series from data/**/skywire_vitals.json
# + generates a PNG chart and a short markdown report.

import json, pathlib, re, statistics as stats, datetime as dt
import matplotlib.pyplot as plt

DATA_DIR = pathlib.Path("data")
OUT_DIR  = pathlib.Path("docs"); OUT_DIR.mkdir(exist_ok=True)
CSV_PATH = OUT_DIR/"skywire_vitals_timeseries.csv"
PNG_PATH = OUT_DIR/"skywire_vitals_chart.png"
REPORT   = OUT_DIR/"skywire_vitals_report.md"

def parse_date(p: pathlib.Path):
    # expects data/YYYY-MM-DD/skywire_vitals.json
    m = re.search(r"data/(\d{4}-\d{2}-\d{2})/skywire_vitals\.json", str(p).replace("\\","/"))
    return dt.date.fromisoformat(m.group(1)) if m else None

def collect():
    rows = []
    for p in sorted(DATA_DIR.glob("*/skywire_vitals.json")):
        d = parse_date(p)
        if not d: continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            agg = obj.get("aggregate", {})
            rows.append({
                "date": d.isoformat(),
                "nodes_active_est": agg.get("nodes_active_est"),
                "latency_ms_avg": agg.get("latency_ms_avg"),
                "success_ratio_avg": agg.get("success_ratio_avg"),
                "uptime_ratio_avg": agg.get("uptime_ratio_avg"),
                "proxy_activity_sum": agg.get("proxy_activity_sum"),
            })
        except Exception:
            pass
    return rows

def write_csv(rows):
    if not rows: return
    headers = list(rows[0].keys())
    with CSV_PATH.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join("" if r[k] is None else str(r[k]) for k in headers) + "\n")

def moving_avg(values, window=7):
    out = []
    for i in range(len(values)):
        win = [v for v in values[max(0,i-window+1):i+1] if v is not None]
        out.append(sum(win)/len(win) if win else None)
    return out

def zscores(values):
    xs = [v for v in values if v is not None]
    if len(xs) < 2: return [None]*len(values)
    mu = stats.mean(xs); sd = stats.pstdev(xs) or 1.0
    return [ (None if v is None else (v-mu)/sd) for v in values ]

def plot(rows):
    if not rows: return
    dates = [r["date"] for r in rows]
    x = list(range(len(dates)))

    def series(key): return [r[key] for r in rows]

    nodes = series("nodes_active_est")
    lat   = series("latency_ms_avg")
    succ  = series("success_ratio_avg")
    prox  = series("proxy_activity_sum")

    nodes_ma = moving_avg(nodes, 7)
    lat_ma   = moving_avg(lat, 7)

    plt.figure(figsize=(10,6), dpi=140)
    # (no specific colors; rely on defaults)
    if any(v is not None for v in nodes):
        plt.plot(x, nodes, label="Nodes est.")
        if any(v is not None for v in nodes_ma): plt.plot(x, nodes_ma, label="Nodes (7d MA)")
    if any(v is not None for v in lat):
        plt.plot(x, lat, label="Latency ms")
        if any(v is not None for v in lat_ma): plt.plot(x, lat_ma, label="Latency (7d MA)")
    if any(v is not None for v in succ):
        plt.plot(x, [v*100 if v is not None else None for v in succ], label="Success %")
    if any(v is not None for v in prox):
        plt.plot(x, prox, label="Proxy activity (tx)")

    plt.xticks(x, dates, rotation=45, ha="right")
    plt.tight_layout()
    plt.legend()
    plt.title("Skywire VitalSigns – Time Series")
    plt.savefig(PNG_PATH)
    plt.close()

def write_report(rows):
    if not rows:
        REPORT.write_text("# Skywire VitalSigns report\n\nAucune donnée encore.\n", encoding="utf-8")
        return

    latest = rows[-1]
    def fmt(v, pct=False):
        if v is None: return "–"
        return f"{v*100:.2f}%" if pct else f"{v:.2f}" if isinstance(v, float) else str(v)

    body = [
        "# Skywire VitalSigns – Daily Report",
        "",
        f"**Latest date**: {latest['date']}",
        "",
        f"- Nodes active (est.): {fmt(latest['nodes_active_est'])}",
        f"- Latency avg (ms): {fmt(latest['latency_ms_avg'])}",
        f"- Success ratio: {fmt(latest['success_ratio_avg'], pct=True)}",
        f"- Uptime avg: {fmt(latest['uptime_ratio_avg'], pct=True)}",
        f"- Proxy activity (tx): {fmt(latest['proxy_activity_sum'])}",
        "",
        f"![chart](./{PNG_PATH.name})",
        "",
        "> Generated automatically by `scripts/skywire_vitals_report.py`."
    ]
    REPORT.write_text("\n".join(body), encoding="utf-8")

if __name__ == "__main__":
    rows = collect()
    write_csv(rows)
    plot(rows)
    write_report(rows)
