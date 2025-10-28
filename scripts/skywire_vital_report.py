#!/usr/bin/env python3
# Skywire Vital Report
# - Agrège tous les data/YYYY-MM-DD/skywire_vitals.json
# - Construit une série temporelle
# - Calcule tendances (↑ ↓ →) et variations %
# - Génère graphiques PNG + résumé Markdown

import json, re, os, pathlib, datetime as dt
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = pathlib.Path("data")
OUT_DIR = pathlib.Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TS_CSV = OUT_DIR / "skywire_vitals_timeseries.csv"
REPORT_MD = OUT_DIR / "skywire_vital_report.md"

DATE_RX = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def collect() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not DATA_DIR.exists():
        return pd.DataFrame()

    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not DATE_RX.match(d.name):
            continue
        j = d / "skywire_vitals.json"
        if not j.exists():
            continue
        try:
            payload = json.loads(j.read_text(encoding="utf-8"))
            agg = payload.get("aggregate", {}) or {}
            rows.append({
                "date": d.name,
                "nodes_active_est": agg.get("nodes_active_est"),
                "latency_ms_avg": agg.get("latency_ms_avg"),
                "uptime_ratio_avg": agg.get("uptime_ratio_avg"),
                "success_ratio_avg": agg.get("success_ratio_avg"),
                "proxy_activity_sum": agg.get("proxy_activity_sum"),
            })
        except Exception:
            # ignore fichiers corrompus
            pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def trend_arrow(delta: float, tol: float = 0.5) -> str:
    """
    delta exprimé dans l’unité de la métrique (ms pour latence, points pour %*100, etc.)
    tol = zone neutre
    """
    if pd.isna(delta):
        return "·"
    if delta > tol:
        return "↑"
    if delta < -tol:
        return "↓"
    return "→"

def pct_change_str(prev, curr) -> str:
    if pd.isna(prev) or pd.isna(curr) or prev == 0:
        return "–"
    pc = (curr - prev) / abs(prev) * 100
    return f"{pc:+.1f}%"

def mk_plot(df: pd.DataFrame, y: str, title: str, fname: str, ylabel: str):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df[y], marker="o")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out

def main():
    df = collect()
    if df.empty:
        REPORT_MD.write_text("# Skywire Vital Report\n\nAucune donnée trouvée.\n", encoding="utf-8")
        return

    # Sauvegarde brute
    df_out = df.copy()
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
    df_out.to_csv(TS_CSV, index=False)

    # Rolling (si assez de points)
    rolling = None
    if len(df) >= 3:
        rolling = df.set_index("date").rolling(window=3, min_periods=1).mean().reset_index()

    # Dernier / précédent
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else pd.Series()

    # Prépare tendances (unités cohérentes)
    lat_delta = (last["latency_ms_avg"] - prev["latency_ms_avg"]) if len(df) >= 2 else None
    succ_delta = ((last["success_ratio_avg"] - prev["success_ratio_avg"]) * 100) if len(df) >= 2 else None
    proxy_delta = (last["proxy_activity_sum"] - prev["proxy_activity_sum"]) if len(df) >= 2 else None
    nodes_delta = (last["nodes_active_est"] - prev["nodes_active_est"]) if len(df) >= 2 else None

    # Graphiques (si colonnes disponibles)
    figs = []
    if df["success_ratio_avg"].notna().any():
        figs.append(mk_plot(df, "success_ratio_avg", "Success ratio (avg)", "success_ratio_avg.png", "ratio"))
    if df["latency_ms_avg"].notna().any():
        figs.append(mk_plot(df, "latency_ms_avg", "Latency average (ms)", "latency_ms_avg.png", "ms"))
    if df["proxy_activity_sum"].notna().any():
        figs.append(mk_plot(df, "proxy_activity_sum", "Proxy activity (transactions)", "proxy_activity_sum.png", "count"))
    if df["nodes_active_est"].notna().any():
        figs.append(mk_plot(df, "nodes_active_est", "Nodes active (est.)", "nodes_active_est.png", "nodes"))

    # Génération rapport Markdown
    lines = []
    lines.append("# Skywire Vital Report")
    lines.append("")
    lines.append(f"**Dernière mesure : {last['date'].strftime('%Y-%m-%d')}**")
    lines.append("")
    lines.append("## Synthèse du jour")
    lines.append("")
    def fmt(v, mult=1, suffix=""):
        if pd.isna(v):
            return "–"
        return f"{v*mult:.2f}{suffix}" if isinstance(v, float) else f"{v}"

    lines.append(f"- **Success ratio** : {fmt(last.get('success_ratio_avg'), 100, '%')} "
                 f"({trend_arrow(succ_delta, tol=0.3)} {pct_change_str(prev.get('success_ratio_avg'), last.get('success_ratio_avg'))})")
    lines.append(f"- **Latency avg** : {fmt(last.get('latency_ms_avg'))} ms "
                 f"({trend_arrow(lat_delta, tol=1.0)} {pct_change_str(prev.get('latency_ms_avg'), last.get('latency_ms_avg'))})")
    lines.append(f"- **Nodes active (est.)** : {fmt(last.get('nodes_active_est'))} "
                 f"({trend_arrow(nodes_delta, tol=1.0)} {pct_change_str(prev.get('nodes_active_est'), last.get('nodes_active_est'))})")
    lines.append(f"- **Proxy activity (tx)** : {fmt(last.get('proxy_activity_sum'))} "
                 f"({trend_arrow(proxy_delta, tol=5.0)} {pct_change_str(prev.get('proxy_activity_sum'), last.get('proxy_activity_sum'))})")
    lines.append("")

    if rolling is not None:
        lines.append("## Moyenne glissante (3 jours)")
        def last_rolling(col):
            v = rolling[col].iloc[-1]
            if pd.isna(v): return "–"
            return f"{v:.2f}"
        lines.append(f"- Success ratio (avg) : {last_rolling('success_ratio_avg')}")
        lines.append(f"- Latency avg (ms) : {last_rolling('latency_ms_avg')}")
        lines.append(f"- Nodes active (est.) : {last_rolling('nodes_active_est')}")
        lines.append(f"- Proxy activity (tx) : {last_rolling('proxy_activity_sum')}")
        lines.append("")

    if figs:
        lines.append("## Graphiques")
        for f in figs:
            # Affichage relatif
            rel = f.as_posix()
            lines.append(f"![{f.stem}]({rel})")
        lines.append("")

    lines.append("> Rapport généré automatiquement par **Sigma – Skywire Vital Report**.")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
