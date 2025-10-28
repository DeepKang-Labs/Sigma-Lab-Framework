#!/usr/bin/env python3
# Skywire Vital Report (DeepKang Labs / Sigma)
# ---------------------------------------------------------------
# Lit tous les data/YYYY-MM-DD/skywire_vitals.json,
# construit une série temporelle, calcule des tendances,
# génère des graphiques + un rapport Markdown,
# et produit des badges SVG (success ratio, latency, updated).
# ---------------------------------------------------------------

import json, re, os, pathlib
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = pathlib.Path("data")
OUT_ROOT = pathlib.Path("reports")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DATE_RX = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def collect_timeseries() -> pd.DataFrame:
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
            # ignorer fichiers corrompus
            pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

# ---------------- Badges (SVG) ----------------
def _badge_svg(label: str, value: str, color: str = "#2ea44f"):
    import html
    label = html.escape(label)
    value = html.escape(value)

    def w(text): return 6 * len(text) + 20  # approximation simple
    wl, wv = w(label), w(value)
    total = wl + wv
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total}" height="20" role="img" aria-label="{label}: {value}">
  <linearGradient id="a" x2="0" y2="100%">
    <stop offset="0" stop-color="#fff" stop-opacity=".7"/>
    <stop offset=".1" stop-opacity=".1"/>
    <stop offset=".9" stop-opacity=".1"/>
  </linearGradient>
  <rect rx="3" width="{total}" height="20" fill="#555"/>
  <rect rx="3" x="{wl}" width="{wv}" height="20" fill="{color}"/>
  <path fill="{color}" d="M{wl} 0h4v20h-4z"/>
  <rect rx="3" width="{total}" height="20" fill="url(#a)"/>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{wl/2}" y="14">{label}</text>
    <text x="{wl + wv/2}" y="14">{value}</text>
  </g>
</svg>'''

def _badge_color_success_ratio(ratio: float | None):
    if ratio is None: return "#6c757d"      # gray
    if ratio >= 0.98: return "#2ea44f"      # green
    if ratio >= 0.90: return "#dbab09"      # yellow
    return "#d73a49"                        # red

def _badge_color_latency(ms: float | None):
    if ms is None: return "#6c757d"
    if ms <= 80: return "#2ea44f"
    if ms <= 200: return "#dbab09"
    return "#d73a49"

def write_badge(path: pathlib.Path, label: str, value: str, color: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_badge_svg(label, value, color), encoding="utf-8")

# ---------------- Utilitaires ----------------
def pct_change_str(prev, curr) -> str:
    if pd.isna(prev) or pd.isna(curr) or prev == 0:
        return "–"
    pc = (curr - prev) / abs(prev) * 100
    return f"{pc:+.1f}%"

def trend_arrow(delta: float | None, tol: float) -> str:
    if delta is None or pd.isna(delta): return "·"
    if delta > tol: return "↑"
    if delta < -tol: return "↓"
    return "→"

def mk_plot(out_dir: pathlib.Path, df: pd.DataFrame, y: str, title: str, ylabel: str):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df[y], marker="o")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / f"{y}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out

def main():
    df = collect_timeseries()
    # Crée le dossier du jour pour stocker le rapport
    if df.empty:
        today_dir = OUT_ROOT / "0000-00-00"
        today_dir.mkdir(parents=True, exist_ok=True)
        (today_dir / "skywire_vital_report.md").write_text(
            "# Skywire Vital Report\n\nNo data found yet.\n",
            encoding="utf-8"
        )
        return

    last_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
    out_dir = OUT_ROOT / last_date
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde CSV de la série temporelle (toutes dates)
    df_out = df.copy()
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")
    (out_dir / "skywire_vitals_timeseries.csv").write_text(
        df_out.to_csv(index=False), encoding="utf-8"
    )

    # Calculs tendances
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else pd.Series(dtype=float)

    def get_delta(col, scale=1.0):
        if len(df) < 2: return None
        a = last.get(col); b = prev.get(col)
        if pd.isna(a) or pd.isna(b): return None
        return (a - b) * scale

    lat_delta   = get_delta("latency_ms_avg", 1.0)
    succ_delta  = get_delta("success_ratio_avg", 100.0)
    proxy_delta = get_delta("proxy_activity_sum", 1.0)
    nodes_delta = get_delta("nodes_active_est", 1.0)

    # Graphiques (génère seulement si données présentes)
    figs = []
    if df["success_ratio_avg"].notna().any():
        figs.append(mk_plot(out_dir, df, "success_ratio_avg", "Success ratio (avg)", "ratio"))
    if df["latency_ms_avg"].notna().any():
        figs.append(mk_plot(out_dir, df, "latency_ms_avg", "Latency average (ms)", "ms"))
    if df["proxy_activity_sum"].notna().any():
        figs.append(mk_plot(out_dir, df, "proxy_activity_sum", "Proxy activity (transactions)", "count"))
    if df["nodes_active_est"].notna().any():
        figs.append(mk_plot(out_dir, df, "nodes_active_est", "Nodes active (est.)", "nodes"))

    # Rapport Markdown
    def fmt(v, mult=1, suffix=""):
        if pd.isna(v): return "–"
        return f"{v*mult:.2f}{suffix}" if isinstance(v, float) else f"{v}"

    lines = []
    lines.append("# Skywire Vital Report")
    lines.append("")
    lines.append(f"**Last measurement: {last_date}**")
    lines.append("")
    lines.append("## Daily Summary")
    lines.append(f"- **Success ratio** : {fmt(last.get('success_ratio_avg'), 100, '%')} "
                 f"({trend_arrow(succ_delta, 0.3)} {pct_change_str(prev.get('success_ratio_avg'), last.get('success_ratio_avg'))})")
    lines.append(f"- **Latency avg** : {fmt(last.get('latency_ms_avg'))} ms "
                 f"({trend_arrow(lat_delta, 1.0)} {pct_change_str(prev.get('latency_ms_avg'), last.get('latency_ms_avg'))})")
    lines.append(f"- **Nodes active (est.)** : {fmt(last.get('nodes_active_est'))} "
                 f"({trend_arrow(nodes_delta, 1.0)} {pct_change_str(prev.get('nodes_active_est'), last.get('nodes_active_est'))})")
    lines.append(f"- **Proxy activity (tx)** : {fmt(last.get('proxy_activity_sum'))} "
                 f"({trend_arrow(proxy_delta, 5.0)} {pct_change_str(prev.get('proxy_activity_sum'), last.get('proxy_activity_sum'))})")
    lines.append("")

    # Graphs dans le rapport
    if figs:
        lines.append("## Charts")
        for f in figs:
            rel = f.name  # chemin relatif dans le dossier du jour
            lines.append(f"![{f.stem}]({rel})")
        lines.append("")

    lines.append("> Auto-generated by **Sigma – Skywire Vital Report**.")
    (out_dir / "skywire_vital_report.md").write_text("\n".join(lines), encoding="utf-8")

    # Badges dynamiques
    badges_dir = out_dir / "badges"
    sr = last.get("success_ratio_avg")
    sr_pct = round(sr * 100, 1) if sr is not None and not pd.isna(sr) else None
    write_badge(
        badges_dir / "success_ratio.svg",
        "success ratio",
        f"{sr_pct:.1f}%" if sr_pct is not None else "n/a",
        _badge_color_success_ratio(sr if sr is not None and not pd.isna(sr) else None)
    )

    lat = last.get("latency_ms_avg")
    write_badge(
        badges_dir / "latency.svg",
        "latency",
        f"{lat:.0f} ms" if lat is not None and not pd.isna(lat) else "n/a",
        _badge_color_latency(lat if lat is not None and not pd.isna(lat) else None)
    )

    write_badge(
        badges_dir / "updated.svg",
        "updated",
        last_date,
        "#0366d6"
    )

if __name__ == "__main__":
    main()
