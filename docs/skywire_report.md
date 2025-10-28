# Skywire Vital Report

The **Skywire Vital Report** module reads all `data/YYYY-MM-DD/skywire_vitals.json` files, 
builds a time series, calculates daily trends (↑ ↓ →), and generates the following artifacts:

- `reports/skywire_vitals_timeseries.csv` — consolidated dataset across all days  
- `reports/skywire_vital_report.md` — summarized daily overview with metrics and charts  
- `reports/*.png` — individual graphs for each metric (latency, success ratio, nodes, proxy activity)

---

## 🧠 Purpose
This module provides a clear visual and analytical representation of the **Skywire network’s health** 
over time — essentially a heartbeat of the decentralized mesh.

It runs automatically via **GitHub Actions** every day at **06:10 UTC** (right after `skywire-vitals` ingestion), 
and can also be triggered manually from the **Actions** tab.

---

## ⚙️ Workflow Overview
1. Collects all available `skywire_vitals.json` files under `/data/YYYY-MM-DD/`.  
2. Builds a structured `pandas` DataFrame and saves it as `skywire_vitals_timeseries.csv`.  
3. Computes:
   - 3-day rolling averages
   - Percent changes between days
   - Trend arrows (↑ decrease / ↓ increase / → stable)
4. Generates clean charts using `matplotlib` and embeds them in `skywire_vital_report.md`.  

---

## 📊 Output Example
