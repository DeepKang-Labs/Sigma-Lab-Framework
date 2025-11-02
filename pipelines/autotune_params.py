"""
Sigma-Lab Autotuning Module
----------------------------------
Automatically adjusts Sigma parameters based on previous run metrics.
"""

import json
import os
from pathlib import Path

PARAMS_FILE = Path("configs/sigma_params.json")
METRICS_FILE = Path("state/last_metrics.json")
OUTPUT_FILE = Path("configs/sigma_params_autotuned.json")

def load_json(file_path):
    if file_path.exists():
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        print(f"[WARN] Missing file: {file_path}")
        return {}

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
        print(f"[INFO] Saved: {file_path}")

def autotune(params, metrics):
    if not params or not metrics:
        print("[WARN] No data to autotune.")
        return params

    tuned = params.copy()

    stability = metrics.get("stability", 0.5)
    resilience = metrics.get("resilience", 0.5)
    sigma_level = metrics.get("sigma_percentage", 50.0)

    # Ajustements progressifs
    if stability < 0.6:
        tuned["λ"] = max(0.15, tuned.get("λ", 0.2) * 0.95)
        print(f"[TUNE] Reduced λ to {tuned['λ']:.3f} for better stability.")

    if resilience < 0.5:
        tuned["μ"] = min(0.25, tuned.get("μ", 0.2) + 0.01)
        print(f"[TUNE] Increased μ to {tuned['μ']:.3f} for stronger resilience.")

    if sigma_level < 40:
        tuned["α"] = min(0.35, tuned.get("α", 0.3) + 0.01)
        print(f"[TUNE] Increased α to {tuned['α']:.3f} to enhance reflexivity.")

    if sigma_level > 70:
        tuned["α"] = max(0.28, tuned.get("α", 0.3) - 0.01)
        print(f"[TUNE] Reduced α to {tuned['α']:.3f} to reduce oscillations.")

    tuned["timestamp_autotuned"] = metrics.get("timestamp", "unknown")
    return tuned

def main():
    print("[INIT] Autotune Sigma parameters...")

    params = load_json(PARAMS_FILE)
    metrics = load_json(METRICS_FILE)

    tuned_params = autotune(params, metrics)

    if tuned_params:
        save_json(tuned_params, OUTPUT_FILE)
        print("[DONE] Autotuning complete ✅")
    else:
        print("[SKIP] Nothing to tune.")

if __name__ == "__main__":
    main()
