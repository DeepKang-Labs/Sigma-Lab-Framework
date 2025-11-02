#!/usr/bin/env python3
import json, sys, math
from pathlib import Path
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
STATE = ROOT / "state"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

hard_errors = []
warnings = []

# ✅ Fonction corrigée : accepte clé grecque OU ASCII
def require_key(d, key, where):
    equivalents = {
        "α": ["α", "alpha"],
        "λ": ["λ", "lambda"],
        "γ": ["γ", "gamma"],
        "μ": ["μ", "mu"],
        "β": ["β", "beta"]
    }
    for k in equivalents.get(key, [key]):
        if k in d:
            return d[k]
    hard_errors.append(f"Missing key {key} in {where}")
    return None


def in_range(v, lo, hi, name):
    if v is None:
        return None
    if not (lo <= v <= hi):
        hard_errors.append(f"{name}={v} out of range [{lo},{hi}]")
    return v


def main():
    params_path = CONFIGS / "sigma_params.json"
    metrics_path = STATE / "last_metrics.json"
    params = {}
    metrics = {}

    # Charger les fichiers JSON
    try:
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
    except Exception as e:
        hard_errors.append(f"Cannot read params: {e}")

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        pass

    # --- Vérification des paramètres clés
    α = require_key(params, "α", "configs/sigma_params.json")
    λ = require_key(params, "λ", "configs/sigma_params.json")
    γ = require_key(params, "γ", "configs/sigma_params.json")
    μ = require_key(params, "μ", "configs/sigma_params.json")
    β = require_key(params, "β", "configs/sigma_params.json")

    if all(v is not None for v in [α, λ, γ, μ]):
        in_range(α, 0.0, 1.0, "α/alpha")
        in_range(λ, 0.0, 1.0, "λ/lambda")
        in_range(γ, 0.0, 1.0, "γ/gamma")
        in_range(μ, 0.0, 1.0, "μ/mu")

    # --- Small gain
    small_gain = sum(v for v in [α, β, γ] if v is not None)
    if small_gain >= 0.45:
        hard_errors.append(f"Small-gain violated: α+β+γ={small_gain}")
    elif small_gain >= 0.40:
        warnings.append(f"Small-gain near limit: {small_gain}")

    # --- Rapport final
    report = {
        "params_file": str(params_path),
        "metrics_file": str(metrics_path),
        "small_gain": small_gain,
        "warnings": warnings,
        "hard_errors": hard_errors
    }

    with open(REPORTS / "invariants_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if hard_errors:
        print("[INVARIANTS][FAIL]")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(1)
    else:
        print("[INVARIANTS][OK]")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(0)


if __name__ == "__main__":
    main()
