#!/usr/bin/env python3
import json, sys, os, math
from pathlib import Path
import numpy as np

# Optional deps guarded
try:
    import networkx as nx
    HAVE_NX = True
except Exception:
    HAVE_NX = False

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
STATE = ROOT / "state"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

hard_errors = []
warnings = []

def load_json(p: Path, default=None):
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default

def require_key(d, key, where):
    if key not in d:
        hard_errors.append(f"Missing key '{key}' in {where}")
        return None
    return d[key]

def in_range(v, lo, hi, name):
    if not (lo <= v <= hi):
        hard_errors.append(f"{name}={v} out of [{lo},{hi}]")
    return v

def soft_range(v, lo, hi, name):
    if not (lo <= v <= hi):
        warnings.append(f"{name}={v} outside soft range [{lo},{hi}]")
    return v

def main():
    params_path = CONFIGS / "sigma_params.json"
    params = load_json(params_path, default={})
    metrics = load_json(STATE / "last_metrics.json", default={})

    # --- 1) Structure & numeric ranges
    α = require_key(params, "α", "configs/sigma_params.json")
    λ = require_key(params, "λ", "configs/sigma_params.json")
    γ = require_key(params, "γ", "configs/sigma_params.json")
    μ = require_key(params, "μ", "configs/sigma_params.json")
    β = params.get("β", 0.0)

    if all(x is not None for x in (α, λ, γ, μ)):
        in_range(α, 0.0, 1.0, "α")
        in_range(λ, 0.0, 1.0, "λ")
        in_range(γ, 0.0, 1.0, "γ")
        in_range(μ, 0.0, 1.0, "μ")
        soft_range(α, 0.25, 0.38, "α (expected stable band)")
        soft_range(λ, 0.12, 0.30, "λ (damping)")
        soft_range(γ, 0.002, 0.02, "γ (creative noise)")
        soft_range(μ, 0.15, 0.30, "μ (agitation)")

    # --- 2) Small-gain condition
    sg = (α or 0) + (β or 0) + (γ or 0)
    if sg >= 0.45:
        hard_errors.append(
            f"Small-gain violated: α+β+γ={sg:.3f} ≥ 0.45"
        )
    elif sg >= 0.40:
        warnings.append(
            f"Small-gain near limit: α+β+γ={sg:.3f} (≥0.40 recommended caution)"
        )

    # --- 3) Optional coupling/W checks
    W_path = CONFIGS / "W.csv"
    if W_path.exists():
        try:
            W = np.loadtxt(W_path, delimiter=",")
            if W.shape[0] != W.shape[1]:
                hard_errors.append(f"W must be square, got {W.shape}")
            if not np.allclose(W, W.T, atol=1e-8):
                hard_errors.append("W must be symmetric.")
            if np.any(np.diag(W) <= 0.0):
                warnings.append("W has non-positive diagonal entries.")
        except Exception as e:
            warnings.append(f"Could not parse W.csv: {e}")

    # --- 4) Optional context graph spectral sanity
    # If a graph is present, report Laplacian spectral radius (sanity only)
    L_spectral = None
    graph_file = CONFIGS / "context_graph.edgelist"
    if HAVE_NX and graph_file.exists():
        try:
            G = nx.read_edgelist(graph_file, nodetype=int, data=(("weight", float),))
            L = nx.laplacian_matrix(G).astype(float).toarray()
            eigs = np.linalg.eigvals(L)
            L_spectral = float(np.max(np.real(eigs)))
            # rule of thumb: require λ to be large enough vs sqrt(rho(A)) ~ sqrt(L_spectral)
            if L_spectral > 0 and λ is not None:
                rhs = 2.0 / (math.sqrt(L_spectral) + max(λ, 1e-9))
                if rhs <= 0:
                    warnings.append("CFL heuristic bound computed non-positive (check parameters).")
        except Exception as e:
            warnings.append(f"Graph spectral check failed: {e}")

    # --- 5) Metrics sanity
    if metrics:
        sp = metrics.get("sigma_percentage")
        if sp is not None:
            in_range(float(sp), 0.0, 100.0, "sigma_percentage")
        stab = metrics.get("stability")
        if stab is not None:
            in_range(float(stab), 0.0, 1.0, "stability")
        res = metrics.get("resilience")
        if res is not None:
            in_range(float(res), 0.0, 60.0, "resilience (sec)")

    # --- 6) Emit machine-readable report + exit code
    report = {
        "params_file": str(params_path),
        "metrics_file": str(STATE / "last_metrics.json"),
        "small_gain": sg,
        "laplacian_spectral_radius": L_spectral,
        "warnings": warnings,
        "hard_errors": hard_errors,
    }
    with (REPORTS / "invariants_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if hard_errors:
        print("[INVARIANTS][FAIL]")
        print(json.dumps(report, indent=2))
        sys.exit(1)
    else:
        print("[INVARIANTS][OK]")
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
