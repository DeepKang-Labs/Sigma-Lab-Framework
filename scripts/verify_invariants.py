#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verify basic and advanced invariants for Sigma-Lab.
- Checks required keys in configs/sigma_params.json
- Bounds & plausibility of params
- Matrix/vector dimensional consistency (theta vs W)
- Basic stability heuristics (small-gain like rule)
- Optional spectral radius (if numpy is available), otherwise warn

Outputs a JSON report to reports/invariants_report.json.
Exit code: 0 if OK/warnings only, 1 if hard errors.
"""

import json, os, sys, math, time

PARAMS_PATH = "configs/sigma_params.json"
METRICS_PATH = "state/last_metrics.json"
REPORTS_DIR = "reports"
REPORT_PATH = os.path.join(REPORTS_DIR, "invariants_report.json")

def load_json_safely(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception as e:
        return {"__load_error__": f"{type(e).__name__}: {e}"}

def check_required_keys(params, hard_errors):
    required = ["alpha","lambda","gamma","mu","beta","theta","W"]
    for k in required:
        if k not in params:
            hard_errors.append(f"Missing key '{k}' in {PARAMS_PATH}")

def check_bounds(params, warnings, hard_errors):
    # Numeric bounds
    def in01(x): return isinstance(x,(int,float)) and (0.0 <= x <= 1.0)
    for k in ["alpha","gamma","mu","beta"]:
        v = params.get(k, None)
        if not in01(v):
            hard_errors.append(f"Parameter '{k}' must be in [0,1], got {v!r}")
    # lambda can be positive small (damping); accept 0..2
    lam = params.get("lambda", None)
    if not (isinstance(lam,(int,float)) and 0.0 <= lam <= 2.0):
        hard_errors.append(f"Parameter 'lambda' must be in [0,2], got {lam!r}")

def check_shapes(params, warnings, hard_errors):
    theta = params.get("theta", None)
    W = params.get("W", None)

    if not isinstance(theta, list) or not all(isinstance(x,(int,float)) for x in theta):
        hard_errors.append("theta must be a list of numbers")
        return

    if not isinstance(W, list) or not all(isinstance(row, list) for row in W):
        hard_errors.append("W must be a matrix (list of lists)")
        return

    n = len(theta)
    if n == 0:
        hard_errors.append("theta must be non-empty")
        return

    if any(len(row) != n for row in W):
        hard_errors.append(f"W must be square {n}x{n} to match theta, got rows {[len(r) for r in W]}")
        return

    # Basic plausibility on W
    for i in range(n):
        if not (isinstance(W[i][i], (int,float)) and abs(W[i][i]-1.0) <= 1e-6):
            warnings.append(f"W[{i},{i}] expected ~1.0 (self-coupling), got {W[i][i]!r}")
        for j in range(n):
            v = W[i][j]
            if not isinstance(v,(int,float)):
                hard_errors.append(f"W[{i},{j}] must be numeric, got {type(v).__name__}")
            if not (0.0 <= v <= 1.0):
                warnings.append(f"W[{i},{j}] outside [0,1]: {v}")

    # Symmetry check (tolerance)
    tol = 1e-6
    asym = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(W[i][j] - W[j][i]) > tol:
                asym.append((i,j,W[i][j],W[j][i]))
    if asym:
        warnings.append(f"W not symmetric (first 3 diffs): {asym[:3]}")

def small_gain_rule(params, warnings, hard_errors):
    # Heuristic: alpha + mu < 1 to avoid runaway positive feedback
    a = params.get("alpha", 0.0)
    m = params.get("mu", 0.0)
    small_gain = float(a) + float(m)
    if small_gain >= 1.0:
        hard_errors.append(f"Small-gain violation: alpha + mu = {small_gain:.3f} >= 1")
    return small_gain

def optional_spectral_radius(W, warnings):
    try:
        import numpy as np
        eigs = np.linalg.eigvals(np.array(W, dtype=float))
        rho = float(max(abs(eigs)))
        return rho, None
    except Exception as e:
        warnings.append(f"Spectral radius skipped ({type(e).__name__}: {e})")
        return None, str(e)

def load_metrics(warnings):
    m = load_json_safely(METRICS_PATH, default=None)
    if m is None:
        warnings.append(f"{METRICS_PATH} not found — skipping metrics checks")
        return {}
    if "__load_error__" in m:
        warnings.append(f"Could not load metrics: {m['__load_error__']}")
        return {}
    return m

def metrics_checks(metrics, warnings, hard_errors):
    # Basic plausibility
    sp = metrics.get("sigma_percentage", None)
    if sp is not None:
        if not isinstance(sp,(int,float)) or not (0.0 <= sp <= 100.0):
            warnings.append(f"sigma_percentage suspicious: {sp!r}")

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    warnings = []
    hard_errors = []

    params = load_json_safely(PARAMS_PATH, default={})
    if "__load_error__" in params:
        hard_errors.append(f"Could not load {PARAMS_PATH}: {params['__load_error__']}")
        params = {}

    check_required_keys(params, hard_errors)
    if not hard_errors:
        check_bounds(params, warnings, hard_errors)
        check_shapes(params, warnings, hard_errors)

    small_gain = None
    rho = None
    if not hard_errors:
        small_gain = small_gain_rule(params, warnings, hard_errors)

    if not hard_errors and isinstance(params.get("W"), list):
        rho, _err = optional_spectral_radius(params["W"], warnings)

    metrics = load_metrics(warnings)
    metrics_checks(metrics, warnings, hard_errors)

    report = {
        "params_file": os.path.abspath(PARAMS_PATH),
        "metrics_file": os.path.abspath(METRICS_PATH),
        "small_gain": small_gain if small_gain is not None else 0.0,
        "laplacian_spectral_radius": rho,   # None if skipped
        "warnings": warnings,
        "hard_errors": hard_errors,
        "timestamp": int(time.time())
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print short status (helps in Actions logs)
    status = "OK" if not hard_errors else "FAIL"
    print(f"[INVARIANTS][{status}]")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    sys.exit(0 if not hard_errors else 1)

if __name__ == "__main__":
    main()
