from __future__ import annotations
import numpy as np
import math
from numbers import Real

def _nonnegative(value):
    return isinstance(value,Real) and not isinstance(value,bool) and math.isfinite(value) and value >= 0

def check_petit_gain(alpha: float, beta: float, gamma: float, max_gain: float = 0.40) -> bool:
    if not all(_nonnegative(v) for v in (alpha,beta,gamma,max_gain)) or max_gain <= 0:
        return False
    return (alpha + beta + gamma) < max_gain

def check_cfl(dt: float, rhoA: float, max_lambda: float, safety: float = 0.8) -> bool:
    if not all(_nonnegative(v) for v in (dt,rhoA,max_lambda,safety)) or dt <= 0 or not 0 < safety <= 1:
        return False
    # Δt < 2 / ( sqrt(ρ(A)) + max(λ_i) ) * safety
    denom = (np.sqrt(max(rhoA, 0.0)) + max_lambda)
    if denom <= 0:
        return False
    return dt < (2.0 / denom) * safety

def clamp_bounds(vals: dict, bounds: dict) -> dict:
    out = dict(vals)
    for k, (mn, mx) in bounds.items():
        if k in out:
            out[k] = min(max(out[k], mn), mx)
    return out
