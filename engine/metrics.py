from __future__ import annotations
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def coh(C: np.ndarray, dC: np.ndarray, w: np.ndarray, mu: float, b_phi: float = 0.0, eps0: float = 1e-6) -> float:
    # coh(t) = σ( sum_i w_i C_i - μ sum_i sqrt(dC_i^2 + ε0) + b(Φ) )
    term = np.dot(w, C) - mu * np.sum(np.sqrt(dC**2 + eps0)) + b_phi
    return float(sigmoid(term))

def meta_coh_scalar(coh_hist: list[float], dt: float) -> float:
    # approx d²coh/dt² via dérivée discrète
    if len(coh_hist) < 3:
        return 0.0
    d1 = (coh_hist[-1] - coh_hist[-2]) / dt
    d0 = (coh_hist[-2] - coh_hist[-3]) / dt
    return (d1 - d0) / dt

def frac_sigma_from_dC(dC_hist: np.ndarray, window: int = 10) -> float:
    # Σ_c = proportion de pas où ||dC|| <= ε_adaptatif, avec hystérésis minimale
    if len(dC_hist) == 0:
        return 0.0
    norms = np.linalg.norm(dC_hist[-window:], axis=1) if len(dC_hist) >= window else np.linalg.norm(dC_hist, axis=1)
    mx = np.max(norms) if norms.size else 0.0
    eps = 0.015 * mx if mx > 0 else 1e-6
    mask = (norms <= eps) & (norms > 0.0)
    return float(np.mean(mask)) if mask.size else 0.0
