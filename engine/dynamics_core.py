from __future__ import annotations
import numpy as np
from typing import Optional

def deep_sigma_ddC(
    C: np.ndarray,
    dC: np.ndarray,
    KPhi: np.ndarray,
    theta: np.ndarray,
    Lambda: np.ndarray,   # diag vector (shape d,) appliqué en Hadamard
    alpha: float,
    beta: float,
    gamma: float,
    meta_coh: np.ndarray,
    coh_inter: np.ndarray,
    noise: np.ndarray,
    L_G: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    d²C/dt² = ∇²[ K(Φ)(C-θ) ] - Λ ⊗ dC + α·meta_coh + β·coh_inter + γ·noise
    Discret: ∇²[X] ≈ - L_G @ X, avec X = K(Φ)(C-θ)
    """
    X = KPhi @ (C - theta)              # (d,)
    lap = -(L_G @ X) if L_G is not None else 0.0
    return lap - (Lambda * dC) + alpha*meta_coh + beta*coh_inter + gamma*noise
