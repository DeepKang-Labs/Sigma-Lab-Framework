from __future__ import annotations
import numpy as np

def laplacian_from_graph(W: np.ndarray) -> np.ndarray:
    """
    Construit L_G (non-normalisé) à partir d'une matrice de poids W (symétrique, diag 0)
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or not W.size:
        raise ValueError('Graph weights must be a nonempty square matrix')
    if not np.all(np.isfinite(W)) or np.any(W < 0) or not np.allclose(W,W.T,rtol=0,atol=1e-12):
        raise ValueError('Graph weights must be finite, non-negative and symmetric')
    # D-A cancels self-loops, matching the non-normalized undirected Laplacian.
    return np.diag(W.sum(axis=1)) - W
