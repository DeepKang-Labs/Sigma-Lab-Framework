from __future__ import annotations
import numpy as np
import networkx as nx

def laplacian_from_graph(W: np.ndarray) -> np.ndarray:
    """
    Construit L_G (non-normalisé) à partir d'une matrice de poids W (symétrique, diag 0)
    """
    G = nx.from_numpy_array(W)
    L = nx.laplacian_matrix(G).astype(float).toarray()
    return L
