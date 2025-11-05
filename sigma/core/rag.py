# sigma/core/rag.py
import os
from typing import List, Dict, Any
from .semantic_index import SemanticIndex

_DEFAULT_IDX: SemanticIndex | None = None


def get_index() -> SemanticIndex:
    """Retourne l'index sémantique global, le crée si nécessaire."""
    global _DEFAULT_IDX
    if _DEFAULT_IDX is None:
        _DEFAULT_IDX = SemanticIndex()
        # Si aucun index initial, construire un nouvel index
        if getattr(_DEFAULT_IDX, "_index", None) is None or getattr(_DEFAULT_IDX._index, "ntotal", 0) == 0:
            _DEFAULT_IDX.build()
    return _DEFAULT_IDX


def retrieve_topk(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Recherche les k résultats les plus proches dans l'index sémantique."""
    return get_index().search(query, k=k)


def add_corpus_texts(texts: List[str], source: str = "dynamic") -> None:
    """Ajoute une liste de textes au corpus sémantique global."""
    idx = get_index()
    idx.add(texts, source=source)


def rebuild_corpus() -> None:
    """Reconstruit entièrement le corpus global."""
    get_index().build()
