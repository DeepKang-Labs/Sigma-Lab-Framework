# sigma/core/rag.py
import os
from typing import List, Dict, Any
from .semantic_index import SemanticIndex

_DEFAULT_IDX = None

def get_index() -> SemanticIndex:
    global _DEFAULT_IDX
    if _DEFAULT_IDX is None:
        _DEFAULT_IDX = SemanticIndex()
        # si aucun index, tenter un build initial
        if _DEFAULT_IDX._index.ntotal == 0:
            _DEFAULT_IDX.build()
    return _DEFAULT_IDX

def retrieve_topk(query: str, k: int = 3) -> List[Dict[str, Any]]:
    return get_index().search(query, k=k)

def add_corpus_texts(texts: List[str], source: str = "dynamic"):
    get_index().add(texts, source=source)

def rebuild_corpus():
    get_index().build()
