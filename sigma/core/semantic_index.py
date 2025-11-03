# sigma/core/semantic_index.py
import os, json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

class SemanticIndex:
    """
    Index FAISS persistant pour passages de connaissance.
    - build() : (re)construit depuis rag_corpus/
    - add()   : ajoute des documents dynamiques
    - search(): top-k passages pertinents
    """
    def __init__(self,
                 corpus_dir: str = "rag_corpus",
                 index_dir: str  = "faiss_index",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_cache: int  = 2000):
        self.corpus_dir = corpus_dir
        self.index_dir  = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.corpus_dir, exist_ok=True)
        self.embedder = SentenceTransformer(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.max_cache = max_cache

        self.index_path  = os.path.join(self.index_dir, "index.faiss")
        self.meta_path   = os.path.join(self.index_dir, "meta.jsonl")
        self._index = None
        self._meta  = []  # [(doc_id, text, source)]

        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            self._index = faiss.read_index(self.index_path)
            # recharger meta
            self._meta = []
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    for line in f:
                        self._meta.append(json.loads(line))
        else:
            self._index = faiss.IndexFlatIP(self.dim)  # cos sim avec vecteurs normalisés

    def _save(self):
        faiss.write_index(self._index, self.index_path)

    def _append_meta(self, rec: Dict[str, Any]):
        with open(self.meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._meta.append(rec)

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")

    def build(self):
        # (re)indexer tout rag_corpus/*
        texts, sources = [], []
        for root, _, files in os.walk(self.corpus_dir):
            for fn in files:
                if not fn.lower().endswith((".md", ".txt", ".json")):
                    continue
                path = os.path.join(root, fn)
                try:
                    if fn.endswith(".json"):
                        data = json.load(open(path, "r", encoding="utf-8"))
                        # tolère soit {"text": "..."} soit list[{"text": "..."}]
                        if isinstance(data, dict) and "text" in data:
                            texts.append(data["text"]); sources.append(path)
                        elif isinstance(data, list):
                            for it in data:
                                if isinstance(it, dict) and "text" in it:
                                    texts.append(it["text"]); sources.append(path)
                    else:
                        txt = open(path, "r", encoding="utf-8").read()
                        texts.append(txt); sources.append(path)
                except Exception:
                    continue

        if not texts:
            return

        self._index = faiss.IndexFlatIP(self.dim)
        self._meta  = []
        vecs = self._embed(texts)
        self._index.add(vecs)
        for i, (t, s) in enumerate(zip(texts, sources)):
            self._append_meta({"doc_id": i, "source": s, "text": t[:20000]})
        self._save()

    def add(self, texts: List[str], source: str = "dynamic"):
        if not texts: return
        vecs = self._embed(texts)
        self._index.add(vecs)
        base = len(self._meta)
        for i, t in enumerate(texts):
            self._append_meta({"doc_id": base + i, "source": source, "text": t[:20000]})
        if len(self._meta) > self.max_cache:
            # simple trim (optionnel)
            pass
        self._save()

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if self._index.ntotal == 0:
            return []
        q = self._embed([query])
        D, I = self._index.search(q, min(k, self._index.ntotal))
        out = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0: continue
            meta = self._meta[idx]
            out.append({
                "score": float(score),
                "doc_id": meta["doc_id"],
                "source": meta["source"],
                "text": meta["text"]
            })
        return out
