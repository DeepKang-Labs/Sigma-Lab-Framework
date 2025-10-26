# tools/mesh_memory.py
# MIT License — DeepKang Labs
# Mesh Memory (append-only JSONL) for loopback learning

from __future__ import annotations
import os, json, time, hashlib
from typing import Dict, Any, Optional, List

def sha256_file(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()
    except Exception:
        return ""

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def append_memory_entry(
    memory_root: str,
    *,
    network: str,
    decision_ids: List[str],
    report_path: Optional[str],
    mappings_path: Optional[str],
    config_path: Optional[str],
    discovery_path: Optional[str],
    sigma_summary: Dict[str, float],
    extra_meta: Optional[Dict[str, Any]] = None
) -> str:
    """
    Append a single line JSON entry to corpus/mesh_memory/index.jsonl
    Returns: absolute path to the JSONL index.
    """
    root = os.path.abspath(memory_root)
    idx = os.path.join(root, "index.jsonl")
    _ensure_dir(root)

    entry = {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "network": network,
        "decision_ids": decision_ids,
        "sigma_summary": sigma_summary,  # e.g. {"non_harm":0.7,"stability":0.6,"resilience":0.5,"equity":0.55}
        "digests": {
            "report_sha256": sha256_file(report_path) if report_path and os.path.exists(report_path) else None,
            "mappings_sha256": sha256_file(mappings_path) if mappings_path and os.path.exists(mappings_path) else None,
            "config_sha256": sha256_file(config_path) if config_path and os.path.exists(config_path) else None,
            "discovery_sha256": sha256_file(discovery_path) if discovery_path and os.path.exists(discovery_path) else None,
        },
        "paths": {
            "report": report_path,
            "mappings": mappings_path,
            "config": config_path,
            "discovery": discovery_path,
        },
        "meta": extra_meta or {}
    }

    with open(idx, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return idx
