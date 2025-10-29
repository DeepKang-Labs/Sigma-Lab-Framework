# scripts/protocols/sigma_mesh.py
from __future__ import annotations
import os, json, glob
from typing import Dict, List


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def write_delta(outbox_dir: str, node_id: str, delta: Dict) -> str:
    ensure_dirs(outbox_dir)
    fname = f"{node_id}_delta.json"
    path = os.path.join(outbox_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(delta, f, indent=2, ensure_ascii=False)
    return path


def collect_inbox_deltas(inbox_dir: str) -> List[Dict]:
    ensure_dirs(inbox_dir)
    items = []
    for p in glob.glob(os.path.join(inbox_dir, "*_delta.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                items.append(json.load(f))
        except Exception:
            continue
    return items


def federated_average(feature_deltas: List[Dict]) -> Dict:
    """
    Agrégation très simple: moyenne pondérée des d_mean par n.
    Attendu: { features: {key: {d_mean, n}}, ... }
    """
    if not feature_deltas:
        return {}

    keys = set()
    for d in feature_deltas:
        keys |= set(d.get("features", {}).keys())

    agg = {"type": "agg_delta", "version": 1, "features": {}}
    for k in keys:
        num = 0.0
        den = 0.0
        for d in feature_deltas:
            feat = d.get("features", {}).get(k)
            if not feat:
                continue
            dm = float(feat.get("d_mean", 0.0))
            n = float(feat.get("n", 0.0))
            num += dm * max(n, 1.0)
            den += max(n, 1.0)
        agg["features"][k] = {"d_mean": (num / den) if den > 0 else 0.0, "n_sum": den}
    return agg
