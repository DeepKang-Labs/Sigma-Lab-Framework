#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeshApply.py — Sigma Mesh Aggregator
------------------------------------
Consumes deltas from mesh/inbox, applies federated averaging,
updates the local model, and archives processed deltas.

Author: DeepKang Labs
"""

import os, json, glob, shutil
from datetime import datetime

ROOT = os.environ.get("ARTIFACTS_ROOT", "artifacts")
MODEL_PATH = os.path.join(ROOT, "model", "local_model.json")
INBOX = os.path.join(ROOT, "mesh", "inbox")
ARCHIVE = os.path.join(ROOT, "mesh", "archive")

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[⚠] Failed to read {path}: {e}")
        return None

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def aggregate_models(local_model, deltas):
    """Weighted averaging of numeric values"""
    agg = local_model.copy()
    counts = {}
    for delta in deltas:
        for key, value in delta.items():
            if isinstance(value, (int, float)):
                agg[key] = agg.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1

    for k, c in counts.items():
        agg[k] = round(agg[k] / c, 6)
    return agg

def main():
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

    # Charger modèle local
    model = load_json(MODEL_PATH)
    if model is None:
        print("[⚠] No local model found, initializing empty one.")
        model = {}

    # Lire les deltas
    delta_files = sorted(glob.glob(os.path.join(INBOX, "*.json")))
    if not delta_files:
        print("[ℹ] No deltas in inbox — nothing to apply.")
        return

    print(f"[ℹ] Found {len(delta_files)} delta(s) to apply.")

    deltas = []
    for df in delta_files:
        d = load_json(df)
        if d: deltas.append(d)

    if not deltas:
        print("[⚠] All deltas invalid or unreadable.")
        return

    # Agrégation pondérée
    new_model = aggregate_models(model, deltas)
    save_json(MODEL_PATH, new_model)

    # Archiver les deltas consommés
    for df in delta_files:
        dst = os.path.join(ARCHIVE, os.path.basename(df))
        os.makedirs(ARCHIVE, exist_ok=True)
        shutil.move(df, dst)

    print(f"[✅] Applied {len(deltas)} delta(s). Model updated → {MODEL_PATH}")

if __name__ == "__main__":
    main()
