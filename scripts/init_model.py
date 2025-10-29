#!/usr/bin/env python3
# init_model.py — créer le modèle local s'il n'existe pas

import os, json, sys

ROOT = os.environ.get("ARTIFACTS_ROOT", "artifacts")
MODEL_PATH = os.path.join(ROOT, "model", "local_model.json")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    data = {
        "version": "0.1.0",
        "updated_at": None,
        "weights": {
            "latency_mean_ms": 0.0,
            "availability_score": 0.0,
            "transport_density": 0.0
        }
    }
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Created zeroed model: {MODEL_PATH}")
else:
    print(f"ℹ️ Model already exists: {MODEL_PATH}")
