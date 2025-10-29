#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesh_federate.py — Sigma Mesh Federator
---------------------------------------

Lit le modèle local, génère un delta simulé (mises à jour aléatoires)
et l’écrit dans le dossier mesh/outbox.
Ce delta sera ensuite consommé par mesh_apply.py.

Auteur : DeepKang Labs
"""

import os
import json
import random
from datetime import datetime

ROOT = os.environ.get("ARTIFACTS_ROOT", "artifacts")
MODEL_PATH = os.path.join(ROOT, "model", "local_model.json")
OUTBOX = os.path.join(ROOT, "mesh", "outbox")

def load_json(path):
    if not os.path.exists(path):
        print(f"[⚠️] Modèle non trouvé : {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_delta(model):
    """Crée un delta (mise à jour simulée du modèle)."""
    weights = model.get("weights", {})
    delta = {}
    for k, v in weights.items():
        # Variation aléatoire de ±0.05
        delta[k] = round(v + random.uniform(-0.05, 0.05), 6)
    delta["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return delta

def main():
    model = load_json(MODEL_PATH)
    if model is None:
        print("[❌] Aucun modèle à fédérer.")
        return

    delta = generate_delta(model)
    os.makedirs(OUTBOX, exist_ok=True)

    out_path = os.path.join(OUTBOX, f"delta_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
    save_json(out_path, delta)

    print(f"[✅] Delta généré et stocké : {out_path}")
    print(json.dumps(delta, indent=2))

if __name__ == "__main__":
    main()
