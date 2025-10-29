# scripts/init_model.py
import os
import json
from pathlib import Path

ARTIFACTS_ROOT = os.environ.get("ARTIFACTS_ROOT", "artifacts")
model_path = os.path.join(ARTIFACTS_ROOT, "model", "local_model.json")
Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)

if not os.path.exists(model_path):
    data = {
        "version": "0.1.0",
        "updated_at": None,
        "weights": {
            "latency_mean_ms": 0.0,
            "availability_score": 0.0,
            "transport_density": 0.0
        }
    }
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Created zeroed model:", model_path)
else:
    print(f"Model already exists:", model_path)
