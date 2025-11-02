from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def load_features_from_artifacts(root: str = "artifacts") -> dict:
    # Adapte ces chemins à vos artefacts existants
    skywire_p = Path(root) / "skywire_sanitized.json"
    btc_p     = Path(root) / "btc_sanitized.json"

    sky = json.loads(skywire_p.read_text()) if skywire_p.exists() else {}
    btc = json.loads(btc_p.read_text()) if btc_p.exists() else {}

    Phi = np.array(sky.get("context_vector", [0.0,0.0,0.0,0.0]), dtype=float)
    W   = np.array(sky.get("W", np.eye(4).tolist()), dtype=float)
    lat = np.array(sky.get("latencies_ms", []), dtype=float)
    ret = np.array(btc.get("returns", []), dtype=float)

    return {
        "Phi": Phi,
        "W": W,
        "signals": {
            "latencies_ms": lat,
            "btc_returns": ret
        }
    }
