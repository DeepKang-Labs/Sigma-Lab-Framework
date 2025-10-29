# scripts/local_model.py
import os
import json
import math
from datetime import datetime

FEATURE_KEYS = [
    "ok_ratio",
    "latency_avg_ms",
    "sample_size",
    "ratio_2xx",
    "ratio_3xx",
    "ratio_4xx",
    "ratio_5xx",
]

DEFAULT_ALPHA = 0.1  # facteur EWMA

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json(p: str, default):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default

def dump_json(p: str, obj):
    safe_mkdir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def ewma_update(mu, var, x, alpha):
    """
    Met à jour moyenne/variance exponentielles (variance non-biaisée approx).
    mu_t = (1-alpha)*mu_{t-1} + alpha*x
    var_t ≈ (1-alpha)*(var_{t-1} + alpha*(x - mu_{t-1})^2)
    """
    if mu is None:  # init
        return x, 0.0
    new_mu = (1 - alpha) * mu + alpha * x
    # variance EWMA simple (stabilité suffisante ici)
    new_var = (1 - alpha) * (var + alpha * (x - mu) * (x - mu))
    return new_mu, new_var

def zscore(x, mu, var, eps=1e-8):
    sigma = math.sqrt(max(var, 0.0) + eps)
    return 0.0 if sigma == 0.0 else (x - mu) / sigma

def main():
    today = os.getenv("TODAY") or datetime.utcnow().strftime("%Y-%m-%d")
    day_dir = os.path.join("data", today)
    features_path = os.path.join(day_dir, "features.json")
    model_out     = os.path.join(day_dir, "sigma_model.json")
    state_path    = os.path.join(".state", "model_state.json")
    mesh_out_dir  = os.path.join("mesh", "outbox")

    features = load_json(features_path, {})
    if not features:
        print("⚠️ No features found, skipping model update.")
        return

    state = load_json(state_path, {"alpha": DEFAULT_ALPHA, "params": {}})
    alpha = float(state.get("alpha", DEFAULT_ALPHA))
    params = state.get("params", {})

    # Mise à jour EWMA par feature
    zs = {}
    for k in FEATURE_KEYS:
        x = float(features.get(k, 0.0))
        st = params.get(k, {"mu": None, "var": 0.0})
        mu0, var0 = st.get("mu"), float(st.get("var", 0.0))

        mu1, var1 = ewma_update(mu0, var0, x, alpha)
        params[k] = {"mu": mu1, "var": var1}

        zs[k] = zscore(x, mu1, var1)

    # Score d’anomalie agrégé (L2 des z-scores bornée)
    anomaly = sum(z*z for z in zs.values()) ** 0.5
    # Snapshot du modèle
    model_snapshot = {
        "ts_utc": datetime.utcnow().isoformat() + "Z",
        "alpha": alpha,
        "z_scores": zs,
        "anomaly_l2": anomaly,
        "features_used": FEATURE_KEYS,
    }

    # Sauvegardes
    dump_json(state_path, {"alpha": alpha, "params": params})
    dump_json(model_out, model_snapshot)

    # Delta minimal à partager (moyennes/variances)
    delta = {
        "ts_utc": model_snapshot["ts_utc"],
        "type": "ewma_params",
        "payload": params,
    }
    delta_path = os.path.join(mesh_out_dir, f"model_delta_{datetime.utcnow().strftime('%H%M%S')}.json")
    dump_json(delta_path, delta)

    print(f"✅ model updated: {model_out}")
    print(f"📦 delta written: {delta_path}")

if __name__ == "__main__":
    main()
