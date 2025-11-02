from __future__ import annotations
import json, yaml
from pathlib import Path
from engine.invariants import clamp_bounds, check_petit_gain

def main():
    params = json.loads(Path("configs/sigma_params.json").read_text())
    metrics = json.loads(Path("state/last_metrics.json").read_text())
    policy = yaml.safe_load(Path("policy/safety_policy.yaml").read_text())

    new = dict(params)
    frac = float(metrics.get("frac_sigma", 0.0))

    # règle simple
    if frac < 0.25:
        new["lambda"] = params["lambda"] * 1.05
    elif frac > 0.60:
        new["lambda"] = params["lambda"] * 0.98

    # bornes
    new = clamp_bounds(new, policy["bounds"])

    # petit-gain (soft rescale)
    if not check_petit_gain(new["alpha"], new.get("beta", 0.0), new["gamma"], policy["constraints"]["petit_gain_max"]):
        total = new["alpha"] + new.get("beta", 0.0) + new["gamma"]
        scale = (policy["constraints"]["petit_gain_max"] - 0.01) / (total + 1e-9)
        new["alpha"] *= scale
        new["gamma"] *= scale

    Path("configs/sigma_params.json").write_text(json.dumps(new, indent=2))
    print("[AUTOTUNE] updated configs/sigma_params.json")

if __name__ == "__main__":
    main()
