from __future__ import annotations
import json, yaml, sys, math
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator

class SigmaParams(BaseModel):
    alpha: float
    beta:  float
    gamma: float
    mu:    float
    lambda_: float = Field(alias="lambda")
    theta: list[float]
    W: list[list[float]]

    @field_validator("theta")
    @classmethod
    def theta_len(cls, v):
        if len(v) != 4:
            raise ValueError("theta must be length 4")
        return v

    @field_validator("W")
    @classmethod
    def w_shape_sym(cls, v):
        if len(v) != 4 or any(len(row) != 4 for row in v):
            raise ValueError("W must be 4x4")
        # Symmetry check (tolerance)
        arr = np.array(v)
        if not np.allclose(arr, arr.T, atol=1e-8):
            raise ValueError("W must be symmetric")
        return v

def main():
    params_path = Path("configs/sigma_params.json")
    policy_path = Path("policy/safety_policy.yaml")

    params = json.loads(params_path.read_text())
    policy = yaml.safe_load(policy_path.read_text())

    # Pydantic validation
    try:
        sp = SigmaParams(**params)
    except ValidationError as e:
        print("Pydantic validation error:", e, file=sys.stderr)
        sys.exit(1)

    alpha = sp.alpha; beta = sp.beta; gamma = sp.gamma
    lam = sp.lambda_

    # Bounds check from policy
    bounds = policy.get("bounds", {})
    def in_bounds(name, val):
        if name not in bounds: return True
        mn, mx = bounds[name]
        return (mn <= val <= mx)

    for k, val in [("alpha",alpha),("beta",beta),("gamma",gamma),("mu",sp.mu),("lambda",lam)]:
        if not in_bounds(k, val):
            print(f"[FAIL] {k}={val} outside bounds {bounds.get(k)}", file=sys.stderr)
            sys.exit(1)

    # petit-gain
    if (alpha + beta + gamma) >= policy["constraints"]["petit_gain_max"]:
        print(f"[FAIL] petit-gain ({alpha+beta+gamma}) violates max {policy['constraints']['petit_gain_max']}", file=sys.stderr)
        sys.exit(1)

    print("[OK] params & policy validation passed.")

if __name__ == "__main__":
    main()
