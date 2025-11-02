from __future__ import annotations
import json, yaml
from pathlib import Path
import numpy as np

from engine.dynamics_core import deep_sigma_ddC
from engine.integrators import rk4_step
from engine.invariants import check_petit_gain, check_cfl
from engine.metrics import coh, meta_coh_scalar, frac_sigma_from_dC
from engine.context_diffusion import laplacian_from_graph
from pipelines.feature_build import load_features_from_artifacts

def main():
    # 1) charger config + policy
    params = json.loads(Path("configs/sigma_params.json").read_text())
    policy = yaml.safe_load(Path("policy/safety_policy.yaml").read_text())

    alpha = float(params["alpha"]); beta = float(params.get("beta", 0.0))
    gamma = float(params["gamma"]); mu = float(params["mu"])
    lam   = float(params["lambda"])
    theta = np.array(params["theta"], dtype=float)
    W     = np.array(params["W"], dtype=float)

    # 2) invariants statiques
    if not check_petit_gain(alpha, beta, gamma, policy["constraints"]["petit_gain_max"]):
        raise SystemExit("petit-gain violated")

    # 3) artefacts → features
    feats = load_features_from_artifacts()
    L_G = laplacian_from_graph(W)   # simple Laplacien graphe

    # 4) setup simulation
    d = 4
    C  = np.array([0.58, 0.52, 0.45, 0.56], dtype=float)
    dC = np.zeros(d, dtype=float)
    Lambda = np.array([lam]*d, dtype=float)
    w = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

    dt = 0.1
    # ρ(A) approx: norme du terme de diffusion → ici bornée par ||L_G|| * ||W||
    rhoA = float(np.max(np.abs(np.linalg.eigvals(-L_G)))) if L_G.size else 0.0
    if not check_cfl(dt, rhoA, lam, policy["constraints"]["cfl_safety"]):
        raise SystemExit("CFL violated")

    T = 600  # 60s si dt=0.1
    C_hist  = []
    dC_hist = []
    coh_hist = []

    # boucle
    for t_idx in range(T):
        # cohérence / méta-coh
        c_now = coh(C, dC, w=w, mu=mu, b_phi=0.0)
        coh_hist.append(c_now)
        mcoh = meta_coh_scalar(coh_hist, dt)
        meta_vec = np.full(d, mcoh, dtype=float)
        inter_vec = np.full(d, c_now, dtype=float)

        noise = np.random.normal(0.0, 1.0, size=d)

        def f(_t, y):
            # y = concat(C, dC)
            C_y = y[:d]
            dC_y = y[d:]
            ddC = deep_sigma_ddC(
                C=C_y, dC=dC_y, KPhi=W, theta=theta, Lambda=Lambda,
                alpha=alpha, beta=beta, gamma=gamma,
                meta_coh=meta_vec, coh_inter=inter_vec, noise=noise, L_G=L_G
            )
            return np.concatenate([dC_y, ddC])

        y = np.concatenate([C, dC])
        y = rk4_step(f, y, t_idx*dt, dt)
        C, dC = y[:d], y[d:]

        C_hist.append(C.copy())
        dC_hist.append(dC.copy())

    C_hist  = np.array(C_hist)
    dC_hist = np.array(dC_hist)

    # métriques
    frac_sigma = frac_sigma_from_dC(dC_hist, window=50)

    # 5) écrire artefacts état
    Path("state").mkdir(exist_ok=True)
    out = {
        "frac_sigma": float(frac_sigma),
        "final_C": C_hist[-1].tolist(),
        "final_dC": dC_hist[-1].tolist(),
        "rhoA": rhoA,
        "dt": dt
    }
    Path("state/last_metrics.json").write_text(json.dumps(out, indent=2))

    print("[RUN] done. frac_sigma =", frac_sigma)

if __name__ == "__main__":
    main()
