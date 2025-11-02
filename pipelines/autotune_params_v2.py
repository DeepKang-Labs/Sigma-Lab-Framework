"""
Autotune Sigma v2 — DeepKang-Labs
---------------------------------
Optimisation réflexive des paramètres Sigma par :
- régression locale sur l’historique (gradient estimé)
- fallback SPSA (exploration stochastique) si peu de données
- projection dure sur les contraintes (bornes + petit-gain)
- journalisation détaillée pour audit scientifique

Entrées :
  - configs/sigma_params.json           (paramètres actuels)
  - state/last_metrics.json             (métriques du dernier run)
  - state/autotune_log.csv (optionnel)  (historique)

Sorties :
  - configs/sigma_params_next.json      (paramètres proposés pour le prochain run)
  - state/autotune_log.csv              (mis à jour)
"""

from __future__ import annotations
import json, csv, os, math, time, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# -------------------- FICHIERS --------------------
PARAMS_FILE     = Path("configs/sigma_params.json")
METRICS_FILE    = Path("state/last_metrics.json")
NEXT_FILE       = Path("configs/sigma_params_next.json")
LOG_FILE        = Path("state/autotune_log.csv")

# -------------------- BORNES & CONTRAINTES --------------------
BOUNDS = {
    "α": (0.25, 0.35),
    "λ": (0.15, 0.25),
    "γ": (0.002, 0.020),
    "μ": (0.15, 0.30),
    # "β" peut être présent ; sinon on suppose 0
}
SMALL_GAIN_LIMIT = 0.40  # marge de sécurité : α + β + γ < 0.40

# Pas d’apprentissage “doux”
BASE_LR = {
    "α": 0.010,
    "λ": 0.010,
    "γ": 0.002,
    "μ": 0.010,
}

# Amplitude de perturbation SPSA
SPSA_DELTA = {
    "α": 0.005,
    "λ": 0.005,
    "γ": 0.001,
    "μ": 0.005,
}

MAX_STEP_SCALE = 0.8  # sécurité, réduit un step trop grand

# -------------------- UTILS IO --------------------
def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def _save_json(obj: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[INFO] Saved {path}")

def _append_log(row: Dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp","alpha","lambda","gamma","mu","beta",
            "sigma_percentage","stability","resilience","objective"
        ])
        if write_header:
            w.writeheader()
        w.writerow(row)

def _load_history(n_last: int = 50) -> List[Dict]:
    if not LOG_FILE.exists():
        return []
    rows = []
    with open(LOG_FILE, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if n_last is None:
        return rows
    return rows[-n_last:]

# -------------------- CONTRAINTES --------------------
def _project_bounds(p: Dict[str,float]) -> Dict[str,float]:
    q = p.copy()
    for k, (lo, hi) in BOUNDS.items():
        if k in q:
            q[k] = float(min(max(q[k], lo), hi))
    return q

def _enforce_small_gain(p: Dict[str,float]) -> Dict[str,float]:
    q = p.copy()
    alpha = q.get("α", 0.30)
    beta  = q.get("β", 0.0)
    gamma = q.get("γ", 0.006)
    s = alpha + beta + gamma
    if s >= SMALL_GAIN_LIMIT:
        # on réduit proportionnellement α et γ (β laissé tel quel si fourni)
        scale = (SMALL_GAIN_LIMIT - 1e-6) / max(s, 1e-9)
        q["α"] = alpha * scale
        q["γ"] = gamma * scale
    return q

def _stable_projection(p: Dict[str,float]) -> Dict[str,float]:
    return _enforce_small_gain(_project_bounds(p))

# -------------------- OBJECTIF --------------------
def compute_objective(metrics: Dict[str, float]) -> float:
    """
    Objectif à maximiser. Pondération prudente :
    - Sigma % (poids 0.7)
    - stabilité (0..1) (poids 0.2)
    - résilience (temps de retour ; plus petit est mieux) -> transformée en score [0..1]
    """
    sigma_pct = float(metrics.get("sigma_percentage", 0.0))
    stability = float(metrics.get("stability", 0.0))       # attendu [0..1]
    resilience_sec = float(metrics.get("resilience", 3.0)) # plus petit = mieux

    # Normalisation résilience (0..1) ~ 0s -> 1.0 ; 10s -> ~0.0 (cap)
    res_score = max(0.0, min(1.0, 1.0 - resilience_sec/10.0))

    objective = 0.7 * sigma_pct/100.0 + 0.2 * stability + 0.1 * res_score
    return float(objective)

# -------------------- GRADIENT LOCAL PAR RÉGRESSION --------------------
def estimate_gradient_from_history(history: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Régression linéaire locale : objective ~ b0 + b1*α + b2*λ + b3*γ + b4*μ
    Retourne (coeffs, mean_params) pour recenter les steps si besoin.
    """
    if len(history) < 8:
        return None, None

    X, y = [], []
    for row in history:
        try:
            a = float(row["alpha"]); l = float(row["lambda"])
            g = float(row["gamma"]); m = float(row["mu"])
            obj = float(row.get("objective", 0.0))
        except Exception:
            continue
        X.append([1.0, a, l, g, m])
        y.append(obj)

    if len(X) < 8:
        return None, None

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Moindres carrés
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    # gradient approx = dérivées partielles (b1..b4)
    grad = beta_hat[1:]
    # moyenne des paramètres observés (utile pour centrage)
    mean_params = X[:,1:].mean(axis=0)
    return grad, mean_params

# -------------------- SPSA (fallback exploration) --------------------
def spsa_step(current: Dict[str,float]) -> Dict[str,float]:
    cand = current.copy()
    for k, delta in SPSA_DELTA.items():
        if k not in cand: 
            continue
        sign = 1.0 if random.random() < 0.5 else -1.0
        cand[k] = cand[k] + sign * delta
    return _stable_projection(cand)

# -------------------- PIPE PRINCIPAL --------------------
def main():
    # 1) Charger données
    params  = _load_json(PARAMS_FILE)
    metrics = _load_json(METRICS_FILE)

    # Defaults safe si champs manquants
    params.setdefault("α", 0.30)
    params.setdefault("λ", 0.20)
    params.setdefault("γ", 0.006)
    params.setdefault("μ", 0.20)
    params.setdefault("β", params.get("β", 0.0))

    objective = compute_objective(metrics) if metrics else 0.0

    # 2) Journaliser le point courant
    now = int(time.time())
    log_row = {
        "timestamp": now,
        "alpha": params["α"],
        "lambda": params["λ"],
        "gamma": params["γ"],
        "mu": params["μ"],
        "beta": params.get("β", 0.0),
        "sigma_percentage": metrics.get("sigma_percentage", ""),
        "stability": metrics.get("stability", ""),
        "resilience": metrics.get("resilience", ""),
        "objective": objective,
    }
    _append_log(log_row)
    print(f"[INFO] Logged current point. Objective={objective:.4f}")

    # 3) Historique pour estimer gradient
    history = _load_history(n_last=60)
    grad, mean_params = estimate_gradient_from_history(history)

    current = {k: float(params[k]) for k in ["α","λ","γ","μ"]}

    if grad is not None:
        print(f"[INFO] Local gradient (α,λ,γ,μ) ≈ {grad}")
        # Step = lr * grad (ascension du gradient)
        step = np.array([BASE_LR["α"], BASE_LR["λ"], BASE_LR["γ"], BASE_LR["μ"]]) * grad
        # Clip doux
        step = np.clip(step, -MAX_STEP_SCALE*np.array(list(BASE_LR.values())),
                              MAX_STEP_SCALE*np.array(list(BASE_LR.values())))
        proposal = {
            "α": current["α"] + float(step[0]),
            "λ": current["λ"] + float(step[1]),
            "γ": current["γ"] + float(step[2]),
            "μ": current["μ"] + float(step[3]),
            "β": params.get("β", 0.0)
        }
        proposal = _stable_projection(proposal)
        print(f"[INFO] Gradient step proposal: {proposal}")
    else:
        # Pas assez de données -> SPSA d’exploration
        print("[WARN] Not enough history for regression; using SPSA exploratory step.")
        proposal = spsa_step({**current, "β": params.get("β", 0.0)})

    # 4) Sauvegarder la proposition
    out = {
        **params,
        "α": proposal["α"],
        "λ": proposal["λ"],
        "γ": proposal["γ"],
        "μ": proposal["μ"],
        "β": proposal.get("β", params.get("β", 0.0)),
        "_policy": {
            "small_gain_limit": SMALL_GAIN_LIMIT,
            "bounds": BOUNDS,
            "method": "gradient_local" if grad is not None else "spsa",
        },
        "_from_objective": objective,
        "_timestamp": now
    }
    _save_json(out, NEXT_FILE)
    print("[DONE] Autotune v2 complete ✅")

if __name__ == "__main__":
    main()
