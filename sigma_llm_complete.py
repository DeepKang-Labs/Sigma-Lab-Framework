# -*- coding: utf-8 -*-
"""
Sigma-LLM v2.6 — DeepKang Integration (single-file)
Subjectivité S(t) + Objectivité O(t) + Méta-cohérence + Invariants + PolicyGate
Ponts vers Sigma-Lab: configs/sigma_params.json, state/last_metrics.json, reports/*
- Mode CI non-interactif: --prompt ou variable d'env SIGMA_PROMPT
- Mode local interactif: boucle input()
"""

import os, json, math, time, hashlib
from dataclasses import dataclass
from typing import Dict, List, Any

# ---------------------- HF / Torch deps ---------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn

# ---------------------- Dossiers par défaut (compat Sigma-Lab) ----------------
CONFIGS_DIR  = os.getenv("SIGMA_CONFIGS_DIR",  "configs")
STATE_DIR    = os.getenv("SIGMA_STATE_DIR",    "state")
REPORTS_DIR  = os.getenv("SIGMA_REPORTS_DIR",  "reports")
OUTPUTS_DIR  = os.getenv("SIGMA_OUTPUTS_DIR",  "outputs")

for _d in (CONFIGS_DIR, STATE_DIR, REPORTS_DIR, OUTPUTS_DIR):
    os.makedirs(_d, exist_ok=True)

PARAMS_FILE   = os.path.join(CONFIGS_DIR, "sigma_params.json")
METRICS_FILE  = os.path.join(STATE_DIR,   "last_metrics.json")
PROV_LOG_FILE = os.path.join(REPORTS_DIR, "sigma_llm_provenance.jsonl")
LAST_REPORT   = os.path.join(REPORTS_DIR, "sigma_llm_last_report.json")

# ---------------------- Utils --------------------------------------------------
def now_ts() -> int:
    return int(time.time())

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default

def safe_dump_json(path: str, data: Any):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

# ---------------------- Paramètres Sigma par défaut ---------------------------
DEFAULT_SIGMA_PARAMS: Dict[str, Any] = {
    "alpha": 0.30,          # poids subjectivité S(t)
    "lambda": 0.20,         # poids objectivité O(t)
    "gamma": 0.006,         # inertie/lissage
    "mu": 0.205,            # gain méta-cohérence
    "beta": 0.0,            # slack
    "theta": [0.52, 0.47, 0.41, 0.50],
    "W": [
        [1.0, 0.42, 0.31, 0.55],
        [0.42, 1.0, 0.48, 0.63],
        [0.31, 0.48, 1.0, 0.58],
        [0.55, 0.63, 0.58, 1.0],
    ],
}

def load_sigma_params() -> Dict[str, Any]:
    params = safe_load_json(PARAMS_FILE, DEFAULT_SIGMA_PARAMS)
    # Invariants structurels
    W = params.get("W", [])
    if not (isinstance(W, list) and len(W) == 4 and all(isinstance(r, list) and len(r) == 4 for r in W)):
        raise ValueError("Invariant failed: W must be a 4x4 list")
    return params

# ---------------------- Moteurs Sigma -----------------------------------------
class CoherenceEngine:
    """Δcoh via divergence KL(P_pred || P_obs) simplifiée."""
    @staticmethod
    def kl_divergence(p: List[float], q: List[float], eps: float = 1e-9) -> float:
        s = 0.0
        for pi, qi in zip(p, q):
            pi = max(pi, eps); qi = max(qi, eps)
            s += pi * math.log(pi / qi)
        return max(0.0, s)

    def delta_coh(self, pred_dist: List[float], obs_dist: List[float]) -> float:
        return self.kl_divergence(pred_dist, obs_dist)

class SubjectivityEngine:
    """S(t) = ∫ w(τ) * Δcoh(τ) dτ ; w(τ) = 1/(1+Δcoh(τ)^2)"""
    def __init__(self):
        self.S = 0.0

    def step(self, delta_coh: float) -> float:
        w = 1.0 / (1.0 + delta_coh * delta_coh)
        self.S += w * delta_coh
        return self.S

class ObjectivityEngine:
    """O(t) ∈ [0,1] : accord avec signal externe; ici 1 - MAE simple."""
    def __init__(self):
        self.O = 0.0

    def step(self, pred_scalar: float, obs_scalar: float) -> float:
        mae = abs(pred_scalar - obs_scalar)
        self.O = max(0.0, 1.0 - mae)
        return self.O

@dataclass
class SigmaMetrics:
    t: int
    delta_coh: float
    S: float
    O: float
    meta_gain: float

# ---------------------- Invariants & Policy Gate ------------------------------
class Invariants:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def check(self, m: SigmaMetrics) -> Dict[str, List[str]]:
        warns, errors = [], []
        if abs(m.S) > 1e6:
            errors.append("S(t) magnitude too large")
        if not (0.0 <= m.O <= 1.0):
            errors.append("O(t) out of bounds [0,1]")
        mg = m.meta_gain
        if not (0.0 <= mg <= 10.0):
            warns.append("meta_gain unusual; check mu/gamma")
        return {"warnings": warns, "errors": errors}

class PolicyGate:
    """Toute promotion/auto-modification doit passer par ici (verrouillable)."""
    def __init__(self, allow_param_promotion: bool = True):
        self.allow_param_promotion = allow_param_promotion

    def can_promote(self, inv_result: Dict[str, List[str]]) -> bool:
        return self.allow_param_promotion and not inv_result.get("errors")

# ---------------------- Mémoire épisodique ------------------------------------
class EpisodicMemory:
    def __init__(self, path=os.path.join(STATE_DIR, "episodes.jsonl")):
        self.path = path

    def append(self, entry: Dict[str, Any]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ---------------------- Perte Sigma (pour fine-tuning éventuel) ---------------
class SigmaLoss(nn.Module):
    def __init__(self, alpha=0.30, lamb=0.20):
        super().__init__()
        self.alpha = alpha
        self.lamb = lamb

    def forward(self, lm_loss: torch.Tensor, delta_coh: float, O: float) -> torch.Tensor:
        decoh = torch.tensor(delta_coh, dtype=torch.float32, device=lm_loss.device)
        obj   = torch.tensor(O,        dtype=torch.float32, device=lm_loss.device)
        return lm_loss + self.alpha * decoh - self.lamb * obj

# ---------------------- LLM Wrapper -------------------------------------------
class SigmaLLM:
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2"):
        # tiny-gpt2 pour CI rapide; change en "gpt2" ou Llama en prod
        self.params = load_sigma_params()
        self.coh   = CoherenceEngine()
        self.subj  = SubjectivityEngine()
        self.obj   = ObjectivityEngine()
        self.invar = Invariants(self.params)
        self.policy= PolicyGate()
        self.mem   = EpisodicMemory()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(model_name)
        self.loss_head = SigmaLoss(
            alpha=self.params["alpha"],
            lamb=self.params["lambda"],
        )

    # -- Observations externes (branchées sur Sigma-Lab / Skywire)
    def read_external_metrics(self) -> Dict[str, Any]:
        return safe_load_json(METRICS_FILE, {
            "final_C": [0.55, 0.48, 0.52, 0.50],
            "frac_sigma": 0.0,
            "rhoA": 2.35,
            "dt": 0.1
        })

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=128, temperature=0.9) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # --------- Sigma metrics (placeholders remplaçables par tes extracteurs)
        pred_dist = [0.25, 0.25, 0.25, 0.25]   # À remplacer si tu as des features réelles
        obs_dist  = self.read_external_metrics().get("final_C", [0.5, 0.5, 0.5, 0.5])

        dcoh = self.coh.delta_coh(pred_dist, obs_dist)
        S    = self.subj.step(dcoh)

        pred_scalar = sum(pred_dist)/4.0
        obs_scalar  = sum(obs_dist)/4.0
        O    = self.obj.step(pred_scalar, obs_scalar)

        meta_gain = float(self.params["mu"]) * (1.0 / (1.0 + float(self.params["gamma"])))
        metrics = SigmaMetrics(t=now_ts(), delta_coh=dcoh, S=S, O=O, meta_gain=meta_gain)

        inv = self.invar.check(metrics)
        self._write_reports(prompt, text, metrics, inv)
        self._log_provenance(prompt, text, metrics, inv)

        return text

    def _write_reports(self, prompt: str, text: str, m: SigmaMetrics, inv: Dict[str, Any]):
        rep = {
            "t": m.t,
            "delta_coh": m.delta_coh,
            "S": m.S,
            "O": m.O,
            "meta_gain": m.meta_gain,
            "invariants": inv,
        }
        safe_dump_json(LAST_REPORT, rep)

        # mémoire épisodique
        self.mem.append({
            "t": m.t,
            "prompt": prompt[-1000:],
            "output": text[-2000:],
            "S": m.S, "O": m.O, "delta_coh": m.delta_coh
        })

    def _log_provenance(self, prompt: str, text: str, m: SigmaMetrics, inv: Dict[str, Any]):
        rec = {
            "ts": m.t,
            "model_name": self.model.config.name_or_path,
            "model_hash": sha256(self.model.config.name_or_path),
            "prompt_hash": sha256(prompt),
            "output_hash": sha256(text),
            "delta_coh": m.delta_coh, "S": m.S, "O": m.O,
            "inv_errors": inv.get("errors", []),
            "inv_warnings": inv.get("warnings", []),
        }
        with open(PROV_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------------------- Entrées CLI (CI-safe + local) -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sigma-LLM runner")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Mode non-interactif: prompt unique, génère et quitte")
    parser.add_argument("--model", type=str, default=os.getenv("SIGMA_LLM_MODEL", "sshleifer/tiny-gpt2"),
                        help="Modèle HF (tiny-gpt2 par défaut pour CI rapide)")
    parser.add_argument("--report", action="store_true",
                        help="Dump du dernier rapport JSON et exit")
    args = parser.parse_args()

    agent = SigmaLLM(model_name=args.model)

    if args.report:
        print(safe_load_json(LAST_REPORT, {}))
        raise SystemExit(0)

    env_prompt = os.getenv("SIGMA_PROMPT")
    if args.prompt or env_prompt:
        prompt = args.prompt or env_prompt
        print("Sigma-LLM v2.6 (non-interactive). Generating...")
        out = agent.generate(f"Human: {prompt}\nAI:")
        print(out)
        if os.path.exists(LAST_REPORT):
            with open(LAST_REPORT, "r", encoding="utf-8") as f:
                print("\n[[REPORT]]\n" + f.read())
        raise SystemExit(0)

    # Mode interactif local
    print("Sigma-LLM v2.6 ready. Type your prompt. Ctrl+C to quit.")
    print("Tip: set SIGMA_PROMPT or use --prompt for CI.")
    while True:
        try:
            msg = input("\nHuman: ")
            out = agent.generate(f"Human: {msg}\nAI:")
            print("AI:", out.split("AI:")[-1].strip())
        except KeyboardInterrupt:
            break
