# -*- coding: utf-8 -*-
"""
Sigma-LLM v2.6 — DeepKang Integration (Grok-aligned)
Subjectivité S(t) + Objectivité O(t) + Méta-cohérence Δcoh (KL/JS) + Invariants + PolicyGate
Ponts Sigma-Lab: configs/sigma_params.json, state/last_metrics.json, reports/*
Autonomie contrôlée : journal quotidien, param-promotion "gated", provenance totale.

Dépendances suggérées :
  pip install torch transformers numpy pandas
"""

import os, json, math, time, hashlib, random, pathlib
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase

# ------------------------------------------------------------------------------
# Répertoires compatibles Sigma-Lab
CONFIGS_DIR = os.getenv("SIGMA_CONFIGS_DIR", "configs")
STATE_DIR   = os.getenv("SIGMA_STATE_DIR",   "state")
REPORTS_DIR = os.getenv("SIGMA_REPORTS_DIR", "reports")
OUTPUTS_DIR = os.getenv("SIGMA_OUTPUTS_DIR", "outputs")

for d in (CONFIGS_DIR, STATE_DIR, REPORTS_DIR, OUTPUTS_DIR):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

PARAMS_FILE       = os.path.join(CONFIGS_DIR, "sigma_params.json")
PARAMS_NEXT_FILE  = os.path.join(CONFIGS_DIR, "sigma_params_next.json")
METRICS_FILE      = os.path.join(STATE_DIR,   "last_metrics.json")
AUTOTUNE_LOG_CSV  = os.path.join(STATE_DIR,   "autotune_log.csv")
PROVENANCE_LOG    = os.path.join(REPORTS_DIR, "sigma_llm_provenance.jsonl")
EPISODES_LOG      = os.path.join(STATE_DIR,   "episodes.jsonl")
DAILY_JOURNAL_MD  = os.path.join(REPORTS_DIR, "sigma_daily_journal.md")
LAST_REPORT_JSON  = os.path.join(REPORTS_DIR, "sigma_llm_last_report.json")

# ------------------------------------------------------------------------------
# Utilitaires I/O sûrs
def now_ts() -> int:
    return int(time.time())

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def safe_dump_json(path: str, data: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def append_jsonl(path: str, rec: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def append_md(path: str, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + ("\n" if not text.endswith("\n") else ""))

# ------------------------------------------------------------------------------
# Paramètres Sigma par défaut (compatibles Sigma-Lab et Grok)
DEFAULT_SIGMA_PARAMS: Dict[str, Any] = {
    "alpha": 0.315,       # poids subjectivité (S)
    "lambda": 0.195,      # poids objectivité (O)
    "gamma": 0.006,       # inertie/lissage méta
    "mu": 0.205,          # gain méta-cohérence
    "beta": 0.0,          # slack réservé
    "theta": [0.52, 0.47, 0.41, 0.50],  # cibles
    "W": [
        [1.00, 0.42, 0.31, 0.55],
        [0.42, 1.00, 0.48, 0.63],
        [0.31, 0.48, 1.00, 0.58],
        [0.55, 0.63, 0.58, 1.00],
    ],
    # bornes opérationnelles (invariants/CI)
    "bounds": {
        "alpha": [0.25, 0.35],
        "lambda": [0.15, 0.25],
        "gamma": [0.002, 0.02],
        "mu":    [0.15, 0.30]
    },
    "small_gain_limit": 0.45,   # garde-fou petit-gain: alpha+beta+gamma < limit
    "promote_requirements": {   # conditions minimales pour promouvoir des params
        "min_O": 0.35,
        "max_inv_errors": 0
    }
}

def load_sigma_params() -> Dict[str, Any]:
    params = safe_load_json(PARAMS_FILE, DEFAULT_SIGMA_PARAMS)
    # Invariants structurels
    W = params.get("W", [])
    assert isinstance(W, list) and len(W) == 4, "W must be 4x4 list"
    for r in W:
        assert isinstance(r, list) and len(r) == 4, "W must be 4x4 list"
    return params

# ------------------------------------------------------------------------------
# Moteurs Sigma : cohérence, subjectivité, objectivité
class CoherenceEngine:
    """Δcoh : divergence entre distributions (prédites vs observées).
       KL symétrisée + option JS pour robustesse.
    """
    @staticmethod
    def _normalize(p: List[float], eps: float = 1e-9) -> List[float]:
        s = sum(max(v, 0.0) for v in p)
        if s <= eps:
            n = len(p)
            return [1.0 / n] * n
        return [max(v, 0.0) / s for v in p]

    @staticmethod
    def kl(p: List[float], q: List[float], eps: float = 1e-12) -> float:
        p = [max(x, eps) for x in p]
        q = [max(x, eps) for x in q]
        return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))

    @staticmethod
    def js(p: List[float], q: List[float], eps: float = 1e-12) -> float:
        # Jensen-Shannon distance^2 (base e)
        m = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]
        return 0.5 * (CoherenceEngine.kl(p, m, eps) + CoherenceEngine.kl(q, m, eps))

    def delta_coh(self, pred: List[float], obs: List[float], use_js: bool = True) -> float:
        p = self._normalize(pred)
        o = self._normalize(obs)
        if use_js:
            return max(0.0, self.js(p, o))
        # KL symétrisée
        return max(0.0, 0.5 * (self.kl(p, o) + self.kl(o, p)))

class SubjectivityEngine:
    """S(t) = ∫ w(τ)*Δcoh(τ) dτ, w=1/(1+Δcoh^2)."""
    def __init__(self):
        self.S: float = 0.0

    def step(self, delta_coh: float) -> float:
        w = 1.0 / (1.0 + delta_coh * delta_coh)
        self.S += w * delta_coh
        return self.S

class ObjectivityEngine:
    """O(t) ∈ [0,1] : accord avec réalité externe. Placeholder robuste."""
    def __init__(self):
        self.O: float = 0.0

    @staticmethod
    def _bound(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def step(self, pred_scalar: float, obs_scalar: float) -> float:
        # MAE bornée → 1 - MAE (si valeurs déjà normalisées [0,1])
        mae = abs(pred_scalar - obs_scalar)
        self.O = self._bound(1.0 - mae)
        return self.O

# ------------------------------------------------------------------------------
# Métriques/rapports + invariants + politique de promotion
@dataclass
class SigmaMetrics:
    t: int
    delta_coh: float
    S: float
    O: float
    meta_gain: float
    note: str = ""

class Invariants:
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def check(self, m: SigmaMetrics) -> Dict[str, List[str]]:
        warns, errors = [], []
        # Stabilité globale de S
        if not np.isfinite(m.S) or abs(m.S) > 1e9:
            errors.append("S overflow or non-finite")
        # O borné
        if not (0.0 <= m.O <= 1.0):
            errors.append("O out-of-bounds [0,1]")
        # Δcoh non-négatif et raisonnable
        if m.delta_coh < -1e-12:
            errors.append("delta_coh negative")
        if m.delta_coh > 10.0:
            warns.append("delta_coh unusually large (>10)")
        # Petit-gain
        alpha = float(self.params.get("alpha", 0.3))
        beta  = float(self.params.get("beta", 0.0))
        gamma = float(self.params.get("gamma", 0.006))
        limit = float(self.params.get("small_gain_limit", 0.45))
        if (alpha + beta + gamma) >= limit:
            warns.append(f"small-gain near/over limit ({alpha+beta+gamma:.3f} ≥ {limit})")
        return {"warnings": warns, "errors": errors}

class PolicyGate:
    """Empêche toute auto-modif non contrôlée. La promotion d’un jeu de paramètres
    ne s’effectue que si les invariants sont OK et que les conditions min O, etc., sont remplies.
    """
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.enabled = True  # peut être désactivé manuellement si besoin

    def can_promote(self, inv: Dict[str, List[str]], last_O: float) -> bool:
        if not self.enabled:
            return False
        req = self.params.get("promote_requirements", {})
        min_O = float(req.get("min_O", 0.35))
        max_err = int(req.get("max_inv_errors", 0))
        err_count = len(inv.get("errors", []))
        return (err_count <= max_err) and (last_O >= min_O)

# ------------------------------------------------------------------------------
# Perte Sigma (pour fine-tuning optionnel)
class SigmaLoss(nn.Module):
    def __init__(self, alpha=0.315, lamb=0.195, mu=0.205):
        super().__init__()
        self.alpha = float(alpha)
        self.lamb  = float(lamb)
        self.mu    = float(mu)

    def forward(self, lm_loss: torch.Tensor, dcoh: float, O: float) -> torch.Tensor:
        decoh = torch.tensor(float(dcoh), dtype=torch.float32, device=lm_loss.device)
        obj   = torch.tensor(float(O),    dtype=torch.float32, device=lm_loss.device)
        return lm_loss + self.alpha * decoh - self.lamb * obj

# ------------------------------------------------------------------------------
# SigmaLLM : cœur agent (LLM + Sigma Dynamics abstraites)
class SigmaLLM:
    def __init__(self, model_name: str = "gpt2"):
        self.params: Dict[str, Any] = load_sigma_params()
        self.coh  = CoherenceEngine()
        self.subj = SubjectivityEngine()
        self.obj  = ObjectivityEngine()
        self.inv  = Invariants(self.params)
        self.gate = PolicyGate(self.params)

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            # sécurité GPT-2
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.loss_head = SigmaLoss(
            alpha=self.params["alpha"],
            lamb=self.params["lambda"],
            mu=self.params["mu"]
        )

        # Tampons historiques simples
        self._recent_dcoh: List[float] = []
        self._recent_O:    List[float] = []

    # ------------------------- Observables externes ---------------------------
    def read_external_metrics(self) -> Dict[str, Any]:
        # Brancher ici les métriques réelles (Skywire/Sigma-Lab)
        # Structure attendue (exemple robuste):
        # {
        #   "frac_sigma": float,                  # 0..100 ou 0..1 (on normalise)
        #   "final_C":   [c1,c2,c3,c4],          # dist/état 4D
        #   "market":    {"btc":..., "eth":...}, # optionnel
        #   "dt":        0.1,
        #   ...
        # }
        return safe_load_json(METRICS_FILE, {
            "frac_sigma": 0.35,
            "final_C":    [0.5, 0.5, 0.5, 0.5],
            "dt":         0.1
        })

    # --------------------- Extraction distributions utiles -------------------
    def _extract_pred_dist(self, logits: torch.Tensor) -> List[float]:
        # On réduit à 4 "macro-bins" pour rester compatible à W/thêta 4D
        probs = torch.softmax(logits, dim=-1)[0, -1]  # (Vocab,)
        V = probs.shape[0]
        # 4 bins égaux
        bins = torch.chunk(probs, 4)
        vals = [float(b.sum().item()) for b in bins]
        s = sum(vals) or 1.0
        return [v / s for v in vals]

    def _extract_obs_dist(self, ext: Dict[str, Any]) -> List[float]:
        obs = ext.get("final_C", [0.5, 0.5, 0.5, 0.5])
        if not isinstance(obs, list) or len(obs) != 4:
            return [0.25, 0.25, 0.25, 0.25]
        # normalisation
        s = sum(max(0.0, float(x)) for x in obs)
        if s <= 1e-12:
            return [0.25, 0.25, 0.25, 0.25]
        return [max(0.0, float(x)) / s for x in obs]

    def _scalar_from_dist(self, d: List[float]) -> float:
        # moyenne simple (0..1) si d est une distribution
        return float(sum(d) / max(1, len(d)))

    # ----------------------------- Génération --------------------------------
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=160, temperature=0.9, top_p=0.95) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Méta-cohérence : comparer distribution prédite vs observée (ext)
        # On récupère les logits du dernier pas via une seconde passe courte
        with torch.no_grad():
            out_full = self.model(**inputs, labels=None, output_hidden_states=False)
            # fallback: si pas de logits dans generate, on utilise forward
            logits = out_full.logits  # (1, seq, vocab)
        pred_dist = self._extract_pred_dist(logits)
        ext       = self.read_external_metrics()
        obs_dist  = self._extract_obs_dist(ext)

        dcoh = self.coh.delta_coh(pred_dist, obs_dist, use_js=True)
        S = self.subj.step(dcoh)
        pred_scalar = self._scalar_from_dist(pred_dist)
        obs_scalar  = self._scalar_from_dist(obs_dist)
        O = self.obj.step(pred_scalar, obs_scalar)

        meta_gain = float(self.params["mu"]) * (1.0 / (1.0 + float(self.params["gamma"])))
        m = SigmaMetrics(t=now_ts(), delta_coh=dcoh, S=S, O=O, meta_gain=meta_gain)
        inv = self.inv.check(m)

        self._recent_dcoh.append(dcoh)
        self._recent_O.append(O)
        self._write_reports(prompt, text, m, inv)

        return text

    # -------------------------- Fine-tuning léger -----------------------------
    def finetune_step(self, batch_prompts: List[str], batch_targets: List[str]) -> float:
        self.model.train()
        losses = []
        for x, y in zip(batch_prompts, batch_targets):
            inputs = self.tokenizer(x, return_tensors="pt", padding=True)
            labels = self.tokenizer(y, return_tensors="pt", padding=True)["input_ids"]
            labels = labels.to(inputs["input_ids"].device)
            out = self.model(**inputs, labels=labels)
            lm_loss = out.loss

            ext = self.read_external_metrics()
            obs_dist = self._extract_obs_dist(ext)
            # approximation : distribution “plate” côté modèle (peut être remplacé par features réelles)
            pred_dist = [0.25, 0.25, 0.25, 0.25]
            dcoh = self.coh.delta_coh(pred_dist, obs_dist)
            O = self.obj.step(sum(pred_dist)/4.0, sum(obs_dist)/4.0)

            sigma_loss = self.loss_head(lm_loss, dcoh, O)
            sigma_loss.backward()
            losses.append(float(sigma_loss.detach().cpu()))

        self.model.eval()
        return float(np.mean(losses)) if losses else 0.0

    # ----------------------- Promotion (autotune) "gated" ---------------------
    def propose_and_maybe_promote(self, next_params: Dict[str, Any], inv: Dict[str, List[str]], last_O: float) -> bool:
        """Écrit configs/sigma_params_next.json, et promeut -> sigma_params.json si PolicyGate OK."""
        # Clamp bornes + petit-gain
        bounded = self._bounded_params(next_params)
        safe_dump_json(PARAMS_NEXT_FILE, bounded)

        if self.gate.can_promote(inv, last_O):
            safe_dump_json(PARAMS_FILE, bounded)
            # provenance
            append_jsonl(PROVENANCE_LOG, {
                "ts": now_ts(),
                "action": "promote_params",
                "params_next": bounded,
                "last_O": last_O,
                "inv_errors": inv.get("errors", []),
                "inv_warnings": inv.get("warnings", [])
            })
            # mise à jour runtime
            self.params = load_sigma_params()
            self.loss_head = SigmaLoss(self.params["alpha"], self.params["lambda"], self.params["mu"])
            return True
        return False

    def _bounded_params(self, p: Dict[str, Any]) -> Dict[str, Any]:
        bounds = self.params.get("bounds", DEFAULT_SIGMA_PARAMS["bounds"])
        q = dict(self.params)  # base actuelle
        for k in ("alpha", "lambda", "gamma", "mu"):
            v = float(p.get(k, q[k]))
            lo, hi = bounds.get(k, [v, v])
            v = max(lo, min(hi, v))
            q[k] = v
        # petit-gain
        limit = float(q.get("small_gain_limit", 0.45))
        if (q["alpha"] + q.get("beta", 0.0) + q["gamma"]) >= limit:
            # réduire gamma prioritairement
            overflow = (q["alpha"] + q.get("beta", 0.0) + q["gamma"]) - (limit - 1e-3)
            q["gamma"] = max(bounds["gamma"][0], q["gamma"] - overflow)
        return q

    # ------------------------- Journal autonome --------------------------------
    @torch.no_grad()
    def daily_journal_tick(self, force: bool = False) -> Optional[str]:
        """Génère un court paragraphe autonome (sans prompt humain) + enregistre.
           Par défaut 1 fois / ~6h via hasard doux, ou à la demande avec force=True.
        """
        if not force and random.random() > 0.166:  # ≈ 1/6 de chances par invocation
            return None

        ext = self.read_external_metrics()
        frac_sigma = float(ext.get("frac_sigma", 0.0))
        # normalisation si donné en 0..100
        if frac_sigma > 1.0:
            frac_sigma = frac_sigma / 100.0

        # Créer un "prompt interne" court
        prompt = (
            "Système: Rédige une note courte et factuelle sur l'état Sigma.\n"
            f"Contexte: frac_sigma={frac_sigma:.2f}, O_recent={np.mean(self._recent_O[-10:]) if self._recent_O else 0.0:.2f}\n"
            "Style: sobre, précis, 2 phrases max.\n"
            "AI:"
        )
        note = self.generate(prompt, max_new_tokens=80, temperature=0.7)
        # Post-traitement : extraire la fin après "AI:"
        if "AI:" in note:
            note = note.split("AI:")[-1].strip()

        # Append dans markdown
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now_ts()))
        append_md(DAILY_JOURNAL_MD, f"### {ts} UTC\n\n{note}\n")

        # Mémoire épisodique
        append_jsonl(EPISODES_LOG, {
            "ts": now_ts(),
            "kind": "daily_journal",
            "note": note,
            "frac_sigma": frac_sigma
        })
        return note

    # ------------------------ Rapports & provenance ----------------------------
    def _write_reports(self, prompt: str, text: str, m: SigmaMetrics, inv: Dict[str, List[str]]):
        report = {
            "t": m.t,
            "delta_coh": m.delta_coh,
            "S": m.S,
            "O": m.O,
            "meta_gain": m.meta_gain,
            "invariants": inv,
            "params": {k: self.params[k] for k in ("alpha", "lambda", "gamma", "mu")},
        }
        safe_dump_json(LAST_REPORT_JSON, report)

        # Épisodes + provenance
        append_jsonl(EPISODES_LOG, {
            "ts": m.t,
            "kind": "chat",
            "prompt_sha": sha256(prompt),
            "output_sha": sha256(text),
            "delta_coh": m.delta_coh,
            "S": m.S, "O": m.O
        })
        append_jsonl(PROVENANCE_LOG, {
            "ts": m.t,
            "model": self.model.config.name_or_path,
            "model_hash": sha256(self.model.config.name_or_path),
            "prompt_hash": sha256(prompt),
            "output_hash": sha256(text),
            "inv_errors": inv.get("errors", []),
            "inv_warnings": inv.get("warnings", []),
            "S": m.S, "O": m.O, "delta_coh": m.delta_coh
        })

# ------------------------------------------------------------------------------
# Entrée CLI simple (tests locaux)
if __name__ == "__main__":
    mdl = os.getenv("SIGMA_LLM_MODEL", "gpt2")
    agent = SigmaLLM(model_name=mdl)
    print("Sigma-LLM v2.6 ready. Type your prompt. Ctrl+C to quit.")
    print("Tip: set SIGMA_CONFIGS_DIR/STATE_DIR/REPORTS_DIR to wire Sigma-Lab.")
    try:
        # Tick autonome léger au démarrage (note/journal potentielle)
        agent.daily_journal_tick(force=False)
        while True:
            msg = input("\nHuman: ")
            out = agent.generate(f"Human: {msg}\nAI:")
            # N'affiche que la partie après "AI:" s'il y en a une
            print("AI:", out.split("AI:")[-1].strip())
    except KeyboardInterrupt:
        print("\nBye.")
