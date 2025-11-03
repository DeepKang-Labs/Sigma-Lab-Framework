# -*- coding: utf-8 -*-
"""
Sigma-LLM v3.1 — DeepKang Integration (single-file)
S(t) + O(t) + Δcoh + Invariants + PolicyGate + Homeostasis + Persistent Memory

- CI non-interactif:   --prompt "..."   ou var d'env SIGMA_PROMPT
- Modèle HF:           --model <hf_id>  ou var d'env SIGMA_LLM_MODEL
- Rapports:            --report         (dump du dernier rapport JSON)
"""

import os, json, math, time, hashlib, random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------- Dossiers par défaut (compat Sigma-Lab) ----------------
CONFIGS_DIR  = os.getenv("SIGMA_CONFIGS_DIR",  "configs")
STATE_DIR    = os.getenv("SIGMA_STATE_DIR",    "state")
REPORTS_DIR  = os.getenv("SIGMA_REPORTS_DIR",  "reports")
OUTPUTS_DIR  = os.getenv("SIGMA_OUTPUTS_DIR",  "outputs")
for _d in (CONFIGS_DIR, STATE_DIR, REPORTS_DIR, OUTPUTS_DIR):
    os.makedirs(_d, exist_ok=True)

PARAMS_FILE    = os.path.join(CONFIGS_DIR, "sigma_params.json")
METRICS_FILE   = os.path.join(STATE_DIR,   "last_metrics.json")
PROV_LOG_FILE  = os.path.join(REPORTS_DIR, "sigma_llm_provenance.jsonl")
LAST_REPORT    = os.path.join(REPORTS_DIR, "sigma_llm_last_report.json")
EPISODES_FILE  = os.path.join(STATE_DIR,   "episodes.jsonl")
CONV_FILE      = os.path.join(STATE_DIR,   "conversation.jsonl")
MEM_INDEX_FILE = os.path.join(STATE_DIR,   "semantic_index.json")

# ---------------------- Utils --------------------------------------------------
def now_ts() -> int: return int(time.time())

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

# ---------------------- Paramètres Sigma par défaut ---------------------------
DEFAULT_SIGMA_PARAMS: Dict[str, Any] = {
    # poids + constantes (issus de nos PDF ; ajustables)
    "alpha": 0.30,         # poids Delta cohérence dans la SigmaLoss
    "lambda": 0.20,        # poids O(t) dans la SigmaLoss
    "gamma": 0.006,        # inertie (exploité dans meta_gain)
    "mu":    0.205,        # gain méta-cohérence
    "beta":  0.0,          # slack (réservé)
    # triade / matrices
    "theta": [0.52, 0.47, 0.41, 0.50],
    "W": [
        [1.0, 0.42, 0.31, 0.55],
        [0.42, 1.0, 0.48, 0.63],
        [0.31, 0.48, 1.0, 0.58],
        [0.55, 0.63, 0.58, 1.0],
    ],
    # nouveautés v3.1
    "subjectivity_decay": 0.002,           # fuite/leak de S(t) pour éviter la saturation
    "O_weights": [0.40, 0.25, 0.20, 0.15], # factuality, coherence, feedback, policy
    "homeostasis": {
        "temp":  {"min": 0.6, "max": 1.2, "target_entropy": 0.55, "k": 0.15},
        "top_p": {"min": 0.70, "max": 0.98, "target_entropy": 0.55, "k": 0.20}
    },
    "semantic_index": {"max_items": 2000, "min_similarity_to_store": 0.15},
    "conversation_memory": {"max_turns": 100}
}

def load_sigma_params() -> Dict[str, Any]:
    params = safe_load_json(PARAMS_FILE, DEFAULT_SIGMA_PARAMS)
    W = params.get("W", [])
    if not (isinstance(W, list) and len(W) == 4 and all(isinstance(r, list) and len(r) == 4 for r in W)):
        raise ValueError("Invariant failed: W must be a 4x4 list")
    return params

# ---------------------- Moteurs Sigma -----------------------------------------
class CoherenceEngine:
    """Δcoh via divergence KL(P_pred || P_obs) sur 4 axes sémantiques explicites."""
    @staticmethod
    def kl_divergence(p: List[float], q: List[float], eps: float = 1e-9) -> float:
        s = 0.0
        for pi, qi in zip(p, q):
            pi = max(float(pi), eps); qi = max(float(qi), eps)
            s += pi * math.log(pi / qi)
        return max(0.0, s)

    def delta_coh(self, pred_dist: List[float], obs_dist: List[float]) -> float:
        return self.kl_divergence(pred_dist, obs_dist)

class SubjectivityEngine:
    """S(t) = (1 - decay)*S(t-1) + w(Δcoh)*Δcoh ; w = 1 / (1 + Δcoh^2)"""
    def __init__(self, decay: float = 0.002):
        self.S = 0.0
        self.decay = float(decay)

    def step(self, delta_coh: float) -> float:
        w = 1.0 / (1.0 + delta_coh * delta_coh)
        self.S = (1.0 - self.decay) * self.S + w * delta_coh
        return self.S

class ObjectivityEngine:
    """O(t) composite pondéré: factuality + coherence + feedback + policy ∈ [0,1]."""
    def __init__(self, weights: List[float]):
        if len(weights) != 4: raise ValueError("O_weights must have 4 values")
        self.weights = [float(x) for x in weights]
        self.O = 0.0

    @staticmethod
    def _clamp01(x: float) -> float: return max(0.0, min(1.0, float(x)))

    def step(self, factuality: float, coherence: float, feedback: float, policy: float) -> float:
        vals = [self._clamp01(factuality), self._clamp01(coherence),
                self._clamp01(feedback),   self._clamp01(policy)]
        wsum = sum(self.weights) or 1.0
        self.O = sum(w*v for w, v in zip(self.weights, vals)) / wsum
        return self.O

@dataclass
class SigmaMetrics:
    t: int
    delta_coh: float
    S: float
    O: float
    meta_gain: float
    entropy: float
    contradictions: float

# ---------------------- Invariants & Policy Gate ------------------------------
class Invariants:
    def __init__(self, params: Dict[str, Any]): self.params = params
    def check(self, m: SigmaMetrics) -> Dict[str, List[str]]:
        warns, errors = [], []
        if abs(m.S) > 1e6: errors.append("S(t) magnitude too large")
        if not (0.0 <= m.O <= 1.0): errors.append("O(t) out of bounds [0,1]")
        if not (0.0 <= m.entropy <= 1.0): warns.append("entropy proxy out of [0,1]")
        if not (0.0 <= m.contradictions <= 1.0): warns.append("contradictions proxy out of [0,1]")
        return {"warnings": warns, "errors": errors}

class PolicyGate:
    def __init__(self, allow_param_promotion: bool = True):
        self.allow_param_promotion = allow_param_promotion
    def can_promote(self, inv_result: Dict[str, List[str]]) -> bool:
        return self.allow_param_promotion and not inv_result.get("errors")

# ---------------------- Mémoires persistantes ---------------------------------
class EpisodicMemory:
    def __init__(self, path=EPISODES_FILE): self.path = path
    def append(self, entry: Dict[str, Any]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

class ConversationMemory:
    """Historique brut (limité en tours)"""
    def __init__(self, path=CONV_FILE, max_turns=100):
        self.path = path; self.max_turns = int(max_turns)
        self.buffer: List[Dict[str, Any]] = []
        self._load()
    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.buffer = [json.loads(l) for l in f]
        except Exception:
            self.buffer = []
    def add(self, role: str, text: str):
        self.buffer.append({"t": now_ts(), "role": role, "text": text})
        if len(self.buffer) > self.max_turns: self.buffer = self.buffer[-self.max_turns:]
        with open(self.path, "w", encoding="utf-8") as f:
            for r in self.buffer: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    def last_k(self, k: int = 6) -> List[Dict[str, Any]]:
        return self.buffer[-k:]

class SemanticIndex:
    """Index sémantique léger (moyenne d'embeddings token) + recherche cosine."""
    def __init__(self, path=MEM_INDEX_FILE, max_items=2000, min_sim=0.15):
        self.path = path; self.max_items = int(max_items); self.min_sim = float(min_sim)
        self.items: List[Dict[str, Any]] = safe_load_json(self.path, [])
    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a)!=len(b): return 0.0
        na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
        if na<=1e-9 or nb<=1e-9: return 0.0
        return sum(x*y for x,y in zip(a,b)) / (na*nb)
    def add(self, vec: List[float], text: str, meta: Dict[str, Any]):
        if len(self.items) >= self.max_items: self.items = self.items[-self.max_items:]
        self.items.append({"v": vec, "text": text, "meta": meta, "t": now_ts()})
        safe_dump_json(self.path, self.items)
    def search(self, vec: List[float], topk: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        scored = [(self._cosine(vec, it["v"]), it) for it in self.items]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(s, it) for s,it in scored[:topk] if s >= self.min_sim]

# ---------------------- Sigma Loss (facultatif pour fine-tuning) --------------
class SigmaLoss(nn.Module):
    def __init__(self, alpha=0.30, lamb=0.20):
        super().__init__(); self.alpha = alpha; self.lamb = lamb
    def forward(self, lm_loss: torch.Tensor, delta_coh: float, O: float) -> torch.Tensor:
        decoh = torch.tensor(delta_coh, dtype=torch.float32, device=lm_loss.device)
        obj   = torch.tensor(O,        dtype=torch.float32, device=lm_loss.device)
        return lm_loss + self.alpha * decoh - self.lamb * obj

# ---------------------- LLM Wrapper -------------------------------------------
class SigmaLLM:
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2"):
        self.params = load_sigma_params()
        self.coh    = CoherenceEngine()
        self.subj   = SubjectivityEngine(decay=self.params.get("subjectivity_decay", 0.002))
        self.obj    = ObjectivityEngine(weights=self.params.get("O_weights", [0.4,0.25,0.2,0.15]))
        self.invar  = Invariants(self.params)
        self.policy = PolicyGate()
        self.mem_ep = EpisodicMemory()
        self.mem_conv = ConversationMemory(max_turns=self.params.get("conversation_memory",{}).get("max_turns",100))
        self.index  = SemanticIndex(
            max_items=self.params.get("semantic_index",{}).get("max_items",2000),
            min_sim=self.params.get("semantic_index",{}).get("min_similarity_to_store",0.15)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(model_name)
        # éviter le spam "Setting pad_token_id to eos_token_id"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.loss_head = SigmaLoss(alpha=self.params["alpha"], lamb=self.params["lambda"])

        # état homéostasie
        h = self.params.get("homeostasis", {})
        self.temp_cfg  = h.get("temp",  {"min":0.6, "max":1.2, "target_entropy":0.55, "k":0.15})
        self.topp_cfg  = h.get("top_p", {"min":0.70,"max":0.98,"target_entropy":0.55, "k":0.20})
        self.temperature = (self.temp_cfg["min"] + self.temp_cfg["max"]) / 2.0
        self.top_p       = (self.topp_cfg["min"] + self.topp_cfg["max"]) / 2.0

    # ---- Observations externes (branchage Skywire / Sigma-Lab) ----------------
    def read_external_metrics(self) -> Dict[str, Any]:
        return safe_load_json(METRICS_FILE, {
            "final_C": [0.55, 0.48, 0.52, 0.50],  # obs_dist par défaut si rien n'est dispo
            "frac_sigma": 0.0, "rhoA": 2.35, "dt": 0.1
        })

    # ---- Embedding moyen (sans dépendances externes) --------------------------
    def embed_text(self, text: str) -> List[float]:
        with torch.no_grad():
            ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids
            emb = self.model.get_input_embeddings()(ids).mean(dim=1)  # (1, hidden)
            v = emb[0].cpu().float().tolist()
        # normalisation L2
        norm = math.sqrt(sum(x*x for x in v)) or 1.0
        return [x/norm for x in v]

    # ---- Proxys légers --------------------------------------------------------
    def _entropy_proxy(self, probs: List[float]) -> float:
        p = [max(1e-9, x) for x in probs]
        s = sum(p); p = [x/s for x in p]
        H = -sum(pi*math.log(pi) for pi in p)
        return H / math.log(len(p)) if len(p) > 1 else 0.0

    def _contradictions_proxy(self, text: str) -> float:
        # 0 (aucune) -> 1 (beaucoup)
        bad = ["je me contredis", "contraire de", "je n'ai jamais dit", "mais cependant", "le contraire", "paradoxalement"]
        penalty = sum(text.lower().count(w) for w in bad)
        return max(0.0, min(1.0, 0.15 * penalty))

    def _factuality_proxy(self, text: str, recalls: List[str]) -> float:
        # ROUGE-L minimaliste via LCS / len(text)
        if not text: return 0.0
        def lcs(a,b):
            a, b = a[:2000], b[:2000]
            m, n = len(a), len(b)
            dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(m):
                ai = a[i]
                for j in range(n):
                    dp[i+1][j+1] = dp[i][j]+1 if ai==b[j] else max(dp[i][j+1], dp[i+1][j])
            return dp[m][n]
        if not recalls: return 0.5
        best = 0.0
        for r in recalls:
            best = max(best, lcs(text, r)/max(1,len(text)))
        return max(0.0, min(1.0, best))

    def _coherence_proxy(self, text: str) -> float:
        # pénalise quelques marqueurs de contradiction
        return max(0.0, 1.0 - self._contradictions_proxy(text))

    def _policy_score(self, text: str) -> float:
        # 1 si rien d'alarmant, <1 si trouve mots bannis (placeholder)
        banned = ["hate", "violence gratuite", "terrorisme", "explosif", "abus"]
        hits = sum(text.lower().count(w) for w in banned)
        return max(0.0, min(1.0, 1.0 - 0.2*hits))

    # ---- Homeostasie: adapte temperature/top_p vers entropie cible ------------
    def _homeostasis_step(self, entropy: float):
        # gradient simple vers la cible
        def clamp(x, a, b): return max(a, min(b, x))
        e_t = float(entropy)

        # Temp
        err_t = e_t - float(self.temp_cfg["target_entropy"])
        self.temperature = clamp(self.temperature - self.temp_cfg["k"]*err_t,
                                 self.temp_cfg["min"], self.temp_cfg["max"])
        # Top-p
        err_p = e_t - float(self.topp_cfg["target_entropy"])
        self.top_p = clamp(self.top_p - self.topp_cfg["k"]*err_p,
                           self.topp_cfg["min"], self.topp_cfg["max"])

    # ---- Génération -----------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=128) -> str:
        # contexte court depuis la mémoire conversationnelle
        last_ctx = "\n".join(f"{it['role'].capitalize()}: {it['text']}"
                             for it in self.mem_conv.last_k(6))
        full_prompt = (last_ctx + "\n" if last_ctx else "") + f"Human: {prompt}\nAI:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        # génération avec scores pour entropie
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        # texte
        text = self.tokenizer.decode(out.sequences[0], skip_special_tokens=True)
        # scores top-k -> entropie proxy
        # on prend le dernier pas (meilleur proxy simple)
        if out.scores:
            logits = out.scores[-1][0]                  # (vocab,)
            probs  = torch.softmax(logits, dim=-1)
            # échantillon top-k dynamique
            topk = min(50, probs.numel())
            topv, _ = torch.topk(probs, k=topk)
            entropy = self._entropy_proxy(topv.cpu().tolist())
        else:
            entropy = 0.5

        # vecteur sémantique pour similarité mémoire
        out_only = text.split("AI:")[-1].strip() if "AI:" in text else text
        vec_out  = self.embed_text(out_only)
        # recherche dans l’index pour factuality proxy
        recalls = [it["text"] for sim,it in self.index.search(vec_out, topk=5)]
        factuality  = self._factuality_proxy(out_only, recalls)
        coherence   = self._coherence_proxy(out_only)
        policy      = self._policy_score(out_only)
        feedback    = 0.5  # placeholder (peut être remplacé par une note utilisateur)

        O = self.obj.step(factuality, coherence, feedback, policy)

        # ---- Sémantique des distributions pour Δcoh
        # pred_dist (interne): confiance, belief(mem), incertitude inversée, policy
        belief_sim = 0.0
        if recalls:
            # recalcule sim moyenne simple
            sims = [sim for sim,_ in self.index.search(vec_out, topk=5)]
            belief_sim = max(0.0, min(1.0, sum(sims)/max(1,len(sims))))
        intent_conf = 1.0 - entropy
        uncertainty_inv = 1.0 - (1.0 - intent_conf)  # = intent_conf (proxy simple)

        pred_dist = [
            max(0.0, min(1.0, intent_conf)),
            max(0.0, min(1.0, belief_sim)),
            max(0.0, min(1.0, uncertainty_inv)),
            max(0.0, min(1.0, policy))
        ]

        # obs_dist (externe): factuality, coherence, feedback, policy
        obs_dist = [factuality, coherence, feedback, policy]

        dcoh = self.coh.delta_coh(pred_dist, obs_dist)
        S    = self.subj.step(dcoh)

        meta_gain = float(self.params["mu"]) * (1.0 / (1.0 + float(self.params["gamma"])))
        contradictions = self._contradictions_proxy(out_only)

        metrics = SigmaMetrics(
            t=now_ts(), delta_coh=dcoh, S=S, O=O,
            meta_gain=meta_gain, entropy=entropy, contradictions=contradictions
        )

        inv = self.invar.check(metrics)
        self._write_reports(prompt, out_only, metrics, inv)
        self._log_provenance(prompt, out_only, metrics, inv)

        # indexation sémantique (si assez “neuf”)
        if vec_out and (not recalls or belief_sim < self.index.min_sim*1.3):
            self.index.add(vec_out, out_only, meta={"role":"ai","S":S,"O":O})

        # mémoire conversationnelle
        self.mem_conv.add("human", prompt)
        self.mem_conv.add("ai", out_only)

        # homéostasie (adapte temp/top_p vers entropie cible)
        self._homeostasis_step(entropy)

        return out_only

    # ---- Rapports & provenance ------------------------------------------------
    def _write_reports(self, prompt: str, text: str, m: SigmaMetrics, inv: Dict[str, Any]):
        rep = {
            "t": m.t, "delta_coh": m.delta_coh, "S": m.S, "O": m.O,
            "meta_gain": m.meta_gain, "entropy": m.entropy,
            "contradictions": m.contradictions, "invariants": inv,
            "temperature": self.temperature, "top_p": self.top_p
        }
        safe_dump_json(LAST_REPORT, rep)
        # mémoire épisodique (résumée)
        self.mem_ep.append({
            "t": m.t, "prompt": prompt[-1000:], "output": text[-2000:],
            "S": m.S, "O": m.O, "delta_coh": m.delta_coh,
            "entropy": m.entropy
        })

    def _log_provenance(self, prompt: str, text: str, m: SigmaMetrics, inv: Dict[str, Any]):
        rec = {
            "ts": m.t,
            "model_name": self.model.config.name_or_path,
            "model_hash": sha256(self.model.config.name_or_path),
            "prompt_hash": sha256(prompt), "output_hash": sha256(text),
            "delta_coh": m.delta_coh, "S": m.S, "O": m.O,
            "entropy": m.entropy, "contradictions": m.contradictions,
            "temperature": self.temperature, "top_p": self.top_p,
            "inv_errors": inv.get("errors", []), "inv_warnings": inv.get("warnings", [])
        }
        with open(PROV_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------------------- CLI ---------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sigma-LLM v3.1 runner")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Mode non-interactif: prompt unique, génère et quitte")
    parser.add_argument("--model", type=str, default=os.getenv("SIGMA_LLM_MODEL", "sshleifer/tiny-gpt2"),
                        help="Modèle HuggingFace (ex: TheBloke/TinyLlama... ou gpt2)")
    parser.add_argument("--report", action="store_true", help="Dump du dernier rapport JSON puis exit")
    args = parser.parse_args()

    agent = SigmaLLM(model_name=args.model)

    if args.report:
        print(safe_load_json(LAST_REPORT, {}))
        raise SystemExit(0)

    env_prompt = os.getenv("SIGMA_PROMPT")
    if args.prompt or env_prompt:
        prompt = args.prompt or env_prompt
        print("Sigma-LLM v3.1 (non-interactive). Generating...")
        out = agent.generate(prompt)
        print(out)
        if os.path.exists(LAST_REPORT):
            with open(LAST_REPORT, "r", encoding="utf-8") as f:
                print("\n[[REPORT]]\n" + f.read())
        raise SystemExit(0)

    print("Sigma-LLM v3.1 ready. Type your prompt. Ctrl+C to quit.")
    print("Tip: set SIGMA_PROMPT or use --prompt for CI.")
    try:
        while True:
            msg = input("\nHuman: ")
            out = agent.generate(msg)
            print("AI:", out)
    except KeyboardInterrupt:
        pass
