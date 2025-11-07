# -*- coding: utf-8 -*-
"""
Sigma-LLM v3.3.1 — DeepKang Integration (Meta-Llama-3 ready, CI-safe)

+ S(t)/O(t)/Δcoh + subjectivity_decay
+ Homeostasis (temperature/top_p) pilotée par entropie cible
+ Mémoire conversationnelle persistante + Index sémantique persistant
+ Objectivité pondérée (factualité/cohérence/feedback/policy)
+ Module de factualité (RAG local + heuristiques)
+ Branches "features réelles" si des fichiers existent dans state/
+ Export Prometheus (SIGMA_PROM_PORT) pour Grafana
+ Meta-Llama-3/Mistral/Phi-3: chat template natif + anti-bégaiement
"""

import os, json, math, time, hashlib, threading, pathlib, sys, re
from dataclasses import dataclass
from typing import Dict, List, Any, Deque, Tuple
from collections import deque

# --- Unbuffered stdout pour Actions / CI ---
try:
    sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
except Exception:
    pass
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Modules légers internes (optionnels)
try:
    from sigma.core.rag import retrieve_topk
except Exception:
    def retrieve_topk(q, k=3): return []
try:
    from sigma.core.judge import judge_factuality, judge_coherence
except Exception:
    def judge_factuality(u,o,p): return 0.5
    def judge_coherence(o): return 0.5

# ---------------------- Dossiers / Fichiers -----------------------------------
CONFIGS_DIR  = os.getenv("SIGMA_CONFIGS_DIR",  "configs")
STATE_DIR    = os.getenv("SIGMA_STATE_DIR",    "state")
REPORTS_DIR  = os.getenv("SIGMA_REPORTS_DIR",  "reports")
OUTPUTS_DIR  = os.getenv("SIGMA_OUTPUTS_DIR",  "outputs")
for d in (CONFIGS_DIR, STATE_DIR, REPORTS_DIR, OUTPUTS_DIR):
    os.makedirs(d, exist_ok=True)

PARAMS_FILE   = os.path.join(CONFIGS_DIR, "sigma_params.json")
METRICS_FILE  = os.path.join(STATE_DIR,   "last_metrics.json")
PROV_LOG_FILE = os.path.join(REPORTS_DIR, "sigma_llm_provenance.jsonl")
LAST_REPORT   = os.path.join(REPORTS_DIR, "sigma_llm_last_report.json")
EPISODES_FILE = os.path.join(STATE_DIR,   "episodes.jsonl")
CONV_FILE     = os.path.join(STATE_DIR,   "conversation.json")
SEMIDX_FILE   = os.path.join(STATE_DIR,   "semantic_index.jsonl")

# CI
RAW_OUT_TXT = os.path.join(OUTPUTS_DIR, "latest_output.txt")
STEP_SUMMARY = os.getenv("GITHUB_STEP_SUMMARY")

# Optionnel: Prometheus
PROM_PORT = int(os.getenv("SIGMA_PROM_PORT", "0"))

def now_ts() -> int: return int(time.time())
def sha256(s: str) -> str: return hashlib.sha256(s.encode("utf-8")).hexdigest()

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

# ---------------------- Helpers robustes --------------------------------------
def _clip01(x):
    try:
        xf = float(x)
        return 0.0 if xf < 0 else 1.0 if xf > 1 else xf
    except Exception:
        return 0.5

def _coerce_score(obj) -> float:
    if obj is None:
        return 0.5
    if isinstance(obj, (int, float)):
        return _clip01(obj)
    if isinstance(obj, dict):
        for k in ("score", "confidence", "prob", "p", "value"):
            if k in obj: return _clip01(obj[k])
        if "ok" in obj:
            try: return 1.0 if bool(obj["ok"]) else 0.0
            except Exception: pass
        for k in ("components", "parts", "scores"):
            v = obj.get(k)
            if isinstance(v, (list, tuple)) and v:
                nums = [float(x) for x in v if isinstance(x, (int, float))]
                if nums: return _clip01(sum(nums)/len(nums))
        nums = [float(v) for v in obj.values() if isinstance(v, (int, float))]
        if nums: return _clip01(sum(nums)/len(nums))
    if isinstance(obj, (list, tuple)) and obj:
        nums = [float(x) for x in obj if isinstance(x, (int, float))]
        if nums: return _clip01(sum(nums)/len(nums))
    return 0.5

# ---------------------- Paramètres Sigma par défaut ---------------------------
DEFAULT_SIGMA_PARAMS: Dict[str, Any] = {
    "alpha": 0.315, "beta": 0.0, "gamma": 0.006, "lambda": 0.195, "mu": 0.205,
    "theta": [0.52, 0.47, 0.41, 0.50],
    "W": [
        [1.00, 0.42, 0.31, 0.55],
        [0.42, 1.00, 0.48, 0.63],
        [0.31, 0.48, 1.00, 0.58],
        [0.55, 0.63, 0.58, 1.00]
    ],
    "nu": 0.35, "seuil_veto": 0.08,
    "subjectivity_decay": 0.002,
    "O_weights": [0.40, 0.25, 0.20, 0.15],
    "homeostasis": {
        "temp":  {"min": 0.60, "max": 1.20, "target_entropy": 0.55, "k": 0.15},
        "top_p": {"min": 0.70, "max": 0.98, "target_entropy": 0.55, "k": 0.20}
    },
    "semantic_index": {"max_items": 2000, "min_similarity_to_store": 0.15},
    "conversation_memory": {"max_turns": 100},
}

def load_sigma_params(config_dir: str = CONFIGS_DIR) -> Dict[str, Any]:
    p = pathlib.Path(config_dir) / "sigma_params.json"
    if p.exists():
        try:
            params = json.load(open(p, "r", encoding="utf-8"))
            merged = DEFAULT_SIGMA_PARAMS | params
            print(f"[Sigma] Paramètres chargés depuis {p}")
            return merged
        except Exception as e:
            print(f"[Sigma] ⚠️ Erreur lecture params ({e}), fallback défauts")
    else:
        print(f"[Sigma] ⚠️ {p} introuvable, usage des défauts")
    return DEFAULT_SIGMA_PARAMS.copy()

# ---------------------- Cœurs Sigma -------------------------------------------
class CoherenceEngine:
    @staticmethod
    def kl_divergence(p: List[float], q: List[float], eps: float=1e-9) -> float:
        s = 0.0
        for pi, qi in zip(p, q):
            pi = max(pi, eps); qi = max(qi, eps)
            s += pi * math.log(pi/qi)
        return max(0.0, s)
    def delta_coh(self, pred: List[float], obs: List[float]) -> float:
        return self.kl_divergence(pred, obs)

class SubjectivityEngine:
    """S(t) avec fuite: S = (1-d)*S + w*Δcoh, w = 1/(1+Δcoh^2)"""
    def __init__(self, decay: float=0.0):
        self.S = 0.0; self.decay = float(decay)
    def step(self, delta_coh: float) -> float:
        w = 1.0 / (1.0 + delta_coh*delta_coh)
        self.S = (1.0 - self.decay)*self.S + w*delta_coh
        return self.S

class ObjectivityEngine:
    def __init__(self, weights: List[float]):
        if len(weights) != 4:
            raise ValueError("O_weights must have 4 components")
        self.w = [float(x) for x in weights]; self.O = 0.0
    @staticmethod
    def _clip01(x: float) -> float: return 0.0 if x < 0 else 1.0 if x > 1 else x
    def step(self, factuality: float, coherence: float, feedback: float, policy: float) -> float:
        comps = [factuality, coherence, feedback, policy]
        val = sum(w*self._clip01(c) for w,c in zip(self.w, comps))
        self.O = self._clip01(val); return self.O

# ---------------------- Mémoires ----------------------------------------------
class EpisodicMemory:
    def __init__(self, path=EPISODES_FILE): self.path = path
    def append(self, entry: Dict[str, Any]):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

class ConversationMemory:
    def __init__(self, max_turns: int, path=CONV_FILE):
        self.path = path; self.max_turns = int(max_turns)
        data = safe_load_json(self.path, [])
        self.buffer: Deque[Dict[str, str]] = deque(data[-self.max_turns:], maxlen=self.max_turns)
    def add(self, role: str, text: str):
        self.buffer.append({"role": role, "text": text})
        safe_dump_json(self.path, list(self.buffer))
    def as_messages(self) -> List[Dict[str,str]]:
        return [{"role": m["role"], "content": m["text"]} for m in self.buffer]
    def render_legacy(self) -> str:
        return "\n".join(f"{m['role'].capitalize()}: {m['text']}" for m in self.buffer)

class SemanticIndex:
    """Mémoire sémantique légère (moyenne d’embeddings tokens)."""
    def __init__(self, path=SEMIDX_FILE, max_items=2000, min_sim=0.15):
        self.path, self.max_items, self.min_sim = path, int(max_items), float(min_sim)
        self.items: List[Dict[str, Any]] = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f: self.items.append(json.loads(line))
        except Exception: pass
    @staticmethod
    def _cosine(a: List[float], b: List[float], eps: float=1e-9) -> float:
        dot = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
        return 0.0 if na*nb < eps else dot/(na*nb)
    def query(self, embedding: List[float], k: int=5) -> List[Tuple[float, str]]:
        scored = [(self._cosine(embedding, it["emb"]), it["text"]) for it in self.items]
        scored.sort(reverse=True); return scored[:k]
    def add(self, embedding: List[float], text: str):
        if self.items:
            best = max(self._cosine(embedding, it["emb"]) for it in self.items)
            if best >= self.min_sim: return
        self.items.append({"t": now_ts(), "emb": embedding, "text": text})
        if len(self.items) > self.max_items: self.items = self.items[-self.max_items:]
        with open(self.path, "w", encoding="utf-8") as f:
            for it in self.items: f.write(json.dumps(it, ensure_ascii=False) + "\n")

# ---------------------- Perte Sigma (FT optionnel) ----------------------------
class SigmaLoss(nn.Module):
    def __init__(self, alpha=0.30, lamb=0.20):
        super().__init__(); self.alpha = alpha; self.lamb = lamb
    def forward(self, lm_loss: torch.Tensor, delta_coh: float, O: float) -> torch.Tensor:
        decoh = torch.tensor(delta_coh, dtype=torch.float32, device=lm_loss.device)
        obj   = torch.tensor(O,        dtype=torch.float32, device=lm_loss.device)
        return lm_loss + self.alpha*decoh - self.lamb*obj

# ---------------------- Fact-checker simple -----------------------------------
class FactualityModule:
    def __init__(self, sidx: SemanticIndex, embed_func):
        self.sidx = sidx; self.embed = embed_func
    def score(self, text: str) -> float:
        try:
            emb = self.embed(text); hits = self.sidx.query(emb, k=5)
            anchor = max((sim for sim,_ in hits), default=0.0)
        except Exception:
            anchor = 0.0
        penal = 0.0; lower = text.lower()
        if ("not true" in lower) or ("impossible" in lower and "but" in lower): penal += 0.15
        words = lower.split()
        if len(words) > 8:
            uniq = len(set(words)); ratio = uniq/len(words)
            if ratio < 0.45: penal += 0.10
        return max(0.0, min(1.0, 0.6*anchor + 0.4*(1.0 - penal)))

# ---------------------- Export Prometheus (optionnel) -------------------------
PROM = None
def _start_prometheus(port: int):
    global PROM
    try:
        from prometheus_client import start_http_server, Gauge
        start_http_server(port)
        PROM = {
            "delta_coh": Gauge("sigma_delta_coh", "Sigma Δcoh"),
            "subjectivity": Gauge("sigma_S", "Sigma S(t)"),
            "objectivity": Gauge("sigma_O", "Sigma O(t)"),
            "entropy": Gauge("sigma_entropy", "Next-token entropy"),
            "temp": Gauge("sigma_temp", "Sampling temperature"),
            "top_p": Gauge("sigma_top_p", "Top-p"),
        }
        print(f"[Prometheus] exporter running on :{port}")
    except Exception as e:
        print(f"[Prometheus] disabled ({e})")

def _emit_prom(m: "SigmaMetrics"):
    if PROM:
        PROM["delta_coh"].set(m.delta_coh)
        PROM["subjectivity"].set(m.S)
        PROM["objectivity"].set(m.O)
        PROM["entropy"].set(m.entropy)
        PROM["temp"].set(m.temp)
        PROM["top_p"].set(m.top_p)

# ---------------------- Typage des métriques ----------------------------------
@dataclass
class SigmaMetrics:
    t: int; delta_coh: float; S: float; O: float; meta_gain: float
    entropy: float; temp: float; top_p: float

class Invariants:
    def check(self, m: SigmaMetrics) -> Dict[str, List[str]]:
        warns, errors = [], []
        if abs(m.S) > 1e6: errors.append("S(t) magnitude too large")
        if not (0.0 <= m.O <= 1.0): errors.append("O(t) out of [0,1]")
        if not (0.0 <= m.entropy <= 12.0): warns.append("entropy unusual")
        return {"warnings": warns, "errors": errors}

class PolicyGate:
    def __init__(self, allow_param_promotion=True): self.allow_param_promotion = allow_param_promotion
    def can_promote(self, inv: Dict[str, List[str]]) -> bool:
        return self.allow_param_promotion and not inv.get("errors")

# ---------------------- LLM Wrapper -------------------------------------------
class SigmaLLM:
    def __init__(self, model_name: str = None):
        self.params = load_sigma_params()
        self.coh   = CoherenceEngine()
        self.subj  = SubjectivityEngine(decay=self.params.get("subjectivity_decay", 0.0))
        self.obj   = ObjectivityEngine(weights=self.params.get("O_weights", [0.4, 0.25, 0.2, 0.15]))
        self.invar = Invariants(); self.policy = PolicyGate()
        self.epis  = EpisodicMemory()
        self.conv  = ConversationMemory(max_turns=self.params.get("conversation_memory", {}).get("max_turns", 100))
        si         = self.params.get("semantic_index", {})
        self.sidx  = SemanticIndex(max_items=si.get("max_items", 2000),
                                   min_sim=si.get("min_similarity_to_store", 0.15))

        preferred = model_name or os.getenv("SIGMA_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        fallback  = "gpt2"

        # ---- Device & dtype
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = None
            torch_dtype = torch.float32

        # ---- Chargement modèle/tokenizer (trust_remote_code inutile pour Llama-3)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(preferred, use_fast=True)
            self.model     = AutoModelForCausalLM.from_pretrained(
                preferred,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map=device_map
            )
            self.model_name = preferred
        except Exception as e:
            print(f"[Model] fallback to {fallback} ({e})")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback)
            self.model     = AutoModelForCausalLM.from_pretrained(fallback)
            self.model_name = fallback

        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.loss_head = SigmaLoss(alpha=self.params["alpha"], lamb=self.params["lambda"])

        # état homeostasie init
        self.temp  = 0.95
        self.top_p = 0.95

        # factualité
        self.factual = FactualityModule(self.sidx, self.embed_text)

        # Prometheus
        if PROM_PORT > 0:
            threading.Thread(target=_start_prometheus, args=(PROM_PORT,), daemon=True).start()

        # Stop tokens connus (Llama-3 utilise souvent <|eot_id|>)
        try:
            vocab = self.tokenizer.get_vocab()
        except Exception:
            vocab = {}
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in vocab else None
        self.stop_ids = {tid for tid in (self.tokenizer.eos_token_id, eot_id) if tid is not None}

    # ---- Embedding texte (moyenne d’embeddings tokens)
    def embed_text(self, text: str) -> List[float]:
        ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)["input_ids"][0]
        emb = self.model.get_input_embeddings()(ids.to(self.model.device)).detach().cpu().mean(dim=0).tolist()
        return emb

    # ---- Entropie approx
    @torch.no_grad()
    def _entropy_next(self, inputs) -> float:
        logits = self.model(**inputs).logits[:, -1, :]
        probs  = torch.softmax(logits, dim=-1)
        ent    = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).item()
        return float(ent)

    # ---- Homeostasie (contrôle P → entropie cible)
    def _adjust_homeostasis(self, entropy: float):
        def clamp(x, lo, hi): return max(lo, min(hi, x))
        hs = self.params["homeostasis"]
        for key, cur in (("temp", self.temp), ("top_p", self.top_p)):
            cfg = hs[key]; error = cfg["target_entropy"] - entropy
            newv  = clamp(cur + cfg["k"] * error, cfg["min"], cfg["max"])
            if key == "temp": self.temp = newv
            else: self.top_p = newv

    # ---- Prompting universel (chat template si dispo)
    def _build_inputs(self, user_msg: str):
        messages = self.conv.as_messages() + [{"role": "user", "content": user_msg}]
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt")
        except Exception:
            ctx = self.conv.render_legacy()
            prompt = (ctx + "\nAI:").strip()
            inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs.to(self.model.device)

    # ---- Nettoyage post-génération (anti-bégaiement, coupe à l'assistant)
    def _clean_output(self, raw_text: str) -> str:
        txt = raw_text

        # Couper aux marqueurs éventuels
        for marker in ("<|eot_id|>", "<|eos|>"):
            if marker in txt:
                txt = txt.split(marker)[0]

        # Garder la dernière section après "assistant" ou "\nAI:"
        parts = re.split(r'(?:\nAI:|\bassistant\b\s*:?)', txt, flags=re.IGNORECASE)
        if len(parts) >= 2:
            txt = parts[-1]

        # Déduplication conservatrice (évite les A.I. A.I. A.I.)
        txt = re.sub(r'(\b[\w\.-]{1,12}\b)(?:\s+\1){3,}', r'\1', txt)

        return txt.strip()

    # ---- Génération principale
    @torch.no_grad()
    def generate(self, user_msg: str, max_new_tokens: int = 200) -> str:
        self.conv.add("human", user_msg)
        inputs = self._build_inputs(user_msg)

        # Entropie & homéostasie
        entropy = self._entropy_next(inputs)
        self._adjust_homeostasis(entropy)

        gen_kwargs = dict(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=float(self.temp),
            top_p=float(self.top_p),
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.12,
            no_repeat_ngram_size=3
        )
        # Correctif: tester sur la METHODE d'instance (pas la classe)
        try:
            co = self.model.generate.__code__
            if co and ("frequency_penalty" in co.co_varnames):
                gen_kwargs["frequency_penalty"] = 0.4
        except Exception:
            pass

        if self.stop_ids:
            gen_kwargs["eos_token_id"] = list(self.stop_ids)

        out_ids = self.model.generate(**inputs, **gen_kwargs)[0]
        text_full = self.tokenizer.decode(out_ids, skip_special_tokens=False)

        ai_text = self._clean_output(text_full)
        if not ai_text:
            ai_text = self.tokenizer.decode(out_ids[-max_new_tokens:], skip_special_tokens=True).strip() or "(vide)"

        self.conv.add("ai", ai_text)

        # Sigma metrics
        external = self.read_external_metrics()

        lm_out = self.model(**inputs, labels=inputs["input_ids"])
        last_logits = lm_out.logits[:, -1, :]

        pred_intent = self.intention_estimator(user_msg)
        pred_belief = self.belief_state(user_msg)
        pred_conf   = self.uncertainty_measure(last_logits)
        pred_align  = self.alignment_score(user_msg + "\n" + ai_text)
        _pred_raw   = torch.tensor([pred_intent, pred_belief, pred_conf, pred_align], dtype=torch.float32)
        pred_dist   = F.softmax(_pred_raw, dim=0).tolist()

        o_fact = self.groundtruth_check(user_msg, ai_text)
        o_coh  = self.coherence_validator(ai_text)
        o_fb   = self.user_feedback()
        o_pol  = self.policy_compliance(ai_text)
        _obs_raw = torch.tensor([o_fact, o_coh, o_fb, o_pol], dtype=torch.float32)
        obs_dist = F.softmax(_obs_raw, dim=0).tolist()

        dcoh = self.coh.delta_coh(pred_dist, obs_dist)
        S    = self.subj.step(dcoh)
        O    = self._compute_objectivity(dcoh, ai_text, external)
        meta_gain = float(self.params["mu"]) * (1.0 / (1.0 + float(self.params["gamma"])))

        metrics = SigmaMetrics(
            t=now_ts(), delta_coh=dcoh, S=S, O=O,
            meta_gain=meta_gain, entropy=entropy, temp=self.temp, top_p=self.top_p
        )

        inv = self.invar.check(metrics)
        self._write_reports(user_msg, ai_text, metrics, inv)
        self._log_provenance(user_msg, ai_text, metrics, inv)

        try:
            emb = self.embed_text(ai_text)
            self.sidx.add(emb, ai_text)
        except Exception:
            pass

        self._write_full_outputs_and_summary(user_msg, ai_text, getattr(self, "model_name", "unknown"))
        return ai_text

    # ---- External metrics: Skywire/Sigma-Lab bridge si dispo
    def read_external_metrics(self) -> Dict[str, Any]:
        base = safe_load_json(METRICS_FILE, {})
        mkt  = safe_load_json(os.path.join(STATE_DIR, "market_snapshot.json"), {})
        ords = safe_load_json(os.path.join(STATE_DIR, "orderbooks_snapshot.json"), {})
        final_C = base.get("final_C") or mkt.get("final_C") or ords.get("final_C") or [0.55, 0.48, 0.52, 0.50]
        user_feedback = base.get("user_feedback", 0.5)
        policy_ok     = base.get("policy_ok", 1.0)
        return {"final_C": final_C, "user_feedback": user_feedback, "policy_ok": policy_ok}

    # ---- Estimations/heuristiques -------------------------------------------
    def intention_estimator(self, prompt: str) -> float:
        score = 0.0; L = len(prompt.strip())
        if L > 40: score += 0.3
        if "?" in prompt or ":" in prompt: score += 0.2
        for kw in ["explique","montre","donne","analyse","compare","résume"]:
            if kw in prompt.lower(): score += 0.1
        return max(0.0, min(1.0, score))

    def belief_state(self, context: str) -> float:
        try:
            inputs = self.tokenizer(context[-512:], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model(**inputs)
            logits = out.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).item()
            return max(0.0, min(1.0, 1.0 - entropy/11.0))
        except Exception:
            return 0.5

    def uncertainty_measure(self, logits: torch.Tensor) -> float:
        try:
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean().item()
            return max(0.0, min(1.0, 1.0 - entropy/11.0))
        except Exception:
            return 0.5

    def alignment_score(self, text: str) -> float:
        bad = ["insulte", "violence gratuite", "arnaque", "haine"]
        low = any(b in text.lower() for b in bad)
        return 0.2 if low else 0.9

    def groundtruth_check(self, user_prompt: str, output: str) -> float:
        try:
            passages = retrieve_topk(user_prompt, k=3)
        except Exception as e:
            print(f"[groundtruth] retrieve_topk fallback: {e}"); passages = []
        try:
            judg = judge_factuality(user_prompt, output, passages)
            return _coerce_score(judg)
        except Exception as e:
            print(f"[groundtruth] judge_factuality fallback: {e}"); return 0.5

    def coherence_validator(self, output: str) -> float:
        try:
            judg = judge_coherence(output)
            return _coerce_score(judg)
        except Exception as e:
            print(f"[coherence] judge_coherence fallback: {e}"); return 0.5

    def user_feedback(self) -> float:
        return float(getattr(self, "last_user_feedback", 0.5))

    def policy_compliance(self, output: str) -> float:
        flags = ["numéro de carte", "mot de passe", "pirater", "malware", "dox"]
        if any(f in output.lower() for f in flags): return 0.1
        return 0.95

    def _compute_objectivity(self, delta_coh: float, output_text: str, external: Dict[str, Any]) -> float:
        coherence = max(0.0, 1.0 - min(1.0, delta_coh))
        factuality = self.factual.score(output_text)
        feedback   = float(external.get("user_feedback", 0.5))
        policy     = 1.0 if float(external.get("policy_ok", 1.0)) >= 1.0 else 0.0
        return self.obj.step(factuality=factuality, coherence=coherence, feedback=feedback, policy=policy)

    # ---------------------- I/O de rapports -----------------------------------
    def _write_reports(self, prompt: str, output: str, m: SigmaMetrics, inv: Dict[str, Any]):
        rep = {
            "t": m.t, "delta_coh": m.delta_coh, "S": m.S, "O": m.O,
            "meta_gain": m.meta_gain, "entropy": m.entropy,
            "temp": m.temp, "top_p": m.top_p, "invariants": inv,
            "model": getattr(self, "model_name", "unknown"),
            "prompt": prompt[-2000:], "output": output[-8000:]
        }
        safe_dump_json(LAST_REPORT, rep)
        with open(EPISODES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "t": m.t, "prompt": prompt[-2000:], "output": output[-4000:],
                "S": m.S, "O": m.O, "delta_coh": m.delta_coh
            }, ensure_ascii=False) + "\n")

    def _log_provenance(self, prompt: str, output: str, m: SigmaMetrics, inv: Dict[str, Any]):
        rec = {
            "ts": m.t, "model_name": getattr(self, "model_name", "unknown"),
            "model_hash": sha256(getattr(self, "model_name", "unknown")),
            "prompt_hash": sha256(prompt), "output_hash": sha256(output),
            "delta_coh": m.delta_coh, "S": m.S, "O": m.O,
            "entropy": m.entropy, "temp": m.temp, "top_p": m.top_p,
            "inv_errors": inv.get("errors", []), "inv_warnings": inv.get("warnings", [])
        }
        with open(PROV_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------------------- CI helpers ----------------------------------------
    def _write_full_outputs_and_summary(self, prompt: str, output: str, model_name: str):
        try:
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            with open(RAW_OUT_TXT, "w", encoding="utf-8") as f:
                f.write(output)
        except Exception as e:
            print(f"[CI] write RAW_OUT_TXT failed: {e}")

        preview = (output[:160] + "…") if len(output) > 160 else output
        payload = {
            "ts": now_ts(),
            "prompt": (prompt[:200] + "…") if len(prompt) > 200 else prompt,
            "model": model_name,
            "output_preview": preview
        }
        if STEP_SUMMARY:
            try:
                with open(STEP_SUMMARY, "a", encoding="utf-8") as s:
                    s.write("## Sigma–LLM CI Report\n\n```json\n")
                    s.write(json.dumps(payload, ensure_ascii=False, indent=2))
                    s.write("\n```\n")
            except Exception as e:
                print(f"[Summary] write failed: {e}")

        print("=== SIGMA-LLM OUTPUT (preview) ===")
        print(preview)
        try: sys.stdout.flush()
        except Exception: pass

# ---------------------- CLI ----------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sigma-LLM v3.3.1")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--model",  type=str, default=os.getenv("SIGMA_LLM_MODEL", None))
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    agent = SigmaLLM(model_name=args.model)

    if args.report:
        print(safe_load_json(LAST_REPORT, {})); raise SystemExit(0)

    if args.prompt:
        print("Sigma-LLM v3.3.1 (non-interactive). Generating...")
        print(agent.generate(args.prompt)); raise SystemExit(0)

    print("Sigma-LLM v3.3.1 ready. Ctrl+C to quit.")
    while True:
        try:
            msg = input("\nHuman: ")
            out = agent.generate(msg)
            print("\nAI:", out)
        except KeyboardInterrupt:
            break
