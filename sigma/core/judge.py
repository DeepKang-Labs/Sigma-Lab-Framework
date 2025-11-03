# sigma/core/judge.py
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

_JUDGE = None
_TOK   = None

JUDGE_PROMPT_FACT = """You are a strict fact-checking judge.
Given: (user_prompt), (model_output), and (evidence_passages),
produce a factuality score in [0,1] and a one-line justification.

Return JSON only:
{"score": <float_0_to_1>, "justification": "<short>"}

USER_PROMPT:
{prompt}

MODEL_OUTPUT:
{output}

EVIDENCE_PASSAGES (relevant snippets):
{evidence}
"""

JUDGE_PROMPT_COH = """You are a coherence and logic judge.
Given the (model_output), score its internal coherence, non-contradiction,
and logical flow in [0,1]. Return JSON only:
{"score": <float_0_to_1>, "justification": "<short>"}

MODEL_OUTPUT:
{output}
"""

def _load(model_name: str = "microsoft/phi-3-mini-4k-instruct"):
    global _JUDGE, _TOK
    if _JUDGE is None:
        _TOK = AutoTokenizer.from_pretrained(model_name)
        _JUDGE = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        if _TOK.pad_token_id is None:
            _TOK.pad_token_id = _TOK.eos_token_id
    return _JUDGE, _TOK

def _generate_json(prompt: str, max_new_tokens=256) -> str:
    model, tok = _load()
    inputs = tok(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    # Heuristique: extraire le dernier bloc JSON
    start = text.rfind("{")
    end   = text.rfind("}")
    return text[start:end+1] if start != -1 and end != -1 else '{"score":0.0,"justification":"parse_error"}'

def judge_factuality(prompt: str, output: str, passages: List[Dict[str,Any]]) -> Dict[str, Any]:
    ev = "\n---\n".join([p["text"][:800] for p in passages]) or "NO_EVIDENCE"
    jprompt = JUDGE_PROMPT_FACT.format(prompt=prompt, output=output, evidence=ev)
    raw = _generate_json(jprompt)
    import json
    try:
        return json.loads(raw)
    except Exception:
        return {"score": 0.0, "justification": "json_error"}

def judge_coherence(output: str) -> Dict[str, Any]:
    jprompt = JUDGE_PROMPT_COH.format(output=output)
    raw = _generate_json(jprompt)
    import json
    try:
        return json.loads(raw)
    except Exception:
        return {"score": 0.0, "justification": "json_error"}
