# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path

# Pour un test rapide, on force un tout petit modèle.
os.environ.setdefault("SIGMA_LLM_MODEL", "sshleifer/tiny-gpt2")

def test_generate_and_invariants():
    import sigma_llm_complete as sl

    # seed état propre
    Path("state").mkdir(exist_ok=True)
    with open("state/last_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"final_C":[0.55,0.48,0.52,0.50],
                   "user_feedback":0.6, "policy_ok":1.0}, f)

    agent = sl.SigmaLLM()
    out = agent.generate("Bonjour Sigma, fais un court salut.")
    assert isinstance(out, str) and len(out) > 0

    rep = sl.safe_load_json("reports/sigma_llm_last_report.json", {})
    assert "delta_coh" in rep and "S" in rep and "O" in rep
    assert 0.0 <= rep["O"] <= 1.0
    assert "invariants" in rep and isinstance(rep["invariants"], dict)
