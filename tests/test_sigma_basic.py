# -*- coding: utf-8 -*-
import os
import pytest
pytestmark = pytest.mark.llm
import json
from pathlib import Path

# Pour un test rapide, on force un tout petit modèle.

def test_generate_and_invariants(llm_agent):
    sl, agent = llm_agent

    # seed état propre
    with open(sl.METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump({"final_C":[0.55,0.48,0.52,0.50],
                   "user_feedback":0.6, "policy_ok":1.0}, f)

    out = agent.generate("Say hello briefly in English.", max_new_tokens=32)
    assert isinstance(out, str) and len(out) > 0

    rep = sl.safe_load_json(sl.LAST_REPORT, {})
    assert "delta_coh" in rep and "S" in rep and "O" in rep
    assert 0.0 <= rep["O"] <= 1.0
    assert "invariants" in rep and isinstance(rep["invariants"], dict)
