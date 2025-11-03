# -*- coding: utf-8 -*-
import os
os.environ.setdefault("SIGMA_LLM_MODEL", "sshleifer/tiny-gpt2")

def test_homeostasis_bounds_and_adjust():
    import sigma_llm_complete as sl
    agent = sl.SigmaLLM()

    # valeurs de départ
    t0, p0 = agent.temp, agent.top_p

    # Entropie très basse -> le contrôleur devrait pousser vers la cible
    agent._adjust_homeostasis(entropy=0.05)
    assert agent.params["homeostasis"]["temp"]["min"] <= agent.temp <= agent.params["homeostasis"]["temp"]["max"]
    assert agent.params["homeostasis"]["top_p"]["min"] <= agent.top_p <= agent.params["homeostasis"]["top_p"]["max"]

    # Entropie très haute -> autre direction mais toujours bornée
    agent._adjust_homeostasis(entropy=8.0)
    assert agent.params["homeostasis"]["temp"]["min"] <= agent.temp <= agent.params["homeostasis"]["temp"]["max"]
    assert agent.params["homeostasis"]["top_p"]["min"] <= agent.top_p <= agent.params["homeostasis"]["top_p"]["max"]
