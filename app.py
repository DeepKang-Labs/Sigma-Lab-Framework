# -*- coding: utf-8 -*-
"""
app.py — Interface Gradio pour Sigma-LLM
Auteur : DeepKang Labs (Yuri Kang)
"""

from sigma_llm_complete import SigmaLLM
import gradio as gr

# Initialisation du modèle (tu peux changer gpt2 → llama2, mistral, etc.)
agent = SigmaLLM(model_name="gpt2")

# Fonction de dialogue
def chat_fn(message, history):
    prompt = f"Human: {message}\nAI:"
    response = agent.generate(prompt)
    return response.split("AI:")[-1].strip()

# Interface Gradio
gr.ChatInterface(
    fn=chat_fn,
    title="Sigma-LLM Reflexive Agent",
    description="Subjectivité S(t) • Objectivité O(t) • Méta-cohérence Δcoh — Powered by Sigma Dynamics"
).launch()
