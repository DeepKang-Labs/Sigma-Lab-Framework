# app.py
# Gradio + import robuste de SigmaLLM

import os, sys, importlib.util

# --- Import robuste de SigmaLLM ---
try:
    # 1) import standard si le fichier est dans le même dossier
    from sigma_llm_complete import SigmaLLM  # doit marcher si sigma_llm_complete.py est à la racine
except ModuleNotFoundError:
    # 2) fallback: chargement explicite par chemin de fichier
    HERE = os.path.dirname(os.path.abspath(__file__))
    CANDIDATE = os.path.join(HERE, "sigma_llm_complete.py")
    if os.path.exists(CANDIDATE):
        spec = importlib.util.spec_from_file_location("sigma_llm_complete", CANDIDATE)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sigma_llm_complete"] = mod
        spec.loader.exec_module(mod)
        SigmaLLM = mod.SigmaLLM
    else:
        raise  # on remonte si le fichier n'existe pas là

# --- Interface minimale Gradio pour tester ---
import gradio as gr

# Choisis le modèle local (tu peux passer SIGMA_LLM_MODEL=gpt2, meta-llama, etc.)
MODEL_NAME = os.getenv("SIGMA_LLM_MODEL", "gpt2")
agent = SigmaLLM(model_name=MODEL_NAME)

def chat_fn(message, history):
    prompt = f"Human: {message}\nAI:"
    out = agent.generate(prompt, max_new_tokens=128, temperature=0.9)
    # on prend tout après le dernier "AI:" si présent
    reply = out.split("AI:")[-1].strip()
    return reply

with gr.Blocks(title="Sigma-LLM Reflexive Agent") as demo:
    gr.Markdown("### Sigma-LLM — Subjectivité S(t) + Objectivité O(t) + Méta-cohérence Δcoh")
    chat = gr.ChatInterface(fn=chat_fn)
    gr.Markdown(
        f"**cwd:** `{os.getcwd()}`  \n"
        f"**sigma_llm_complete.py présent ?** `{os.path.exists(os.path.join(os.getcwd(),'sigma_llm_complete.py'))}`"
    )

if __name__ == "__main__":
    # Lance en local sur 7860 (Codespaces/localhost)
    demo.launch(server_name="0.0.0.0", server_port=7860)
