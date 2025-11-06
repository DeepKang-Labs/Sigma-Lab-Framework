# app.py — Gradio + import robuste de SigmaLLM (fix no-response)

import os, sys, importlib.util, traceback

# --- Import robuste de SigmaLLM ---
def import_sigma_llm():
    try:
        # 1) si le fichier est à la racine du repo
        from sigma_llm_complete import SigmaLLM
        return SigmaLLM
    except Exception:
        # 2) fallback: chargement explicite par chemin
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, "sigma_llm_complete.py")
        if not os.path.exists(candidate):
            raise
        spec = importlib.util.spec_from_file_location("sigma_llm_complete", candidate)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sigma_llm_complete"] = mod
        spec.loader.exec_module(mod)
        return mod.SigmaLLM

SigmaLLM = import_sigma_llm()

# --- Choix du modèle (env ou défauts sûrs CPU) ---
MODEL_NAME = os.getenv("SIGMA_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
FALLBACK   = "meta-llama/Meta-Llama-3-8B-Instruct"

# --- Instanciation agent avec fallback + logs ---
def make_agent():
    try:
        print(f"[SigmaLLM] Loading model: {MODEL_NAME}", flush=True)
        return SigmaLLM(model_name=MODEL_NAME)
    except Exception as e:
        print(f"[SigmaLLM] Primary model failed: {e}\n→ Fallback: {FALLBACK}", flush=True)
        return SigmaLLM(model_name=FALLBACK)

agent = make_agent()

# --- Gradio UI ---------------------------------------------------------------
import gradio as gr

def chat_fn(message, history):
    """
    Gradio ChatInterface signature: (message: str, history: list[tuple[str,str]])
    On renvoie une string; en cas d'erreur on affiche le traceback dans le chat.
    """
    try:
        # IMPORTANT: generate() attend le message utilisateur 'brut'
        reply = agent.generate(message, max_new_tokens=160)
        # certaines implémentations renvoient tout le transcript; on nettoie si besoin
        if isinstance(reply, str) and "AI:" in reply:
            reply = reply.split("AI:")[-1].strip()
        return reply if reply else "(réponse vide)"
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        print(f"[chat_fn] ERROR: {e}\n{tb}", flush=True)
        return f"⚠️ Erreur: {e}\n```\n{tb}\n```"

with gr.Blocks(title="Sigma-LLM Reflexive Agent") as demo:
    gr.Markdown("### Sigma-LLM — S(t) / O(t) / Δcoh — Gradio UI")
    with gr.Row():
        gr.Markdown(
            f"**Model:** `{MODEL_NAME}`  \n"
            f"**CWD:** `{os.getcwd()}`  \n"
            f"**sigma_llm_complete.py présent ?** `{os.path.exists(os.path.join(os.getcwd(),'sigma_llm_complete.py'))}`"
        )
    chat = gr.ChatInterface(
        fn=chat_fn,
        chatbot=gr.Chatbot(height=420),
        textbox=gr.Textbox(placeholder="Tape ton message…", autofocus=True),
        title="Sigma-LLM",
        description="Agent réflexif CPU-friendly (meta-llama/Meta-Llama-3-8B-Instruct)."
    )

if __name__ == "__main__":
    # pour Codespaces/localhost
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
