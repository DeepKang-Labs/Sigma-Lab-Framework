# Sigma local chat UI: Transformers or a loopback llama.cpp server.

import os, sys, importlib.util, traceback, json, pathlib
import gradio as gr

# ───────────────────────────────────────────────────────────────
# Import robuste de SigmaLLM (à la racine ou par chemin explicite)
# ───────────────────────────────────────────────────────────────
def import_sigma_llm():
    if os.getenv('SIGMA_LLM_BACKEND', 'transformers') == 'llama.cpp':
        from sigma.local_chat import LocalChatLLM
        return LocalChatLLM
    try:
        from sigma_llm_complete import SigmaLLM
        return SigmaLLM
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, "sigma_llm_complete.py")
        if not os.path.exists(candidate):
            raise ImportError("sigma_llm_complete.py introuvable à la racine du repo")

        spec = importlib.util.spec_from_file_location("sigma_llm_complete", candidate)
        if spec is None or spec.loader is None:
            raise ImportError("Impossible de créer un ModuleSpec pour sigma_llm_complete.py")

        mod = importlib.util.module_from_spec(spec)
        sys.modules["sigma_llm_complete"] = mod
        spec.loader.exec_module(mod)
        if not hasattr(mod, "SigmaLLM"):
            raise ImportError("SigmaLLM non trouvé dans sigma_llm_complete.py")
        return mod.SigmaLLM

SigmaLLM = None  # Import the optional model stack only when loading an agent.

# ───────────────────────────────────────────────────────────────
# Préférence modèle + chaîne de fallbacks
# ───────────────────────────────────────────────────────────────
PREFERRED = os.getenv("SIGMA_LLM_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")

# Dossiers utiles (pour affichage & reset mémoire)
CFG = os.getenv("SIGMA_CONFIGS_DIR", "configs")
ST  = os.getenv("SIGMA_STATE_DIR",   "state")
RP  = os.getenv("SIGMA_REPORTS_DIR", "reports")
OUT = os.getenv("SIGMA_OUTPUTS_DIR", "outputs")
for d in (CFG, ST, RP, OUT):
    os.makedirs(d, exist_ok=True)

LATEST_OUT = os.path.join(OUT, "latest_output.txt")

# ───────────────────────────────────────────────────────────────
# Fabrique/Reload d’agent avec fallback
# ───────────────────────────────────────────────────────────────
_agent = None
_active_model = None

def make_agent(model_name: str):
    """Connect to or load the explicitly selected model."""
    global _agent, _active_model, SigmaLLM
    if SigmaLLM is None:
        try:
            SigmaLLM = import_sigma_llm()
        except ImportError as error:
            raise RuntimeError('The interface is available, but the optional LLM stack is missing. Install requirements-llm.txt and configure model access.') from error
    tried = [model_name]
    last_err = None
    for m in tried:
        try:
            print(f"[SigmaLLM] Loading model: {m}", flush=True)
            _agent = SigmaLLM(model_name=m)
            _active_model = _agent.model_name
            return _agent
        except Exception as e:
            last_err = e
            print(f"[SigmaLLM] Failed loading {m}: {e}", flush=True)
    raise RuntimeError(f"Impossible de charger un modèle. Dernière erreur: {last_err}")

def get_agent():
    global _agent
    if _agent is None:
        _agent = make_agent(PREFERRED)
    return _agent

# ───────────────────────────────────────────────────────────────
# Utilitaires mémoire
# ───────────────────────────────────────────────────────────────
def reset_memory():
    """Efface la mémoire conversationnelle & fichiers rapides, puis recrée l'agent."""
    for p in [
        os.path.join(ST, "conversation.json"),
        os.path.join(ST, "episodes.jsonl"),
        os.path.join(ST, "semantic_index.jsonl"),
    ]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    try:
        if os.path.exists(LATEST_OUT):
            os.remove(LATEST_OUT)
    except Exception:
        pass
    make_agent(_active_model or PREFERRED)

# ───────────────────────────────────────────────────────────────
# Fonction de chat (branchée sur SigmaLLM.generate)
# ───────────────────────────────────────────────────────────────
def chat_fn(message, history, temperature, top_p):
    """
    Gradio ChatInterface:
      - message: str
      - history: list[(user, assistant)]
      - temperature/top_p: sliders (additional_inputs)
    """
    try:
        agent = get_agent()
    except Exception as error:
        return 'Model unavailable: '+str(error)

    try:
        reply = agent.generate(message, max_new_tokens=400, history=history,
                               temperature=temperature, top_p=top_p)
        # garde: sauvegarder aussi pour debug rapide
        try:
            with open(LATEST_OUT, "w", encoding="utf-8") as f:
                f.write(reply or "")
        except Exception:
            pass
        return reply or "(réponse vide)"
    except Exception as e:
        tb = traceback.format_exc(limit=4)
        print(f"[chat_fn] ERROR: {e}\n{tb}", flush=True)
        return f"⚠️ Erreur: {e}\n```\n{tb}\n```"

# ───────────────────────────────────────────────────────────────
# Callbacks UI: changer le modèle, reset mémoire, infos système
# ───────────────────────────────────────────────────────────────
AVAILABLE = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]
if os.getenv('SIGMA_LLM_BACKEND') == 'llama.cpp':
    AVAILABLE = [PREFERRED]

def on_change_model(new_model):
    try:
        make_agent(new_model)
        return f"Model loaded: {_active_model}"
    except Exception as e:
        return f"❌ Échec chargement {new_model}: {e}"

def on_reset_memory():
    try:
        reset_memory()
        return "🧹 Mémoire réinitialisée."
    except Exception as e:
        return f"❌ Reset échoué: {e}"

def info_text():
    model = _active_model or PREFERRED
    return f'**Modèle sélectionné :** {model}  \nLes rapports sont enregistrés localement.'

# ───────────────────────────────────────────────────────────────
# Gradio UI
# ───────────────────────────────────────────────────────────────
with gr.Blocks(title="Sigma-LLM Reflexive Agent", analytics_enabled=False) as demo:
    gr.Markdown("## Sigma-LLM — Assistant local")

    with gr.Row():
        model_dd = gr.Dropdown(
            choices=AVAILABLE,
            value=PREFERRED if PREFERRED in AVAILABLE else AVAILABLE[0],
            label="Modèle",
            interactive=True,
        )
        temp = gr.Slider(minimum=0.10, maximum=1.30, value=0.6, step=0.01, label="temperature")
        topp = gr.Slider(minimum=0.10, maximum=1.00, value=0.9, step=0.01, label="top_p")
        btn_reload = gr.Button("🔄 Recharger modèle")
        btn_reset  = gr.Button("Nouvelle conversation")

    # zone d’état
    status_msg = gr.Markdown()
    status = gr.Markdown(value=info_text())

    def _reload(m):
        msg = on_change_model(m)
        return msg, info_text()

    btn_reload.click(_reload, inputs=model_dd, outputs=[status_msg, status])

    gr.Markdown("### 💬 Chat")
    chat = gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        additional_inputs=[temp, topp],  # ✅ sliders réellement reliés à chat_fn
        chatbot=gr.Chatbot(type="messages", height=460, avatar_images=(None, None), allow_tags=False),
        textbox=gr.Textbox(placeholder="Tape ton message…", autofocus=True),
        title="Sigma-LLM",
        description=("Conversation avec le modèle local. Demandez une réponse dans la langue de votre choix."
                     if os.getenv('SIGMA_LLM_BACKEND') == 'llama.cpp' else
                     "The selected model loads on first use. Sigma metrics are experimental proxies; they do not verify factuality or safety."),
        cache_examples=False,
    )
    btn_reset.click(lambda: ('New conversation started.', info_text(), [], [], None),
                    outputs=[status_msg, status, chat.chatbot, chat.chatbot_state, chat.saved_input])

# ───────────────────────────────────────────────────────────────
# Lancement serveur (Codespaces/localhost)
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv('SIGMA_BIND_HOST','127.0.0.1')
    print(f"[SigmaLLM] Interface starting on {host}:{port}; model loading is deferred.", flush=True)
    demo.launch(server_name=host, server_port=port, show_error=True)
