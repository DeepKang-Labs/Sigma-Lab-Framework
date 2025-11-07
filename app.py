# app.py — Gradio UI pour Sigma-LLM (Llama-3 ready, Codespaces friendly)

import os, sys, importlib.util, traceback, json, pathlib
import gradio as gr

# ───────────────────────────────────────────────────────────────
# Import robuste de SigmaLLM (à la racine ou par chemin explicite)
# ───────────────────────────────────────────────────────────────
def import_sigma_llm():
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

SigmaLLM = import_sigma_llm()

# ───────────────────────────────────────────────────────────────
# Préférence modèle + chaîne de fallbacks
# ───────────────────────────────────────────────────────────────
PREFERRED = os.getenv("SIGMA_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
FALLBACKS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3-mini-4k-instruct",
    "gpt2",
]

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
    """Instancie SigmaLLM avec fallback en cascade si besoin."""
    global _agent, _active_model
    tried = [model_name] + [m for m in FALLBACKS if m != model_name]
    last_err = None
    for m in tried:
        try:
            print(f"[SigmaLLM] Loading model: {m}", flush=True)
            _agent = SigmaLLM(model_name=m)
            _active_model = m
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
    agent = get_agent()

    # pilote direct des curseurs de l’agent
    try:
        agent.temp = float(temperature)
        agent.top_p = float(top_p)
    except Exception:
        pass

    try:
        reply = agent.generate(message, max_new_tokens=200)
        if isinstance(reply, str) and "AI:" in reply:
            reply = reply.split("AI:")[-1].strip()
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
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3-mini-4k-instruct",
    "gpt2",
]

def on_change_model(new_model):
    try:
        make_agent(new_model)
        return f"✅ Modèle chargé: {new_model}"
    except Exception as e:
        return f"❌ Échec chargement {new_model}: {e}"

def on_reset_memory():
    try:
        reset_memory()
        return "🧹 Mémoire réinitialisée."
    except Exception as e:
        return f"❌ Reset échoué: {e}"

def info_text():
    here = os.getcwd()
    info = {
        "active_model": _active_model or PREFERRED,
        "cwd": here,
        "outputs": str(pathlib.Path(OUT).resolve()),
        "reports": str(pathlib.Path(RP).resolve()),
        "state": str(pathlib.Path(ST).resolve()),
    }
    return "```\n" + json.dumps(info, indent=2, ensure_ascii=False) + "\n```"

# ───────────────────────────────────────────────────────────────
# Gradio UI
# ───────────────────────────────────────────────────────────────
with gr.Blocks(title="Sigma-LLM Reflexive Agent") as demo:
    gr.Markdown("## 🧠 Sigma-LLM — S(t) / O(t) / Δcoh — Interface interactive")

    with gr.Row():
        model_dd = gr.Dropdown(
            choices=AVAILABLE,
            value=PREFERRED if PREFERRED in AVAILABLE else AVAILABLE[0],
            label="Modèle",
            interactive=True,
        )
        temp = gr.Slider(minimum=0.60, maximum=1.30, value=0.95, step=0.01, label="temperature")
        topp = gr.Slider(minimum=0.70, maximum=0.99, value=0.95, step=0.01, label="top_p")
        btn_reload = gr.Button("🔄 Recharger modèle")
        btn_reset  = gr.Button("🧹 Reset mémoire")

    # zone d’état
    status_msg = gr.Markdown()
    status = gr.Markdown(value=info_text())

    def _reload(m):
        msg = on_change_model(m)
        return msg, info_text()

    btn_reload.click(_reload, inputs=model_dd, outputs=[status_msg, status])
    btn_reset.click(lambda: (on_reset_memory(), info_text()), outputs=[status_msg, status])

    gr.Markdown("### 💬 Chat")
    chat = gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[temp, topp],  # ✅ sliders réellement reliés à chat_fn
        chatbot=gr.Chatbot(height=460, avatar_images=(None, None)),
        textbox=gr.Textbox(placeholder="Tape ton message…", autofocus=True),
        title="Sigma-LLM",
        description="Agent réflexif (Llama-3 ready). Les sorties sont archivées dans outputs/ et reports/.",
        theme="soft",
        cache_examples=False,
    )

# ───────────────────────────────────────────────────────────────
# Lancement serveur (Codespaces/localhost)
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Pré-initialise pour feedback immédiat dans l’UI
    try:
        make_agent(PREFERRED)
    except Exception as e:
        print(f"[boot] warning: preferred model not available ({e}) — UI démarre avec fallback à la première requête.")

    port = int(os.getenv("PORT", "7860"))
    print(f"[SigmaLLM] Gradio démarre sur 0.0.0.0:{port}", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
