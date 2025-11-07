# app.py — Gradio UI pour Sigma-LLM (Llama-3 ready, Codespaces friendly)

import os, sys, importlib.util, traceback, json, shutil, pathlib
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
            raise
        spec = importlib.util.spec_from_file_location("sigma_llm_complete", candidate)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sigma_llm_complete"] = mod
        spec.loader.exec_module(mod)
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
    # Si tout a échoué, on propage l’erreur la plus récente
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
    """Efface la mémoire conversationnelle & fichiers rapides."""
    # Efface les fichiers de conversation / épisodes (mais garde la config)
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
    # Vide le dernier output pour ne pas confondre
    try:
        latest = os.path.join(OUT, "latest_output.txt")
        if os.path.exists(latest):
            os.remove(latest)
    except Exception:
        pass
    # Réinstancie l’agent pour repartir propre
    make_agent(_active_model or PREFERRED)

# ───────────────────────────────────────────────────────────────
# Fonction de chat (Branchée sur SigmaLLM.generate)
# ───────────────────────────────────────────────────────────────
def chat_fn(message, history, temperature, top_p):
    """Gradio ChatInterface: reçoit message/historique + sliders."""
    agent = get_agent()

    # pilote direct des curseurs de l’agent
    try:
        agent.temp = float(temperature)
        agent.top_p = float(top_p)
    except Exception:
        pass

    try:
        reply = agent.generate(message, max_new_tokens=200)
        # Sanity: si un transcript complet arrive, on ne renvoie que la dernière partie
        if isinstance(reply, str) and "AI:" in reply:
            reply = reply.split("AI:")[-1].strip()
        return reply or "(réponse vide)"
    except Exception as e:
        tb = traceback.format_exc(limit=3)
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

    status = gr.Markdown(value=info_text())

    def _reload(m):
        msg = on_change_model(m)
        return msg, info_text()

    btn_reload.click(_reload, inputs=model_dd, outputs=[gr.Markdown(), status])
    btn_reset.click(lambda: (on_reset_memory(), info_text()), outputs=[gr.Markdown(), status])

    gr.Markdown("### 💬 Chat")
    chat = gr.ChatInterface(
        fn=lambda msg, hist: chat_fn(msg, hist, temp.value, topp.value),
        chatbot=gr.Chatbot(height=460, avatar_images=(None, None)),
        textbox=gr.Textbox(placeholder="Tape ton message…", autofocus=True, submit_on_enter=True),
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
    # Codespaces: Gradio sait ouvrir l’URL publique automatiquement
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
