# -*- coding: utf-8 -*-
"""
Sigma-LLM v3.2 — Web Interface (Gradio + Prometheus + Reflexive Mode)
Compatibilité : Workflow GitHub + Sigma-LLM v3.2
"""

import os, sys, importlib.util, json, time
import gradio as gr

# ================================
# 🔹 Chargement robuste du module
# ================================
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE = os.path.join(HERE, "sigma_llm_complete.py")

try:
    from sigma_llm_complete import SigmaLLM
except ModuleNotFoundError:
    if os.path.exists(CANDIDATE):
        spec = importlib.util.spec_from_file_location("sigma_llm_complete", CANDIDATE)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sigma_llm_complete"] = mod
        spec.loader.exec_module(mod)
        SigmaLLM = mod.SigmaLLM
    else:
        raise FileNotFoundError("❌ Impossible de charger SigmaLLM : fichier sigma_llm_complete.py introuvable.")

# ================================
# 🔹 Chargement du modèle
# ================================
MODEL_NAME = os.getenv("SIGMA_LLM_MODEL", "gpt2").strip() or "gpt2"

print(f"[App] Démarrage Sigma-LLM sur modèle : {MODEL_NAME}")
try:
    agent = SigmaLLM(model_name=MODEL_NAME)
except Exception as e:
    print(f"[App] ⚠️ Échec du chargement du modèle {MODEL_NAME} : {e}")
    print("[App] Tentative de fallback sur gpt2…")
    agent = SigmaLLM(model_name="gpt2")

# ================================
# 🔹 Fonctions utilitaires
# ================================
def chat_fn(message, history):
    """Appel unique Sigma-LLM avec enregistrement conversationnel"""
    prompt = f"Human: {message}\nAI:"
    try:
        out = agent.generate(prompt, max_new_tokens=256)
        reply = out.split("AI:")[-1].strip()
        return reply
    except Exception as e:
        return f"[Erreur interne] {e}"

def export_metrics():
    """Affiche les dernières métriques Sigma"""
    report_path = os.path.join("reports", "sigma_llm_last_report.json")
    if os.path.exists(report_path):
        try:
            data = json.load(open(report_path, "r", encoding="utf-8"))
            summary = json.dumps(data, indent=2, ensure_ascii=False)
            return summary
        except Exception as e:
            return f"⚠️ Impossible de lire le rapport : {e}"
    else:
        return "Aucun rapport Sigma-LLM trouvé."

def save_prompt(prompt):
    """Sauvegarde rapide du dernier prompt"""
    state_file = os.path.join("state", "last_prompt.txt")
    os.makedirs("state", exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as f:
        f.write(prompt)
    return f"✅ Prompt sauvegardé ({len(prompt)} caractères)."

# ================================
# 🔹 Interface Gradio enrichie
# ================================
with gr.Blocks(title="Sigma-LLM Reflexive Agent (v3.2)") as demo:
    gr.Markdown("## 🧠 Sigma-LLM Reflexive Agent — v3.2")
    gr.Markdown(
        "- **S(t)** : Subjectivité dynamique  \n"
        "- **O(t)** : Objectivité pondérée  \n"
        "- **Δcoh** : Méta-cohérence réflexive  \n"
        "- **Homeostasis** : contrôle entropique automatique  \n"
        "- **Prometheus** : export métriques (si activé via `SIGMA_PROM_PORT`)"
    )

    chat = gr.ChatInterface(fn=chat_fn, title="Dialogue réflexif Sigma-LLM")

    with gr.Accordion("📊 Outils de diagnostic Sigma-Lab", open=False):
        btn_metrics = gr.Button("Afficher dernières métriques")
        out_metrics = gr.Textbox(label="Dernier rapport Sigma", lines=15)
        btn_metrics.click(export_metrics, outputs=out_metrics)

        prompt_input = gr.Textbox(label="Dernier prompt à sauvegarder", lines=2)
        btn_save = gr.Button("Sauvegarder ce prompt")
        out_save = gr.Textbox(label="État sauvegarde")
        btn_save.click(save_prompt, inputs=prompt_input, outputs=out_save)

    gr.Markdown(
        f"**cwd :** `{os.getcwd()}`  \n"
        f"**sigma_llm_complete.py présent ?** `{os.path.exists(CANDIDATE)}`  \n"
        f"**Modèle actuel :** `{MODEL_NAME}`"
    )

# ================================
# 🔹 Lancement du serveur
# ================================
if __name__ == "__main__":
    PORT = int(os.getenv("PORT", "7860"))
    HOST = os.getenv("HOST", "0.0.0.0")
    print(f"[App] Interface Gradio disponible sur http://{HOST}:{PORT}")
    demo.launch(server_name=HOST, server_port=PORT)
