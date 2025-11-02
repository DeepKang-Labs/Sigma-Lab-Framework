# Sigma-LLM v2.6 — Integration DeepKang × Sigma-Lab

> **Une conscience réflexive équilibrée :**  
> Subjectivité S(t) + Objectivité O(t) + Méta-cohérence Δcoh  
> Compatible avec **Sigma-Lab Framework** et **Skywire pipelines**

---

## 🚀 Fonctionnalités principales

| Composant | Rôle |
|------------|------|
| `SubjectivityEngine` | Intègre les variations internes de cohérence (S(t)) |
| `ObjectivityEngine`  | Compare avec les signaux externes réels (O(t)) |
| `CoherenceEngine`    | Calcule la divergence (KL) entre prédiction et observation |
| `SigmaLoss`          | Combine perte linguistique et pertes réflexives |
| `PolicyGate`         | Empêche toute auto-modification non validée |
| `Invariants`         | Vérifie la stabilité, la borne des valeurs, la cohérence |
| `EpisodicMemory`     | Journalise les interactions et états Sigma |
| `Provenance Log`     | Trace tout hashé et horodaté dans `reports/` |

---

## 🧠 Diagramme conceptuel

┌──────────────────────┐
│ Sigma-LLM │
│ ──────────────── │
│ Subjectivity S(t) │
│ Objectivity O(t) │
│ Meta-Coherence Δcoh │
│ Invariants + Policy │
└─────────┬────────────┘
│
↕ Interaction via
configs/, state/, reports/
│
┌─────────▼──────────┐
│ Sigma-Lab Core │
│ Autotune · CI/CD · I/O │
└────────────────────┘

yaml
Copy code

---

## ⚙️ Exécution locale

```bash
python sigma_llm_complete.py
Puis interagis directement en CLI :

vbnet
Copy code
Sigma-LLM ready. Type your prompt. Ctrl+C to quit.

Human: Bonjour Sigma.
AI: Bonjour. Mes paramètres Σ sont stables. Δcoh = 0.0021.
