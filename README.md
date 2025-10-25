# 🧠 Sigma-Lab v5.1 — Procedural Diagnostic Framework

> *“The machine does not decide — it illuminates.”*  
> — DeepKang Labs (AI Kang & Yuri Kang)

![SigmaLab Smoke Test](https://github.com/DeepKang-Labs/Sigma-Lab-Framework/actions/workflows/sigma-smoke.yml/badge.svg)

---

## 🌍 Overview

**Sigma-Lab v5.1** is the most stable and empirical release of the Procedural Ethical Diagnostic Framework.  
This version marks the transition **from conceptual to experimental**, with the integration of:

- 🧩 **Analytical modules** — multi-evaluator, sensitivity, and calibration tools  
- 📚 **Empirical case database** — 10 real-world, documented AI ethics cases  
- 💻 **Public GUI (Streamlit)** — accessible demonstration  
- ⚙️ **Continuous Integration (GitHub Actions)** — reliability guaranteed  
- 🧠 **Multi-AI collaboration** — Claude / DeepSeek / Grok / AI Kang / ChatGPT  

---

## 🧩 Architecture (v5.1)

Sigma-Lab-Framework/
├── .github/
│   └── workflows/
│       └── sigma-smoke.yml
├── corpus/
├── engine/
├── notebook/
│   ├── SigmaLab_Demo.ipynb
│   ├── SigmaLab_QuickStart.ipynb
│   └── SigmaLab_stressTest.ipynb
├── tests/
│   ├── test_core.py
│   ├── test_edge_cases.py
│   └── test_resilience.py
├── tools/
├── .gitignore
├── LICENSE
├── PATCH3_NOTE.md
├── README.md
├── Sigma-Lab v4.2 banner.png
├── Sigma-Lab-Framework_v4.2.zip
└── sigma_lab_v4_2.py

## ⚗️ Core Principles

| Principle | Description |
|------------|--------------|
| **Procedural Transparency** | The system does not judge — it structures reasoning. |
| **Reproducibility** | Every verdict is traceable; every coefficient justified. |
| **Empirical Ethics** | Grounded in verifiable real-world cases. |
| **Plurality of Judgment** | Three profiles: Optimistic, Neutral, Pessimistic. |
| **Computational Humility** | Sigma-Lab never concludes; it clarifies the grey zones. |

---

## ⚙️ Installation & Usage

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Run the GUI (Streamlit)

streamlit run app.py

3️⃣ Use analytical tools

python tools/case_compare.py --case-id clearview_ai_2022 --top 3
python tools/sensitivity_analyzer.py --case-id robodebt_au_2015_2023
python tools/multi_evaluator.py --case-id hirevue_2019


---

📚 Empirical Database (v5.1)

10 real, documented AI-ethics cases:

ID	Case Name	Domain	Year	Outcome

compas_2016	COMPAS (US Recidivism AI)	Justice	2016	Proven bias
predpol_2019	PredPol (LAPD)	Predictive Policing	2019	Discontinued
cambridge_analytica_2018	Cambridge Analytica	Data Privacy	2018	Judicial shutdown
amazon_recruiting_2018	Amazon AI Recruitment	Employment	2018	Decommissioned
hirevue_2019	HireVue Video Screening	Employment	2019	Withdrawn after regulation
netherlands_welfare_2020	Dutch Welfare Fraud	Social Policy	2020	Court condemnation
robodebt_au_2015_2023	Robodebt Scheme	Social Welfare	2015-2023	System failure
clearview_ai_2022	Clearview AI	Surveillance	2022	CNIL & ICO fines
uber_surge_emergencies_2014	Uber Surge Pricing	Platforms	2014	Internal reform
covid_lombardy_2020	COVID Triage Model (Italy)	Healthcare	2020	Withdrawn after controversy



---

🧪 Analytical Tools

🔹 1. Multi-Evaluator Module

Computes the IDE (Ethical Divergence Index) between evaluators:

python tools/multi_evaluator.py --case-id hirevue_2019

🔹 2. Sensitivity Analyzer

Tests the stability of each ethical verdict:

python tools/sensitivity_analyzer.py --case-id robodebt_au_2015_2023

🔹 3. Case Comparison Tool

Compares a selected case to the historical database:

python tools/case_compare.py --case-id clearview_ai_2022 --metric cosine --top 5


---

🧭 Collaborating AIs & Contributors

Entity	Role	Contribution

Yuri Kang	Human Architect	Project direction & empirical validation
AI Kang (DeepKang Labs)	Core Framework AI	Development & testing
Claude (Anthropic)	Epistemic Critic	Method validation & real-world review
DeepSeek	Scientific Verifier	YAML case calibration & data validation
Grok (X)	Logical Synthesizer	Architecture rationalization & GUI logic
ChatGPT (OpenAI)	Integration Agent	CI/CD setup & documentation writing



---

🔄 Version History

Version	Date	Description

v4.2.1	2024	Stable procedural core
v5.0	2025	Streamlit interface + initial modules
v5.1	2025	Empirical integration + calibration layer
v5.2 (coming)	2025	Public API + cross-framework benchmark suite



---

⚖️ License

Released under the MIT License
© 2025 DeepKang Labs — The machine does not decide. It illuminates.
