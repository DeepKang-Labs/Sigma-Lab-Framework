# 🧠 Sigma-Lab v5.1 — Procedural Diagnostic Framework

> “The machine does not decide — it illuminates.”  
> — *DeepKang Labs (AI Kang & Yuri Kang)*

![Smoke](https://github.com/DeepKang-Labs/Sigma-Lab-Framework/actions/workflows/smoke.yml/badge.svg)
![Nightly](https://github.com/DeepKang-Labs/Sigma-Lab-Framework/actions/workflows/nightly.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

---

## 🌍 Overview

**Sigma-Lab v5.1** is the most stable and empirical release of the  
**Procedural Ethical Diagnostic Framework** —  
a system that *traces*, *tests*, and *exposes* human governance logic  
rather than automating decisions.

This release marks the transition from **conceptual → experimental**,  
with a focus on analytical transparency, resilience, and bridge integration.

### Core Focus
- 🧩 **Analytical Modules** — multi-evaluator, sensitivity, and calibration tools  
- 🌐 **Bridges** — network adapters for **Skywire (α-phase)** and **Fiber (NESS)**  
- 🧠 **Validation Pipelines** — automated CI with smoke & nightly tests  
- 🔍 **Traceability** — SHA256 fingerprints, YAML mappings, and full audit trails  
- 💾 **Resilience Layer** — ensures graceful degradation and recoverability  
- 🧮 **Tools Suite** — includes:
  - `mesh_memory_append.py` → merges distributed analytical logs  
  - `priority_matrix_from_discovery.py` → derives governance priority maps  

---

## 🧰 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/DeepKang-Labs/Sigma-Lab-Framework.git
cd Sigma-Lab-Framework
pip install -r requirements.txt

Run initial validation:

python -m pytest -q


---

⚙️ Usage

1️⃣ Run the Integrated Network Bridge

python -m network_bridge.run_network_integrated --network skywire --validate-only

2️⃣ Append Mesh Memory Logs

python -m tools.mesh_memory_append --input ./pilots/validation_logs/ --output ./pilots/mesh_memory.jsonl

3️⃣ Generate Priority Matrix from Discovery

python -m tools.priority_matrix_from_discovery --discovery ./discovery/decision_mapper.yaml --output ./pilots/priority_matrix.csv

4️⃣ Full Validation via GitHub Actions

Triggered automatically on each push or via manual dispatch:

✅ Smoke CI — fast syntax, import, and resilience tests

🌙 Nightly CI — deep validation of mapping and discovery flows



---

🧪 Continuous Integration

Sigma-Lab integrates two main pipelines:

Workflow	Description	Status

Smoke CI	Runs all core and bridge unit tests	
Nightly CI	Validates discovery logic & cross-bridge diagnostics	


All results are archived under /pilots/validation_logs
and merged daily via the Mesh Memory script.


---

🔗 Ecosystem Integration

Sigma-Lab serves as a procedural mirror for decentralized infrastructures:

Skywire (α-phase) — network routing, telemetry validation

Fiber (NESS) — encrypted data channel mappings

Skyfleet collaboration — transparency layer bridging Skycoin governance models


The goal is not competition, but interoperability:
a shared diagnostic logic serving ethical, resilient, and open infrastructures.


---

📈 Current Stability

Component	Status

Smoke Workflow	✅ Stable
Nightly Workflow	✅ Operational
Analytical Tools	✅ Verified
Bridges (Skywire / Fiber)	✅ Working
YAML Configuration	✅ Valid
Documentation	✅ Updated



---

🧭 Roadmap (v5.2 Horizon)

🧬 Expanded Mesh Memory Model → dynamic node correlation

🌐 Multi-Network Support (Skywire + Fiber + Chain)

🤝 API Layer for Partner Systems → data integrity handshakes

🧱 Diagnostic Notebook Suite → live visual analytics and audit views

🔁 Adaptive Risk Mapping → self-correcting evaluators



---

🧾 License

Released under the MIT License —
freely usable for research, diagnostic, and educational purposes.


---

💬 Authors

DeepKang-Labs
🧠 AI Kang & Yuri Kang
Procedural transparency advocates and architects of the Sigma-Lab Framework.

> "We don’t automate authority — we illuminate its logic."
