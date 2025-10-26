# 🧠 Sigma-Lab v5.1 — Procedural Diagnostic Framework  
**“The machine does not decide — it illuminates.”**  
— *DeepKang Labs (AI Kang & Yuri Kang)*  

[![smoke](https://github.com/DeepKang-Labs/Sigma-Lab-Framework/actions/workflows/smoke.yml/badge.svg?branch=main)](https://github.com/DeepKang-Labs/Sigma-Lab-Framework/actions/workflows/smoke.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-yellow.svg)](https://www.python.org/)
[![status](https://img.shields.io/badge/status-In Testing (α)-orange.svg)](#)

---

## 🌍 Overview

**Sigma-Lab v5.1** is the most stable and empirical release of the **Procedural Ethical Diagnostic Framework** —  
a system that exposes, traces and tests human governance logic rather than automating decisions.

This version marks the transition **from conceptual → experimental**, with a focus on:

- 🧩 **Analytical modules** — multi-evaluator, sensitivity, and calibration tools.  
- ⚙️ **Bridges** — network adapters for **Skywire** (α-phase) and **Fiber (NESS)** (pre-activation).  
- 📈 **Validation pipelines** — full GitHub Actions CI with smoke & resilience tests.  
- 🧮 **Traceability** — SHA256 fingerprints and auditable YAML-based mappings.

---

## ✨ Philosophy — *AI + Human = 3*

> Sigma-Lab is not a moral oracle; it’s a procedural mirror.

Every “diagnosis” emerges from **human-defined thresholds** and **explicit YAML parameters**,  
allowing teams to *argue*, *revise*, and *own* their ethical logic rather than delegate it.

```yaml
weights:
  non_harm: 0.45
  stability: 0.20
  resilience: 0.20
  equity: 0.15
thresholds:
  non_harm_floor: 0.30
  veto_irreversibility: 0.70

Run modes:

linear → no dependencies, fully deterministic

simple → lightweight expression evaluation

auto → chooses the most efficient path



---

🧠 Core Components

Module	Description	Status

sigma_lab_v4_2.py	Main analytical engine (diagnostic, audit trail, resilience)	✅ Stable
network_bridge/	Unified bridge for Skywire & Fiber (NESS)	🚧 α testing
skywire_bridge/	Legacy adapter (kept for comparison)	💤 Deprecated
tests/	10+ CI-validated test cases (core, resilience, edge cases)	✅ Passing



---

🚀 Quick Start

1️⃣ Install dependencies

pip install -r requirements.txt

2️⃣ Run validation (no side effects)

python network_bridge/run_network_integrated.py \
  --network skywire \
  --discovery ./discovery \
  --mappings ./network_bridge/mappings_skywire.yaml \
  --config ./sigma_config_placeholder.yaml \
  --validate-only --formula-eval auto --pretty

Switch to Fiber (NESS):

python network_bridge/run_network_integrated.py \
  --network fiber \
  --discovery ./discovery \
  --mappings ./network_bridge/mappings_fiber.yaml \
  --config ./sigma_config_placeholder.yaml \
  --validate-only --formula-eval auto --pretty


---

🧾 Output Example (simplified)

{
  "status": "success",
  "scores": {
    "non_harm": 0.88,
    "stability": 0.71,
    "resilience": 0.63,
    "equity": 0.89
  },
  "audit": {
    "schema_version": "1.0",
    "timestamp_utc": "2025-10-26T02:43:21Z",
    "sha256": "fe3ac...8b2f"
  }
}


---

🧪 Continuous Integration

✅ Smoke Tests

Located in .github/workflows/smoke.yml.
Runs pytest -q on all modules including network bridges and resilience simulations.

🧩 Resilience Suite

Stress-tests (test_resilience.py) ensure reproducibility, graceful failure, and determinism under load.



---

🧩 Governance

Committee: DeepKang Labs Network Governance Committee (DL-NGC)
License: MIT
Maintainers: @DeepKang-Labs
Contact: x.com/Mc_MomaL Yuri Kang


---

🌌 Roadmap

Phase	Target	Status

α 1	Skywire Bridge (live case validation)	✅ In testing
α 2	Fiber (NESS) Bridge (compatibility check)	🚧 Preparation
β 1	Multi-context calibrator	🧩 Design
β 2	Human-in-the-loop Dashboard	🧭 Concept
1.0	Sigma-Lab full decentralized deployment	🌍 Vision



---

> 🕊️ “Between algorithmic logic and human judgment, Sigma-Lab builds a bridge.”
