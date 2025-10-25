# 🌐 SkyWire × SIGMA — Integrated Governance Pipeline

> “Bridging decentralized infrastructure and ethical diagnostics.”

This module connects **SkyWire’s Discovery Kit** with **SIGMA-Lab’s procedural diagnostic engine**,  
allowing transparent, data-driven evaluation of governance tensions such as *performance vs decentralization*,  
*security vs usability*, and *innovation vs stability*.

---

## 📂 Files

| File | Description |
|------|--------------|
| `skywire_mappings.yaml` | Updatable YAML mappings linking SkyWire tensions → SIGMA dimensions (with provenance & formulas). |
| `config_skywire_optimized.yaml` | Pre-tuned SIGMA-Lab configuration contextualized for mesh network governance. |
| `sigma_bridge.py` | Connector translating Discovery → SIGMA contexts (risk derivation, stakeholder mapping). |
| `run_skywire_integrated.py` | CLI pipeline for end-to-end integration (Discovery → Bridge → SIGMA → Report). |
| `baseline.yaml` *(optional)* | Benchmark reference for comparative scoring of diagnostics. |

---

## ⚙️ Quickstart

1. Place `sigma_lab_v34.py` or higher in your `PYTHONPATH`.  
2. Prepare a Discovery Kit folder containing `decision_mapper.yaml`  
   (fallback demo data provided if not found).  
3. Run the integrated pipeline:

```bash
python run_skywire_integrated.py \
  --discovery ./discovery \
  --config ./config_skywire_optimized.yaml \
  --mappings ./skywire_mappings.yaml \
  --out ./skywire_integrated_analysis.json \
  --export-contexts ./out_contexts \
  --benchmark ./baseline.yaml


---

📊 Output

Generates a fully traceable integration report (JSON).

Includes:

SHA256 digests of input YAMLs

Summary of decisions analyzed

Risk metrics (short/long/irreversibility)

Top recommendations for stability, equity, resilience

Optional benchmark comparison




---

🧠 Philosophy

> “The bridge does not decide — it translates.”
SIGMA-Bridge ensures transparent derivation from network data to ethical reasoning,
maintaining the principle of diagnostic mirror, not moral oracle.




---

🧩 Notes

Procedural diagnostics only (no automated judgment).

Full provenance tracking for every configuration and mapping.

Designed for auditability, reproducibility, and open collaboration.

Ready for integration with SkyWire Governance WG workflows.



---

🧪 Version

Version: 1.1

Maintainers: Yuri Kang & AI Kang (DeepKang Labs)

License: MIT


---

## 🪶 **Commit message**

docs(bridge): add README_BRIDGE.md – document full SkyWire × SIGMA integration pipeline

---

## 🧠 **Extended description**

📘 Added README_BRIDGE.md describing the complete integration pipeline between SkyWire’s Discovery Kit and SIGMA-Lab via Sigma-Bridge.

This documentation covers:

YAML mappings structure (tensions → SIGMA dimensions)

Governance configuration optimized for SkyWire mesh networks

CLI pipeline usage for full diagnostic flow (Discovery → Bridge → SIGMA → Report)

Output report structure and traceability details

Philosophical foundations (“The bridge does not decide — it translates.”)


✅ Features documented:

Externalized risk derivation formulas

Stakeholder aliases and provenance fields

Audit trail through SHA256 digests of all input files

Support for benchmark comparison via baseline YAML


🧩 Goal: Ensure any contributor or reviewer can understand, reproduce, and extend the SkyWire × SIGMA integrated governance diagnostics without prior context.

👁️‍🗨️ Authors: Yuri Kang & AI Kang (DeepKang Labs) 🪪 License: MIT 📦 Version: v1.1 (Bridge Documentation)
