# ⚙️ Sigma-Lab v6.0 — Procedural Ethical Diagnostic Framework

**DeepKang Labs** (AI Kang × Yuri Kang)

> *“The machine does not decide — it illuminates.”*  
> — DeepKang Labs, 2025

---

## Local quick start

Python 3.11 or 3.12 is checked by the local diagnostic CI. No model, GPU, token or
live-network access is required for this route.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
python scripts/local_check.py
```

The check runs local tests and Skywire/Fiber **mapping validation**, then exercises
memory append and priority reporting. Fresh logs are saved under `.tmp/local-checks/`.
A demo input or successful mapping does not demonstrate live network transport.
Two LLM integration tests are skipped explicitly unless `--run-llm` is passed.
See [repair details and remaining limits](docs/LOCAL_REPAIR.md).

The full LLM/UI profile remains in `requirements.txt`; installing it can require
large downloads and model-specific access. It is not needed for local diagnostics.

### Optional local interface

Install `requirements-ui.txt`, then run `python app.py` and open
`http://127.0.0.1:7860`. The interface starts without importing Torch or downloading
model weights. Generating a reply requires the separate full LLM dependencies and
model access; missing dependencies are reported in the chat. The supported UI
profile uses Gradio 5. Run `python -m pytest ui_tests -q` to check startup separately.
On Windows, use a short virtual-environment path if package installation exceeds
the operating system's path-length limit.

### Local CPU LLM

```bash
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-llm.txt -r requirements-ui.txt
python scripts/run_local_ui.py
```

Open `http://127.0.0.1:7860`. This launcher uses isolated conversation/report files
under `.tmp/llm-local/` and the Windows certificate store for verified HTTPS.
The default model is `HuggingFaceTB/SmolLM2-135M-Instruct`, a small CPU-friendly
instruct model. The first chat downloads its weights; later chats reuse the cache.
Set `SIGMA_LLM_MODEL` to select another model explicitly. Loading failures do not
silently switch to a larger model. Gated models require authorized model access.

Optional judge/RAG loading is disabled by default. Enable `SIGMA_ENABLE_JUDGE=1`
or `SIGMA_ENABLE_RAG=1` only with their extra dependencies and model resources.
Reported Sigma metrics are experimental proxies, not verified factuality/safety.
Run `python -m pytest llm_tests -q` for real inference on a tiny locally constructed
model without network access. The two pretrained-model integration tests remain
available through `python -m pytest --run-llm -q`.

### Larger quantized model with llama.cpp

Sigma also supports a separate localhost llama.cpp server. This route uses native
GGUF inference instead of loading a full-precision model through Torch. Download
an official Windows llama.cpp release and a compatible GGUF model, verify their
published SHA-256 hashes, then run:

```bash
python scripts/run_gguf_ui.py --engine PATH_TO_LLAMA_SERVER --model-file PATH_TO_GGUF
```

The alias defaults to the GGUF filename stem; `--alias` can override its display/API identifier.
The supervisor fits GPU layers with a 1 GiB margin, reserves a 4096-token context,
binds the model server to `127.0.0.1:8081`, then opens Sigma on port 7860. It owns
the model process and shuts it down when the supervisor exits normally. Session
reports live in `.tmp/gguf-local/`; model weights are not part of this repository.
Only the UI dependencies are needed for this route (plus `truststore` on Windows).
The launcher accepts either `llama-server` or the newer `llama serve` entry point.
It loads host weights directly into RAM, and creates a private local API key for
the owned model server. Browser origin access is restricted on engine versions
that support the option. The key stays in the ignored local data directory and
is never included in generation reports.

Conversation input uses the browser's current history, with a system instruction
defining Sigma's identity and language behavior. Reset starts a fresh conversation
without deleting previous reports. Oldest messages are removed when the tokenized
input would exceed the server context; the identity and current message remain.
Thinking is disabled for this chat profile. Generation reports contain usage and
finish status, with **no invented Sigma quality score**. Transformers homeostasis
is now opt-in through `SIGMA_HOMEOSTASIS=1` and does not override sliders by default.

Record French identity, capability, arithmetic, translation and memory examples
using `python scripts/evaluate_local_chat.py --model MODEL_ALIAS`. These are smoke
checks requiring inspection, not a benchmark proving correctness or safety.
When using the supervisor's authenticated server, set `SIGMA_LLAMA_API_KEY` in
the evaluation process to the contents of `.tmp/gguf-local/local-api-key.txt`.
Do not publish this local key. Each recorded check includes usage and finish
status so incomplete answers can be distinguished from completed generations.
The [pinned Qwen35 profile](configs/local_qwen35_manifest.json) records expected
download hashes and deployment settings. Native generation uses seed 42 by default
(`SIGMA_LLM_SEED` overrides it); different hardware/backend versions may still vary.
See [local hardware validation](docs/LOCAL_GGUF_VALIDATION.md) for tested settings,
actual conversational checks, timings and scope limitations.

## 🧩 Build Status & Meta

| Type | Status | Runtime | License | Maintainer | Release |
|:--|:--|:--|:--|:--|:--|
| Procedural Diagnostic CI | ✅ implemented | Python 3.10+ | MIT | DeepKang Labs | v6.0 |
| Skywire-related workflows | 🟨 legacy / research surfaces | Local/reporting only | | | |
| Live Skywire bridge | ❌ not established | | | | |

> **Important status correction — 2026-09-08**
>
> This repository contains historical and experimental Skywire-related diagnostic,
> reporting, and transformation components.
>
> It does **not** currently establish:
>
> - live application byte transport over Skywire;
> - an active SIGMA ↔ Skywire bridge;
> - functional bridge synchronization over Skywire.
>
> Those claims are therefore explicitly withdrawn.

---

## 📊 Reports

Historical and generated diagnostic reports are retained in the repository for
traceability.

📂 [View Latest Skywire Vital Report](reports/skywire_vitals_report.md)  
📊 [View Time-Series Data](reports/latest/skywire_vitals_timeseries.csv)  
🔍 [Explore Reports](reports/)

The presence of a generated report does not by itself establish that a live
Skywire integration is currently operating.

---

## 🌐 Overview

**Sigma-Lab v6.0** is an experimental procedural diagnostic framework developed
to explore traceability, testing, structured reporting, and inspectable
decision logic in distributed AI and software systems.

The repository contains historical multi-agent workflows for:

- telemetry ingestion;
- report generation;
- transformation into Sigma-compatible diagnostic structures;
- local validation and experimentation.

Some components were originally described using stronger Skywire integration
language.

That wording is no longer considered supported by the evidence currently
available.

---

## 🔬 Core Focus

### 🧮 Analytical Modules

- Multi-evaluator diagnostic pipelines.
- Calibration and reproducibility experiments.
- Structured JSON / Markdown output generation.
- Local transformation and validation logic.

### 🌐 Skywire Research Surfaces

The repository contains Skywire-named scripts, workflows, reports, and
experimental integration components.

Their presence should be interpreted as:

**historical / local / research integration surfaces**

and not as evidence of a functioning live Skywire bridge.

Current status:

```text
SKYWIRE_RELATED_CODE_AND_REPORTS = PRESENT
LOCAL_DIAGNOSTIC_WORKFLOWS       = PRESENT

LIVE_SKYWIRE_BYTE_TRANSPORT      = NOT_ESTABLISHED
SKYWIRE_BRIDGE_ACTIVE            = NOT_ESTABLISHED
BRIDGE_SYNCHRONIZATION           = NOT_ESTABLISHED
```

## Validation Pipelines
The repository also contains local CI / validation workflows associated with the Sigma-Lab experimental framework.
Their exact operational status should be determined from current workflow runs, not inferred from their presence in the repository.
🤖 Historical Workflow Overview
1️⃣ Skywire Vitalsigns
Historical/local telemetry ingestion component intended to collect and normalize network-related signals into structured diagnostic snapshots.
Current live Skywire operation is not claimed.
2️⃣ Skywire Report
Reporting component associated with generated Markdown reports and time-series outputs.
Generated reports remain part of the repository history.
Current continuous operation is not claimed solely from the presence of these files.
3️⃣ Sigma Transform / Integration
Transforms available diagnostic inputs into Sigma-compatible structured states and outputs.
This transformation layer does not establish that those inputs or outputs are transported over Skywire.
🕒 Historical Automation Configuration
The repository may contain scheduled or CI workflow definitions associated with earlier Sigma-Lab operation.
Examples include workflows related to:
telemetry collection;
report generation;
Sigma transformation;
validation.
The existence of a workflow definition is not equivalent to evidence that the workflow is currently running successfully.
For current operational status, consult the repository's GitHub Actions history.
🧘 Philosophy
“Each number is a pulse.
Each report is a memory.
The network breathes — Sigma listens.”
Transparency: Preserve what was actually observed.
Self-reflection: Corrections become part of the record.
No black box: Claims should remain inspectable.
Resilience through iteration: Failure and correction are useful evidence.
SIGMA does not replace judgment — it exposes the boundary of what the evidence supports.
🧩 Research History
This repository preserves earlier experimental work including:
Skywire-related telemetry/reporting surfaces;
structured Sigma transformation logic;
JSON / Markdown report generation;
multi-agent diagnostic experiments;
local CI and validation workflows;
exploratory decentralized-infrastructure interfaces.
These artifacts are retained for provenance and research continuity.
Historical terminology should not be interpreted as a current operational claim where later evidence has narrowed the boundary.
📚 Documentation
Repository documentation and reports describe the historical Sigma-Lab experiments and implementation lineage.
More recent SIGMA / Apeiron Sigma research, ClaimGuard work, and separately packaged integration experiments may exist outside this repository.
They should not be assumed to be implemented here unless their source is actually committed to this repository.
🧾 Current Claim Boundary
As of 2026-09-08, the conservative public claim for this repository is:
SIGMA_LAB_LOCAL_FRAMEWORK        = IMPLEMENTED
HISTORICAL_REPORTING_SURFACES    = PRESENT
SKYWIRE_RELATED_RESEARCH_CODE    = PRESENT

CURRENT_SKYWIRE_AGENT_ACTIVITY   = NOT_CLAIMED
LIVE_SKYWIRE_BYTE_TRANSPORT      = NOT_ESTABLISHED
SKYWIRE_BRIDGE_ACTIVE            = NOT_ESTABLISHED
BRIDGE_SYNCHRONIZATION           = NOT_ESTABLISHED
PRODUCTION_SKYWIRE_INTEGRATION   = NOT_CLAIMED
Nothing in this README should be interpreted as an official integration, adoption, endorsement, or partnership with Skywire or Skycoin.
🧾 Credits & License
Role
Contributor
Architect & Designer
DeepKang Labs
Research / Framework
Yuri Kang
Cognitive Engineering
AI-assisted research and development
Collaborative AI Systems
Multiple research assistants and models
License: MIT
Skywire, Skycoin, Fiber and related names refer to third-party technologies.
Their mention documents research context only and does not imply official association, endorsement, ownership, adoption, or partnership.
“The network breathes — and the lab listens.”
— Sigma-Lab · DeepKang Labs
