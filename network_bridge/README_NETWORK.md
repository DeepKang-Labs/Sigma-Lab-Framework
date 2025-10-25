# Network Bridge (Generic): Skywire & Fiber (NESS)

**Status**
- ✅ Sigma-Lab v5.1: Core stable (MIT)
- 🚧 Skywire Bridge: In testing (Phase α)
- 📋 Fiber Bridge: **Prepared, NOT activated**
- 🔐 License: MIT

**Committee (governance)**
- DeepKang Labs Network Governance Committee (DL-NGC)

## Philosophy — AI + Human = 3
Sigma-Lab is a **procedural mirror**, not a moral oracle. The bridge exposes
human choices (weights, risk formulas, thresholds) in YAML so that teams can
**argue, revise, and own** the policy. You can run **linear mode** (no external
deps) for maximum simplicity, or **simple expressions** for quicker iteration.

## What this does
- Loads a Discovery-like YAML of decision points (or a deterministic demo).
- Applies **mappings** (tensions → SIGMA dimensions) with traceable derivations.
- Produces SIGMA contexts + runs Sigma-Lab (or a demo fallback).
- Exports an **integration report** with SHA256 fingerprints for auditability.

## CLI (validate-only; no side effects)
```bash
python network_bridge/run_network_integrated.py \
  --network skywire \
  --discovery ./discovery \
  --mappings ./network_bridge/mappings_skywire.yaml \
  --config ./sigma_config_placeholder.yaml \
  --out ./pilots/validation_logs/skywire_validate_only.json \
  --validate-only \
  --formula-eval auto \
  --pretty

Switch to Fiber:

python network_bridge/run_network_integrated.py \
  --network fiber \
  --discovery ./discovery \
  --mappings ./network_bridge/mappings_fiber.yaml \
  --config ./sigma_config_placeholder.yaml \
  --out ./pilots/validation_logs/fiber_validate_only.json \
  --validate-only \
  --formula-eval auto \
  --pretty

--formula-eval modes

linear  → no external deps; use derive_risk_linear (field mapping + scale)

simple  → evaluate short expressions via simpleeval

auto    → prefer derive_risk_linear if present, otherwise try expressions


Timeline (agreed)

Week 1–2: Skywire Phase α testing (real case, 3 evaluators)

Week 3–4: Results publication

Week 5+: Fiber activation (only if Phase α successful)


> ⚠️ PREPARATORY DISCLAIMER (Fiber): This bridge is prepared but NOT validated yet. Activation will occur only after Skywire Phase α publication.
