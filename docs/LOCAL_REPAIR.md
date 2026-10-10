# Local diagnostic restoration - 2026-10-11

This repair restores a reproducible local route. It does not establish operational
Skywire transport, model quality or scientific validity of diagnostic proxies.

## Defects corrected

* The smoke workflow passed mapping-validation options that its runner no longer
  accepted. The runner now supports that local route for Skywire and Fiber, records
  demo/file provenance, validates configuration and emits an explicit local scope.
* Invalid risks were silently clamped and could receive an acceptance verdict.
  They now produce input errors, invalid status and no acceptance. Diagnostics
  operate on a copy rather than modifying the caller's context.
* Deliberation used an equal average despite configured policy weights. It now uses
  finite normalized weights, supports wrapped YAML values and partial configuration,
  and reports the aggregate used. This intentionally changes results where policy
  weights differ from equal weighting; it is not a new empirical validation.
* Default audit provenance now identifies illustrative defaults rather than
  implying an external hospital policy authority.
* Missing/non-finite/out-of-range vitals no longer receive optimistic health scores.
* Context exports cannot escape the configured directory through decision IDs.
* Mesh append and priority commands no longer fail on Windows console arrows.
  An unreadable existing memory is not silently overwritten with an empty history.
* Priority placeholders are labeled as unsupported-schema fallbacks, not computed
  rankings. Actual weighted-node inputs remain labeled computed.
* Simulation uses an explicit reproducible seed and refuses non-finite output.
  The undirected Laplacian uses NumPy directly, eliminating undeclared NetworkX/
  SciPy requirements for this route. Invalid graph/CFL/gain parameters are rejected.
* Local tests/install no longer require the model stack. LLM integration tests
  remain available through `--run-llm`; their skips are explicit.

## Verification

Windows/Python 3.12: 56 tests passed and two model integration tests were skipped.
Regression coverage includes concurrent diagnostics, malformed inputs/configs,
mapping CLI plus memory append, valid/missing vitals, export containment,
memory preservation, known graph Laplacian, RK4 decay accuracy and two identical
600-step simulations in an isolated directory. Run `python scripts/local_check.py`
to generate fresh local logs and summaries. GitHub's Local diagnostics workflow
checks Python 3.11/3.12 on Windows and Linux.

## Remaining external or unverified surfaces

* Real Skywire/Fiber transport and mesh synchronization need an actual supported
  endpoint/protocol, operator configuration, authorization and observed traffic.
  YAML transformation is not a transport integration.
* The Gradio/LLM route requires its separate dependencies, model weights, adequate
  RAM/GPU resources where applicable, and any gated-model access. It was not run
  in this repair. `requirements.txt` remains the legacy full-stack profile; the
  lightweight profile is `requirements-core.txt`.
* Historical scheduled workflows, Docker/mesh services, live data feeds and model
  quality were not validated by local tests. Their status requires separate runs;
  this repair neither fabricates external results nor restores unsupported claims.
* The risk/health/coherence formulas are experimental proxies, not established
  measurements of ethics, safety or factual truth.

The repository's safety policy requests PR review for rule/engine changes. The
repair is submitted on a separate branch for review rather than merged automatically.
