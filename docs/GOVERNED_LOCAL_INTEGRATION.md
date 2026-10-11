# SIGMA Local Assistant: controlled integration

Status: initial local prototype, not a production isolation boundary.
Capability ≠ Authority.

## Executable boundary

The ordinary chat has no development tools. `scripts/demo_dev_gateway.py` is
a separate opt-in demonstration using the already-running local model. It
supplies the single sandbox file to Qwen, accepts one strictly parsed JSON
request, produces a unified diff and performs a fixed test. It never applies
the proposal. A successful test does not authorize a write.

The model-facing `Gateway.dispatch` supports only `read`, `propose`, and
`test`. No shell, process execution, network, deletion, Git operation,
approval flag or apply operation is exposed. Only `calculator.py` in the
configured isolated workspace is readable; links, junctions, hard links,
traversal and files over 8 KiB are rejected. State and audit files live outside
that workspace. Those broker-managed records are part of the authorized
installation; model requests cannot choose their paths.

Tests compare a restricted AST against two exact function contracts. They do
not execute, import, eval or spawn proposed Python. Results are PASS, FAIL,
or INCONCLUSIVE when a proposal exceeds that tiny contract. These are not
general Python tests or a security assessment of arbitrary programs.

The separate operator command is:

```text
python scripts/dev_gateway.py approve PROPOSAL_ID
```

It requires an interactive terminal, displays the diff, and requires typing
`APPLY PROPOSAL_ID`. Approval expires after ten minutes, binds the current
base hash and complete replacement, is checked again after review, and can
only be consumed once. A lock serializes operator reviews. A piped answer,
`approved: true` in model JSON, or a model apply request is rejected.
The operator command must never become an LLM tool or a chat API endpoint.

After approval the broker records intent, saves the previous content,
prepares the replacement and replaces the single target atomically. Audit
JSONL records requests, decisions and completed operations. An interrupted
approval can leave a receipt, backup or lock requiring manual recovery;
there is no automatic retry granting fresh authority. To roll back a file,
review its backup, propose that content again and explicitly approve the
reverse diff. No real sandbox modification was approved during delivery.

This prototype does not defend against a hostile process running as the
same Windows user, forged terminal input by privileged automation, or
concurrent modification outside the broker. Logs are durable local records,
not cryptographically independent evidence. Production expansion requires
separate OS identities, confined workers and an authenticated operator
service. Human approval is an operator action, not a judgment by the model.

## Future Governed Runtime Orchestrator

1. Planner/model proposes a typed request without authority.
2. Policy broker checks capability, exact resource scope and current state.
3. Operator channel grants a short-lived, single-use authorization bound to
   request digest, base digest, resource, action and policy version.
4. Confined executor consumes that authorization immediately before mutation.
   Expired/revoked grants fail closed. Unknown evidence is INCONCLUSIVE.
5. Independent observer verifies downstream effects and captures evidence.
   A claimed success by the executor is not downstream proof.

The current read/propose/test broker and interactive gate instantiate a small
part of this architecture. Signed grants, mandate revocation, process
isolation and an independent observer are future work, not implemented here.

## Repository compatibility review

The open repair PR #1 in Sigma-Lab-Framework was inspected through GitHub;
it was a draft and unmerged at inspection, head `f6d3c75`. This integration
continues that reviewed branch; it does not automatically merge it.

Reusable Framework components: the loopback chat adapter, visible-session
history, request sampling, generation metadata, conservative diagnostic
validation, policy schemas, contained exports and deterministic local
simulation. Diagnostic scores are advisory experimental proxies and must
never issue write authority. Existing engine/rule edits remain subject to
repository review policy.

Sigma-Dynamics was read at `3a18f2f56ca5a82da22cdc0e615691247a4d41ea`.
Potential reuse: CSV feature mappings, numerical simulation, threshold/veto
diagnostics and plots, through a separate read-only data adapter. Its online
collectors and arbitrary output paths are outside this gateway's scope.
No Dynamics execution or regression suite was validated by the Framework
tests. Static findings requiring separate work before integration:

- `sigma_real_eval.py` and `sigma_stream_eval.py` use
  `sigmoid(lambda * (now - past))`, which weights older observations more
  heavily despite comments describing recency preference.
- `sigma_stream_eval.py` imports YAML, while its requirements omit PyYAML.
  Empty context rule lists match vacuously and can classify every sample as
  a crisis. Missing/nonfinite data needs explicit fail-closed validation.
- `network_bridge/skywire_bridge.py` labels external cryptocurrency proxies
  as network metrics; it is not a verified Skywire measurement adapter.
  Treat source provenance and proxy semantics explicitly.
- Simulation uses a module-global seeded RNG; successive runs in the same
  process advance state. Fresh-process seeding is not complete replayability.

The GitHub public repository listing returned only Framework and Dynamics.
Private IAST artifacts remain separate; no private content was published or
copied into this public integration. This document describes architectural
compatibility, not validation of Cards 1A/1B against this assistant.

## Windows lifecycle

This PC's launcher uses the existing verified GGUF and b9000 engine, a named
launch mutex, process ownership checks and an isolated Edge app profile.
Repeated launch reuses the supervisor; an orphan model blocks a second
load instead of silently creating another one. Occupied unrelated ports are
not terminated. Launch after Windows restart works without an autostart task;
the user double-clicks the Desktop shortcut.

The supervisor creates a private local stop token in ignored runtime state.
The stop launcher checks process identity and submits the token locally.
The supervisor interrupts its UI loop and releases its owned native server.
Windows native termination is process cleanup, not a guarantee of graceful
completion of an active inference. Wait for an answer before stopping.
Closing the app window alone leaves the local model available. No antivirus
setting, model weight or operating-system startup setting is changed.
