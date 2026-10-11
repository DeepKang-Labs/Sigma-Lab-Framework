# Local GGUF conversation profile

Validated on 2026-10-11 with Windows, 32 GB RAM, an Intel i7-7700K and an
NVIDIA RTX 2070 with 8 GB VRAM. This is a local text assistant, with no browser,
email, phone or file-inspection tools available to the model.

## Reproducible profile

The [manifest](../configs/local_qwen35_manifest.json) pins Qwen3.5-35B-A3B
Q4_K_M, its repository revision and SHA-256, and the official llama.cpp b9000
Windows Vulkan package and SHA-256. Both downloaded artifacts passed their
published checksum checks. Weights occupy 22,016,023,168 bytes and are not
included in Git.

The reference deployment uses 4096 context tokens, one slot, four CPU threads,
automatic GPU fitting with a 1024 MiB margin, direct host RAM loading, and
thinking disabled. Temperature is 0.6, top-p 0.9 and seed 42. Different hardware
or engine versions can still produce different answers.

Two newer modular Windows launchers were quarantined by the installed antivirus
with `IDP.Generic`. The official monolithic b9000 release ran with protection
unchanged. No antivirus exception was added and no quarantined file was restored.

## Checks performed

- 67 core/adapter tests passed, including both cached pretrained-model tests.
- 11 offline LLM runtime regressions passed using a locally constructed tiny GPT-2.
- One interface startup/history test passed: **79 automated tests in total**.
- Twelve real Qwen conversation checks were inspected: French identity,
  capability honesty, exact arithmetic (`17 * 23 = 391`), translation, supplied
  nickname history, fresh-session isolation, and a six-turn conversation replay.
  All twelve finished with `stop`, without token-budget truncation.
- After an explicit French request, the two English questions in the replay
  received French answers. Recognized French/English language requests are derived
  from user messages, ignoring assistant drift, and retained in the system
  instruction even when older context messages are trimmed. This recognizer is
  a limited command heuristic, not a universal language-understanding parser.
- Two additional actual Gradio API exchanges confirmed the loaded Qwen alias,
  visible session history, language preference and complete answers.
- Unauthenticated generation requests received HTTP 401. Health/model metadata
  routes remain public on loopback. The private local API key is excluded from
  generation reports and Git.

The final twelve short exchanges took 6.06–13.47 seconds each on this machine.
The two UI exchanges took 7.02 and 4.55 seconds. Initial model loading can take
several minutes. Longer prompts or responses can take longer.

## Running and reviewing

Install the UI dependency profile, download and verify the pinned artifacts,
then start:

```text
python scripts/run_gguf_ui.py --engine PATH_TO_LLAMA_SERVER --model-file PATH_TO_GGUF
```

The newer `llama` executable is also accepted; the launcher adds its `serve`
subcommand. The supervisor owns the child model server and stops it on normal
exit. It binds the server and UI to localhost, and generates a local API key.

To record checks against the running supervisor, set `SIGMA_LLAMA_API_KEY` in
the evaluation process from `.tmp/gguf-local/local-api-key.txt`, then run:

```text
python scripts/evaluate_local_chat.py --model Qwen3.5-35B-A3B-Q4_K_M
```

The default 400-token output budget matches the UI. Records include messages,
answers, usage, finish status, sampling and elapsed time. Keep the local key
private. A new browser conversation starts with empty history; earlier reports
are not imported as model memory.

## Limits

### Subsequent integration pass (2026-10-11)

The software suite now passes 94 tests locally: 82 Framework/adapter/broker
tests (including the two cached pretrained checks), 11 offline tiny-model
generation regressions and one UI check. The 15 new broker tests cover denied
authority, path scope, hard links, noninteractive approval, expiry, consumed
approval, backup and the fixed AST contract. They do not validate other repos.

A separate expanded actual-Qwen run contains 18 recorded cases with prompts,
responses, times and token usage. Bounded inspection assessed 17 PASS and
one FAIL; all finished without truncation. The failed Python trace returned
37 instead of the correct 27 and also ignored the numeric-only format.
A previous pass returned 47 and showed language drift. Strengthening the
language instruction restored French in the replay, but did not correct this
trace. Observed final times were 3.41–12.05 seconds per case. These counts are
case-specific assessments, not a general model accuracy percentage.

An actual model-generated sandbox proposal passed its three fixed AST cases;
the target file's before/after hashes were identical. A real piped approval
attempt was rejected. No live target mutation was approved. See
[controlled integration](GOVERNED_LOCAL_INTEGRATION.md) for the authority boundary.

These are inspected conversational smoke checks on one PC, not a general quality,
safety or scientific benchmark. The assistant can still make factual mistakes.
Oldest messages are trimmed to fit the context. Long answers can hit the output
budget; the generation report records this with `finish_reason: length`.
Native chat reports do not assign an invented Sigma correctness score.
