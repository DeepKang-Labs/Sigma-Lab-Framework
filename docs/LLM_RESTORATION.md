# Local LLM restoration - 2026-10-11

The CPU route now generates with `HuggingFaceTB/SmolLM2-135M-Instruct`.
This restores local inference, not validated model quality or ethical judgement.

## Corrected defects

* Current prompts were persisted before prompt construction and then appended
  again. Prompts now enter the input once and successful replies persist as
  `user`/`assistant`; historical `human`/`ai` records are translated on read.
* Legacy prompt fallback omitted the current message. Both prompt routes retain it.
* Entire generated sequences were decoded, allowing prompt text into the answer.
  Only newly generated tokens are decoded. Ordinary uses of the word "assistant"
  are preserved instead of splitting a response at that word.
* Context truncation returned a plain dictionary and then called its nonexistent
  `.to()` method. Tensors now move individually, with space reserved for output.
* Loading failures silently cascaded through other, potentially much larger
  models. Loading is now explicit; the interface reports the actual active model.
* Optional evaluators could attempt additional model downloads; judge/RAG use is
  explicit and disabled on the minimal route. Judge prompt JSON braces are escaped
  correctly, and its output parser decodes only generated tokens.
* Partial nested configuration could discard homeostasis defaults. Configuration
  now merges into a deep copy. Malformed configuration and conversation files are
  surfaced instead of silently replacing history. Objectivity weights normalize
  and reject non-finite/negative/zero-sum input; invariants flag non-finite metrics.
* Shared-agent generation is serialized to protect its conversation and controller
  state. Report provenance labels the model-name hash as an identifier hash, not
  weight integrity. Experimental evaluation scope is explicit.
* The minimal dependency route excludes quantization, semantic embedding models
  and FAISS. The CPU Torch profile no longer uses the old 2.3.1 pin.
* `--report` no longer instantiates or downloads a model to read a saved report.

## Verified on the local Windows PC

Python 3.12, Torch 2.14.1+cpu, Transformers 4.57.6, Accelerate 1.15.0 and Gradio
5.50.0: **70 tests passed with all model integration tests enabled**. Gradio emits
62 internal deprecation warnings about its future version 6 API; this profile
excludes version 6. No test was skipped in this full local run.

Two successive pretrained-model exchanges produced a greeting and "Two + Two =
Four", persisted four correctly ordered messages, and wrote a report without
invariant errors. After download, the pretrained-model integration tests passed
with `HF_HUB_OFFLINE=1`. These examples are smoke checks, not benchmarks.

Eleven offline regression tests construct a tiny random GPT-2 model locally and
exercise actual token generation, prompt uniqueness, context budgeting, output
decoding, old roles, failed-generation history preservation, invalid arguments,
configuration, corrupt memory, weights and finite metrics. They do not claim the
random model gives useful answers. GitHub CI runs this route without model downloads.

## Run and scope

Follow the README's Local CPU LLM steps. `scripts/run_local_ui.py` isolates session
files under `.tmp/llm-local/`, uses verified Windows system certificates, and binds
only to localhost. Optional `--cache-dir` reuses a chosen Hugging Face cache.
Prompts and replies persist locally; use a fresh `--data-dir` for separate sessions.

Large/gated models, GPU inference, optional judge/RAG quality, multi-user deployment
and real Skywire transport remain unverified. Model-specific licenses and access
still apply. Sigma scores are heuristic proxies, not established factuality or
safety measurements. This repair remains in the existing review PR.
