#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeshFederate — export a local model delta to mesh outbox.

This is a simple producer that:
- reads LOCAL_MODEL
- computes a small random/heuristic delta (placeholder)
- wraps it per protocol and writes to OUTBOX_DIR/{utc-ts}_{rand}.json
"""

from __future__ import annotations
import os, json, time, random, string
from pathlib import Path
from typing import Dict, Any

try:
    from scripts.mesh.protocol import PROTO_VERSION, make_delta_envelope
except Exception:
    from mesh.protocol import PROTO_VERSION, make_delta_envelope


def _env(k: str, d: str) -> str:
    v = os.getenv(k)
    return v if v and v.strip() else d


ARTIFACTS_ROOT = _env("ARTIFACTS_ROOT", "artifacts")
LOCAL_MODEL    = os.path.join(ARTIFACTS_ROOT, "model", "local_model.json")
OUTBOX_DIR     = os.path.join(ARTIFACTS_ROOT, "mesh", "outbox")

Path(OUTBOX_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(LOCAL_MODEL)).mkdir(parents=True, exist_ok=True)


def _read_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(p: str, obj: Dict[str, Any]) -> None:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _utc_ts() -> int:
    return int(time.time())


def _rand_tag(n: int = 5) -> str:
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def run() -> int:
    if not os.path.exists(LOCAL_MODEL):
        print("[MeshFederate] No local model; nothing to federate.")
        return 0

    model = _read_json(LOCAL_MODEL)
    weights = model.get("model", {}).get("weights", {}) or {}

    # --- Placeholder delta logic ---
    # For now we just create a tiny perturbation on up to 10% of keys (max 20 keys)
    keys = list(weights.keys())
    if not keys:
        print("[MeshFederate] Model has no weights; nothing to federate.")
        return 0

    max_keys = max(1, min(20, int(0.1 * len(keys))))
    sample = random.sample(keys, k=max_keys)
    delta = {k: random.uniform(-0.01, 0.01) for k in sample}  # small nudges
    weight = 1.0

    envelope = make_delta_envelope(
        proto=PROTO_VERSION,
        author=_env("SIGMA_NODE_ID", "anonymous"),
        weight=weight,
        delta=delta,
        meta={"note": "heuristic demo delta"},
    )

    name = f"{_utc_ts()}_{_rand_tag()}.json"
    path = os.path.join(OUTBOX_DIR, name)
    _write_json(path, envelope)

    print(f"[MeshFederate] Wrote delta to {path} (keys={len(delta)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
