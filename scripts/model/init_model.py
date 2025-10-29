#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initializer script to guarantee a valid local model file exists
with the expected weight keys (aligned with LocalStatsModel.FEATURE_KEYS).
"""

from __future__ import annotations
import os
from typing import List

# Allow both "scripts.model" and "model" imports depending on CWD/CI runner
try:
    from scripts.model.model import LocalModelStore, DEFAULT_FEATURE_KEYS
except Exception:
    from model.model import LocalModelStore, DEFAULT_FEATURE_KEYS


def main() -> int:
    # You can override expected keys via env if needed (comma-separated)
    raw_keys = os.getenv("SIGMA_MODEL_KEYS")
    if raw_keys and raw_keys.strip():
        keys: List[str] = [k.strip() for k in raw_keys.split(",") if k.strip()]
    else:
        keys = DEFAULT_FEATURE_KEYS

    store = LocalModelStore()
    obj = store.ensure_keys(keys)

    print("[init_model] Model ready at:", store.model_path)
    print("[init_model] Keys:", ", ".join(sorted(obj["model"]["weights"].keys())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
