#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple local model store for Sigma.

- Keeps a JSON file at {ARTIFACTS_ROOT}/model/local_model.json
- Ensures a consistent set of weight keys (aligned with LocalStatsModel.FEATURE_KEYS):
  ["proxies", "vpn", "transports", "dmsg_entries", "rf_status_ok"]
"""

from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import Dict, Any


DEFAULT_FEATURE_KEYS = ["proxies", "vpn", "transports", "dmsg_entries", "rf_status_ok"]


def _env(k: str, d: str) -> str:
    v = os.getenv(k)
    return v if v and v.strip() else d


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class LocalModelStore:
    def __init__(self, artifacts_root: str | None = None):
        self.artifacts_root = _env("ARTIFACTS_ROOT", artifacts_root or "artifacts")
        self.model_dir = os.path.join(self.artifacts_root, "model")
        self.model_path = os.path.join(self.model_dir, "local_model.json")
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return os.path.exists(self.model_path)

    def load(self) -> Dict[str, Any]:
        with open(self.model_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, obj: Dict[str, Any]) -> None:
        with open(self.model_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    def bootstrap(self, weight_keys: list[str] | None = None) -> Dict[str, Any]:
        """
        Create a new local model with zeroed weights if none exists.
        """
        if self.exists():
            return self.load()

        keys = weight_keys or DEFAULT_FEATURE_KEYS
        model = {
            "model": {
                "version": 1,
                "weights": {k: 0.0 for k in keys},
            },
            "meta": {
                "created_utc": _utc_iso(),
                "note": "bootstrap zero-weights",
            },
        }
        self.save(model)
        return model

    def ensure_keys(self, required_keys: list[str]) -> Dict[str, Any]:
        """
        Ensure model has required keys; add missing keys set to 0.0.
        """
        obj = self.load() if self.exists() else self.bootstrap(required_keys)
        weights = obj.setdefault("model", {}).setdefault("weights", {})
        changed = False
        for k in required_keys:
            if k not in weights:
                weights[k] = 0.0
                changed = True
        if changed:
            obj.setdefault("meta", {})["updated_utc"] = _utc_iso()
            self.save(obj)
        return obj
