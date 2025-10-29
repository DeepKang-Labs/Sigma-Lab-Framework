#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh protocol helpers for Sigma.

Envelope format (JSON)
----------------------
{
  "proto": 1,
  "type": "model_delta",
  "author": "...",
  "weight": 1.0,
  "delta": { "w1": +0.001, "w2": -0.004, ... },
  "meta": { "note": "optional" },
  "sig":  null
}
"""

from __future__ import annotations
from typing import Dict, Any, Tuple

PROTO_VERSION = 1


def is_valid_delta(obj: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not_a_dict"
    if obj.get("type") != "model_delta":
        return False, "wrong_type"
    if int(obj.get("proto", -1)) != PROTO_VERSION:
        return False, "proto_mismatch"
    if "delta" not in obj or not isinstance(obj["delta"], dict):
        return False, "no_delta"
    # basic numeric check
    for k, v in obj["delta"].items():
        try:
            float(v)
        except Exception:
            return False, f"non_numeric:{k}"
    return True, "ok"


def extract_delta_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "proto": int(obj.get("proto", PROTO_VERSION)),
        "author": obj.get("author", "unknown"),
        "weight": float(obj.get("weight", 1.0) or 1.0),
        "delta": dict(obj.get("delta", {})),
        "meta": dict(obj.get("meta", {})),
    }


def make_delta_envelope(
    proto: int,
    author: str,
    weight: float,
    delta: Dict[str, float],
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "proto": int(proto),
        "type": "model_delta",
        "author": author,
        "weight": float(weight),
        "delta": {k: float(v) for k, v in delta.items()},
        "meta": meta or {},
        "sig": None,  # place-holder for future signing
    }
