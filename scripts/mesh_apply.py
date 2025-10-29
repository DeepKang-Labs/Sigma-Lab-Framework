#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeshApply — apply federated deltas from mesh inbox to the local model.

Inputs
------
- ARTIFACTS_ROOT (env) default: artifacts
- INBOX_DIR       = {ARTIFACTS_ROOT}/mesh/inbox
- LOCAL_MODEL     = {ARTIFACTS_ROOT}/model/local_model.json

Outputs
-------
- Updated LOCAL_MODEL (after weighted average)
- A small audit file: {ARTIFACTS_ROOT}/mesh/last_apply.json

Behavior
--------
- Collect all *.json deltas in INBOX_DIR
- Validate protocol (see scripts.mesh.protocol)
- Weighted average by 'weight' field (fallback = 1.0)
- Merge only keys present in local model's "weights" dict
- Archive consumed deltas to .../mesh/archive/{utc-ts}_{hash}.json
"""

from __future__ import annotations
import os, json, glob, time, hashlib, shutil
from typing import Dict, Any, List, Tuple
from pathlib import Path

try:
    from scripts.mesh.protocol import (
        PROTO_VERSION,
        is_valid_delta,
        extract_delta_payload,
    )
except Exception:
    # GitHub Actions sometimes runs from repo root; allow relative import
    from mesh.protocol import PROTO_VERSION, is_valid_delta, extract_delta_payload


def _env(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val and val.strip() else default


ARTIFACTS_ROOT = _env("ARTIFACTS_ROOT", "artifacts")
INBOX_DIR      = os.path.join(ARTIFACTS_ROOT, "mesh", "inbox")
ARCHIVE_DIR    = os.path.join(ARTIFACTS_ROOT, "mesh", "archive")
LOCAL_MODEL    = os.path.join(ARTIFACTS_ROOT, "model", "local_model.json")
LAST_APPLY     = os.path.join(ARTIFACTS_ROOT, "mesh", "last_apply.json")

Path(INBOX_DIR).mkdir(parents=True, exist_ok=True)
Path(ARCHIVE_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(LOCAL_MODEL)).mkdir(parents=True, exist_ok=True)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_local_model(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        # bootstrap an empty model
        base = {"model": {"version": 1, "weights": {}}, "meta": {"created_utc": _utc_iso()}}
        _write_json(path, base)
        return base
    return _read_json(path)


def _collect_inbox() -> List[str]:
    return sorted(glob.glob(os.path.join(INBOX_DIR, "*.json")))


def _compute_weighted_average(
    base_weights: Dict[str, float],
    deltas: List[Tuple[Dict[str, float], float]]
) -> Dict[str, float]:
    """
    base_weights: current local weights (keys define the space)
    deltas: list of (delta_weights, weight)
    Returns a new dict (same keys as base)
    """
    if not deltas:
        return base_weights.copy()

    # initialize accumulators
    acc: Dict[str, float] = {k: 0.0 for k in base_weights.keys()}
    sumw: float = 0.0

    for dw, w in deltas:
        ww = float(w) if w and w > 0 else 1.0
        for k in acc.keys():
            if k in dw and isinstance(dw[k], (int, float)):
                acc[k] += ww * float(dw[k])
        sumw += ww

    if sumw <= 0:
        return base_weights.copy()

    # new weights = base + averaged delta
    out = {}
    for k in base_weights.keys():
        avg_delta = acc[k] / sumw
        out[k] = float(base_weights.get(k, 0.0)) + avg_delta

    return out


def _archive_delta(path: str) -> str:
    with open(path, "rb") as f:
        b = f.read()
    h = _sha256_bytes(b)
    name = os.path.basename(path)
    dst = os.path.join(ARCHIVE_DIR, f"{int(time.time())}_{h}_{name}")
    shutil.move(path, dst)
    return dst


def run() -> int:
    local = _load_local_model(LOCAL_MODEL)
    base_weights: Dict[str, float] = local.get("model", {}).get("weights", {}) or {}

    inbox_files = _collect_inbox()
    report_items = []
    applied: List[Tuple[Dict[str, float], float]] = []

    for fp in inbox_files:
        try:
            raw = _read_json(fp)
            ok, reason = is_valid_delta(raw)
            if not ok:
                report_items.append({"file": fp, "status": "rejected", "reason": reason})
                _archive_delta(fp)
                continue

            payload = extract_delta_payload(raw)
            if payload["proto"] != PROTO_VERSION:
                report_items.append({"file": fp, "status": "rejected", "reason": "proto_mismatch"})
                _archive_delta(fp)
                continue

            d_weights = payload.get("delta", {})
            weight    = float(payload.get("weight", 1.0) or 1.0)

            # Filter to known keys only (defensive)
            d_filtered = {k: float(v) for k, v in d_weights.items() if k in base_weights}
            if not d_filtered:
                report_items.append({"file": fp, "status": "skipped", "reason": "no_matching_keys"})
                _archive_delta(fp)
                continue

            applied.append((d_filtered, weight))
            archived_to = _archive_delta(fp)
            report_items.append({"file": fp, "status": "applied", "archived": archived_to, "weight": weight})
        except Exception as e:
            report_items.append({"file": fp, "status": "error", "reason": repr(e)})
            # try to archive anyway to avoid re-processing loop
            try:
                _archive_delta(fp)
            except Exception:
                pass

    # If nothing to apply, still write last_apply report and exit
    if not applied:
        _write_json(LAST_APPLY, {
            "utc": _utc_iso(),
            "proto": PROTO_VERSION,
            "inbox_count": len(inbox_files),
            "applied": 0,
            "note": "no applicable deltas",
            "items": report_items,
        })
        print("[MeshApply] No applicable deltas; local model unchanged.")
        return 0

    new_weights = _compute_weighted_average(base_weights, applied)
    local["model"]["weights"] = new_weights
    local.setdefault("meta", {})["updated_utc"] = _utc_iso()
    _write_json(LOCAL_MODEL, local)

    _write_json(LAST_APPLY, {
        "utc": _utc_iso(),
        "proto": PROTO_VERSION,
        "inbox_count": len(inbox_files),
        "applied": len(applied),
        "items": report_items,
        "local_model": {
            "path": LOCAL_MODEL,
            "keys": len(new_weights),
        },
    })

    print(f"[MeshApply] Applied {len(applied)} delta(s). Local model updated with {len(new_weights)} keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
