#!/usr/bin/env python3
# build_manifest.py — résumé des artefacts du run

import os, json

ROOT = os.environ.get("ARTIFACTS_ROOT", "artifacts")

summary = {
    "heartbeat_exists": os.path.exists(os.path.join(ROOT, "heartbeat", "heartbeat.json")),
    "model_exists": os.path.exists(os.path.join(ROOT, "model", "local_model.json")),
    "outbox": len(os.listdir(os.path.join(ROOT, "mesh", "outbox"))) if os.path.isdir(os.path.join(ROOT, "mesh", "outbox")) else 0,
    "inbox": len(os.listdir(os.path.join(ROOT, "mesh", "inbox"))) if os.path.isdir(os.path.join(ROOT, "mesh", "inbox")) else 0,
    "archive": len(os.listdir(os.path.join(ROOT, "mesh", "archive"))) if os.path.isdir(os.path.join(ROOT, "mesh", "archive")) else 0,
}

os.makedirs(ROOT, exist_ok=True)

manifest_path = os.path.join(ROOT, "manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("📦 Manifest créé :", json.dumps(summary, indent=2))
