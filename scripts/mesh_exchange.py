#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesh_exchange.py — Sigma Mesh exchange (HTTP or Git mailbox)
- Sends deltas from artifacts/mesh/outbox to peers
- Fetches peers' deltas into artifacts/mesh/inbox
- Backends: "http" (production) or "git" (CI mailbox)

Author: DeepKang Labs
"""
import os, json, glob, shutil, subprocess, tempfile
from datetime import datetime
from typing import List

import yaml
import requests

ROOT = os.environ.get("ARTIFACTS_ROOT", "artifacts")
INBOX = os.path.join(ROOT, "mesh", "inbox")
OUTBOX = os.path.join(ROOT, "mesh", "outbox")
ARCHIVE = os.path.join(ROOT, "mesh", "archive")

CONFIG_PATH = "configs/mesh.yaml"   # <-- plural 'configs'

def ensure_dirs():
    for p in (INBOX, OUTBOX, ARCHIVE):
        os.makedirs(p, exist_ok=True)

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -------------------------- HTTP backend --------------------------
def http_send(delta_path: str, peer_url: str, token: str, node_id: str) -> bool:
    with open(delta_path, "rb") as f:
        files = {"file": (os.path.basename(delta_path), f, "application/json")}
        headers = {"X-Sigma-Token": token, "X-Sigma-Node": node_id}
        try:
            r = requests.post(f"{peer_url}/mesh/inbox", files=files, headers=headers, timeout=20)
            ok = (r.status_code == 200)
            print(f"[HTTP→] {os.path.basename(delta_path)} -> {peer_url} :: {r.status_code}")
            return ok
        except Exception as e:
            print(f"[HTTP!] send error {peer_url}: {e}")
            return False

def http_fetch(peer_url: str, token: str, node_id: str) -> int:
    headers = {"X-Sigma-Token": token, "X-Sigma-Node": node_id}
    try:
        r = requests.get(f"{peer_url}/mesh/outbox/list", headers=headers, timeout=20)
        r.raise_for_status()
        files = r.json().get("files", [])
    except Exception as e:
        print(f"[HTTP!] list error {peer_url}: {e}")
        return 0

    pulled = 0
    for fname in files:
        try:
            r = requests.get(f"{peer_url}/mesh/outbox/get/{fname}", headers=headers, timeout=20)
            if r.status_code == 200:
                target = os.path.join(INBOX, f"{peer_url.replace('://','_').replace('/','_')}__{fname}")
                with open(target, "wb") as f:
                    f.write(r.content)
                print(f"[HTTP←] got {fname} from {peer_url}")
                pulled += 1
        except Exception as e:
            print(f"[HTTP!] get error {peer_url}: {e}")
    return pulled

def run_http(cfg):
    node_id = cfg["node_id"]
    token = cfg["http"]["shared_token"]
    peers = cfg["http"].get("peers", [])

    # send all local outbox deltas to every peer, then archive locally
    for path in sorted(glob.glob(os.path.join(OUTBOX, "*.json"))):
        for p in peers:
            http_send(path, p["url"] + "/api", token, node_id)
        shutil.move(path, os.path.join(ARCHIVE, os.path.basename(path)))
        print(f"[✓] archived {os.path.basename(path)}")

    total = 0
    for p in peers:
        total += http_fetch(p["url"] + "/api", token, node_id)
    print(f"[Σ] pulled {total} delta(s)")

# -------------------------- Git mailbox backend -------------------
def run(cmd: List[str], cwd: str, env=None):
    print(f"[git] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd, env=env)

def run_git(cfg):
    node_id = cfg["node_id"]
    repo_url = cfg["git"]["repo_url"]
    branch = cfg["git"]["branch"]
    author_name = cfg["git"]["author_name"]
    author_email = cfg["git"]["author_email"]

    gh_token = os.environ.get("GH_TOKEN")
    if not gh_token:
        print("[!] GH_TOKEN missing; skip git backend.")
        return

    proto, rest = repo_url.split("://", 1)
    auth_url = f"{proto}://{gh_token}@{rest}"

    with tempfile.TemporaryDirectory() as tmp:
        run(["git", "init"], cwd=tmp)
        run(["git", "config", "user.name", author_name], cwd=tmp)
        run(["git", "config", "user.email", author_email], cwd=tmp)
        run(["git", "remote", "add", "origin", auth_url], cwd=tmp)

        # checkout mailbox branch (create if needed)
        try:
            run(["git", "fetch", "origin", branch], cwd=tmp)
            run(["git", "checkout", branch], cwd=tmp)
        except subprocess.CalledProcessError:
            run(["git", "checkout", "--orphan", branch], cwd=tmp)
            open(os.path.join(tmp, ".gitkeep"), "w").close()
            run(["git", "add", "."], cwd=tmp)
            run(["git", "commit", "-m", "init mailbox"], cwd=tmp)
            run(["git", "push", "origin", branch], cwd=tmp)

        mailbox = os.path.join(tmp, ".mesh-mailbox")
        my_out = os.path.join(mailbox, node_id, "outbox")
        my_in = os.path.join(mailbox, node_id, "inbox")
        os.makedirs(my_out, exist_ok=True)
        os.makedirs(my_in, exist_ok=True)

        # pull latest
        run(["git", "pull", "--rebase", "origin", branch], cwd=tmp)

        # stage local outbox -> mailbox
        for pth in sorted(glob.glob(os.path.join(OUTBOX, "*.json"))):
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            dest = os.path.join(my_out, f"{ts}__{os.path.basename(pth)}")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(pth, dest)
            print(f"[→] staged {os.path.basename(pth)}")

        run(["git", "add", "."], cwd=tmp)
        try:
            run(["git", "commit", "-m", f"mesh: push outbox for {node_id}"], cwd=tmp)
        except subprocess.CalledProcessError:
            pass  # nothing to commit

        run(["git", "push", "origin", branch], cwd=tmp)
        run(["git", "pull", "--rebase", "origin", branch], cwd=tmp)

        # copy peers' outbox to local INBOX
        peers_root = os.path.join(mailbox)
        pulled = 0
        for peer_id in os.listdir(peers_root):
            if peer_id == node_id:
                continue
            p_out = os.path.join(peers_root, peer_id, "outbox")
            if not os.path.isdir(p_out):
                continue
            for f in sorted(glob.glob(os.path.join(p_out, "*.json"))):
                dest = os.path.join(INBOX, f"{peer_id}__{os.path.basename(f)}")
                shutil.copy2(f, dest)
                pulled += 1
        print(f"[Σ] pulled {pulled} delta(s) from mailbox")

        # archive local outbox
        for pth in glob.glob(os.path.join(OUTBOX, "*.json")):
            shutil.move(pth, os.path.join(ARCHIVE, os.path.basename(pth)))
            print(f"[✓] archived {os.path.basename(pth)}")

def main():
    ensure_dirs()
    cfg = load_yaml(CONFIG_PATH)
    backend = cfg.get("backend", "http").lower()
    print(f"[i] mesh backend = {backend}")
    if backend == "http":
        run_http(cfg)
    elif backend == "git":
        run_git(cfg)
    else:
        print(f"[!] unknown backend: {backend}")

if __name__ == "__main__":
    main()
