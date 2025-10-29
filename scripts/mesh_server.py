#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesh_server.py — FastAPI endpoints for Sigma Mesh.

Env:
  ARTIFACTS_ROOT (default: artifacts)
  SIGMA_TOKEN    (shared secret)
  SIGMA_NODE_ID  (node id)
"""
import os, glob
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import FileResponse
import uvicorn

ROOT = os.environ.get("ARTIFACTS_ROOT", "artifacts")
TOKEN = os.environ.get("SIGMA_TOKEN", "CHANGE_ME_SECRET")
NODE_ID = os.environ.get("SIGMA_NODE_ID", "node-prod")

INBOX  = os.path.join(ROOT, "mesh", "inbox")
OUTBOX = os.path.join(ROOT, "mesh", "outbox")
os.makedirs(INBOX, exist_ok=True)
os.makedirs(OUTBOX, exist_ok=True)

app = FastAPI(title=f"Sigma Mesh Node ({NODE_ID})")

def auth(tok: str):
    if not tok or tok != TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/api/mesh/inbox")
async def inbox(file: UploadFile = File(...), x_sigma_token: str = Header(None), x_sigma_node: str = Header(None)):
    auth(x_sigma_token)
    target = os.path.join(INBOX, file.filename)
    with open(target, "wb") as f:
        f.write(await file.read())
    return {"ok": True, "stored_as": file.filename, "from": x_sigma_node}

@app.get("/api/mesh/outbox/list")
def outbox_list(x_sigma_token: str = Header(None), x_sigma_node: str = Header(None)):
    auth(x_sigma_token)
    files = [os.path.basename(p) for p in glob.glob(os.path.join(OUTBOX, "*.json"))]
    return {"node": NODE_ID, "files": files}

@app.get("/api/mesh/outbox/get/{fname}")
def outbox_get(fname: str, x_sigma_token: str = Header(None), x_sigma_node: str = Header(None)):
    auth(x_sigma_token)
    path = os.path.join(OUTBOX, fname)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="application/json", filename=fname)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
