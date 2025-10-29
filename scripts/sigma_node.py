# scripts/sigma_node.py
from __future__ import annotations
import os, json, datetime
from typing import Dict
import yaml

from scripts.skywire_client import SkywireClient
from scripts.protocols.sigma_mesh import write_delta, collect_inbox_deltas, federated_average
from models.local_stats_model import LocalStatsModel


def env_or(cfg: Dict, path: str, env_name: str, default):
    # path: "skywire.rpc_mode" etc.
    cur = cfg
    for part in path.split("."):
        cur = cur.get(part, {})
    return os.getenv(env_name, cur if cur else default)


def today_folder(root: str) -> str:
    d = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    path = os.path.join(root, d)
    os.makedirs(path, exist_ok=True)
    return path


def read_json_if(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    # --- charge config ---
    with open("configs/sigma_node.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    node_id = os.getenv("SIGMA_NODE_ID", cfg.get("node_id", "sigma-local-001"))

    rpc_mode   = env_or(cfg, "skywire.rpc_mode",   "SKYWIRE_RPC_MODE", "http")
    http_base  = env_or(cfg, "skywire.http_base",  "SKYWIRE_HTTP_BASE", "https://sd.skycoin.com")
    timeout_s  = int(os.getenv("SKYWIRE_TIMEOUT_S", cfg.get("skywire", {}).get("timeout_s", 8)))

    data_root  = os.getenv("SIGMA_DATA_ROOT", cfg.get("data_root", "data"))
    outbox_dir = os.getenv("SIGMA_MESH_OUTBOX", cfg.get("mesh", {}).get("outbox_dir", "mesh/outbox"))
    inbox_dir  = os.getenv("SIGMA_MESH_INBOX",  cfg.get("mesh", {}).get("inbox_dir",  "mesh/inbox"))
    agg_every  = int(os.getenv("SIGMA_MESH_AGG_EVERY", cfg.get("mesh", {}).get("aggregate_every", 3)))

    # --- collecte d’un échantillon ---
    client = SkywireClient(mode=rpc_mode, http_base=http_base, timeout_s=timeout_s)
    sample = client.ping_public()  # dict avec 5 features

    # --- chemin état du modèle (par date) ---
    day_dir = today_folder(data_root)
    state_path = os.path.join(day_dir, f"{node_id}_model_state.json")

    # --- charge ancien état / init modèle ---
    model = LocalStatsModel()
    prev_state = read_json_if(state_path) or {}

    if prev_state:
        model.load_state_dict(prev_state.get("stats", {}))

    # --- met à jour le modèle avec l’échantillon ---
    model.update_with_sample(sample)

    # --- new state + delta ---
    new_state = {
        "node_id": node_id,
        "stats": model.state_dict(),
        "last_sample": sample,
    }
    delta = model.delta_against(prev_state.get("stats", {}))

    write_json(state_path, new_state)

    # --- mesh: sortie de delta ---
    written = write_delta(outbox_dir, node_id, delta)

    # --- tentative d’agrégation si assez de deltas en inbox ---
    inbox = collect_inbox_deltas(inbox_dir)
    agg = None
    if len(inbox) >= agg_every:
        agg = federated_average(inbox)
        agg_path = os.path.join(day_dir, f"{node_id}_agg_delta.json")
        write_json(agg_path, agg)

    # --- statut court à STDOUT ---
    print("[SigmaNode] node:", node_id)
    print("[SigmaNode] sample:", sample)
    print("[SigmaNode] delta_written:", written)
    if agg:
        print("[SigmaNode] aggregated:", agg)

    # Option: produire un ‘heartbeat’ léger
    hb = {"node": node_id, "sample": sample, "delta_file": written}
    write_json(os.path.join(day_dir, f"{node_id}_heartbeat.json"), hb)


if __name__ == "__main__":
    main()
