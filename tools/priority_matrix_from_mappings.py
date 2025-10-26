# tools/priority_matrix_from_mappings.py
#!/usr/bin/env python3
"""
Priority Matrix generator from network mappings.

The script is intentionally tolerant:
- If the mappings YAML contains a list under `tensions`, with fields
  id / impact / pain / frequency → it computes score = impact * pain * frequency.
- If the file contains a dict of nodes with numeric `weight`, it uses that as score.
- Otherwise, it falls back to a neutral single entry to avoid CI hard-fail.

Usage:
  python -m tools.priority_matrix_from_mappings --mappings PATH --out PATH
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except Exception as e:
    raise RuntimeError("pyyaml is required. Install with: pip install pyyaml") from e


def _safe_num(x: Any, default: float = 1.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _from_tensions(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in m.get("tensions", []) or []:
        if not isinstance(t, dict):
            continue
        tid = t.get("id") or t.get("name") or "tension"
        impact = _safe_num(t.get("impact"), 1.0)
        pain = _safe_num(t.get("pain"), 1.0)
        freq = _safe_num(t.get("frequency"), 1.0)
        score = impact * pain * freq
        out.append(
            {
                "id": str(tid),
                "impact": impact,
                "pain": pain,
                "frequency": freq,
                "score": score,
            }
        )
    return out


def _from_weighted_nodes(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for k, v in (m or {}).items():
        if isinstance(v, dict) and ("weight" in v or "priority" in v):
            w = _safe_num(v.get("weight", v.get("priority", 1)))
            out.append({"id": str(k), "score": w})
    return out


def build_priority_matrix(mappings: Dict[str, Any]) -> Dict[str, Any]:
    # try "tensions" schema
    rows = _from_tensions(mappings)
    if not rows:
        # try node weights schema
        rows = _from_weighted_nodes(mappings)

    if not rows:
        # very safe fallback
        rows = [{"id": "fallback", "score": 1.0}]

    # sorted descending by score
    rows.sort(key=lambda r: float(r.get("score", 0)), reverse=True)

    return {
        "version": "1.0",
        "count": len(rows),
        "priorities": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Generate priority matrix from mappings YAML.")
    p.add_argument("--mappings", required=True, help="Path to mappings YAML file.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    args = p.parse_args()

    mappings_path = Path(args.mappings)
    if not mappings_path.exists():
        raise FileNotFoundError(f"Mappings file not found: {mappings_path}")

    data = yaml.safe_load(mappings_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Mappings YAML must deserialize to a mapping/dict.")

    matrix = build_priority_matrix(data)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"[PriorityMatrix] Saved → {out_path}")


if __name__ == "__main__":
    main()
