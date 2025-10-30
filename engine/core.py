# engine/core.py
# SigmaAnalyzer — minimal core for evaluating network health from metrics
# Works with dicts like:
# {
#   "node_count": int,
#   "avg_uptime": float in [0,1],
#   "avg_latency": float (ms),
#   "success_ratio": float in [0,1]
# }

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
import json
import math
from pathlib import Path
from datetime import datetime


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _norm_latency_ms(ms: float) -> float:
    """
    Normalize latency (ms) to [0..1], higher is better.
    Piecewise:
      - <= 50ms  -> 1.00
      - 50..100  -> 0.9 .. 1.0 (linear)
      - 100..500 -> 0.1 .. 0.9 (linear)
      - >= 500   -> 0.05
    """
    if ms <= 50:
        return 1.0
    if ms <= 100:
        # 50 ->1.0, 100->0.9
        return 1.0 - 0.1 * ((ms - 50) / 50.0)
    if ms <= 500:
        # 100->0.9, 500->0.1
        return 0.9 - 0.8 * ((ms - 100) / 400.0)
    return 0.05


def _equity_from_nodes(n: int) -> float:
    """
    Very rough equity proxy from node_count.
    Uses log scaling with saturation:  1 - exp(-n/k)
    Choose k≈150 to reach ~0.63 at 150 nodes, ~0.86 at 300, ~0.95 at 450+.
    """
    if n <= 0:
        return 0.0
    k = 150.0
    return _clamp(1.0 - math.exp(-n / k))


@dataclass
class SigmaAnalyzer:
    # weights must sum ~1.0
    weights: Dict[str, float] = field(default_factory=lambda: {
        "stability": 0.35,   # uptime & success
        "latency":   0.25,   # lower ms => higher score
        "resilience":0.25,   # conservative proxy from success/uptime
        "equity":    0.15,   # decentralization proxy from node count
    })

    verdict_bands: Tuple[Tuple[float, str], ...] = (
        (85.0, "Excellent"),
        (70.0, "Healthy"),
        (50.0, "Watch"),
        (0.0,  "Critical"),
    )

    def evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute component scores, weighted overall score, verdict & advice.
        metrics keys expected:
          - node_count (int)
          - avg_uptime (0..1)
          - avg_latency (ms)
          - success_ratio (0..1)
        """
        node_count   = int(metrics.get("node_count", 0))
        avg_uptime   = float(metrics.get("avg_uptime", 0.0))
        avg_latency  = float(metrics.get("avg_latency", 0.0))
        success_rat  = float(metrics.get("success_ratio", 0.0))

        # Component scores in [0..1]
        stability   = _clamp(avg_uptime * success_rat)               # strict: both must be high
        latency     = _clamp(_norm_latency_ms(avg_latency))
        resilience  = _clamp(min(avg_uptime, success_rat) * 1.05)    # slightly optimistic clip
        equity      = _clamp(_equity_from_nodes(node_count))

        comp = {
            "stability": round(stability, 4),
            "latency": round(latency, 4),
            "resilience": round(resilience, 4),
            "equity": round(equity, 4),
        }

        # Weighted overall score on 100
        overall = 100.0 * sum(comp[k] * self.weights.get(k, 0.0) for k in comp)
        overall = round(overall, 2)

        verdict = self._verdict_for(overall)
        advice  = self._advice(comp, node_count, avg_latency)

        return {
            "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs": {
                "node_count": node_count,
                "avg_uptime": avg_uptime,
                "avg_latency_ms": avg_latency,
                "success_ratio": success_rat,
            },
            "component_scores": comp,               # 0..1
            "weights": self.weights,                # echo for transparency
            "overall_score": overall,               # 0..100
            "verdict": verdict,
            "advice": advice,
        }

    def evaluate_from_file(self, path: str | Path) -> Dict[str, Any]:
        """
        Convenience: accept a Skywire vitals JSON (with "payloads")
        and derive aggregate metrics before scoring.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        payloads = raw.get("payloads", [])
        n = len(payloads)

        # Extract tolerant fields
        def get(p: dict, k: str, default: float = 0.0) -> float:
            v = p.get(k, default)
            try:
                return float(v)
            except Exception:
                return default

        avg_uptime = sum(get(p, "uptime", 0.0) for p in payloads) / max(n, 1)
        avg_latency = sum(get(p, "latency_ms", 0.0) for p in payloads) / max(n, 1)
        success_ratio = sum(get(p, "success_ratio", 0.0) for p in payloads) / max(n, 1)

        return self.evaluate({
            "node_count": n,
            "avg_uptime": avg_uptime,
            "avg_latency": avg_latency,
            "success_ratio": success_ratio,
        })

    # -------------- internals --------------

    def _verdict_for(self, overall: float) -> str:
        for threshold, label in self.verdict_bands:
            if overall >= threshold:
                return label
        return "Unknown"

    def _advice(self, comp: Dict[str, float], node_count: int, avg_latency_ms: float) -> list[str]:
        adv: list[str] = []
        if comp["stability"] < 0.7:
            adv.append("Increase successful task completion and uptime across nodes.")
        if comp["latency"] < 0.7:
            adv.append(f"Reduce average latency (now ~{int(avg_latency_ms)} ms); optimize routes/peers.")
        if comp["resilience"] < 0.7:
            adv.append("Improve fault tolerance; ensure retries/backoff and node self-healing.")
        if comp["equity"] < 0.7:
            target = 300 if node_count < 300 else node_count + 100
            adv.append(f"Increase participating nodes (now {node_count}); aim for ~{target}+.")
        if not adv:
            adv.append("Maintain current posture; monitor trend lines and anomalies.")
        return adv
