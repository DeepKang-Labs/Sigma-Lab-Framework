# models/local_stats_model.py
from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class OnlineStat:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # somme des carrés centrés (pour variance)

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def var(self) -> float:
        return self.m2 / self.n if self.n > 1 else 0.0

    def to_dict(self) -> Dict:
        return {"n": self.n, "mean": self.mean, "var": self.var}

    @classmethod
    def from_dict(cls, d: Dict) -> "OnlineStat":
        obj = cls()
        obj.n = int(d.get("n", 0))
        obj.mean = float(d.get("mean", 0.0))
        # recalcul m2 à partir de var*n (approx si n<2)
        v = float(d.get("var", 0.0))
        obj.m2 = v * obj.n
        return obj


class LocalStatsModel:
    """Petit modèle ‘online’ sur 5 features simples."""

    FEATURE_KEYS = ["proxies", "vpn", "transports", "dmsg_entries", "rf_status_ok"]

    def __init__(self) -> None:
        self.stats: Dict[str, OnlineStat] = {k: OnlineStat() for k in self.FEATURE_KEYS}

    def update_with_sample(self, sample: Dict[str, float]) -> None:
        for k in self.FEATURE_KEYS:
            x = float(sample.get(k, 0.0))
            self.stats[k].update(x)

    def state_dict(self) -> Dict:
        return {k: self.stats[k].to_dict() for k in self.FEATURE_KEYS}

    def load_state_dict(self, d: Dict) -> None:
        for k in self.FEATURE_KEYS:
            if k in d:
                self.stats[k] = OnlineStat.from_dict(d[k])

    def delta_against(self, old: Dict) -> Dict:
        """Delta = nouvelle_mean - ancienne_mean (par feature) + meta(n)."""
        delta = {"type": "local_stats_delta", "version": 1, "features": {}}
        for k in self.FEATURE_KEYS:
            prev_m = float(old.get(k, {}).get("mean", 0.0))
            now_m = self.stats[k].mean
            delta["features"][k] = {"d_mean": now_m - prev_m, "n": self.stats[k].n}
        return delta
