#!/usr/bin/env python3
# ============================================================
# Network Bridge (Generic) — Skywire & Fiber (NESS)
# MIT License — DeepKang Labs
# ============================================================

from __future__ import annotations

import os
import json
import time
import hashlib
import yaml
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher


# ---------------- Utility ----------------

def clip01(x: float) -> float:
    """Clamp a float to [0,1]."""
    return float(max(0.0, min(1.0, x)))


def _similar(a: str, b: str) -> float:
    """Lightweight fuzzy similarity for keyword matching."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _hash_file(path: str) -> str:
    """SHA256 of file; empty string if not found."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()
    except Exception:
        return ""


# ---------------- Data classes ----------------

@dataclass
class TensionMapping:
    mapping_id: str
    network_tension: str
    sigma_dimensions: List[str]
    stakeholder_impact: Dict[str, float]


# ---------------- Engine ----------------

class NetworkBridge:
    """
    Generic bridge:
      - loads discovery data (real or demo)
      - transforms to SIGMA contexts via YAML mappings
      - can export contexts and build an integration report

    formula_eval_mode:
      - "linear":    use derive_risk_linear only (no external deps)
      - "simple":    allow short expressions via simpleeval (lazy import)
      - "auto":      prefer linear when present, else expressions
    """

    def __init__(
        self,
        discovery_kit_path: str,
        sigma_config_path: str,
        mappings_path: str,
        export_contexts_dir: Optional[str] = None,
        network_name: str = "skywire",
        formula_eval_mode: str = "auto",  # auto | linear | simple
    ):
        self.discovery_kit_path = discovery_kit_path
        self.sigma_config_path = sigma_config_path
        self.mappings_path = mappings_path
        self.export_contexts_dir = export_contexts_dir
        self.network_name = network_name
        self.formula_eval_mode = formula_eval_mode

        self._aliases: Dict[str, Any] = {}
        self._mappings_raw: Dict[str, Any] = {}
        self.mappings: List[TensionMapping] = self._load_mappings_yaml(mappings_path)

    # ---------- Loading ----------

    def _load_mappings_yaml(self, path: str) -> List[TensionMapping]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self._mappings_raw = data
        self._aliases = data.get("aliases", {})
        out: List[TensionMapping] = []
        for m in data.get("mappings", []):
            out.append(
                TensionMapping(
                    mapping_id=m["mapping_id"],
                    network_tension=(
                        m.get("skywire_tension")
                        or m.get("fiber_tension")
                        or m.get("network_tension", "")
                    ),
                    sigma_dimensions=m["sigma_dimensions"],
                    stakeholder_impact=m.get("stakeholder_impact", {}),
                )
            )
        return out

    def load_discovery_data(self) -> Dict[str, Any]:
        """Load discovery/decision_mapper.yaml, else return a deterministic demo."""
        path = os.path.join(self.discovery_kit_path, "decision_mapper.yaml")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return self._get_demo_discovery_data()

    def _get_demo_discovery_data(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "discovery_date": time.strftime("%Y-%m-%d"),
            "decision_points": [
                {
                    "decision_id": "dp_demo_001",
                    "description": "Balance latency vs. decentralization for routing",
                    "context": f"Demo context for {self.network_name}",
                    "stakeholders": ["relay_nodes", "end_users", "core_devs"],
                    "governance_dimensions": {
                        "technical_complexity": 6,
                        "economic_impact": 5,
                        "user_experience_impact": 8,
                        "security_implications": 4,
                    },
                    "current_pain_level": 7,
                }
            ],
        }

    # ---------- Validation ----------

    def validate_discovery(self, data: Dict[str, Any]) -> List[str]:
        """Basic shape & range validation for discovery data."""
        warnings: List[str] = []
        for dp in data.get("decision_points", []):
            if "decision_id" not in dp:
                warnings.append("Decision without 'decision_id'")
            gd = dp.get("governance_dimensions", {})
            for k in ["technical_complexity", "economic_impact", "user_experience_impact", "security_implications"]:
                v = gd.get(k, None)
                if v is None or not (0 <= v <= 10):
                    warnings.append(f"{dp.get('decision_id','?')}: {k} out of [0..10] or missing")
        return warnings

    # ---------- Mapping & derivation ----------

    def _find_best_mapping(self, description: str) -> Dict[str, Any]:
        """Pick the mapping whose keywords best match the decision description."""
        best = None
        best_score = 0.0
        desc = (description or "").lower()
        for m in self._mappings_raw.get("mappings", []):
            kw = m.get("keywords", [])
            score = max((_similar(desc, k) for k in kw), default=0.0)
            if score > best_score:
                best, best_score = m, score
        return best or (self._mappings_raw.get("mappings", [])[0])

    def _normalize_stakeholder(self, name: str) -> str:
        return self._aliases.get("stakeholders", {}).get(name, name)

    def _safe_eval_expr(self, expr: str, env: Dict[str, float]) -> float:
        """
        Evaluate short expressions if enabled.
        - In 'linear' mode or on any error, return 0.5.
        """
        if self.formula_eval_mode == "linear":
            return 0.5
        if not expr:
            return 0.5
        try:
            # Lazy import to avoid hard dependency when not needed
            from simpleeval import simple_eval  # type: ignore
        except Exception:
            return 0.5
        try:
            val = simple_eval(expr, names=env)
            return clip01(float(val))
        except Exception:
            return 0.5

    def _derive_risks(self, decision: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, float]:
        """Compute SIGMA short/long/irreversibility risks via linear mapping or expressions."""
        gd = decision.get("governance_dimensions", {})
        env = {
            "technical_complexity": float(gd.get("technical_complexity", 0.0)),
            "economic_impact": float(gd.get("economic_impact", 0.0)),
            "user_experience_impact": float(gd.get("user_experience_impact", 0.0)),
            "security_implications": float(gd.get("security_implications", 0.0)),
        }

        # Path A: linear mapping (no expressions)
        lin = mapping.get("derive_risk_linear")
        if self.formula_eval_mode in ("linear", "auto") and lin:
            scale = float(lin.get("scale", 10.0))

            def sel(key: str) -> float:
                src = lin.get(key)
                if not src:
                    return 0.5
                val = env.get(src, 5.0)
                return clip01(val / scale)

            return {
                "short_term_risk": sel("short_term"),
                "long_term_risk": sel("long_term"),
                "irreversibility_risk": sel("irreversibility"),
            }

        # Path B: expressions (simpleeval)
        dr = mapping.get("derive_risk", {})
        if self.formula_eval_mode in ("simple", "auto") and dr:
            st = self._safe_eval_expr(dr.get("short_term", "0.5"), env)
            lt = self._safe_eval_expr(dr.get("long_term", "0.5"), env)
            irr = self._safe_eval_expr(dr.get("irreversibility", "0.5"), env)
            return {
                "short_term_risk": st if st <= 1 else clip01(st / 10.0),
                "long_term_risk": lt if lt <= 1 else clip01(lt / 10.0),
                "irreversibility_risk": irr if irr <= 1 else clip01(irr / 10.0),
            }

        # Fallback neutral
        return {"short_term_risk": 0.5, "long_term_risk": 0.5, "irreversibility_risk": 0.5}

    def _create_stakeholders(self, decision: Dict[str, Any], mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        base = mapping.get("stakeholder_impact", {})
        out: List[Dict[str, Any]] = []
        for st in decision.get("stakeholders", []):
            normalized = self._normalize_stakeholder(st)
            if normalized in base:
                out.append({
                    "name": normalized,
                    "vulnerability": clip01(0.6 if normalized in ("users", "end_users") else 0.5),
                    "impact_benefit": clip01(base[normalized]),
                })
            else:
                out.append({"name": normalized, "vulnerability": 0.5, "impact_benefit": 0.5})
        return out

    def _map_stability_risks(self, decision: Dict[str, Any]) -> Dict[str, float]:
        gd = decision.get("governance_dimensions", {})
        return {
            "technical_complexity": clip01(gd.get("technical_complexity", 0) / 10.0),
            "economic_instability": clip01(gd.get("economic_impact", 0) / 10.0),
            "ux_degradation": clip01(gd.get("user_experience_impact", 0) / 10.0),
        }

    def _map_resilience_features(self, decision: Dict[str, Any]) -> Dict[str, float]:
        gd = decision.get("governance_dimensions", {})
        return {
            "adaptability": clip01((10 - gd.get("technical_complexity", 5)) / 10.0),
            "economic_resilience": clip01((10 - gd.get("economic_impact", 5)) / 10.0),
            "security_resilience": clip01(gd.get("security_implications", 5) / 10.0),
        }

    # ---------- Transform → SIGMA contexts ----------

    def transform_to_sigma_contexts(self, discovery_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        contexts: List[Dict[str, Any]] = []
        for dp in discovery_data.get("decision_points", []):
            mapping = self._find_best_mapping(dp.get("description", ""))
            risks = self._derive_risks(dp, mapping)
            stakeholders = self._create_stakeholders(dp, mapping)
            ctx = {
                "name": f"{self.network_name.upper()}: {dp.get('description','(no description)')}",
                "short_term_risk": risks["short_term_risk"],
                "long_term_risk": risks["long_term_risk"],
                "irreversibility_risk": risks["irreversibility_risk"],
                "stakeholders": stakeholders,
                "stability_risks": self._map_stability_risks(dp),
                "resilience_features": self._map_resilience_features(dp),
                "metadata": {
                    "original_decision_id": dp.get("decision_id", "unknown"),
                    "pain_level": dp.get("current_pain_level", None),
                    "decision_frequency": dp.get("decision_frequency", None),
                    "discovery_source": "NetworkBridge demo"
                        if "dp_demo" in dp.get("decision_id", "")
                        else "Discovery",
                },
                "transform_provenance": {
                    "mappings_file": self.mappings_path,
                    "mapping_used": mapping.get("mapping_id", ""),
                    "risk_formulas": mapping.get("derive_risk", {}),
                    "risk_linear": mapping.get("derive_risk_linear", {}),
                    "formula_eval_mode": self.formula_eval_mode,
                }
            }
            contexts.append(ctx)
        return contexts

    def export_sigma_contexts(self, contexts: List[Dict[str, Any]]):
        if not self.export_contexts_dir:
            return
        os.makedirs(self.export_contexts_dir, exist_ok=True)
        for c in contexts:
            fn = f"{c['metadata']['original_decision_id']}.yaml"
            with open(os.path.join(self.export_contexts_dir, fn), "w", encoding="utf-8") as f:
                yaml.safe_dump(c, f, sort_keys=False, allow_unicode=True)

    # ---------- Recommendations & Report ----------

    def _generate_cross_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        recs: List[str] = []
        low_counts: Dict[str, int] = {}
        for r in results:
            for d, s in r.get("scores", {}).items():
                if s < 0.6:
                    low_counts[d] = low_counts.get(d, 0) + 1
        n = max(1, len(results))
        for d, k in low_counts.items():
            if k >= 0.5 * n:
                if d == "stability":
                    recs.append("Harden stability mechanisms (rate-limits, backpressure, circuit breakers).")
                elif d == "equity":
                    recs.append("Reduce disparity across stakeholder groups (rebalance incentives, fairness-aware rewards).")
                elif d == "non_harm":
                    recs.append("Mitigate high-risk vectors (rollback, kill-switch, incident response).")
                elif d == "resilience":
                    recs.append("Increase redundancy & failover; conduct recovery drills.")
        return recs

    def generate_integration_report(
        self,
        discovery_data: Dict[str, Any],
        sigma_results: List[Dict[str, Any]],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        stability_avg = 0.0
        if sigma_results:
            stability_avg = sum(r.get("scores", {}).get("stability", 0.0) for r in sigma_results) / len(sigma_results)
        veto_fail = [r for r in sigma_results if r.get("vetoes")]

        src_path = os.path.join(self.discovery_kit_path, "decision_mapper.yaml")
        report = {
            "integration_report": {
                "version": "1.0",
                "generated_by": "NetworkBridge",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "network": self.network_name,
                "summary": {
                    "total_decisions_analyzed": len(discovery_data.get("decision_points", [])),
                    "high_risk_decisions": len(veto_fail),
                    "average_stability_score": round(stability_avg, 3),
                },
                "inputs_digest": {
                    "discovery_file": src_path if os.path.exists(src_path) else "(demo)",
                    "discovery_sha256": _hash_file(src_path) if os.path.exists(src_path) else "(demo)",
                    "mappings_file": self.mappings_path,
                    "mappings_sha256": _hash_file(self.mappings_path),
                    "sigma_config": self.sigma_config_path,
                    "sigma_config_sha256": _hash_file(self.sigma_config_path),
                },
            },
            "discovery_phase": discovery_data,
            "sigma_analysis": sigma_results,
            "recommendations": self._generate_cross_recommendations(sigma_results),
        }
        if extra_meta:
            report["meta"] = extra_meta
        return report
```0
