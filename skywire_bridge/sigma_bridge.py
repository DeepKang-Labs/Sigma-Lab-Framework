# ======================================
# File: skywire_bridge/sigma_bridge.py
# ======================================
#!/usr/bin/env python3
"""
SIGMA-BRIDGE v1.2 — SkyWire Discovery → SIGMA-LAB connector
- Mappings externalisés (YAML) avec provenance
- Dérivation transparente des risques depuis dimensions governance Discovery
- Validation, export des contextes SIGMA, benchmark optionnel
- Recommandations actionnables + traçabilité complète
- Licence: MIT (aligné avec le dépôt principal)
"""
from __future__ import annotations

import os, json, yaml, hashlib, time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher

def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _hash_file(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()
    except Exception:
        return ""

@dataclass
class TensionMapping:
    mapping_id: str
    skywire_tension: str
    sigma_dimensions: List[str]
    stakeholder_impact: Dict[str, float]

class SigmaBridge:
    def __init__(
        self,
        discovery_kit_path: str,
        sigma_config_path: str,
        mappings_path: str = "./skywire_bridge/skywire_mappings.yaml",
        export_contexts_dir: Optional[str] = None,
    ):
        self.discovery_kit_path = discovery_kit_path
        self.sigma_config_path = sigma_config_path
        self.mappings_path = mappings_path
        self.export_contexts_dir = export_contexts_dir
        self._aliases: Dict[str, Dict[str, str]] = {}
        self._mappings_raw: Dict[str, Any] = {}
        self.mappings = self._load_tension_mappings_yaml(mappings_path)

    def _load_tension_mappings_yaml(self, path: str) -> List[TensionMapping]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self._mappings_raw = data
        self._aliases = data.get("aliases", {})
        out: List[TensionMapping] = []
        for m in data.get("mappings", []):
            out.append(
                TensionMapping(
                    mapping_id=m["mapping_id"],
                    skywire_tension=m["skywire_tension"],
                    sigma_dimensions=m["sigma_dimensions"],
                    stakeholder_impact=m.get("stakeholder_impact", {}),
                )
            )
        return out

    def load_discovery_data(self) -> Dict[str, Any]:
        path = os.path.join(self.discovery_kit_path, "decision_mapper.yaml")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return self._get_demo_discovery_data()

    def _get_demo_discovery_data(self) -> Dict[str, Any]:
        return {
            "version": "1.0",
            "discovery_date": "2025-10-25",
            "decision_points": [
                {
                    "decision_id": "dp_001",
                    "description": "Arbitrage latence vs énergie (routing temps-réel)",
                    "context": "Path selection & congestion",
                    "stakeholders": ["nœuds relais", "utilisateurs finaux", "développeurs core"],
                    "conflicting_objectives": [
                        {"objective": "Latence minimale", "metric": "ms", "target": "<50ms"},
                        {"objective": "Efficacité énergétique", "metric": "joules/paquet", "target": "minimiser"},
                    ],
                    "current_pain_level": 8,
                    "decision_frequency": "continuelle",
                    "governance_dimensions": {
                        "technical_complexity": 7,
                        "economic_impact": 6,
                        "user_experience_impact": 9,
                        "security_implications": 4,
                    },
                }
            ],
        }

    def validate_discovery(self, data: Dict[str, Any]) -> List[str]:
        warnings: List[str] = []
        for dp in data.get("decision_points", []):
            if "decision_id" not in dp:
                warnings.append("Decision sans decision_id")
            gd = dp.get("governance_dimensions", {})
            for k in ["technical_complexity", "economic_impact", "user_experience_impact", "security_implications"]:
                v = gd.get(k, None)
                if v is None or not (0 <= v <= 10):
                    warnings.append(f"{dp.get('decision_id','?')}: {k} hors bornes [0..10] ou manquant")
        return warnings

    def _find_best_mapping(self, description: str) -> Dict[str, Any]:
        best = None
        best_score = 0.0
        desc = description.lower()
        for m in self._mappings_raw.get("mappings", []):
            kw = m.get("keywords", [])
            score = max((_similar(desc, k) for k in kw), default=0.0)
            if score > best_score:
                best, best_score = m, score
        return best or self._mappings_raw["mappings"][0]

    def _derive_risks(self, decision: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, float]:
        gd = decision.get("governance_dimensions", {})
        env = {
            k: float(gd.get(k, 0.0))
            for k in ["technical_complexity", "economic_impact", "user_experience_impact", "security_implications"]
        }
        def _eval(expr: str) -> float:
            try:
                return clip01(eval(expr, {}, env))
            except Exception:
                return 0.5
        dr = mapping.get("derive_risk", {})
        return {
            "short_term_risk": _eval(dr.get("short_term", "0.5")),
            "long_term_risk": _eval(dr.get("long_term", "0.5")),
            "irreversibility_risk": _eval(dr.get("irreversibility", "0.5")),
        }

    def _normalize_stakeholder(self, name: str) -> str:
        return self._aliases.get("stakeholders", {}).get(name, name)

    def _create_stakeholders(self, decision: Dict[str, Any], mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        base = mapping.get("stakeholder_impact", {})
        out: List[Dict[str, Any]] = []
        for st in decision.get("stakeholders", []):
            normalized = self._normalize_stakeholder(st)
            if normalized in base:
                out.append({
                    "name": normalized,
                    "vulnerability": clip01(0.6 if normalized == "users" else 0.5),
                    "impact_benefit": clip01(base[normalized]),
                })
            else:
                out.append({
                    "name": normalized,
                    "vulnerability": 0.5,
                    "impact_benefit": 0.5,
                })
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

    def transform_to_sigma_contexts(self, discovery_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        contexts: List[Dict[str, Any]] = []
        for dp in discovery_data.get("decision_points", []):
            mapping = self._find_best_mapping(dp.get("description", ""))
            risks = self._derive_risks(dp, mapping)
            stakeholders = self._create_stakeholders(dp, mapping)
            ctx = {
                "name": f"SkyWire: {dp.get('description','(no description)')}",
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
                    "discovery_source": "SkyWire Discovery Kit"
                },
                "transform_provenance": {
                    "mappings_file": self.mappings_path,
                    "mapping_used": mapping.get("mapping_id", ""),
                    "risk_formulas": mapping.get("derive_risk", {}),
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
                    recs.append("Strengthen protocol-level stability (rate-limits, backpressure, circuit breakers).")
                elif d == "equity":
                    recs.append("Reduce disparity across stakeholder groups (rebalance incentives, fairness-aware policies).")
                elif d == "non_harm":
                    recs.append("Mitigate high-risk vectors (rollbacks, kill-switches, rapid incident response).")
                elif d == "resilience":
                    recs.append("Increase redundancy & failover; schedule recovery drills.")
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
        veto_fail = [r for r in sigma_results if any(r.get("vetoes", []))]

        src_path = os.path.join(self.discovery_kit_path, "decision_mapper.yaml")
        report = {
            "integration_report": {
                "version": "1.2",
                "generated_by": "SIGMA-BRIDGE",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
