#!/usr/bin/env python3
# ============================================================
# Sigma-Lab v4.2 - Procedural Diagnostic Core (MIT)
# Patched: add 'timestamp_utc' in audit and inline 'audit' in results
# ============================================================

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import math
import json
import random

# ---------------- Utils ----------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0

def logistic(x: float, k: float = 10.0, x0: float = 0.5) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - x0)))
    except Exception:
        return 0.5

def exp_value(x: float, a: float = 2.5) -> float:
    try:
        return clamp(1.0 - math.exp(-a * clamp(x)))
    except Exception:
        return clamp(x)

def gini(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = sorted([max(0.0, v) for v in values])
    n = len(arr)
    cum = 0.0
    for i, v in enumerate(arr, 1):
        cum += i * v
    total = sum(arr)
    if total == 0:
        return 0.0
    return (2.0 * cum) / (n * total) - (n + 1) / n

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

# ---------------- Data model ----------------

@dataclass
class Provenance:
    source: str
    doc_ref: Optional[str] = None
    rationale: Optional[str] = None

@dataclass
class Weights:
    non_harm: float = 0.45
    stability: float = 0.20
    resilience: float = 0.20
    equity: float = 0.15
    provenance: Optional[Provenance] = None

@dataclass
class Thresholds:
    non_harm_floor: float = 0.30
    veto_irreversibility: float = 0.70
    provenance: Optional[Provenance] = None

@dataclass
class Stakeholder:
    name: str
    vulnerability: float = 0.5
    impact_benefit: float = 0.5

@dataclass
class OptionContext:
    name: str
    short_term_risk: Optional[float] = None
    long_term_risk: Optional[float] = None
    irreversibility_risk: Optional[float] = None
    stakeholders: Any = field(default_factory=list)  # list[str] or list[Stakeholder]
    stability_risks: Dict[str, float] = field(default_factory=dict)
    resilience_features: Dict[str, float] = field(default_factory=dict)
    uncertainty: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# ---------------- Core ----------------

class SigmaLab:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = self._normalize_config(config)

    # ----- config & demo -----

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = {
            "weights": asdict(Weights(provenance=Provenance(
                "Hospital Ethics Board", "POL-H-12", "Clinical prioritization policy"
            ))),
            "thresholds": asdict(Thresholds(provenance=Provenance(
                "Hospital Ethics Board", "POL-H-09", "Harm floors & irreversibility veto"
            ))),
            "harm_model": {
                "base_weight": 0.7,
                "irreversibility_weight": 0.3,
                "base_agg": "max",        # "max" | "mean" | "weighted"
                "base_agg_weight": 0.5,   # used when base_agg == "weighted"
                "formula": "expected_harm = base_risk * clamp(base_weight + irreversibility_weight * irreversibility)",
                "rationale": (
                    "Expected-utility style amplification: irreversibility raises effective risk. "
                    "base_risk aggregated via max/mean/weighted (configurable)."
                ),
            },
            "stability_model": {"mean_weight": 0.6, "max_weight": 0.4},
            "resilience_model": {"avg_weight": 0.85, "breadth_weight": 0.15},
            "guardrails": {
                "max_parameter_adjustment": 0.3,
                "policy_lock": False,
                "require_cross_validation": False,
            },
            "value_functions": {
                "non_harm": {"kind": "logistic", "params": {"k": 10.0, "x0": 0.5}},
                "stability": {"kind": "linear", "params": {}},
                "resilience": {"kind": "piecewise", "params": {"points": [(0, 0), (0.5, 0.6), (1, 1)]}},
                "equity": {"kind": "exp", "params": {"a": 2.5}},
            },
            "verdict_acceptance_threshold": 0.65,
        }
        for k, v in (config or {}).items():
            cfg[k] = v
        # ensure provenance are dicts
        if isinstance(cfg["weights"].get("provenance"), Provenance):
            cfg["weights"]["provenance"] = asdict(cfg["weights"]["provenance"])
        if isinstance(cfg["thresholds"].get("provenance"), Provenance):
            cfg["thresholds"]["provenance"] = asdict(cfg["thresholds"]["provenance"])
        return cfg

    @staticmethod
    def demo_context(domain: str = "public") -> Tuple[Dict[str, Any], OptionContext]:
        cfg = {}
        ctx = OptionContext(
            name="demo",
            short_term_risk=0.3,
            long_term_risk=0.25,
            irreversibility_risk=0.5,
            stakeholders=[
                Stakeholder("patients", 0.8, 0.6),
                Stakeholder("clinicians", 0.4, 0.5),
                Stakeholder("administration", 0.3, 0.4),
            ],
            stability_risks={"complexity": 0.3, "interop": 0.25},
            resilience_features={"redundancy": 0.6, "fallbacks": 0.5},
        )
        return cfg, ctx

    # ----- validation & coercion -----

    def _coerce_stakeholders(self, arr: Any) -> List[Stakeholder]:
        out: List[Stakeholder] = []
        if not arr:
            return [Stakeholder("public", 0.5, 0.5)]
        for s in arr:
            if isinstance(s, Stakeholder):
                out.append(Stakeholder(s.name, clamp(s.vulnerability), clamp(s.impact_benefit)))
            elif isinstance(s, dict):
                out.append(Stakeholder(
                    name=str(s.get("name", "stakeholder")),
                    vulnerability=clamp(s.get("vulnerability", 0.5)),
                    impact_benefit=clamp(s.get("impact_benefit", 0.5)),
                ))
            else:
                out.append(Stakeholder(str(s), 0.5, 0.5))
        return out

    def validate_context(self, ctx: OptionContext) -> List[str]:
        errs: List[str] = []
        st = clamp(ctx.short_term_risk if ctx.short_term_risk is not None else 0.5)
        lt = clamp(ctx.long_term_risk if ctx.long_term_risk is not None else 0.5)
        irr = clamp(ctx.irreversibility_risk if ctx.irreversibility_risk is not None else 0.5)
        ctx.short_term_risk = st
        ctx.long_term_risk = lt
        ctx.irreversibility_risk = irr
        ctx.stakeholders = self._coerce_stakeholders(ctx.stakeholders)
        return errs

    # ----- value models -----

    def _value_non_harm(self, expected_harm: float) -> float:
        vf = self.cfg["value_functions"]["non_harm"]
        if vf["kind"] == "logistic":
            k = float(vf["params"].get("k", 10.0))
            x0 = float(vf["params"].get("x0", 0.5))
            return clamp(1.0 - logistic(expected_harm, k=k, x0=x0))
        return clamp(1.0 - expected_harm)

    def _value_stability(self, val: float) -> float:
        return clamp(val)

    def _value_resilience(self, avg: float, breadth: float) -> float:
        pw = self.cfg["value_functions"]["resilience"]["params"]["points"]
        x = clamp(avg)
        for (x1, y1), (x2, y2) in zip(pw[:-1], pw[1:]):
            if x1 <= x <= x2:
                if x2 == x1:
                    return clamp(y1)
                t = (x - x1) / (x2 - x1)
                return clamp(y1 + t * (y2 - y1))
        return clamp(pw[-1][1])

    def _value_equity(self, benefits: List[float]) -> float:
        g = clamp(gini(benefits))
        return clamp(exp_value(1.0 - g, a=2.5))

    # ----- main API -----

    def diagnose(self, ctx: OptionContext, verdict_opt_in: bool = True) -> Dict[str, Any]:
        warnings: List[str] = []
        input_errors = self.validate_context(ctx)

        st, lt, irr = ctx.short_term_risk, ctx.long_term_risk, ctx.irreversibility_risk
        hm = self.cfg["harm_model"]
        base_agg = hm.get("base_agg", "max")
        if base_agg == "mean":
            base_risk = (st + lt) / 2.0
        elif base_agg == "weighted":
            w = clamp(hm.get("base_agg_weight", 0.5))
            base_risk = clamp(w * st + (1 - w) * lt)
        else:
            base_risk = max(st, lt)

        expected_harm = clamp(base_risk * clamp(hm.get("base_weight", 0.7) +
                                                hm.get("irreversibility_weight", 0.3) * irr))

        st_mod = self.cfg["stability_model"]
        st_val = clamp(st_mod.get("mean_weight", 0.6) * (st + lt) / 2.0 +
                       st_mod.get("max_weight", 0.4) * max(st, lt))

        res_mod = self.cfg["resilience_model"]
        res_avg = clamp(sum(ctx.resilience_features.values()) / max(1, len(ctx.resilience_features)))
        res_breadth = clamp(len([v for v in ctx.resilience_features.values() if v > 0.5]) /
                            max(1, len(ctx.resilience_features)))

        benefits = [clamp(s.impact_benefit) for s in ctx.stakeholders]
        eq_val = self._value_equity(benefits)

        scores = {
            "non_harm": self._value_non_harm(expected_harm),
            "stability": self._value_stability(1.0 - st_val if st_val > 0.5 else 1.0 - 0.5 * st_val),
            "resilience": self._value_resilience(res_avg, res_breadth),
            "equity": eq_val,
        }

        details = {
            "non_harm": {"expected_harm": expected_harm, "base_risk": base_risk, "irr": irr,
                         "factor": clamp(hm.get("base_weight", 0.7) + hm.get("irreversibility_weight", 0.3) * irr)},
            "stability": {"mean": (st + lt) / 2.0, "max": max(st, lt), "raw": st_val},
            "resilience": {"avg": res_avg, "breadth": res_breadth},
            "equity": {"gini": gini(benefits), "benefits": benefits},
        }

        vetoes: List[str] = []
        thr = self.cfg["thresholds"]
        if scores["non_harm"] < thr.get("non_harm_floor", 0.3):
            vetoes.append(f"non-harm floor violated ({scores['non_harm']:.2f} < {thr.get('non_harm_floor', 0.3):.2f})")
        if irr > thr.get("veto_irreversibility", 0.7):
            vetoes.append(f"irreversibility veto ({irr:.2f} > {thr.get('veto_irreversibility', 0.7):.2f})")

        verdict = None
        if verdict_opt_in:
            mean_score = sum(scores.values()) / 4.0
            if not vetoes and mean_score >= self.cfg.get("verdict_acceptance_threshold", 0.65):
                verdict = "ACCEPT"

        ts = now_iso_utc()
        result: Dict[str, Any] = {
            "status": "success",
            "warnings": warnings,
            "input_errors": input_errors,
            "scores": scores,
            "details": details,
            "vetoes": vetoes,
            "verdict": verdict,
            "timestamp": ts,
            "semantics": {
                "note": "Procedural diagnostic only. No moral truth claim.",
                "verdict_threshold": self.cfg.get("verdict_acceptance_threshold", 0.65),
            },
            "diagnostic": {"scores": scores, "vetoes": vetoes},
        }

        # Inline audit required by tests
        result["audit"] = {
            "timestamp_utc": ts,
            "weights": self.cfg["weights"],
            "thresholds": self.cfg["thresholds"],
            "verdict_threshold": self.cfg.get("verdict_acceptance_threshold", 0.65),
        }

        return result

    # ----- audit export -----

    def export_audit_trail(self, result: Dict[str, Any]) -> Dict[str, Any]:
        weights = self.cfg.get("weights", {})
        thresholds = self.cfg.get("thresholds", {})

        cfg_blob = {
            "weights": weights,
            "thresholds": thresholds,
            "harm_model": self.cfg.get("harm_model", {}),
            "stability_model": self.cfg.get("stability_model", {}),
            "resilience_model": self.cfg.get("resilience_model", {}),
            "guardrails": self.cfg.get("guardrails", {}),
            "value_functions": self.cfg.get("value_functions", {}),
            "verdict_acceptance_threshold": self.cfg.get("verdict_acceptance_threshold", 0.65),
        }

        run_ctx = {
            "verdict_threshold": self.cfg.get("verdict_acceptance_threshold", 0.65),
            "thresholds": thresholds,
            "weights": weights,
        }

        audit = {
            "schema_version": "1.0",
            "timestamp": now_iso_utc(),
            "timestamp_utc": now_iso_utc(),
            "config_snapshot": cfg_blob,
            "run_context": run_ctx,
            "result": result,
        }
        # sanity JSON
        json.loads(json.dumps(audit, ensure_ascii=False, sort_keys=True))
        return audit

# --------------- helper (legacy external API parity) ----------------

def demo_context(domain: str = "public"):
    lab = SigmaLab({})
    return lab._normalize_config({}), SigmaLab.demo_context(domain)[1]
