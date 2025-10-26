#!/usr/bin/env python3
# ===============================================================
# SIGMA-LAB v4.2 — Procedural Diagnostic Framework (test-friendly)
# Authors: DeepKang Labs
# License: MIT
# ===============================================================

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import argparse
import json
import math
import numpy as np

# ===============================================================
# UTILITIES
# ===============================================================

def clip01(x: float) -> float:
    try:
        xv = float(x)
    except Exception:
        return 0.0
    return float(max(0.0, min(1.0, xv)))

def safe_mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ===============================================================
# PROVENANCE
# ===============================================================

@dataclass
class Provenance:
    source: str
    doc_ref: Optional[str] = None
    rationale: Optional[str] = None

# ===============================================================
# VALUE FUNCTIONS
# ===============================================================

@dataclass
class ValueFunction:
    """
    kind: 'linear' | 'exp' | 'logistic' | 'piecewise'
    exp: f(x) = 1 - exp(-a x), a>0 (bounded in [0,1))
    logistic: f(x) = 1/(1+exp(-k(x-x0)))
    piecewise: params{'points':[(x,y),...]} with 0<=x,y<=1
    """
    kind: str = "linear"
    params: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[Provenance] = None

    def __call__(self, x: float) -> float:
        x = clip01(x)
        k = self.kind.lower()
        if k == "linear":
            return x
        elif k == "exp":
            a = float(self.params.get("a", 3.0))
            return clip01(1.0 - math.exp(-a * x))
        elif k == "logistic":
            k_ = float(self.params.get("k", 12.0))
            x0 = float(self.params.get("x0", 0.5))
            return clip01(1.0 / (1.0 + math.exp(-k_ * (x - x0))))
        elif k == "piecewise":
            raw_pts = self.params.get("points", [(0, 0), (1, 1)])
            pts = sorted([(float(a), float(b)) for a, b in raw_pts], key=lambda t: t[0])
            xs, ys = zip(*pts)
            return float(np.interp(x, xs, ys))
        return x

# ===============================================================
# ETHICAL WEIGHTS
# ===============================================================

@dataclass
class EthicalWeights:
    non_harm: float = 0.4
    stability: float = 0.2
    resilience: float = 0.2
    equity: float = 0.2
    provenance: Optional[Provenance] = None

    def norm(self) -> Dict[str, float]:
        v = np.array([
            abs(float(self.non_harm)),
            abs(float(self.stability)),
            abs(float(self.resilience)),
            abs(float(self.equity)),
        ], dtype=float)
        s = float(v.sum())
        if s < 1e-12:
            return {"non_harm": 0.25, "stability": 0.25, "resilience": 0.25, "equity": 0.25}
        return {
            "non_harm": float(v[0] / s),
            "stability": float(v[1] / s),
            "resilience": float(v[2] / s),
            "equity": float(v[3] / s),
        }

# ===============================================================
# THRESHOLDS & GUARDRAILS
# ===============================================================

@dataclass
class Thresholds:
    non_harm_floor: float = 0.35
    veto_irreversibility: float = 0.7
    provenance: Optional[Provenance] = None

@dataclass
class Guardrails:
    max_parameter_adjustment: float = 0.3
    policy_lock: bool = False
    require_cross_validation: bool = False

# ===============================================================
# RISK MODELS
# ===============================================================

@dataclass
class HarmModel:
    base_weight: float = 0.7
    irreversibility_weight: float = 0.3
    base_agg: str = "max"  # "max" | "mean" | "weighted"
    base_agg_weight: float = 0.5  # if base_agg == "weighted"
    formula: str = ("expected_harm = base_risk * clamp(base_weight + "
                    "irreversibility_weight * irreversibility)")
    rationale: str = ("Expected-utility style amplification: irreversibility raises effective risk. "
                      "base_risk aggregated via max/mean/weighted (configurable).")
    provenance: Optional[Provenance] = None

@dataclass
class StabilityModel:
    mean_weight: float = 0.6
    max_weight: float = 0.4
    provenance: Optional[Provenance] = None

@dataclass
class ResilienceModel:
    avg_weight: float = 0.85
    breadth_weight: float = 0.15
    provenance: Optional[Provenance] = None

# ===============================================================
# CONTEXT DATA
# ===============================================================

@dataclass
class Stakeholder:
    name: str
    vulnerability: float = 0.5  # 0..1
    impact_benefit: float = 0.5 # 0..1
    weight: float = 1.0

@dataclass
class Uncertainty:
    short_term_risk: Tuple[float, float] = (0.0, 1.0)
    long_term_risk: Tuple[float, float] = (0.0, 1.0)
    irreversibility: Tuple[float, float] = (0.0, 1.0)

@dataclass
class OptionContext:
    name: str
    short_term_risk: Optional[float] = None
    long_term_risk: Optional[float] = None
    irreversibility_risk: float = 0.0
    stakeholders: List[Stakeholder] = field(default_factory=list)
    stability_risks: Dict[str, float] = field(default_factory=dict)
    resilience_features: Dict[str, float] = field(default_factory=dict)
    uncertainty: Optional[Uncertainty] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# ===============================================================
# CONFIG
# ===============================================================

@dataclass
class SigmaConfig:
    weights: EthicalWeights = field(default_factory=EthicalWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)
    harm_model: HarmModel = field(default_factory=HarmModel)
    stability_model: StabilityModel = field(default_factory=StabilityModel)
    resilience_model: ResilienceModel = field(default_factory=ResilienceModel)
    guardrails: Guardrails = field(default_factory=Guardrails)
    value_functions: Dict[str, ValueFunction] = field(default_factory=lambda: {
        "non_harm": ValueFunction("linear"),
        "stability": ValueFunction("linear"),
        "resilience": ValueFunction("linear"),
        "equity": ValueFunction("linear"),
    })
    verdict_acceptance_threshold: float = 0.65

# ===============================================================
# EQUITY METRICS
# ===============================================================

def gini_coefficient(x: List[float]) -> float:
    """
    Standard Gini in [0,1] for non-negative values; robust to zeros.
    g = (2 * sum_{i=1..n} i*x_i_sorted) / (n * sum(x)) - (n+1)/n
    """
    a = np.array([max(0.0, float(v)) for v in x], dtype=float)
    if a.size == 0:
        return 0.0
    S = float(a.sum())
    if S <= 1e-12:
        return 0.0
    a_sorted = np.sort(a)
    n = a.size
    i = np.arange(1, n + 1, dtype=float)
    g = (2.0 * float(np.sum(i * a_sorted)) / (n * S)) - (n + 1.0) / n
    return clip01(abs(g))

# ===============================================================
# ENGINE
# ===============================================================

class SigmaLab:
    """
    Procedural diagnostic framework for ethical deliberation.
    Evaluates four dimensions: Non-harm, Stability, Resilience, Equity.
    Returns transparent diagnostics; no moral truth claim.
    """

    def __init__(self, cfg: SigmaConfig):
        self.cfg = cfg

    # ---------- VALIDATION & NORMALIZATION ----------

    def _coerce_stakeholders(self, raw: List[Any]) -> List[Stakeholder]:
        out: List[Stakeholder] = []
        for s in raw or []:
            if isinstance(s, Stakeholder):
                out.append(s)
            elif isinstance(s, dict) and "name" in s:
                out.append(Stakeholder(
                    name=str(s.get("name")),
                    vulnerability=clip01(s.get("vulnerability", 0.5)),
                    impact_benefit=clip01(s.get("impact_benefit", 0.5)),
                    weight=float(s.get("weight", 1.0))
                ))
            elif isinstance(s, str):
                out.append(Stakeholder(name=s, vulnerability=0.5, impact_benefit=0.5, weight=1.0))
        return out

    def validate_context(self, ctx: OptionContext) -> List[str]:
        # Hard errors (raise): None values for essential fields
        essentials = {
            "short_term_risk": ctx.short_term_risk,
            "long_term_risk": ctx.long_term_risk,
        }
        for k, v in essentials.items():
            if v is None:
                # Test expects an exception with the word "error" present
                raise ValueError(f"error: missing essential field '{k}'")

        errs: List[str] = []
        # bounds (soft errors): allow test to proceed while flagging
        for name, value in [
            ("short_term_risk", float(ctx.short_term_risk)),
            ("long_term_risk", float(ctx.long_term_risk)),
            ("irreversibility_risk", float(ctx.irreversibility_risk)),
        ]:
            if not (0.0 <= value <= 1.0):
                errs.append(f"{name} out of [0,1]: {value}")

        # stakeholders
        coerced = self._coerce_stakeholders(ctx.stakeholders)
        if not coerced:
            errs.append("At least one stakeholder is required")
        else:
            for s in coerced:
                if not (0.0 <= s.vulnerability <= 1.0):
                    errs.append(f"Stakeholder '{s.name}' vulnerability out of [0,1]")
                if not (0.0 <= s.impact_benefit <= 1.0):
                    errs.append(f"Stakeholder '{s.name}' impact_benefit out of [0,1]")
                if s.weight < 0:
                    errs.append(f"Stakeholder '{s.name}' weight must be >= 0")

        # attach coerced back
        ctx.stakeholders = coerced
        return errs

    def _validate_config_soft(self) -> List[str]:
        warns: List[str] = []
        hm = self.cfg.harm_model
        valid_aggs = {"max", "mean", "weighted"}
        if hm.base_agg not in valid_aggs:
            warns.append(f"HarmModel.base_agg '{hm.base_agg}' invalid; falling back to 'max'.")
            hm.base_agg = "max"
        if hm.base_agg == "weighted":
            if not (0.0 <= hm.base_agg_weight <= 1.0):
                warns.append("HarmModel.base_agg_weight out of [0,1]; using 0.5")
                hm.base_agg_weight = 0.5
        sm = self.cfg.stability_model
        if abs((sm.mean_weight + sm.max_weight) - 1.0) > 1e-6:
            warns.append("StabilityModel weights should sum to 1.0 for interpretability")
        return warns

    # ---------- HELPER AGGREGATIONS ----------

    def _aggregate_base_risk(self, st: float, lt: float) -> float:
        hm = self.cfg.harm_model
        if hm.base_agg == "max":
            return max(st, lt)
        elif hm.base_agg == "mean":
            return safe_mean([st, lt])
        w = float(hm.base_agg_weight)
        return clip01(w * st + (1.0 - w) * lt)

    def _eval_non_harm(self, ctx: OptionContext) -> Tuple[float, Dict[str, Any]]:
        hm = self.cfg.harm_model
        base_risk = self._aggregate_base_risk(
            clip01(ctx.short_term_risk),
            clip01(ctx.long_term_risk)
        )
        irr = clip01(ctx.irreversibility_risk)
        factor = clip01(hm.base_weight + hm.irreversibility_weight * irr)
        expected_harm = clip01(base_risk * factor)
        score = clip01(1.0 - expected_harm)
        return score, {"expected_harm": expected_harm, "base_risk": base_risk, "irr": irr, "factor": factor}

    def _eval_stability(self, ctx: OptionContext) -> Tuple[float, Dict[str, Any]]:
        sm = self.cfg.stability_model
        vals = [clip01(v) for v in ctx.stability_risks.values()] or [0.0]
        mean_r = safe_mean(vals)
        max_r = max(vals)
        raw = sm.mean_weight * mean_r + sm.max_weight * max_r
        score = clip01(1.0 - raw)
        return score, {"mean": mean_r, "max": max_r, "raw": raw}

    def _eval_resilience(self, ctx: OptionContext) -> Tuple[float, Dict[str, Any]]:
        rm = self.cfg.resilience_model
        feats = [clip01(v) for v in ctx.resilience_features.values()] or [0.0]
        avg = safe_mean(feats)
        breadth = clip01(len([v for v in feats if v > 0.5]) / max(1, len(feats)))
        score = clip01(rm.avg_weight * avg + rm.breadth_weight * breadth)
        return score, {"avg": avg, "breadth": breadth}

    def _eval_equity(self, ctx: OptionContext) -> Tuple[float, Dict[str, Any]]:
        benefits = [clip01(s.impact_benefit) for s in ctx.stakeholders]
        g = gini_coefficient(benefits)
        score = clip01(1.0 - g)
        return score, {"gini": g, "benefits": benefits}

    # ---------- DIAGNOSE ----------

    def diagnose(self, ctx: OptionContext, verdict_opt_in: bool = False) -> Dict[str, Any]:
        """
        Returns a dict that ALWAYS contains key "diagnostic".
        On hard invalid input (e.g., None), raises ValueError with 'error' in the message.
        """
        warnings = self._validate_config_soft()
        try:
            errs = self.validate_context(ctx)
        except ValueError as e:
            # propagate as tests expect an exception with 'error' in the message
            raise

        if errs:
            # Still include 'diagnostic' so tests can find it.
            return {
                "status": "error",
                "warnings": warnings,
                "input_errors": errs,
                "scores": None,
                "details": None,
                "vetoes": [],
                "verdict": None,
                "timestamp": _now_iso(),
                "diagnostic": {
                    "note": "invalid_input",
                    "errors": errs
                },
                "semantics": {
                    "note": "Procedural diagnostic only. No moral truth claim."
                }
            }

        nh, nh_d = self._eval_non_harm(ctx)
        st, st_d = self._eval_stability(ctx)
        re, re_d = self._eval_resilience(ctx)
        eq, eq_d = self._eval_equity(ctx)

        vf = self.cfg.value_functions
        scores = {
            "non_harm": vf["non_harm"](nh),
            "stability": vf["stability"](st),
            "resilience": vf["resilience"](re),
            "equity": vf["equity"](eq),
        }

        vetoes: List[str] = []
        th = self.cfg.thresholds
        if scores["non_harm"] < th.non_harm_floor:
            vetoes.append(f"non-harm floor violated ({scores['non_harm']:.2f} < {th.non_harm_floor:.2f})")
        if clip01(ctx.irreversibility_risk) > th.veto_irreversibility:
            vetoes.append(f"irreversibility veto ({ctx.irreversibility_risk:.2f} > {th.veto_irreversibility:.2f})")

        agg = None
        verdict = None
        if verdict_opt_in:
            w = self.cfg.weights.norm()
            agg = sum(scores[k] * w[k] for k in w.keys())
            verdict = "ACCEPT" if agg >= self.cfg.verdict_acceptance_threshold and not vetoes else "REVIEW"

        diagnostic = {
            "aggregate": agg,
            "has_vetoes": bool(vetoes),
            "vetoes": vetoes
        }

        return {
            "status": "success",
            "warnings": warnings,
            "input_errors": [],
            "scores": scores,
            "details": {
                "non_harm": nh_d,
                "stability": st_d,
                "resilience": re_d,
                "equity": eq_d
            },
            "vetoes": vetoes,
            "verdict": verdict,
            "timestamp": _now_iso(),
            "diagnostic": diagnostic,
            "semantics": {
                "note": "Procedural diagnostic only. No moral truth claim.",
                "verdict_threshold": self.cfg.verdict_acceptance_threshold
            }
        }

    # ---------- AUDIT TRAIL ----------

    def export_audit_trail(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Return a JSON-serializable audit trail with stable fingerprints."""
        # Convert config dataclasses to plain dicts
        cfg_blob = {
            "weights": asdict(self.cfg.weights),
            "thresholds": asdict(self.cfg.thresholds),
            "harm_model": asdict(self.cfg.harm_model),
            "stability_model": asdict(self.cfg.stability_model),
            "resilience_model": asdict(self.cfg.resilience_model),
            "guardrails": asdict(self.cfg.guardrails),
            "value_functions": {
                k: {"kind": v.kind, "params": v.params, "provenance": asdict(v.provenance) if v.provenance else None}
                for k, v in self.cfg.value_functions.items()
            },
            "verdict_acceptance_threshold": self.cfg.verdict_acceptance_threshold
        }

        # Stable JSON for hashing
        cfg_json = json.dumps(cfg_blob, sort_keys=True, ensure_ascii=False)
        res_json = json.dumps(result, sort_keys=True, ensure_ascii=False)

        import hashlib
        def md5(txt: str) -> str:
            h = hashlib.md5()
            h.update(txt.encode("utf-8"))
            return h.hexdigest()

        return {
            "timestamp": _now_iso(),
            "config_fingerprint_md5": md5(cfg_json),
            "result_fingerprint_md5": md5(res_json),
            "config": cfg_blob,
            "result": result
        }

# ===============================================================
# DEMOS
# ===============================================================

def demo_context(domain: str) -> Tuple[SigmaConfig, OptionContext]:
    d = (domain or "healthcare").lower()

    if d == "healthcare":
        cfg = SigmaConfig(
            weights=EthicalWeights(0.45, 0.20, 0.20, 0.15,
                                   provenance=Provenance("Hospital Ethics Board", "POL-H-12",
                                                         "Clinical prioritization policy")),
            thresholds=Thresholds(0.40, 0.70, provenance=Provenance("Hospital Ethics Board", "POL-H-09",
                                                                     "Harm floors & irreversibility veto")),
            value_functions={
                "non_harm": ValueFunction("logistic", {"k": 10.0, "x0": 0.5}),
                "stability": ValueFunction("linear"),
                "resilience": ValueFunction("piecewise", {"points": [(0, 0), (0.5, 0.6), (1, 1)]}),
                "equity": ValueFunction("exp", {"a": 2.5}),
            },
            verdict_acceptance_threshold=0.65
        )
        ctx = OptionContext(
            name="TriageAssist-v1",
            short_term_risk=0.35,
            long_term_risk=0.20,
            irreversibility_risk=0.50,
            stakeholders=[
                Stakeholder("critical_patients", vulnerability=0.9, impact_benefit=0.8),
                Stakeholder("non_critical_patients", vulnerability=0.5, impact_benefit=0.55),
                Stakeholder("staff", vulnerability=0.3, impact_benefit=0.6),
            ],
            stability_risks={"ops": 0.25, "capacity": 0.30},
            resilience_features={"backup_staff": 0.6, "failover_units": 0.5},
            uncertainty=Uncertainty((0.25, 0.50), (0.10, 0.35), (0.40, 0.60)),
            metadata={"summary": "Clinical triage assistant. Policy-compliant pilot."}
        )
        return cfg, ctx

    if d == "ai":
        cfg = SigmaConfig(
            weights=EthicalWeights(0.40, 0.25, 0.20, 0.15,
                                   provenance=Provenance("AI Governance Committee", "AIG-2025-03",
                                                         "Platform risk policy")),
            thresholds=Thresholds(0.35, 0.65, provenance=Provenance("AI Governance Committee", "AIG-01",
                                                                     "Systemic harm veto")),
            value_functions={
                "non_harm": ValueFunction("logistic", {"k": 12.0, "x0": 0.55}),
                "stability": ValueFunction("linear"),
                "resilience": ValueFunction("linear"),
                "equity": ValueFunction("piecewise", {"points": [(0, 0), (0.7, 0.8), (1, 1)]}),
            },
            verdict_acceptance_threshold=0.65
        )
        ctx = OptionContext(
            name="BiasAudit-Release",
            short_term_risk=0.25,
            long_term_risk=0.45,
            irreversibility_risk=0.60,
            stakeholders=[
                Stakeholder("minority_users", vulnerability=0.8, impact_benefit=0.55),
                Stakeholder("majority_users", vulnerability=0.3, impact_benefit=0.65),
                Stakeholder("dev_team", vulnerability=0.2, impact_benefit=0.6),
            ],
            stability_risks={"ops": 0.20, "scale": 0.30, "security": 0.25},
            resilience_features={"rollback": 0.7, "canary": 0.6},
            uncertainty=Uncertainty((0.20, 0.40), (0.35, 0.55), (0.50, 0.70)),
            metadata={"summary": "Bias audit and controlled rollout plan."}
        )
        return cfg, ctx

    # public (default)
    cfg = SigmaConfig(
        weights=EthicalWeights(0.35, 0.25, 0.20, 0.20,
                               provenance=Provenance("City Council", "CIV-2025", "Public welfare policy")),
        thresholds=Thresholds(0.30, 0.70, provenance=Provenance("City Council", "CIV-2024",
                                                                 "Public risk thresholds")),
        value_functions={
            "non_harm": ValueFunction("linear"),
            "stability": ValueFunction("linear"),
            "resilience": ValueFunction("piecewise", {"points": [(0, 0), (0.4, 0.5), (1, 1)]}),
            "equity": ValueFunction("logistic", {"k": 10.0, "x0": 0.5}),
        },
        verdict_acceptance_threshold=0.65
    )
    ctx = OptionContext(
        name="UrbanSensors-Deploy",
        short_term_risk=0.20,
        long_term_risk=0.40,
        irreversibility_risk=0.45,
        stakeholders=[
            Stakeholder("residents", vulnerability=0.6, impact_benefit=0.62),
            Stakeholder("small_business", vulnerability=0.4, impact_benefit=0.58),
            Stakeholder("municipality", vulnerability=0.3, impact_benefit=0.6),
        ],
        stability_risks={"ops": 0.25, "privacy": 0.35, "maintenance": 0.30},
        resilience_features={"redundancy": 0.55, "incident_response": 0.60},
        uncertainty=Uncertainty((0.15, 0.35), (0.30, 0.50), (0.35, 0.55)),
        metadata={"summary": "Civic deployment of environmental sensors."}
    )
    return cfg, ctx

# ===============================================================
# CLI (optional local run)
# ===============================================================

def _pretty(obj: Any, pretty: bool) -> str:
    return json.dumps(obj, indent=2 if pretty else None, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIGMA-LAB v4.2 — Procedural Diagnostic Framework (test-friendly)")
    parser.add_argument("--demo", choices=["healthcare", "ai", "public"], default="healthcare",
                        help="Run a built-in demo context")
    parser.add_argument("--verdict", action="store_true", help="Include procedural verdict")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--audit", action="store_true", help="Print audit trail")
    args = parser.parse_args()

    cfg, ctx = demo_context(args.demo)
    engine = SigmaLab(cfg)
    try:
        out = engine.diagnose(ctx, verdict_opt_in=bool(args.verdict))
    except ValueError as e:
        print(str(e))
    else:
        if args.audit:
            audit = engine.export_audit_trail(out)
            print(_pretty(audit, bool(args.pretty)))
        else:
            print(_pretty(out, bool(args.pretty)))
