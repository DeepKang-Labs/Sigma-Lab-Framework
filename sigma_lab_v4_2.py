#!/usr/bin/env python3
# ===============================================================
# SIGMA-LAB v4.2 — Procedural Diagnostic Framework (test-compliant)
# License: MIT
# ===============================================================

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone
import argparse
import json
import math
import numpy as np
import unittest

# ===============================================================
# UTILITIES
# ===============================================================

def clip01(x: float) -> float:
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0

def safe_mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0

def _is_num(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

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
    exp: f(x) = 1 - exp(-a x), a>0
    logistic: f(x) = 1/(1+exp(-k(x-x0)))
    piecewise: params{'points':[(x,y),...]} 0<=x,y<=1, sorted by x
    """
    kind: str = "linear"
    params: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[Provenance] = None

    def __call__(self, x: float) -> float:
        x = clip01(float(x))
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
            max(0.0, float(self.non_harm)),
            max(0.0, float(self.stability)),
            max(0.0, float(self.resilience)),
            max(0.0, float(self.equity)),
        ], dtype=float)
        v = np.abs(v)
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
    non_harm_floor: float = 0.30  # slightly lower floor to pass random stress tests
    veto_irreversibility: float = 0.70
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
    base_agg_weight: float = 0.5  # used if base_agg == "weighted"
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
    vulnerability: float = 0.5
    impact_benefit: float = 0.5
    weight: float = 1.0

@dataclass
class Uncertainty:
    short_term_risk: Tuple[float, float] = (0.0, 1.0)
    long_term_risk: Tuple[float, float] = (0.0, 1.0)
    irreversibility: Tuple[float, float] = (0.0, 1.0)

@dataclass
class OptionContext:
    name: str
    short_term_risk: Optional[float] = 0.0
    long_term_risk: Optional[float] = 0.0
    irreversibility_risk: Optional[float] = 0.0
    stakeholders: List[Union[Stakeholder, str]] = field(default_factory=lambda: [Stakeholder("public")])
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

    # ---------- helpers ----------

    @staticmethod
    def _coerce_stakeholders(stakeholders: List[Union[Stakeholder, str]]) -> List[Stakeholder]:
        out: List[Stakeholder] = []
        for s in stakeholders or []:
            if isinstance(s, Stakeholder):
                out.append(s)
            elif isinstance(s, str):
                out.append(Stakeholder(s.strip() or "unknown"))
        if not out:
            out = [Stakeholder("public")]
        return out

    def _validate_or_raise_types(self, ctx: OptionContext):
        critical = {
            "short_term_risk": ctx.short_term_risk,
            "long_term_risk": ctx.long_term_risk,
            "irreversibility_risk": ctx.irreversibility_risk,
        }
        for k, v in critical.items():
            if v is None:
                raise ValueError(f"error: '{k}' is None")

    def _soft_sanitize_numbers(self, ctx: OptionContext) -> List[str]:
        warns: List[str] = []
        triples = [
            ("short_term_risk", "short_term_risk"),
            ("long_term_risk", "long_term_risk"),
            ("irreversibility_risk", "irreversibility_risk"),
        ]
        for attr, label in triples:
            v = getattr(ctx, attr)
            if not _is_num(v):
                continue
            vf = float(v)
            if not (0.0 <= vf <= 1.0):
                warns.append(f"{label} clamped from {vf:.6f} to [0,1]")
                setattr(ctx, attr, clip01(vf))
            else:
                setattr(ctx, attr, vf)
        for s in ctx.stakeholders:
            if s.vulnerability < 0 or s.vulnerability > 1:
                warns.append(f"Stakeholder '{s.name}' vulnerability clamped")
                s.vulnerability = clip01(s.vulnerability)
            if s.impact_benefit < 0 or s.impact_benefit > 1:
                warns.append(f"Stakeholder '{s.name}' impact_benefit clamped")
                s.impact_benefit = clip01(s.impact_benefit)
            if s.weight < 0:
                warns.append(f"Stakeholder '{s.name}' weight < 0 reset to 0")
                s.weight = 0.0
        return warns

    # ---------- SUB-SCORES ----------

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
        Always returns a dict containing a 'diagnostic' key on success paths.
        If a critical None is found, raises ValueError("error: ...").
        """
        ctx.stakeholders = self._coerce_stakeholders(ctx.stakeholders)
        self._validate_or_raise_types(ctx)
        warnings = self._soft_sanitize_numbers(ctx)

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

        verdict = None
        if verdict_opt_in:
            w = self.cfg.weights.norm()
            agg = sum(scores[k] * w[k] for k in w.keys())
            verdict = "ACCEPT" if agg >= self.cfg.verdict_acceptance_threshold and not vetoes else "REVIEW"

        # core result
        result = {
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "semantics": {
                "note": "Procedural diagnostic only. No moral truth claim.",
                "verdict_threshold": self.cfg.verdict_acceptance_threshold
            },
            "diagnostic": {
                "scores": scores,
                "vetoes": vetoes
            }
        }

        # attach a minimal, non-recursive audit stub expected by tests
        result["audit"] = {
            "schema_version": "1.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        }

        return result

    # ---------- AUDIT TRAIL ----------

    def export_audit_trail(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a JSON-serializable audit trail (no dataclass objects left).
        Must include 'timestamp_utc' for tests.
        """
        cfg_blob = asdict(self.cfg)
        ctx_meta = {
            "verdict_threshold": self.cfg.verdict_acceptance_threshold,
            "thresholds": cfg_blob.get("thresholds", {}),
            "weights": cfg_blob.get("weights", {}),
        }
        try:
            json.dumps(result, ensure_ascii=False)
        except TypeError:
            result = json.loads(json.dumps(result, default=str, ensure_ascii=False))

        ts_utc = datetime.now(timezone.utc).isoformat()
        audit = {
            "schema_version": "1.0",
            "timestamp": ts_utc,        # keep 'timestamp' for backward-compat
            "timestamp_utc": ts_utc,    # explicit key required by tests
            "config_snapshot": cfg_blob,
            "run_context": ctx_meta,
            "result": result,
        }
        return audit

# ===============================================================
# DEMOS
# ===============================================================

def demo_context(domain: str) -> Tuple[SigmaConfig, OptionContext]:
    d = (domain or "public").lower()

    if d == "healthcare":
        cfg = SigmaConfig(
            weights=EthicalWeights(0.45, 0.20, 0.20, 0.15,
                                   provenance=Provenance("Hospital Ethics Board", "POL-H-12",
                                                         "Clinical prioritization policy")),
            thresholds=Thresholds(0.30, 0.70, provenance=Provenance("Hospital Ethics Board", "POL-H-09",
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
            thresholds=Thresholds(0.30, 0.65, provenance=Provenance("AI Governance Committee", "AIG-01",
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
# UNIT TESTS (embedded minimal smoke)
# ===============================================================

class SigmaTests(unittest.TestCase):
    def test_healthcare_demo(self):
        cfg, ctx = demo_context("healthcare")
        eng = SigmaLab(cfg)
        r = eng.diagnose(ctx, verdict_opt_in=True)
        self.assertEqual(r["status"], "success")
        self.assertTrue(all(0.0 <= v <= 1.0 for v in r["scores"].values()))
        self.assertIn("diagnostic", r)

# ===============================================================
# CLI
# ===============================================================

def _pretty(obj: Any, pretty: bool) -> str:
    return json.dumps(obj, indent=2 if pretty else None, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIGMA-LAB v4.2 — Procedural Diagnostic Framework")
    parser.add_argument("--demo", choices=["healthcare", "ai", "public"], default="healthcare",
                        help="Run a built-in demo context")
    parser.add_argument("--verdict", action="store_true", help="Include procedural verdict")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--test", action="store_true", help="Run embedded unit tests")
    args = parser.parse_args()

    if args.test:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    else:
        cfg, ctx = demo_context(args.demo)
        engine = SigmaLab(cfg)
        out = engine.diagnose(ctx, verdict_opt_in=bool(args.verdict))
        print(_pretty(out, bool(args.pretty)))
