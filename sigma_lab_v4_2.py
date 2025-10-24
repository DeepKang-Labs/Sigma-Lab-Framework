#!/usr/bin/env python3
"""
SIGMA-LAB v4.2 — Procedural Ethics Framework (DeepKang-Labs)

A transparent, auditable diagnostic tool to structure ethical deliberation.
No moral truth. No auto-decisions. Procedural mirror only.

CLI:
  python -m sigma_lab.sigma_lab_v42 --demo ai --pretty
  python -m sigma_lab.sigma_lab_v42 --demo healthcare --pretty
  python -m sigma_lab.sigma_lab_v42 --demo public --pretty
  python -m sigma_lab.sigma_lab_v42 --config examples/demo_ai.yaml --context examples/demo_ai.yaml --pretty
  python -m sigma_lab.sigma_lab_v42 --help
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import argparse, json, math, hashlib, sys
import numpy as np
import yaml
import os

# ---------------- Utilities ----------------

def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def safe_mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0

def gini_coefficient(x: List[float]) -> float:
    a = np.array([max(0.0, float(v)) for v in x], dtype=float)
    if a.size == 0:
        return 0.0
    a_sorted = np.sort(a)
    n = a.size
    cum = np.cumsum(a_sorted)
    denom = a_sorted.sum()
    if denom <= 1e-12:
        return 0.0
    i = np.arange(1, n + 1)
    g = 1.0 - 2.0 * np.sum((n - i + 0.5) * a_sorted) / (n * denom)
    return clip01(float(abs(g)))

def load_yaml(path: str) -> Dict[str, Any]:
    """Robust YAML loader with helpful errors."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading {path}: {str(e)}")

# ---------------- Models ----------------

@dataclass
class Provenance:
    source: str
    doc_ref: Optional[str] = None
    rationale: Optional[str] = None

@dataclass
class ValueFunction:
    """Value function: linear | exp(a) | logistic(k,x0) | piecewise(points)."""
    kind: str = "linear"
    params: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[Provenance] = None

    def __call__(self, x: float) -> float:
        x = clip01(x)
        k = self.kind.lower()
        if k == "linear":
            return x
        if k == "exp":
            a = float(self.params.get("a", 3.0))
            return clip01((math.exp(a * x) - 1.0) / (math.exp(a) - 1.0 + 1e-12))
        if k == "logistic":
            kk = float(self.params.get("k", 12.0))
            x0 = float(self.params.get("x0", 0.5))
            return clip01(1.0 / (1.0 + math.exp(-kk * (x - x0))))
        if k == "piecewise":
            pts = self.params.get("points", [(0, 0), (1, 1)])
            pts = sorted([(float(a), float(b)) for a, b in pts], key=lambda t: t[0])
            xs, ys = zip(*pts)
            return float(np.interp(x, xs, ys))
        return x

@dataclass
class EthicalWeights:
    non_harm: float = 0.4
    stability: float = 0.2
    resilience: float = 0.2
    equity: float = 0.2
    provenance: Optional[Provenance] = None

    def norm(self) -> Dict[str, float]:
        v = np.array([self.non_harm, self.stability, self.resilience, self.equity], dtype=float)
        s = float(v.sum())
        if s < 1e-12:
            # All zero → uniform distribution for safety
            return {"non_harm": 0.25, "stability": 0.25, "resilience": 0.25, "equity": 0.25}
        return {
            "non_harm": float(v[0] / s),
            "stability": float(v[1] / s),
            "resilience": float(v[2] / s),
            "equity": float(v[3] / s),
        }

@dataclass
class Thresholds:
    non_harm_floor: float = 0.35
    veto_irreversibility: float = 0.7
    provenance: Optional[Provenance] = None

@dataclass
class Stakeholder:
    name: str
    vulnerability: float = 0.0
    impact_benefit: float = 0.0
    weight: float = 1.0
    notes: Optional[str] = None

@dataclass
class Uncertainty:
    short_term_risk: Tuple[float, float] = (0.0, 1.0)
    long_term_risk: Tuple[float, float] = (0.0, 1.0)
    irreversibility: Tuple[float, float] = (0.0, 1.0)

@dataclass
class HarmModel:
    base_weight: float = 0.7
    irreversibility_weight: float = 0.3
    base_agg: str = "max"  # "max" | "mean" | "weighted"
    base_agg_weight: float = 0.5
    formula: str = ("expected_harm = base_risk * clamp(base_weight + "
                    "irreversibility_weight * irreversibility)")
    rationale: str = ("Expected-utility style amplification: irreversibility raises effective risk.")
    provenance: Optional[Provenance] = None

@dataclass
class StabilityModel:
    mean_weight: float = 0.6
    max_weight: float = 0.4
    rationale: str = "Hotspot-aware stability = 1 - (0.6*mean + 0.4*max) risk"
    provenance: Optional[Provenance] = None

@dataclass
class VerdictPolicy:
    enabled: bool = False
    accept_threshold: float = 0.65
    rationale: str = "Procedural gatekeeping only; not a moral verdict."
    provenance: Optional[Provenance] = None

@dataclass
class SigmaConfig:
    weights: EthicalWeights = field(default_factory=EthicalWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)
    value_functions: Dict[str, ValueFunction] = field(default_factory=lambda: {
        "non_harm": ValueFunction("linear"),
        "stability": ValueFunction("linear"),
        "resilience": ValueFunction("linear"),
        "equity": ValueFunction("linear"),
    })
    harm_model: HarmModel = field(default_factory=HarmModel)
    stability_model: StabilityModel = field(default_factory=StabilityModel)
    verdict_policy: VerdictPolicy = field(default_factory=VerdictPolicy)
    mode: str = "PROFILE_ONLY"  # PROFILE_ONLY | MCDA
    no_aggregation: bool = True
    verdict_acceptance_threshold: float = 0.65
    provenance: Optional[Provenance] = None

@dataclass
class OptionContext:
    """Context describing a single option to diagnose."""
    name: str
    short_term_risk: float
    long_term_risk: float
    irreversibility_risk: float
    stakeholders: List[Stakeholder]
    stability_risks: Dict[str, float] = field(default_factory=dict)
    resilience_features: Dict[str, float] = field(default_factory=dict)
    uncertainty: Optional[Uncertainty] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# ---------------- Engine ----------------

class SigmaLab:
    """
    Procedural diagnostic framework for ethical deliberation.

    Evaluates options across:
      - Non-Harm (expected harm, irreversibility aware)
      - Stability (hotspot-aware operational risk)
      - Resilience (simple average of features)
      - Equity (benefit distribution via Gini)

    Example:
      >>> cfg, ctx = demo_context("ai")
      >>> engine = SigmaLab(cfg)
      >>> result = engine.diagnose(ctx, verdict_opt_in=False)
    """

    def __init__(self, config: SigmaConfig):
        self.cfg = config

    # ---------- Validation ----------

    def validate_context(self, ctx: OptionContext) -> List[str]:
        errs = []
        for name, v in [("short_term_risk", ctx.short_term_risk),
                        ("long_term_risk", ctx.long_term_risk),
                        ("irreversibility_risk", ctx.irreversibility_risk)]:
            if not (0 <= v <= 1):
                errs.append(f"{name} out of [0,1]: {v}")
        if not ctx.stakeholders:
            errs.append("At least one stakeholder required.")

        for s in ctx.stakeholders:
            if not (0 <= s.vulnerability <= 1):
                errs.append(f"vulnerability out of [0,1] for {s.name}")
            if not (0 <= s.impact_benefit <= 1):
                errs.append(f"impact_benefit out of [0,1] for {s.name}")
            if s.weight < 0:
                errs.append(f"weight negative for {s.name}")

        if ctx.uncertainty:
            for nm, (lo, hi) in [("short_term_risk", ctx.uncertainty.short_term_risk),
                                 ("long_term_risk", ctx.uncertainty.long_term_risk),
                                 ("irreversibility", ctx.uncertainty.irreversibility)]:
                if not (0 <= lo <= hi <= 1):
                    errs.append(f"uncertainty range invalid for {nm}: [{lo},{hi}]")
        return errs

    # ---------- Evaluations ----------

    def _harm_base(self, ctx: OptionContext) -> float:
        hm = self.cfg.harm_model
        st = clip01(ctx.short_term_risk)
        lt = clip01(ctx.long_term_risk)
        if hm.base_agg == "max":
            return max(st, lt)
        if hm.base_agg == "mean":
            return (st + lt) / 2.0
        if hm.base_agg == "weighted":
            w = clip01(hm.base_agg_weight)
            return w * st + (1 - w) * lt
        # fallback
        return max(st, lt)

    def _eval_non_harm(self, ctx: OptionContext) -> Tuple[float, Dict[str, Any]]:
        hm = self.cfg.harm_model
        base_risk = self._harm_base(ctx)
        irr = clip01(ctx.irreversibility_risk)
        factor = clip01(hm.base_weight + hm.irreversibility_weight * irr)

        expected_harms = []
        for s in ctx.stakeholders:
            eh = clip01(base_risk * factor) * clip01(s.vulnerability)
            expected_harms.append(eh)

        agg_harm = clip01(safe_mean(expected_harms))
        score = clip01(1.0 - agg_harm)
        return score, {
            "stakeholder_expected_harm": expected_harms,
            "aggregate_expected_harm": agg_harm,
            "base_risk": base_risk,
            "irr_factor": factor
        }

    def _eval_stability(self, ctx: OptionContext) -> Tuple[float, Dict[str, Any]]:
        sm = self.cfg.stability_model
        risks = [clip01(v) for v in ctx.stability_risks.values()] or [0.0]
        mean_r = safe_mean(risks)
        max_r = max(risks)
        raw = clip01(1.0 - (clip01(sm.mean_weight) * mean_r + clip01(sm.max_weight) * max_r))
        return raw, {"mean_risk": mean_r, "max_risk": max_r, "raw_stability": raw}

    def _eval_resilience(self, ctx: OptionContext) -> Tuple[float, Dict[str, Any]]:
        feats = [clip01(v) for v in ctx.resilience_features.values()] or [0.0]
        avg = safe_mean(feats)
        return avg, {"avg_features": avg, "features_count": len(feats)}

    def _eval_equity(self, ctx: OptionContext) -> Tuple[float, Dict[str, Any]]:
        # Interpret impact_benefit as normalized benefits per stakeholder
        benefits = [clip01(s.impact_benefit) for s in ctx.stakeholders] or [0.0]
        gi = gini_coefficient(benefits)
        equity_score = clip01(1.0 - gi)  # higher equity when gini is low
        return equity_score, {"gini": gi, "benefits": benefits}

    # ---------- Profile & verdict ----------

    def profile(self, ctx: OptionContext) -> Dict[str, Any]:
        nh_raw, nh_d = self._eval_non_harm(ctx)
        st_raw, st_d = self._eval_stability(ctx)
        re_raw, re_d = self._eval_resilience(ctx)
        eq_raw, eq_d = self._eval_equity(ctx)

        vf = self.cfg.value_functions
        scores = {
            "non_harm": vf["non_harm"](nh_raw),
            "stability": vf["stability"](st_raw),
            "resilience": vf["resilience"](re_raw),
            "equity": vf["equity"](eq_raw),
        }
        diagnostics = {
            "raw": {"non_harm": nh_raw, "stability": st_raw, "resilience": re_raw, "equity": eq_raw},
            "details": {"non_harm": nh_d, "stability": st_d, "resilience": re_d, "equity": eq_d}
        }
        return {"scores": scores, "diagnostics": diagnostics}

    def mcda_score(self, scores: Dict[str, float]) -> Optional[float]:
        if self.cfg.no_aggregation:
            return None
        w = self.cfg.weights.norm()
        return clip01(sum(scores[k] * w[k] for k in scores.keys()))

    def outranking_veto(self, ctx: OptionContext, scores: Dict[str, float]) -> Dict[str, Any]:
        th = self.cfg.thresholds
        vetoes = []
        if scores["non_harm"] < th.non_harm_floor:
            vetoes.append(f"Non-harm below floor ({scores['non_harm']:.2f} < {th.non_harm_floor:.2f})")
        if clip01(ctx.irreversibility_risk) > th.veto_irreversibility:
            vetoes.append(f"Irreversibility risk veto ({ctx.irreversibility_risk:.2f} > {th.veto_irreversibility:.2f})")
        return {"vetoes": vetoes, "pass": len(vetoes) == 0}

    def _procedural_verdict(self, scores: Dict[str, float], veto: Dict[str, Any]) -> Dict[str, Any]:
        if not veto["pass"]:
            return {"status": "REJECT", "reason": "; ".join(veto["vetoes"])}
        agg = self.mcda_score(scores)
        if agg is None:
            return {"status": "REVIEW", "reason": "Aggregation disabled; human deliberation required."}
        thr = clip01(self.cfg.verdict_acceptance_threshold)
        if agg >= thr:
            return {"status": "ACCEPTABLE", "reason": f"Weighted profile >= {thr:.2f} (agg={agg:.2f})"}
        return {"status": "REVIEW", "reason": f"Weighted profile < {thr:.2f} (agg={agg:.2f})"}

    def audit_trail(self) -> Dict[str, Any]:
        prov = {
            "weights": self.cfg.weights.provenance.__dict__ if self.cfg.weights.provenance else None,
            "thresholds": self.cfg.thresholds.provenance.__dict__ if self.cfg.thresholds.provenance else None,
            "value_functions": {k: (v.provenance.__dict__ if v.provenance else None) for k, v in self.cfg.value_functions.items()},
            "harm_model": self.cfg.harm_model.provenance.__dict__ if self.cfg.harm_model.provenance else None,
            "stability_model": self.cfg.stability_model.provenance.__dict__ if self.cfg.stability_model.provenance else None,
            "verdict_policy": self.cfg.verdict_policy.provenance.__dict__ if self.cfg.verdict_policy.provenance else None,
            "config": self.cfg.provenance.__dict__ if self.cfg.provenance else None
        }
        fp = hashlib.md5(json.dumps(prov, sort_keys=True, default=str).encode()).hexdigest()
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": prov,
            "config_fingerprint": fp
        }

    def diagnose(self, ctx: OptionContext, verdict_opt_in: bool = False, pretty: bool = False) -> Dict[str, Any]:
        """
        Run a full procedural diagnosis. Returns a JSON-serializable dict.
        Does not raise on invalid inputs; returns 'input_errors' instead.
        """
        errors = self.validate_context(ctx)
        if errors:
            return {"input_errors": errors}

        prof = self.profile(ctx)
        scores = prof["scores"]
        veto = self.outranking_veto(ctx, scores)
        audit = self.audit_trail()
        result = {
            "mode": self.cfg.mode,
            "no_aggregation": self.cfg.no_aggregation,
            "scores": scores,
            "diagnostics": prof["diagnostics"],
            "veto": veto,
            "provenance_audit": audit,
            "semantics": {
                "note": "Procedural diagnostic only. No moral truth claim.",
                "verdict_procedural": None
            }
        }
        if not self.cfg.no_aggregation:
            result["mcda_score"] = self.mcda_score(scores)
        if verdict_opt_in:
            result["semantics"]["verdict_procedural"] = self._procedural_verdict(scores, veto)
        return result

# ---------------- Demo & YAML ----------------

def config_from_yaml(path: str) -> SigmaConfig:
    data = load_yaml(path)
    def _prov(d): return Provenance(**d) if isinstance(d, dict) else None
    vfs = {}
    for k in ["non_harm", "stability", "resilience", "equity"]:
        spec = data.get("value_functions", {}).get(k, {"kind": "linear", "params": {}})
        vf = ValueFunction(kind=spec.get("kind", "linear"), params=spec.get("params", {}), provenance=_prov(spec.get("provenance")))
        vfs[k] = vf
    weights = data.get("weights", {})
    weights_obj = EthicalWeights(**weights.get("values", weights))
    if "provenance" in data.get("weights", {}):
        weights_obj.provenance = _prov(data["weights"]["provenance"])
    thresholds = data.get("thresholds", {})
    th_obj = Thresholds(**thresholds.get("values", thresholds))
    if "provenance" in data.get("thresholds", {}):
        th_obj.provenance = _prov(data["thresholds"]["provenance"])
    hm = HarmModel(**data.get("harm_model", {}))
    sm = StabilityModel(**data.get("stability_model", {}))
    vp = VerdictPolicy(**data.get("verdict_policy", {}))
    cfg = SigmaConfig(
        weights=weights_obj,
        thresholds=th_obj,
        value_functions=vfs,
        harm_model=hm,
        stability_model=sm,
        verdict_policy=vp,
        mode=str(data.get("mode", "PROFILE_ONLY")),
        no_aggregation=bool(data.get("no_aggregation", True)),
        verdict_acceptance_threshold=float(data.get("verdict_acceptance_threshold", 0.65)),
        provenance=_prov(data.get("provenance"))
    )
    return cfg

def context_from_yaml(path: str) -> OptionContext:
    data = load_yaml(path)
    stakeholders = [Stakeholder(**s) for s in data.get("stakeholders", [])]
    unc = data.get("uncertainty", None)
    uncertainty = Uncertainty(**unc) if unc else None
    return OptionContext(
        name=data["name"],
        short_term_risk=float(data["short_term_risk"]),
        long_term_risk=float(data["long_term_risk"]),
        irreversibility_risk=float(data["irreversibility_risk"]),
        stakeholders=stakeholders,
        stability_risks=data.get("stability_risks", {}),
        resilience_features=data.get("resilience_features", {}),
        uncertainty=uncertainty,
        metadata=data.get("metadata", {})
    )

def demo_context(domain: str) -> Tuple[SigmaConfig, OptionContext]:
    d = domain.lower()
    if d == "healthcare":
        cfg = SigmaConfig(
            weights=EthicalWeights(0.45, 0.2, 0.2, 0.15, provenance=Provenance("Hospital Ethics Board","POL-H-12","Clinical prioritization policy")),
            thresholds=Thresholds(0.4, 0.7, provenance=Provenance("Hospital Ethics Board","POL-H-09","Harm floors & irreversibility veto")),
            no_aggregation=True, mode="PROFILE_ONLY"
        )
        ctx = OptionContext(
            name="TriageAssist-v1",
            short_term_risk=0.35, long_term_risk=0.20, irreversibility_risk=0.50,
            stakeholders=[
                Stakeholder("critical_patients", vulnerability=0.9, impact_benefit=0.8),
                Stakeholder("non_critical_patients", vulnerability=0.5, impact_benefit=0.55),
                Stakeholder("staff", vulnerability=0.3, impact_benefit=0.6),
            ],
            stability_risks={"ops": 0.25, "capacity": 0.30},
            resilience_features={"backup_staff": 0.6, "failover_units": 0.5},
            uncertainty=Uncertainty((0.25, 0.5), (0.10, 0.35), (0.40, 0.60))
        )
        return cfg, ctx

    if d == "ai":
        cfg = SigmaConfig(
            weights=EthicalWeights(0.4, 0.25, 0.2, 0.15, provenance=Provenance("AI Governance Committee","AIG-03","Platform risk policy")),
            thresholds=Thresholds(0.35, 0.65, provenance=Provenance("AI Governance Committee","AIG-01","Irreversibility = systemic harm")),
            no_aggregation=False, mode="MCDA", verdict_acceptance_threshold=0.65
        )
        ctx = OptionContext(
            name="BiasAudit-Release",
            short_term_risk=0.25, long_term_risk=0.45, irreversibility_risk=0.60,
            stakeholders=[
                Stakeholder("minority_users", vulnerability=0.8, impact_benefit=0.55),
                Stakeholder("majority_users", vulnerability=0.3, impact_benefit=0.65),
                Stakeholder("dev_team", vulnerability=0.2, impact_benefit=0.6),
            ],
            stability_risks={"ops": 0.2, "scale": 0.3, "security": 0.25},
            resilience_features={"rollback": 0.7, "canary": 0.6},
            uncertainty=Uncertainty((0.2, 0.4), (0.35, 0.55), (0.5, 0.7))
        )
        return cfg, ctx

    # public
    cfg = SigmaConfig(
        weights=EthicalWeights(0.35, 0.25, 0.2, 0.2, provenance=Provenance("City Council","CIV-2025","Public welfare policy")),
        thresholds=Thresholds(0.3, 0.7, provenance=Provenance("City Council","CIV-2024","Public risk thresholds")),
        no_aggregation=True, mode="PROFILE_ONLY"
    )
    ctx = OptionContext(
        name="UrbanSensors-Deploy",
        short_term_risk=0.20, long_term_risk=0.40, irreversibility_risk=0.45,
        stakeholders=[
            Stakeholder("residents", vulnerability=0.6, impact_benefit=0.62),
            Stakeholder("small_business", vulnerability=0.4, impact_benefit=0.58),
            Stakeholder("municipality", vulnerability=0.3, impact_benefit=0.6),
        ],
        stability_risks={"ops": 0.25, "privacy": 0.35, "maintenance": 0.3},
        resilience_features={"redundancy": 0.55, "incident_response": 0.6},
        uncertainty=Uncertainty((0.15, 0.35), (0.30, 0.50), (0.35, 0.55))
    )
    return cfg, ctx

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="SIGMA-LAB v4.2 — Procedural Ethics Framework")
    ap.add_argument("--config", type=str, help="YAML config path")
    ap.add_argument("--context", type=str, help="YAML context path")
    ap.add_argument("--demo", type=str, choices=["healthcare","ai","public"], help="Run a built-in demo")
    ap.add_argument("--verdict", action="store_true", help="Opt-in procedural verdict")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = ap.parse_args()

    if args.demo:
        cfg, ctx = demo_context(args.demo)
    else:
        if not args.config or not args.context:
            print("ERROR: Provide --config and --context, or use --demo.", file=sys.stderr)
            sys.exit(1)
        cfg = config_from_yaml(args.config)
        ctx = context_from_yaml(args.context)

    engine = SigmaLab(cfg)
    out = engine.diagnose(ctx, verdict_opt_in=bool(args.verdict))
    print(json.dumps(out, indent=2 if args.pretty else None, ensure_ascii=False))

if __name__ == "__main__":
    main()
