# Phase α – Skyfleet Runbook

## Objective
Validate Sigma-Lab’s empirical usefulness in real-world governance deliberations.

---

## 🧱 Pre-requisites

- [ ] Case identified (decision_mapper.yaml ready)
- [ ] 2 evaluators confirmed (names + roles)
- [ ] 90 min session scheduled
- [ ] Recording consent obtained

---

## 🕒 Session Structure (90 min)

| Phase | Duration | Objective |
|--------|-----------|-----------|
| 0–15 min | Context presentation | Overview of case and objectives |
| 15–35 min | Parameter filling | Evaluators fill mappings independently |
| 35–55 min | Sigma-Lab execution | Run diagnostic via Skywire/Fiber |
| 55–75 min | Discussion | Compare outputs and interpretations |
| 75–90 min | Debrief | Feedback and documentation |

---

## 🧪 Deliverables

- `decision_mapper.yaml` — case context
- `skywire_alpha_results.json` — Sigma-Lab outputs
- `evaluator_feedback.md` — pre/post feedback forms
- `session_transcript.md` — optional conversation record
- `learnings.md` — synthesis of key insights

---

## ✅ Success Criteria

At least one of:
- [ ] An evaluator changes position (Accept → Uncertain → Reject)
- [ ] Blind spot revealed
- [ ] Divergence prompts deeper discussion

---

## ⚠️ Failure Scenarios

### Scenario A – Evaluators don’t engage
- **Fallback:** Shorten session (60 min) focusing on parameter filling.
- **Learning:** Adjust recruitment or framing for clarity.

### Scenario B – Sigma-Lab crashes
- **Fallback:** Run `validate-only` mode, record manual reasoning.
- **Learning:** Fix runtime bugs before Phase β.

### Scenario C – No divergence detected
- **Fallback:** Interview evaluators on why (threshold, framing).
- **Learning:** Recalibrate scoring sensitivity.

---

## 🧩 Null Result = Valid Result

If evaluators conclude *“Sigma-Lab added no value”*, this is:
- NOT a failure of honesty  
- IS a signal of design gap  
- REQUIRES iteration before Phase β

Action: Publish null result and pivot.

---

## 📈 Next Steps

If success → Prepare Phase β scaling plan.  
If failure → Refine mappings and thresholds.

---

**Facilitator:** Yuri Kang  
**Session Type:** Governance Discovery  
**Version:** Sigma-Lab v5.1
