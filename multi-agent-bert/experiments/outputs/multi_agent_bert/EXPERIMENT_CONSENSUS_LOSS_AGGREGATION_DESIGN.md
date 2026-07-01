# Aggregation-Layer Design — Recovering Consensus Loss Without Over-Trusting Dissent

Design a better decision-fusion method than majority/weighted voting, targeting the measured
**consensus loss** (correct answers the panel holds but the vote discards). **Design only — no
runs, no prompt changes, no new agents, no LLM calls, no training.** Grounded in an **offline
simulation** over the saved A–G captures (agent labels are already stored, so label-based fusion
rules can be re-scored for free). Date: 2026-07-01.

## The binding constraint (measured offline, no LLM calls)
From `EXPERIMENT_CONSENSUS_LOSS_ANALYSIS.md`, the loss is dominated by **lone-correct** cases
(one specialist right, outvoted). The obvious fix — "trust the dissenter" — is tested here:

**Lone-dissenter precision** (specialist X disagrees with an agreeing bloc; is X right?):
| specialist | right / lone-dissents | precision |
|---|---|---|
| **Contextual** | 11 / 19 | **0.58** |
| Intent | 12 / 26 | 0.46 |
| Lexical | 16 / 43 | 0.37 |
| Polarity | 8 / 24 | 0.33 |

**Only Contextual beats a coin flip.** And **unconditionally trusting even Contextual's dissent
loses net** (offline sim on Design G):
| rule | escalated acc | Δ vs actual G |
|---|---|---|
| actual G final | 0.786 | — |
| trust Contextual lone-dissent | 0.762 | **−2** |
| trust Lexical lone-dissent | 0.762 | −2 |
| trust Contextual+Lexical | 0.738 | −4 |

**Conclusion that governs the whole design:** the consensus loss is *real* (oracle 0.83–0.88 vs
final ~0.75) but **not recoverable by any unconditional minority-trust or reweighting rule** —
minority-correct and minority-incorrect dissents are entangled, and no *label-only* rule
separates them. Recovery requires either (i) a **detectable domain** in which a specialist is
reliably right, or (ii) a **calibrated confidence** signal — and prior work showed the agents'
self-confidence is uninformative on this subset. This is why the *only* mechanism that ever
worked was the **IntentGate: a domain-restricted, non-overriding veto** — not a reweighting.

---

## Option-by-option evaluation
### 1. Specialist-specific vetoes (Intent/Contextual/Lexical)
- **Targets:** lone-correct suppression, by converting a suppressed *vote* into an
  un-suppressible *veto* (proven: Intent-as-vote 12/12 suppressed → Intent-as-veto 0 missed).
- **Would have helped:** A/C/E — where Intent/Contextual dissents were 100%/68% suppressed; G
  already banks the Intent veto (+the meta cluster).
- **Harmful-override risk:** real but **bounded IF each veto is (a) domain-restricted and (b)
  non-overriding** (protects the primary, never forces a flip). Unrestricted vetoes = the −2/−4
  above. The IntentGate's 2–3 hurt cases show the residual risk even when restricted.
- **General/task-aware:** yes (domain-of-authority is generic).
- **Complexity:** medium (one guard predicate per veto in consensus).
- **Offline-simulable:** **YES for label-based domain conditions** (e.g. "Contextual reads neutral
  + primary neutral → block polar override", the strongest-agent analogue of the IntentGate);
  **NO for "strong-cue/high-confidence" conditions** (confidence not serialized).

### 2. Confidence-aware arbitration
- **Targets:** lone-correct where the correct dissenter is confident and the wrong majority is not.
- **Would have helped:** *only if confidence tracked correctness* — but the established finding is
  that these agents are over-confident and **poorly calibrated on the hard subset** (confidence
  does not separate their right/wrong cases; cf. router-selectability). So expected effect ≈ noise.
- **Harmful-override risk:** **high** — arbitrating on an uninformative signal adds errors.
- **General:** yes. **Complexity:** medium.
- **Offline-simulable:** **NO** — per-agent confidences were **not saved** in the design captures;
  simulating this requires a **new capture pass that logs confidences (paid LLM calls)**, and even
  then the calibration evidence predicts little gain. **Deprioritize.**

### 3. Role-priority rules (domain → expert wins)
- **Targets:** lone-correct suppression, by making the domain expert the *decider* (mixture-of-
  experts routing by case type, not vote count) — directly attacks minority suppression.
- **Would have helped:** Contextual-wins-on-sarcasm recovers the 13 lone-Contextual; Lexical-wins-
  on-explicit-cue recovers lone-Lexical; gate-wins-on-meta = IntentGate.
- **Harmful-override risk:** **high if it grants OVERRIDE authority** (the −2 sim is exactly a
  Contextual-priority override). The **circular problem:** deciding "this is a sarcasm case → route
  to Contextual" is itself the hard pragmatic judgment; a wrong route hands authority to the wrong
  expert. Safe **only** as a *non-overriding* protection (one-directional veto), which collapses it
  into Option 1.
- **General:** yes. **Complexity:** medium-high (domain predicates).
- **Offline-simulable:** **partially** — domains proxied by label patterns (e.g. "Contextual is the
  lone neutral") are simulable; true domain detection (is it sarcasm?) needs the reasoning text
  (saved as `raw_llm_response`, but fuzzy to parse).

### 4. Meta-consensus classifier (learned or rule-based selector)
- **Targets:** all consensus-loss patterns, by learning which configuration predicts the correct final.
- **Would have helped:** upper bound = oracle 0.83–0.88, but **achievable ≪ oracle** (the correct
  voice is the minority; no saved feature identifies it).
- **Harmful-override risk:** **overfitting** — the escalated set is ~80 samples with correlated,
  poorly-calibrated features; a learned selector will not generalize. **Leakage risk** — training
  on the test captures is invalid; a clean selector needs **dev-set agent captures (paid)**.
- **General:** the rule-based form = Option 3; the learned form is dataset-specific and fragile.
- **Complexity:** high. **Offline-simulable:** the *rule-based* form yes; the *learned* form needs
  new dev captures (paid) + risks overfit. **Deprioritize the learned form.**

### 5. Conservative primary protection (adaptive w_primary)
- **Targets:** **not** the consensus loss (that is agent-vs-agent); it manages **agent-vs-primary**.
- **Would have helped:** on a strong primary it *reduces harmful overrides* (protect a good primary)
  but also reduces rescues → nets ~0 on Ahmed (the consensus-simulation already showed agent-bloc +
  high w_primary recovers the primary exactly). On a **weak** primary it is the key knob (allow more
  agent correction → the +0.059 regime).
- **Harmful-override risk:** **low** (protective). Upside on the strong primary is capped at ~net 0.
- **General:** yes. **Complexity:** low.
- **Offline-simulable:** **YES, fully** (w_primary sweep on saved labels + primary confidence;
  already partially done in the consensus simulation).

## Summary matrix
| option | targets consensus-loss? | harmful-override risk | general | complexity | offline-simulable (no paid) |
|---|---|---|---|---|---|
| 1 specialist vetoes | **yes (proven template)** | bounded if restricted+non-overriding | yes | med | **yes (label-based)** |
| 2 confidence-aware | yes in principle | **high (uncalibrated)** | yes | med | **no (confidences not saved)** |
| 3 role-priority | yes | high if override; low if veto | yes | med-high | partial |
| 4 meta-classifier | yes (upper bound) | **high (overfit/leakage)** | learned=no | high | rule-based yes; learned needs paid dev |
| 5 primary protection | no (agent-vs-primary) | low | yes | low | **yes (fully)** |

---

## Recommendation — one next aggregation experiment (offline-first, zero LLM cost)
**Build an offline consensus-rescoring simulator over the saved A–G captures and sweep the two
families that are (a) label-simulable for free and (b) consistent with the evidence** — then
promote to a paid confirmation *only* a rule that is net-positive offline.

Concretely, the single next experiment is an **offline fusion sweep** evaluating:
1. **Non-overriding specialist vetoes (Option 1), in the IntentGate template** — the highest-value
   candidate is a **Contextual-driven neutral guard** (the strongest agent, 0.58 lone-dissent
   precision): *if Contextual reads neutral and the primary is neutral, block a polar override* —
   the strongest-agent analogue of the working IntentGate. Also test a **Lexical strong-cue
   protection** in label-only proxy form. Measure net vs actual G, and the helped/hurt split, so we
   do not repeat the unconditional −2.
2. **Adaptive w_primary (Option 5)** — a w_primary sweep to map the protection/correction trade-off,
   which is the one knob that transfers to the weak-primary regime.

Why this and not the others:
- The offline sim already **rules out** unconditional minority-trust/reweighting (−2/−4) and shows
  recovery needs a *domain-restricted, non-overriding* mechanism — i.e. Option 1 in the IntentGate
  template, evaluable for free on saved data.
- **Confidence-aware (2) and learned meta-classifier (4) cannot even be simulated** from current
  saves (no confidences) and are contraindicated (poor calibration / overfit / leakage); they would
  require new paid captures before they could be assessed, so they are second-tier.
- Doing the sweep **offline first** means **no paid LLM calls** until a rule demonstrably beats G on
  the saved runs — the disciplined path given every recent change was noise-bound.

**Expected outcome (honest):** even the best restricted veto is likely to yield **0 to +2 samples
on the strong Ahmed primary** (the −2 unconditional result + the entanglement of correct/incorrect
dissent bound the upside), i.e. still noise-adjacent there. Its real value — like the whole agentic
layer — should appear on the **weak C3 primary**, where the consensus loss is larger and the primary
term is small. So the aggregation sweep should ultimately be evaluated on **both** primaries, and the
decisive paid run remains **C3**, not more strong-primary tuning.

## What NOT to do (evidence-based)
- Do not add unconditional minority-trust or a global reweighting (proven −2/−4).
- Do not build a learned meta-selector on ~80 correlated test samples (overfit/leakage).
- Do not rely on agent self-confidence without first re-capturing and *validating* its calibration.

## Artifacts / basis
- `EXPERIMENT_CONSENSUS_LOSS_ANALYSIS.md` (per-specialist suppression, vote-vs-veto),
  `EXPERIMENT_ESCALATED_CEILING_ROOT_CAUSE.md` (floor + correlation),
  `EXPERIMENT_AHMED_ROUTER_SELECTABILITY.md` (confidence uninformative),
  `EXPERIMENT_AHMED_CONSENSUS_SIMULATION.md` (offline w_primary sweep precedent),
  `EXPERIMENT_SENTIMENT_INTENT_GATE_ABLATION.md` (the working veto template).
- Offline lone-dissenter precision + trust-dissenter sim computed from each
  `experiment_ahmed_*/error_attribution/attribution_table.json` (labels only; no LLM calls).
