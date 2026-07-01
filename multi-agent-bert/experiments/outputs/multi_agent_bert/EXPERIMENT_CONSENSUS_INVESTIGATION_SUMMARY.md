# Consensus / Aggregation Investigation — Summary of What Was Tested and What Remains

A single record of the consensus-layer investigation on the Ahmed frozen primary (84 escalated
samples per design). **Summary only — no new runs, no code changes.** Consolidates the
consensus-loss, root-cause, aggregation-design, and offline-rescoring reports. Date: 2026-07-01.

---

## 1. Evidence that consensus loss exists
Across designs, the panel frequently **holds the correct answer in one specialist that the
consensus then discards.** Quantified as suppressed-correct (agent right, final wrong):

| specialist | suppressed-correct (across designs it votes) | of which **lone-correct** |
|---|---|---|
| Lexical | 25 | 17 |
| Contextual | 20 | 13 |
| Intent (as a voter) | 12 | **12 (100%)** |
| Polarity | 7 | 5 |
| Logic | 3 | 2 |

The loss concentrates on the **decorrelated** agents (Lexical/Contextual/Intent) — exactly the
independent signal an ensemble should exploit — while the redundant correlated agents
(Polarity/Logic) are rarely suppressed because they rarely dissent correctly. Consensus loss is
therefore **structural, not stochastic**: when one agent is uniquely right, the correlated
remainder is jointly wrong and outvotes it. (Source: `EXPERIMENT_CONSENSUS_LOSS_ANALYSIS.md`.)

## 2. Oracle-any-voter upper bound
If an oracle could pick the correct label whenever **any** voter holds it:

| design | final acc (escalated) | **oracle-any-voter** | recoverable loss | unrecoverable (floor) |
|---|---|---|---|---|
| A | 0.726 | 0.798 | 6 | 17 |
| C | 0.738 | 0.821 | 7 | 15 |
| **E** | 0.750 | **0.881** | **11** | 10 |
| F | 0.726 | 0.845 | 10 | 13 |
| **G** | 0.786 | 0.833 | **4** | 14 |
| G2 | 0.774 | 0.833 | 5 | 14 |
| v3 | 0.750* | 0.833 | 7 | 14 |

So the panel contains **4–11 more correct answers per 84** than the consensus emits (oracle
0.80–0.88 vs actual ~0.75). Of ~19–23 final errors per design, **~⅔ are unrecoverable** (no agent
had the truth = the information/model floor) and **~⅓ are consensus loss** (recoverable in
principle). G has the lowest recoverable loss (the veto recovers some). \*capture value.

## 3. Lone-dissenter precision by agent
When a specialist disagrees with an otherwise-agreeing bloc, is it right? (offline, from labels)

| specialist | right / lone-dissents | precision |
|---|---|---|
| **Contextual** | 11 / 19 | **0.58** |
| Intent | 12 / 26 | 0.46 |
| Lexical | 16 / 43 | 0.37 |
| Polarity | 8 / 24 | 0.33 |

**Only Contextual dissents better than a coin flip.** The rest are wrong more often than right, so
trusting their dissent adds errors. (Source: `EXPERIMENT_CONSENSUS_LOSS_AGGREGATION_DESIGN.md`.)

## 4–5. Offline rules tested, and why each failed or tied
All simulated offline on the saved G capture (labels only; rules fixed before scoring; true labels
never used to decide firing). Baseline = G capture draw (66/84 escalated, net +3, full 0.9291 —
within temp-0 noise of headline G 0.9279 / +2). (Source: `EXPERIMENT_G_OFFLINE_CONSENSUS_RESCORING.md`.)

| rule | full acc | net | interventions | helped / hurt | vs G |
|---|---|---|---|---|---|
| baseline (G) | 0.9291 | +3 | — | — | = |
| **Contextual neutral guard** | 0.9291 | +3 | **0** | 0 / 0 | **tie — never fires** |
| **Lexical cue protection** | 0.9279 | +2 | 1 | 0 / **1** | **worse (−1)** |
| **Combined guard** | 0.9279 | +2 | 1 | 0 / 1 | **worse (−1)** |
| **w_primary = 1.0 / 1.5 / 2.0** (label-proxy) | 0.9242 | −1 | 6 | 1 / 5 | worse |
| **w_primary = 3.0** (label-proxy) | 0.9254 | 0 | 18 | 7 / 10 | = primary_only |
| **minority-trust (Contextual lone-dissent)** | — | — | — | — | **−2** |
| **minority-trust (Lexical lone-dissent)** | — | — | — | — | −2 |
| **minority-trust (Contextual + Lexical)** | — | — | — | — | −4 |
| **role-priority (Contextual override)** | ≡ minority-trust | — | — | — | −2 |

**Why each failed or tied:**
- **Contextual neutral guard — tie (0 interventions).** Every case it would protect is *already*
  caught by the IntentGate; the gate **saturates the neutral-protection domain**, so a second
  neutral guard on the strongest agent recovers nothing.
- **Lexical cue protection — worse (−1).** The one case it fired on was one the panel had
  *correctly* neutralized; restoring the polar broke it (`neutral→negative`). Consistent with
  Lexical's 0.37 dissent precision — protecting its polar cue adds errors.
- **Combined guard — worse (−1).** = Lexical protection (the Contextual guard never fires); inherits
  its −1.
- **w_primary sweep — no gain.** w=1–2 over-corrects (−1); w=3 makes the primary dominate →
  consensus = primary_only (net 0, 0 overrides). It tops out **at** the primary, never above —
  matching the confidence-based `EXPERIMENT_AHMED_CONSENSUS_SIMULATION.md`.
- **Minority-trust / role-priority override — worse (−2/−4).** Unconditionally adopting a lone
  dissenter loses, because a specialist's *incorrect* dissents cost as much as its *correct* ones
  recover; even the best agent (Contextual, 0.58) nets −2. Role-priority-as-override is exactly this
  and fails identically; the circular problem is that detecting "this is Contextual's domain" is as
  hard as the classification itself.

## 6. Why the IntentGate worked (when re-fusion did not)
The IntentGate is the **one** aggregation change that improved the system (C net +1 → G net +2,
0.9267 → 0.9279; C→W 11 → 8). It succeeds because it is **not** a vote and **not** a reweighting —
it is a **domain-restricted, non-overriding veto**:
- **Non-overriding:** it only *blocks* an unsupported polar override of a neutral primary; it never
  forces a flip. So its downside is bounded (2–3 hurt) and it cannot be outvoted.
- **Domain-restricted:** it acts only where a specialist is reliably right — the meta/mention/
  no-opinion domain — which is *detectable* (that is Intent's whole job).
- **The decisive contrast:** the identical pragmatic signal is **12/12 suppressed as a vote** (Design
  E) but **0 missed as a veto** (Design G). Same information, opposite fate — a veto cannot be
  outvoted by the correlated bloc, whereas a vote is. This is why the gate is the only lever that
  moved the needle, and why re-fusion of the *votes* cannot (§4–5).

## 7. What consensus approaches remain untested
- **Learned meta-consensus** — a classifier over agent labels, confidences, agreement pattern,
  primary confidence, and gate outputs, predicting the correct final per case.
- **Confidence-calibrated arbitration** — trust a confident minority over an uncertain majority,
  *after* calibrating (e.g. temperature/Platt scaling) the agents' confidences.
- **Stronger / heterogeneous agents** — replace the single shared base model (gpt-4o-mini) with a
  panel of *different* models so errors decorrelate (the ensemble premise that is currently violated).
- **Dev-trained selector** — any rule/weights fit on a held-out dev split (not the test captures).

## 8. Why those are risky / not doable now
- **Small sample size.** Only ~80 escalated samples per design; a learned selector or fitted weights
  over correlated features will overfit and not generalize.
- **Test leakage.** All current per-agent captures are on the **test** set; fitting or selecting any
  rule on them and reporting on the same set is invalid. A clean selector needs **dev-set** captures.
- **Poor confidence calibration.** The agents are over-confident and their self-confidence does **not**
  track correctness on the hard subset (established via router-selectability); confidence-based
  arbitration on raw scores is expected to add noise, not signal.
- **No per-agent confidence history.** The design captures serialized **labels only** — confidences
  were not stored — so confidence-aware or learned methods **cannot even be simulated offline**; they
  require a **new paid capture pass** (and, for a dev-trained selector, capturing the escalated dev
  subset too).

## 9. Final conclusion
> **Consensus is a genuine bottleneck — the panel discards 4–11 correct answers per 84 (oracle
> 0.80–0.88 vs ~0.75) — but *simple* aggregation has now been exhaustively ruled out.** Majority/
> weighted voting, specialist neutral/cue guards, w_primary re-tuning, and minority-trust/role-
> priority overrides were all simulated offline: none beats G (they tie, or lose −1 to −4). The
> only aggregation change that ever helped is the **domain-restricted, non-overriding veto**
> (IntentGate), and it is already in G. **The next safe consensus direction is not another
> label-based rule — it requires new information: either calibrated per-agent confidence (which
> must first be captured and validated) or genuinely decorrelated evidence (heterogeneous models /
> dev-trained selection).** On the strong primary the system is at its aggregation ceiling; the one
> regime where any consensus refinement can still pay off is the **weak (C3 generated) primary**,
> where the consensus loss is larger and the primary term is small.

## Source reports
- `EXPERIMENT_CONSENSUS_LOSS_ANALYSIS.md` — suppression counts, vote-vs-veto.
- `EXPERIMENT_CONSENSUS_LOSS_AGGREGATION_DESIGN.md` — lone-dissent precision, option evaluation.
- `EXPERIMENT_G_OFFLINE_CONSENSUS_RESCORING.md` — the offline rule simulation.
- `EXPERIMENT_ESCALATED_CEILING_ROOT_CAUSE.md` — the ceiling / correlation floor.
- `EXPERIMENT_SENTIMENT_INTENT_GATE_ABLATION.md` — the working veto (G).
- `EXPERIMENT_AHMED_CONSENSUS_SIMULATION.md` — confidence-based w_primary sweep (default prompts).
- `EXPERIMENT_AHMED_ROUTER_SELECTABILITY.md` — confidence uninformative on primary errors.
