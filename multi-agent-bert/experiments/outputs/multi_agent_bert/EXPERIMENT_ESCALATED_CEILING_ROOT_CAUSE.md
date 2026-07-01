# Root-Cause Analysis — Why Every Specialist Converges to ~0.72–0.76 on the Escalated Subset

Synthesis across all sentiment agent designs (A / semantic_v1, B, C, D, E, F, G, G2, v3) on the
Ahmed frozen primary. **Analysis only — no runs, no new design, no prompt.** The question:
every redesigned specialist lands at ~0.72–0.76 on the 84 escalated cases despite different
prompts and reasoning decompositions, and component improvements do not translate into
end-to-end gains. What is the underlying mechanism? Date: 2026-07-01.

---

## 1. The observation, quantified
Per-agent accuracy on the **same 84 escalated samples**, every design (frozen primary is a
different model — TF/Keras ensemble, no LLM — shown as reference):

| design | lexical | logic/polarity | contextual | intent(gate) | **primary (ref)** | final |
|---|---|---|---|---|---|---|
| A semantic_v1 | 0.726 | 0.690 | 0.750 | — | **0.750** | 0.726 |
| C | 0.714 | 0.738 | 0.738 | — | **0.750** | 0.738 |
| E | 0.714 | 0.738 | 0.738 | 0.714 | **0.750** | 0.738 |
| F | — | 0.714 | 0.750 | 0.714 | **0.750** | 0.726 |
| G | 0.738 | 0.738 | 0.738 | 0.726 | **0.750** | 0.774 |
| G2 | 0.702 | 0.726 | 0.738 | 0.714 | **0.750** | 0.750 |
| v3 | 0.714 | 0.726 | **0.762** | 0.690 | **0.750** | 0.774 |

Every specialist, every prompt, every decomposition — and the **frozen primary itself** — sits
in **0.69–0.76, centered on 0.75.** That an unrelated model (the primary) also scores exactly
0.75 on this subset is the decisive clue: **0.75 is a property of the subset, not of the agents.**

## 2. The governing equation (fits every experiment)
The net end-to-end effect of the agentic layer is, to first order:

> **Δaccuracy ≈ (agent-ceiling − primary-accuracy-on-escalated) × escalation-rate**

| primary | primary acc on escalated | agent ceiling | Δ (measured) |
|---|---|---|---|
| C3 generated | 0.54 | ~0.75 | **+0.059** |
| EESA XLM-R | 0.56 | ~0.75 | **+0.027** |
| **Ahmed (strong)** | **0.75** | **~0.75** | **≈ 0** (A–v3: −0.005 … +0.002) |

The agent ceiling ~0.75 is **fixed**; the only term that changes Δ is the **primary's accuracy
on the escalated subset** — a property of the primary and router, **not of the agent prompts.**
Prompts do not appear in the equation, so every redesign nets ~the same on a strong primary.
The rest of this document explains (i) why the ceiling is fixed and (ii) why component gains
don't propagate.

## 3. Why the ~0.75 ceiling is fixed — two stacked floors

### (a) Selection + information-theoretic floor — dominant
The router escalates *low-primary-confidence* cases: an adversarial selector that hands the
agents exactly the inputs where the label is least determined by the text. The residual splits
into two irreducible populations (from the error typing in `EXPERIMENT_G_TO_093_GAP_ANALYSIS.md`):
- **~⅓ label-convention** (ad→neutral, advice→neutral, sarcasm→always-negative, mildly-positive
  →positive). The text **under-determines** the gold label — the label carries bits from an
  annotation convention **not present in X**. `H(Y|X) > 0` structurally; no reasoning recovers
  bits the input does not contain.
- **~⅔ cue-less implicit pragmatics** (sarcasm / insult / praise on 2–12-word code-switched
  comments, e.g. "انت no one", "out of season"). The stance lives in shared cultural/contextual
  knowledge, not in the tokens — again, **missing bits**.

A large slice of the residual is therefore **Bayes-irreducible given the input alone.** The
primary scoring 0.925 overall but 0.75 here *quantifies* the difficulty: this band's separability
is ~0.75 for **any text-only predictor**.

### (b) Shared-model competence floor — amplifier
Every specialist is the **same base model** (gpt-4o-mini) reading the **same input**. Ensemble
theory: a vote helps only insofar as members err **independently**. Measured agreement is
**84–92%** → errors are highly correlated → **ensemble error ≈ member error**. Reprompting changes
*which* cases a member gets right (semantic_v1 cut agreement 92%→84.5%) but cannot remove the
**shared blind spot** — implicit pragmatics on terse bilingual text — because that blind spot is a
property of *(base model × input)*, not of the prompt.

Floor (a) is a hard wall no prompt crosses; floor (b) is a soft wall only a *stronger base model*
could raise. Together they pin the ceiling at ~0.75.

## 4. Why component gains don't translate — conservation of difficulty × dilution

### Conservation of difficulty (within a member)
Each redesign **reallocates** a member's correct-set rather than expanding it — it fixes error
type X and, on this hard core, introduces error type Y. Direct evidence:
- **F**: fixed meta-comment structure but over-neutralized → net 0.
- **G2**: recovered 1 implicit insult, **lost 3** platform blocks → net 0.
- **v3**: Contextual **+0.024** (fixed a meta break) but **added** one description-vs-evaluation
  over-neutralization (`00113` "wtf") → net 0.
- **semantic_v1**: net −4 → −2 purely by **trading** break types.
The hard core is a roughly fixed-size set; prompts push errors around inside it.

### Dilution × correlation (across members)
For an improved member to change the **output**, it must **flip the consensus** — be the outvoted
dissenter and now tip it. But members are correlated: when Contextual was wrong the others were
usually wrong too (no lone dissent to amplify); when it newly gets one right, the others were
already right (redundant) or still wrong (it is outvoted 1–2). So marginal single-agent gains land
exactly where they **cannot** move the aggregate. v3 changed only 4/84 Contextual labels; none
flipped a consensus that wasn't already going there.

### Strong-primary parity (the closing term)
With w_primary = 1.0 and the primary at 0.75 = the agents' 0.75, the consensus is at equilibrium:
the agents can only alter the outcome on the ~10–15% where they collectively disagree with the
primary, and there they are right about as often as wrong. Net agent contribution ≈ 0 **by
construction** — the design correctly declining to override a primary as good as the panel.

## 5. Ranking the candidate causes
| candidate | role | binding? |
|---|---|---|
| **Information-theoretic limit of the (selected) input** | sets the hard floor (~⅓ irreducible + missing context) | **PRIMARY / binding** |
| **Shared-base-model correlated competence** | sets the soft floor; ensemble ≈ member | **PRIMARY / binding** |
| Consensus dilution | blocks *translation* of member gains to the output | downstream (why gains don't show) |
| Remaining error *types* | surface symptom of the floors | descriptive, not causal |
| LLM capability | governs the *reducible* part of floor (b) | contributory (a stronger model raises b, not a) |

**Verdict:** the convergence is fundamentally **(selected input) × (shared model)** — the router
hands the agents an intrinsically ~0.75-separable subset, and a correlated ensemble of one base
model cannot exceed that subset's separability by any prompt. Consensus dilution and error-type
analysis are *how* this manifests, not the root.

## 6. Falsifiable proof that this is the mechanism
If the ceiling were prompt/reasoning-limited, the **weak-primary** result would match the strong
one. It does not: the *identical* agents at the *same* ~0.75 ceiling produce **+0.059** on C3 and
**≈0** on Ahmed. Same agents, same ceiling, opposite outcome — the only change is
`primary-on-subset` (0.54 vs 0.75). This single contrast **falsifies "prompt-limited"** and
confirms `Δ = (ceiling − primary_on_subset) × rate`. The agents' value is entirely the *gap*
between a fixed ceiling and the primary.

## 7. What (mechanistically) could move it — and what cannot
- **Reprompting / re-decomposition — cannot.** It moves members inside a fixed feasible set
  (floors a + b). Every A–v3 result confirms this.
- **Feature hints (Ahmed's engineered features) — cannot for these error types.** They add no bits
  for convention/implicit cases; Ahmed's own GPT-3.5 + sentiment-hints *dropped* to 0.69.
- **Only three levers change the number, none of them a prompt:**
  1. **A different base model** with genuinely better code-switched pragmatic inference (raises
     floor b — capability).
  2. **Changing the subset**: a better-calibrated primary makes the escalated band less
     adversarial and lowers the `primary-on-subset` term — i.e. move to the **weak/mid-primary
     regime where the agents demonstrably pay off** (C3 +0.059).
  3. **Adding the missing bits to X**: thread/multimodal context, or adopting the annotation
     convention (= dataset tailoring).

## 8. One-sentence mechanism
> **Every specialist converges to ~0.75 because that is the Bayes-limited separability of the
> router-selected hard subset for a text-only, single-base-model panel; and component
> improvements do not translate because they reallocate a conserved difficulty budget inside a
> correlated ensemble that, against a parity-strength primary, has zero net headroom.**

## Corollary for the project
- On a **strong** primary the agentic layer is at ceiling — further prompt/agent work is
  noise-bound; the honest ceiling is ~0.928–0.930 (see `EXPERIMENT_G_TO_093_GAP_ANALYSIS.md`).
- The agents' value is real but lives **where the primary is weak** (C3 generated). The single
  decisive open experiment is therefore **G (and v3) on the C3 generated primary**, not more
  agent redesign on the strong primary.

## Artifacts / evidence base
- Per-agent tables: `EXPERIMENT_AGENT_BEHAVIOR_COMPARISON.md` (agent ceiling ~0.70–0.75 across
  C3/EESA/Ahmed), each design's `…/error_attribution/attribution_table.json`.
- Selection/calibration: `EXPERIMENT_AHMED_ROUTER_SELECTABILITY.md` (primary confidence
  uninformative on its own errors).
- Error typing & ceiling: `EXPERIMENT_G_TO_093_GAP_ANALYSIS.md`,
  `EXPERIMENT_SENTIMENT_PRAGMATIC_CONTEXTUAL_V3.md` (component-up / system-flat),
  `EXPERIMENT_SENTIMENT_SELECTIVE_INTENT_GATE_G2.md`, `EXPERIMENT_SENTIMENT_INTENT_GATE_ABLATION.md`.
