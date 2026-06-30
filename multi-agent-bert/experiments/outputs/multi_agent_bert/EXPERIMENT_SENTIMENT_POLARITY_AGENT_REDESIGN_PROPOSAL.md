# Sentiment Polarity-Agent Redesign — Design Proposal (no implementation)

Does the current **Lexical + Logic + Contextual** decomposition remain the right
specialist split for sentiment, or should a dedicated **Polarity Agent** replace/augment
it? **Design analysis only — no prompt changes, no runs, no training/generation.** General
sentiment reasoning; **no dataset/benchmark named.** Date: 2026-06-30.

## Evidence motivating a decomposition change
From the B1 semantic_v1 ablation (84 escalated, measured):
- **Logic is still the weakest agent** even after semantic_v1: **0.690** — and it remains
  the most redundant, ~**0.89 pairwise agreement with Lexical**. Its distinctive
  structural mandate is the least productive of the three.
- **Contextual is the strongest**: **0.750** (ties the strong primary), and the only agent
  that independently resists the surface read.
- **Lexical**: 0.726 — the cue-inventory role.
- semantic_v1 cut all-3 agreement **91.7% → 84.5%** (good), but **final consensus 0.726
  is still below the strong primary (0.750)** on the escalated subset.
- Prompt refinement closed **half** the agentic gap (net −4 → −2) but did not lift the
  agents past the **~0.73 agent ceiling** on hard code-switched cases.

**Reading:** the panel still contains **two overlapping surface agents** (Lexical and
Logic, ~0.89 correlated) and **one strong distinct agent** (Contextual). The redundant
pair is the structural inefficiency: it spends two of three votes on near-identical
surface reasoning, so the bloc is effectively ~2 independent opinions, and the weaker of
the pair (Logic) drags the consensus. A decomposition that converts the redundant pair
into **one disciplined polarity decision** + keeps the strong pragmatic agent is the
hypothesis under test.

> **Honest scope caveat (applies to all designs).** The consensus-simulation and
> behaviour-comparison findings show that on a **strong** primary (≈0.92) *no* agent
> configuration beats primary_only — the best a consensus can do is recover it by not
> overriding. So this redesign's realistic target is **(i)** improving the *quality of the
> rescue* where the primary is weak/mid (the C3-generated and mid-strength regime, where
> agents genuinely help) and **(ii)** *reducing harm* on a strong primary by being more
> accurate and better-calibrated. It is **not** expected to make the agents beat a 0.92+
> primary; that is an agent-ceiling problem, not a decomposition problem.

---

## The Polarity Agent — general definition (dataset-agnostic)
- **Role:** decide whether the author **explicitly expresses** positive, negative, or
  neutral sentiment — i.e. output an *expressed polarity decision*, not a cue list.
- **Must separate polarity *expression* from mere *mention*** of sentiment-related words.
- **Handles:** explicit sentiment words, **negation**, **intensifiers**, **mixed/conflicting
  polarity**, **emojis**, and weak/artifact cues — weighing them, not just detecting them.
- **Confidence discipline:** returns **lower confidence** when polarity is weak,
  artifact-based, or the **target is ambiguous** (whose sentiment, toward what).
- **Boundaries:** does **not** perform full pragmatic/social interpretation (sarcasm,
  irony, communicative intent) — that stays with **Contextual**. Does **not** merely
  enumerate lexical cues — it **decides** expressed polarity.

This is generic: "expressed pos/neg/neutral polarity" is a universal sentiment construct,
independent of any corpus, register, or language pair.

---

## Designs compared

### A. Current semantic_v1 — Lexical + Logic + Contextual (baseline)
1. **Independence/diversity:** moderate. Improved by semantic_v1 (84.5% agree) but
   Lexical–Logic still ~0.89 → effectively ~2 independent opinions.
2. **Surface-cue literalism risk:** still present — two surface agents can co-over-read;
   13 breaks remain.
3. **Neutral vs polar boundary:** weakest spot (breaks dominated by neutral→polar); no
   single agent *owns* the polarity decision.
4. **Implementation effort:** zero (this is the shipped semantic_v1).
5. **Task-aware / not dataset-specific:** yes.
6. **C3 generated-primary risk:** none (it *is* the reference; C3 +0.059 was measured on
   the prior default prompts, semantic_v1 effect on C3 still unmeasured = B2).
7. **Consensus:** 3 agents × weight 1 + primary (w=1.0) → bloc weight 3 vs primary 1.

### B. Polarity + Contextual (replace Lexical **and** Logic with one Polarity)
1. **Independence/diversity:** **highest structural diversity** — two genuinely different
   jobs (polarity decision vs pragmatic interpretation), zero redundant surface pair. But
   only **2 voters** → least ensemble averaging; a single Polarity error has no second
   surface agent to dampen it (Contextual still checks).
2. **Surface-cue literalism risk:** **lowest by design** — collapses the two literalist
   agents into one *decider* told to weigh cues, not list them. Concentrates the surface
   judgement in one disciplined place.
3. **Neutral vs polar boundary:** **best positioned** — a dedicated agent whose explicit
   job is the pos/neg/neutral decision with weak-cue→low-confidence and mention-vs-express
   built in.
4. **Implementation effort:** **medium** — author one Polarity agent/prompt, retire two
   agents, and **rebalance consensus** (voter count changes 3→2).
5. **Task-aware:** yes.
6. **C3 risk:** **highest of the redesigns** — fewer rescuers (2 vs 3) reduces the bloc's
   ability to out-vote a weak primary, which is exactly where C3's +0.059 came from. Could
   still rescue if Polarity+Contextual are individually strong, but the margin shrinks.
7. **Consensus:** 2 agents + primary → bloc weight 2 vs primary 1. Tie risk rises (agents
   can split 1–1); the existing **primary-label-if-tied** rule resolves it (leans toward
   the primary — protective on strong primary, possibly under-rescuing on weak primary).
   May want **Polarity slightly heavier** than Contextual, or keep equal initially.

### C. Lexical + Polarity + Contextual (replace **Logic only**)
1. **Independence/diversity:** **good** — removes the weakest, most-redundant agent
   (Logic) and installs a distinct decision role. Three roles: cue inventory (Lexical) /
   polarity decision (Polarity) / pragmatics (Contextual). **Watch-item:** Lexical and
   Polarity both touch sentiment vocabulary → they must be sharply separated (Lexical =
   *evidence/strength only, low-confidence reporter*; Polarity = *the decision*) or they
   re-form a correlated pair — the very problem being fixed.
2. **Surface-cue literalism risk:** **moderate-low** — Lexical still reports cues
   (literalist by design) but is the *reporter*, not the decider; Polarity arbitrates.
   Lower than A (Logic's redundant literalist vote is gone), slightly higher than B
   (Lexical still votes).
3. **Neutral vs polar boundary:** **good** — Polarity owns it; cleaner than A, marginally
   less clean than B (Lexical still contributes a surface vote).
4. **Implementation effort:** **medium, lowest friction** — one new agent replaces Logic;
   **agent count stays 3 → consensus is unchanged**, so the swap is a clean drop-in and the
   effect is cleanly isolatable.
5. **Task-aware:** yes.
6. **C3 risk:** **lowest** — keeps **3 voters / bloc weight 3**, identical to the config
   that produced C3 +0.059; only swaps Logic→Polarity. Rescue power preserved; most likely
   to retain C3 gains.
7. **Consensus:** **unchanged** (3 agents × weight 1 + primary 1.0). No rebalancing needed.

### D. Lexical + Logic + Contextual + Polarity (add Polarity as a 4th agent)
1. **Independence/diversity:** **adds to the correlated cluster**, not away from it —
   Lexical + Logic + Polarity all touch sentiment vocabulary → potentially **three**
   correlated surface agents vs one Contextual. Worsens the bloc imbalance unless Polarity
   is made very distinct.
2. **Surface-cue literalism risk:** **highest at the consensus level** — bloc weight rises
   to **4 vs primary 1**, i.e. **more** override power. On a strong primary (where B1
   showed we need *less* override) this pushes the **wrong way**.
3. **Neutral vs polar boundary:** a dedicated agent is added but its vote is **diluted**
   among four and the surface cluster still dominates.
4. **Implementation effort:** **highest** — new agent **plus** mandatory consensus surgery
   (4-vs-1 is too lopsided; must raise w_primary or down-weight the surface cluster).
5. **Task-aware:** yes.
6. **C3 risk:** mixed — more rescuers *could* help the weak C3 primary, but the same extra
   override power **hurts the strong-primary case more** (opposite of the B1 lesson). Net
   direction depends entirely on consensus retuning.
7. **Consensus:** 4 agents → **most surgery**. Would need w_primary↑ (~2) or a
   surface-cluster down-weight to avoid a 4-vote bloc steamrolling a strong primary; also
   raises the chance of multi-way ties.

---

## Comparison matrix
| criterion | A (Lex+Log+Ctx) | B (Pol+Ctx) | C (Lex+Pol+Ctx) | D (Lex+Log+Ctx+Pol) |
|---|---|---|---|---|
| 1. independence/diversity | moderate | **highest** | good | low (bigger surface cluster) |
| 2. literalism risk | present | **lowest** | moderate-low | highest (bloc 4) |
| 3. neutral↔polar boundary | weakest | **best** | good | diluted |
| 4. implementation effort | none | medium | **medium, lowest-friction** | highest |
| 5. task-aware / generic | yes | yes | yes | yes |
| 6. C3 (weak-primary) risk | n/a (ref) | highest | **lowest** | mixed/uncertain |
| 7. consensus change needed | none | rebalance (3→2) | **none (stays 3)** | most (4-vs-1) |

---

## Consensus handling — summary of item 7
- The current consensus is a **confidence-weighted vote**: each agent contributes
  `weight × confidence`; the primary contributes `w_primary × primary_conf` (w_primary=1.0);
  winner = argmax; non-positional tie-break (primary-label-if-tied → most agents → highest
  single contribution → alphabetical). **Agent count directly sets the bloc weight vs the
  primary**, which is the lever for help-vs-harm.
- **C** needs **no change** (stays 3 voters) — clean isolation of the decomposition effect.
- **B** changes 3→2: keep w_primary=1.0 (bloc 2 vs 1 still overrides on agreement); rely on
  the existing primary-tie rule; optionally weight Polarity ≥ Contextual. Re-check tie
  behaviour.
- **D** changes 3→4: the 4-vote bloc is too strong against a single primary — **must** raise
  w_primary (or down-weight the surface cluster) to preserve the B1 lesson that a strong
  primary should rarely be overridden.
- **General principle (from the consensus simulation):** for a *strong* primary the optimal
  rule is *near-non-overriding*; for a *weak* primary the bloc *should* override. Any
  decomposition should be paired with the **per-primary** calibration that already exists
  (threshold + w_primary), not a one-size weight.

---

## Recommendation — implement **Design C first** (replace Logic only)
**Why C first:**
1. **Targets the proven defect with minimal confounds.** Logic is the *measured* weakest
   (0.690) and most redundant (~0.89 with Lexical). C removes exactly that link and tests
   whether a dedicated **decider** beats the redundant structural agent — the most direct
   test of the decomposition hypothesis.
2. **Cleanest experiment.** Agent count stays 3 → **consensus unchanged** → any change in
   results is attributable to the Logic→Polarity swap alone, comparable head-to-head with
   semantic_v1 and with the C3 reference.
3. **Lowest regression risk on the regime that actually benefits.** It preserves the
   3-voter bloc weight that produced the C3 +0.059 rescue, so it is the least likely to
   erode weak-primary gains (the place the agentic layer earns its keep).
4. **Keeps the strongest agent and a sharpened reporter.** Contextual (0.750) is retained;
   Lexical is re-scoped to a low-confidence *evidence reporter* so it no longer competes
   with the decision role.

**Staged fallback to B:** the one risk in C is Lexical↔Polarity re-correlating (both read
sentiment words). The first validation must **measure their pairwise agreement**. If they
behave as a new redundant pair (agreement back ≳0.9 and Lexical adds no independent signal),
**merge them → Design B** as the second step. So the plan is **C → (measure) → B if needed**,
which also gives a clean ablation of "is a separate cue-reporter worth a vote?".

**Not recommended first:** **D** (adds override power in the wrong direction for strong
primaries and needs the most consensus surgery) and **A** (it is the baseline we are trying
to beat).

## Proposed validation (when approved — not now)
Two primaries that bracket the strength curve, **prompt-only**, no training:
- **Strong primary (Ahmed frozen, th 0.7):** confirm C does **no worse** than semantic_v1
  (target: net ≥ −2, fewer literalism breaks) — the harm-reduction check.
- **Weak primary (C3 generated, th 0.9):** confirm C **retains** the rescue (target:
  net gain ≈ the +0.059 regime) — the gain-preservation check.
- **Measure for the decomposition specifically:** per-agent accuracy (does Polarity beat
  Logic's 0.690?), **Lexical↔Polarity pairwise agreement** (the C→B trigger), all-3
  agreement vs 84.5%, and the neutral↔polar break matrix.

---
*(Design only. No agent/prompt/consensus code changed. No LLM calls, no training, no
generation. Implementation deferred until a design is approved.)*
