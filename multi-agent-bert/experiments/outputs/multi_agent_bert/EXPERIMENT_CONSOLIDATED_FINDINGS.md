# Multi-Agent Sentiment — Consolidated Findings (thesis summary)

One-page consolidation of the campaign: does a multi-agent LLM layer improve an Arabic-English
code-switched sentiment classifier, and does generated data help? All numbers on the EESA test
(818), primary_only vs the best agent config. Date: 2026-07-05.

---

## 1. Headline result — the primary-strength curve (agents help more as the primary weakens)

Best agent config = **G2 (Lexical+Polarity+Contextual+selective IntentGate) @ gpt-4.1-mini**.

| primary (all evaluated on EESA test) | standalone acc | primary on its escalated subset | **+ agents** | gain | net | significant? |
|---|---|---|---|---|---|---|
| **C3** — XLM-R on *generated* data (weak) | 0.6956 | 0.541 | **0.7665** | **+0.071** | +58 | p ≪ 0.001 ✅ |
| **E0** — XLM-R on *EESA* (mid, our recipe) | 0.8533 | 0.621 | **0.8826** | **+0.029** | +24 | p ≈ 0.002 ✅ |
| **Ahmed** — external precomputed (strong) | 0.9254 | 0.750 | **0.9303** | +0.005 | +4 | p ≈ 0.37 ✗ |

**Three findings, one mechanism:**
1. **Agent gain shrinks monotonically as the primary strengthens** (+7 → +3 → +0.5 points);
   significance fades from p≪0.001 to not-significant.
2. **The agents lift the hard (escalated) cases to a stable ceiling ~0.76–0.79 regardless of
   primary strength** (C3 0.766, E0 0.763, Ahmed 0.786).
3. So the gain is fully explained by **how far the primary starts below that ceiling on its own
   hard cases**: *gain ≈ (agent-ceiling − primary-on-escalated) × escalation-rate*. Confirmed at
   three real strengths (the earlier +0.027 estimate for E0 was measured as +0.029 — essentially
   exact).

**Best configuration per regime:** weak primary → **G @ 4.1-mini** (full gate fine); strong
primary → **G2 @ 4.1-mini** (selective gate; the full gate over-vetoes a strong model). On the
weak primary G ≈ G2 (gate choice is second-order when the agents rescue many errors).

---

## 2. Data augmentation — what worked and what didn't (honest: nothing robustly helped)

"Real EESA + generated data" vs "real EESA alone", EESA test:

| experiment | setup | result vs real-only |
|---|---|---|
| LR + GEN-960 | 10 / 25 / 50 % EESA + full gen | −0.034 / −0.018 / −0.002 → **HURT / HURT / tie** |
| E-V1 + V1 (Arabic-dominant gen) | 10 / 25 / 50 % EESA + V1 (gen ≤50%) | −0.022 / +0.001 / +0.001 → **HURT / tie / tie** |
| E3 | full EESA + gen-960 | −0.012 → **HURT** |
| ratio-sweep | EESA% + gen at 0.25/0.5/1.0 | mostly HURT; best +0.006 (noise) |

- **No setting clearly beat real-only.** The only two positive deltas (+0.005, +0.006) are
  4–5 samples, single-seed, inconsistent across ratios → **noise, not wins.**
- Augmentation **hurts** when gen is a large fraction or the real base is small; is at best a
  **tie** when the real base is large and gen is a small capped fraction.
- **Domain-proximity helps *relatively* but not absolutely:** the more-EESA-like V1 (Arabic↑,
  CMI↓) was a *less-harmful* augmenter than GEN-960 (+0.01–0.02 vs 960 at every ratio) — but
  still never beat real-only.
- **Generated-only transfer:** V1 (0.630) transferred *worse* than the matched GEN-480 (0.650)
  — moving the generation control toward EESA's Arabic%/CMI did **not** improve standalone
  transfer, because the proximity came from *reduced* code-switching, the very signal transfer
  needs.

**BUT — two-stage use of gen DOES help (the exception that worked):** using the generated data as
**pretraining** (gen-960 → fine-tune on real EESA%), not mixed in, beats real-only at 25% (+0.022)
and 50% (+0.015) real data — accuracy *and* macro-F1 — and beats naive mixing at every ratio. Same
gen data, different method: mixing distorts the decision boundary; two-stage lets the real
fine-tune have the last word. (Hurts only at 10%, where too little real data can't steer the
gen-heavy init.) See `EXPERIMENT_TWOSTAGE_GEN_AUGMENTATION.md`.

**So the value of generated data is twofold:** (a) as **two-stage pretraining** it augments a real
classifier (+1.5–2.2 pts, ≥25% real); and (b) as a **substitute** for missing real data it builds
the weak primary where the agent layer delivers its largest gain (C3, +0.071). What does NOT work
is **mixing gen into real training** — that is neutral-to-harmful at every ratio.

---

## 3. What does NOT move the ceiling (ruled out on the strong primary)

Four interventions, all ≤ 0 net and non-significant on the strong (Ahmed) primary:
- **Prompt redesigns** (semantic_v1, v3 pragmatic, semantic_v2 disambiguation) — 0 to −4.
- **Sequential/staged reasoning** (v1 review-based, v2 forward-pragmatics) — −1 and −11.
- **A stronger model** (4o-mini → 4.1-mini) — +1 sample, non-significant (fixes ~4 cases, breaks
  ~3: a wash, because the residual errors are cultural/knowledge gaps, not instruction gaps).
- **Gate variants** (no-gate, selective gate) — selective helps *only* on a strong model
  (gate aggressiveness must scale inversely with model strength), never breaks the ceiling.

**Why:** the residual strong-primary errors are ~78% *information floor* — implicit Egyptian
insults / sarcasm the base model doesn't know — which no prompt or topology can fix.

---

## 4. Plain-language thesis takeaways
1. **The multi-agent layer works — where it should.** On a real EESA-trained XLM-R it gives a
   solid, significant **+2.9 points**; on a weak (generated-data) classifier, **+7 points**; on
   an already-strong classifier, ~nothing. The value scales with the base model's weakness.
2. **There is a firm agent ceiling (~0.77 on the hard cases).** The agents can only take the
   uncertain cases so far; the way to push higher is a **better primary**, not more agents.
3. **Generated data does not augment a real classifier** — it's a *stand-in* for scarce real
   data, useful precisely because it creates the weak-primary regime the agents rescue.
4. **You cannot prompt-engineer past the ceiling** on a strong primary; the residual is model
   knowledge, not instructions.

## Source reports
Curve: `EXPERIMENT_EESAXLMR_MIDPRIMARY_G2.md`, `EXPERIMENT_C3_GPT41_GATE_BUNDLE.md`,
`EXPERIMENT_GPT41_GATE_ABLATION.md`, `EXPERIMENT_ESCALATED_CEILING_ROOT_CAUSE.md`.
Augmentation: `EXPERIMENT_V1_LOWERCS_SENSITIVITY.md`,
`EXPERIMENT_E_AUGMENTATION_CONSOLIDATED_SUMMARY.md`, `EXPERIMENT_E_AUGMENTATION_FAILURE_DIAGNOSIS.md`.
Ceiling/negatives: `EXPERIMENT_G_DISAMBIG_AHMED_RESULTS.md`,
`EXPERIMENT_SEQUENTIAL_SENTIMENT_V2_AHMED_RESULTS.md`, `EXPERIMENT_WHY_STRONGER_MODEL_BROKE_CASES.md`.
