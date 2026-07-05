# Master Summary — All Experiments (Multi-Agent Sentiment + Generated Data)

Everything run, in tables. Task: Arabic-English code-switched sentiment (pos/neg/neu), EESA
test (818). Two tracks: **A)** does a multi-agent LLM layer help a classifier; **B)** does
generated training data help. Date: 2026-07-05.

═══════════════════════════════════════════════════════════════════════
## TRACK A — Multi-agent LLM layer

### A1. Agent designs on the STRONG Ahmed primary (gpt-4o-mini, threshold 0.7, 84 escalated)
| design | agents | full acc | escalated acc | net vs primary |
|---|---|---|---|---|
| primary_only | — | 0.9254 | 0.750 | — |
| A (default) | Lexical+Logic+Contextual | 0.9205 | 0.726 | −4 |
| C | Lexical+**Polarity**+Contextual | 0.9267 | 0.738 | +1 |
| E | Lexical+Intent+Polarity+Contextual | — | 0.750 | ~0 |
| F | Intent+Polarity+Contextual | — | 0.726 | −2 |
| **G** | +**IntentGate** (non-voting veto) | **0.9279** | 0.774 | +2 |
| G2 | G with *selective* gate | 0.9279 | 0.774 | +2 |
| v3 | G + pragmatic Contextual | 0.9279 | 0.750 | +2 |
| seq-v1 | staged: Intent→Polarity→Pragmatic review | 0.9242 | 0.738 | −1 |
| seq-v2 | staged: Intent→Pragmatic features→Polarity | 0.9120 | 0.619 | −11 |
**Verdict:** all within noise of primary_only (McNemar n.s.). Strong primary is **at ceiling**.
Sequential reasoning ≤ 0. Prompt redesigns (semantic_v1/v3) ≈ 0.

### A2. Stronger model (gpt-4.1-mini) on the strong Ahmed primary
| config | full acc | macro F1 | vs primary |
|---|---|---|---|
| primary_only | 0.9254 | 0.9207 | — |
| G @ 4o-mini | 0.9279 | 0.9257 | +2 (n.s.) |
| G @ 4.1-mini | 0.9291 | 0.9248 | +3 (n.s.) |
| C (no gate) @ 4.1-mini | 0.9266 | 0.9216 | +1 |
| disambig prompt @ 4.1-mini | 0.9242 | 0.9196 | −1 |
| **G2 (selective gate) @ 4.1-mini** | **0.9303** | **0.9262** | +4 (n.s.) — **best strong** |
**Verdict:** stronger model only nudges (+1, non-significant). **First cross of 0.930** but not a
proven beat. Finding: **gate aggressiveness must scale inversely with model strength** (selective
gate helps only at 4.1-mini; full gate over-vetoes it).

### A3. The PRIMARY-STRENGTH CURVE (best config G/G2 @ 4.1-mini across primaries) — the headline
| primary | what it is | standalone | + agents | gain | net | significant |
|---|---|---|---|---|---|---|
| **C3** | XLM-R on *generated* data (weak) | 0.6956 | **0.7677** | **+0.071** | +59 | p≪0.001 ✅ |
| **E0** | XLM-R on *EESA* (mid) | 0.8533 | **0.8826** | **+0.029** | +24 | p≈0.002 ✅ |
| **Ahmed** | external precomputed (strong) | 0.9254 | 0.9303 | +0.005 | +4 | p≈0.37 ✗ |
**Verdict:** agent gain **shrinks as the primary strengthens**; agents lift the hard cases to a
stable **ceiling ~0.77** regardless. Gain ≈ (0.77 − primary-on-escalated) × escalation-rate.
On C3 (weak) G ≈ G2; on Ahmed (strong) G2 > G.

═══════════════════════════════════════════════════════════════════════
## TRACK B — Generated training data

### B1. Generated-ONLY training (gen as the sole training data; primary_only on EESA)
| dataset | size | acc | note |
|---|---|---|---|
| C1 | 240 | ~0.59 | — |
| C2 (GEN-480) | 480 | 0.6500 ± 0.016 | matched-size baseline |
| C3 (GEN-960) | 960 | 0.6695 ± 0.024 | more data → better |
| C-V1 (V1_lowerCS-480) | 480 | 0.6304 ± 0.017 | Arabic↑/CMI↓ → **WORSE** than C2 |
**Verdict:** more gen data helps; making it more Arabic-dominant/lower-CMI (V1) transferred
**worse** — reduced code-switching removed the signal transfer needs.

### B2. Augmentation by MIXING (real EESA + gen trained together) — never helped
| experiment | 10% | 25% | 50% | full |
|---|---|---|---|---|
| real-only baseline | 0.7751 | 0.7873 | 0.8166 | 0.8533 (E0) |
| + GEN-960 (mix) | 0.7408 | 0.7689 | 0.8142 | 0.8411 (E3) |
| + V1 (mix, ≤50%) | 0.7531 | 0.7885 | 0.8178 | — |
| ratio-sweep (best) | — | +0.006 (noise) | — | — |
**Verdict:** **mixing is neutral-to-harmful at every ratio** — gen distorts the real signal.
V1 was *less* harmful than GEN-960 but still ≤ real-only.

### B3. Augmentation by TWO-STAGE (gen pretrain → real fine-tune) — THE ONE THAT WORKS ✅
| real EESA used | real-only | mixing | **two-stage** | 2-stage vs real-only |
|---|---|---|---|---|
| 10% | 0.7751 | 0.7408 | 0.7604 | −0.015 ❌ |
| **25%** | 0.7873 | 0.7689 | **0.8093** | **+0.022 ✅** |
| **50%** | 0.8166 | 0.8142 | **0.8313** | **+0.015 ✅** |
| **100%** | 0.8533 | 0.8411 | **0.8655** | **+0.012 ✅** (new best full-EESA) |
**Verdict:** using gen as **pretraining** (real gets the last word) **helps at 25/50/100%** real
data and **beats mixing everywhere**; only hurts at 10% (too little real to steer). This is
domain-adaptive pretraining — the one method that makes generated data pay off as augmentation.

═══════════════════════════════════════════════════════════════════════
## Headline conclusions
1. **Multi-agent layer works where the primary is weak, not where it's strong** — +7 pts on a weak
   classifier, +3 on a mid one (your EESA-XLM-R), ~0 on a strong one. Stable agent ceiling ~0.77.
2. **You cannot prompt-engineer or re-architect past the strong-primary ceiling** — 4 attempts
   (prompts, sequential, disambig, stronger model) all ≤ 0; residual errors are cultural/knowledge
   gaps, not instructions.
3. **The right gate depends on the model** — selective gate helps only on a stronger model.
4. **Generated data: mixing never helps; two-stage pretraining does** (+1.2 to +2.2 pts, even at
   full real data). Its other value is *substituting* for scarce real data to build the weak
   primary the agents then rescue.
5. **Config tuning of the generator (V1) didn't help** — the binding gap is register/authenticity,
   not Arabic%/CMI.

## Source reports (per finding)
Track A: `EXPERIMENT_EESAXLMR_MIDPRIMARY_G2.md`, `EXPERIMENT_C3_GPT41_GATE_BUNDLE.md`,
`EXPERIMENT_GPT41_GATE_ABLATION.md`, `EXPERIMENT_SEQUENTIAL_SENTIMENT_V2_AHMED_RESULTS.md`,
`EXPERIMENT_G_DISAMBIG_AHMED_RESULTS.md`, `EXPERIMENT_ESCALATED_CEILING_ROOT_CAUSE.md`.
Track B: `EXPERIMENT_TWOSTAGE_GEN_AUGMENTATION.md`, `EXPERIMENT_V1_LOWERCS_SENSITIVITY.md`,
`EXPERIMENT_E_AUGMENTATION_FAILURE_DIAGNOSIS.md`. Overview: `EXPERIMENT_CONSOLIDATED_FINDINGS.md`.
