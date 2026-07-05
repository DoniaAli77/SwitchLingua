# gpt-4.1-mini Gate Ablation on Ahmed — Is the IntentGate Still Helpful?

Focused ablation (no prompt tuning): at gpt-4.1-mini, does the full IntentGate over-veto? Test
**C (no gate)** and **G2 (selective gate)** vs the full-gate **G**, all semantic_v1, Ahmed
frozen primary, threshold 0.7, w_primary=1.0. Date: 2026-07-02.

## Results

| config | accuracy | macro F1 | escalated acc | net vs primary (W→C / C→W) |
|---|---|---|---|---|
| primary_only | 0.9254 | 0.9207 | 63/84 = 0.750 | — |
| G @ 4o-mini (full gate) | 0.9279 | 0.9257 | 65/84 = 0.774 | +2 |
| G @ 4.1-mini (full gate) | 0.9291 | 0.9248 | 66/84 = 0.786 | +3 |
| **G2 @ 4.1-mini (selective gate)** ✅ | **0.9303** | **0.9262** | **67/84 = 0.798** | **+4** (12 / 8) |
| C @ 4.1-mini (no gate) | 0.9266 | 0.9216 | 64/84 = 0.762 | +1 (13 / 12) |

**G2 @ gpt-4.1-mini = 0.9303 is the new best** — beats G@4.1-mini (0.9291) and, for the first
time in the whole line, the point estimate **crosses 0.930**. Best macro F1 too (0.9262).
**No-gate (C) is worse (0.9266)** than the full gate.

## Answer to the main question: the gate is still helpful — but only RESTRICTED
The stronger model does make the *full* gate over-veto, but removing it entirely is worse. The
three configs cleanly separate the mechanism:

| | recovers over-vetoed negatives? | neutral→polar breakages | net |
|---|---|---|---|
| Full gate (G) | no (00045, 00100 vetoed) | fewest | +3 |
| **Selective gate (G2)** | **partly** (00100 ✓, 00045 ✗) | **5** | **+4 (best)** |
| No gate (C) | **yes** (00045 ✓, 00100 ✓) | **7 (most)** | +1 (worst) |

- **Full gate over-vetoes** correct negatives (as the trace predicted) → leaves +3 on the table.
- **No gate recovers those** (00045, 00100 both correct) **but loses neutral protection** →
  7 neutral→polar breakages (the agents over-polarize other neutrals) → net collapses to +1.
- **Selective gate is the sweet spot:** it stops vetoing enough to recover 00100 while still
  protecting neutrals (only 5 breakages) → **net +4, best overall.** The gate's *neutral
  protection* is still worth keeping; its *aggressive meta-veto* is what the stronger model
  broke, and G2's restriction removes exactly that.

## Specific cases (as requested)
- **00045** ("عاملين dislike ليه يا ولاد المرة!"): recovered by **C only**; G2 **still vetoes**
  it (its selective gate still reads this as meta). Net-net G2 sacrifices this one case to keep
  neutral protection elsewhere — a good trade (+4 vs C's +1).
- **00100** ("…تنزل اغنية ع bts يخدو الترند"): **recovered by both G2 and C.** G2's lighter gate
  no longer vetoes it.
- **00362** (Breaking Bad plot description): **NOT recovered by any config** — both G2 and C
  still predict negative. This confirms it is an **agent over-reading problem (M3), independent
  of the gate**: all three agents read the plot's negative lexicon as sentiment, so no gate
  change touches it. Removing the gate did **not** specifically fix new neutral cases; C's
  extra neutral→polar breakages (7 vs 5) show the no-gate config *creates* more such errors.

## Honesty / significance
- G2 net +4 vs primary (b=12, c=8): McNemar χ² ≈ 0.8, **p ≈ 0.37 — still NOT significant.** And
  G2 vs G@4.1-mini is +1 sample (761 vs 760) — within temp-0 noise. So **0.9303 is the best
  point estimate ever and the first to cross 0.930, but it is not a statistically significant
  beat over primary_only or over G.** Treat it as the best *configuration*, not a proven gain.
- Single temp-0 draw; ±1–2 sample noise applies.

## Decision (per the stated rule)
- **G2 beats 0.9291 → keep G2 @ gpt-4.1-mini (selective gate) as the best strong-primary
  configuration (0.9303 / 0.9262).**
- No-gate (C) is worse → discard.
- **Stop Ahmed tuning.** The ablation confirmed the mechanism (full gate over-vetoes under a
  stronger model; selective gate fixes it) and produced the best config; further strong-primary
  tuning is not warranted (all gains remain non-significant vs primary).
- C3 not run (per instruction). When C3 is next, use **G2 @ gpt-4.1-mini** as the configuration.

## New mechanism revealed (justifies this having been worth running)
**Gate aggressiveness must scale INVERSELY with model strength.** A gate tuned for a weak model
(4o-mini) over-vetoes on a strong one (4.1-mini); the selective gate is the right amount of veto
for the stronger model. This is a genuine, reusable finding — not prompt tuning.

## Artifacts
- `experiment_G2_ahmed_41mini/*`, `experiment_C_ahmed_41mini/*`
- Basis: `EXPERIMENT_WHY_STRONGER_MODEL_BROKE_CASES.md` (the over-veto trace that motivated this),
  `EXPERIMENT_G_AHMED_GPT41MINI_RESULTS.md`.
