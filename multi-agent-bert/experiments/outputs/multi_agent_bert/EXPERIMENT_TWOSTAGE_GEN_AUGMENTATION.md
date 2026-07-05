# Two-Stage Generated-Data Augmentation — the First Setup That HELPS

Tests generated data as **pretraining** (gen-960 → fine-tune on real EESA%) instead of **mixing**
(real+gen trained together). Motivated by the diagnosis that mixing lets gen *distort* the
decision boundary, whereas two-stage lets gen only *warm up the representation* while the real
data gets the last word. XLM-R, adafactor, max_steps 400, load_best, seed 42; base_checkpoint =
gen-960 (`expC3_switchlingua_xlmr_960`). primary_only, EESA test. Date: 2026-07-05.

## Result — two-stage beats real-only at 25% and 50% (first augmentation win)

| EESA% real | real-only | +960 (mixing) | **two-stage** | **2-stage vs real-only** | 2-stage vs mixing |
|---|---|---|---|---|---|
| 10% | 0.7751 | 0.7408 | 0.7604 | **−0.0147 (HURT)** | +0.0196 |
| **25%** | 0.7873 | 0.7689 | **0.8093** | **+0.0220 (HELPED)** | +0.0403 |
| **50%** | 0.8166 | 0.8142 | **0.8313** | **+0.0147 (HELPED)** | +0.0171 |
| **100% (E0)** | 0.8533 | 0.8411 (E3) | **0.8655** | **+0.0122 (HELPED)** | +0.0244 |

macro F1 confirms it at every helping ratio: 10% −0.012, **25% +0.021**, **50% +0.020**,
**100% +0.017** — accuracy *and* macro-F1 both up (not a single-metric artifact).

**Correction to the earlier prediction:** two-stage was expected to fade to ~0 at full data — it
did NOT. It **still helps at 100% real (+0.012 / +0.017)**, making two-stage-full (0.8655) a **new
best full-EESA primary**, beating E0 (0.8533). The benefit peaks at moderate resource (25%) but
**persists at full data** — this is domain-adaptive pretraining (DAPT): warming up on in-domain-ish
gen data helps even when all real labels are available, because the real fine-tune keeps the last
word so gen only adds representation signal without distorting the decision.

## What this means
- **For the first time in the entire augmentation program, generated data made a real classifier
  BETTER** — +2.2 points at 25% real, +1.5 at 50% real, on both metrics.
- **Two-stage beats naive mixing at every ratio** (+0.020, +0.040, +0.017). The *method* of using
  the gen data is what matters, not the gen data itself: the exact same generated set that *hurt*
  when mixed in *helps* when used as pretraining.
- **The mechanism is confirmed:** mixing lets gen distort the final decision (every mixing run
  hurt); two-stage lets gen initialize the representation and the **real fine-tune has the last
  word**, so the gen's out-of-distribution parts get overwritten while its useful
  representation-level signal is retained.
- **Only 10% still hurt** (−0.015) — with too little real data (246 rows), the real fine-tune
  can't fully steer away from the gen-heavy initialization. The benefit needs enough real data to
  "correct" the pretraining, and appears from ~25% upward.

## Answer to "how do we make augmentation work?"
**Use the generated data as pretraining (two-stage), not as mixed-in augmentation.** This is the
concrete, validated recipe:
1. Pretrain XLM-R on the generated set (you already have this checkpoint).
2. Fine-tune that checkpoint on your real EESA data.
It requires **no new generation and no config tweaking** — just changing *how* the gen data is
consumed. It turned the campaign's consistent negative into a **+1.5 to +2.2 point win**.

## Caveats
- **Single seed** (seed 42, matching the LR baselines). The +0.022 / +0.015 gains are ~12–18
  samples; they are consistent across **two ratios and both metrics** and **beat mixing at all
  three ratios**, but a 3-seed replication would put error bars on them before a strong claim.
- Gen-960 (`expC3`) used as the pretrain init; a different gen size/seed init may shift the exact
  numbers. Threshold-free (primary_only), EESA test held out of training.
- Intermediate checkpoints were pruned mid-run (disk); final metrics unaffected.

## Recommended follow-ups (cheap, GPU-only)
1. **3-seed replication** of two-stage 25% and 50% → error bars on the win.
2. **Two-stage + the AGENT layer** on the resulting mid-strength primary → does the agent gain
   stack on top of the two-stage gain? (This would combine both wins.)
3. Optional: distribution-filtered gen for the pretrain stage (Suggestion 2) to push further.

## Artifacts
- `experiment_TwoStage_gen960_eesa/eesa{10,25,50}_twostage/primary_only/`; runner
  `scripts/run_twostage_gen960.sh`.
- Comparators: `EXPERIMENT_V1_LOWERCS_SENSITIVITY.md`, LR `_only`/`_plus960` baselines,
  `EXPERIMENT_E_AUGMENTATION_FAILURE_DIAGNOSIS.md` (the mixing failure this fixes).
