# SwitchLingua Topic Learning Curve — Topic-180 → Topic-360 → Topic-540

Nine XLM-R training runs (3 sizes × seeds 42/43/44), each evaluated `primary_only`
on the **unchanged Silver-1163** corpus. Silver was used for **final evaluation only** —
never for training, never for model selection. No ArEnTC, no agents, no LLM calls,
no newly generated data.

Correctness = agreement with the automatically assigned silver topic label, not human gold.

## Setup

| Item | Value |
|---|---|
| Source data | `switchlingua_topic_train_540_60perlabel.jsonl` (existing, 540 rows, 60/label) |
| Subsets | balanced 20/40/60 per label; **nesting asserted programmatically** per seed |
| Topic-540 | the **original file, unchanged** (so 540/seed-42 is a faithful reproduction attempt) |
| Seeds | 42, 43, 44 — control **both** the subset draw and training |
| Recipe | `xlm-roberta-base`, 4 epochs, batch 16, grad_accum 1, lr 2e-5, max_len 256, fp16, gradient_checkpointing off, `adamw_torch`, frozen 9-label order |
| Model selection | none (`--load_best` off) → final-epoch model, matching the original |
| Evaluation | `evaluate_pipeline.py --pipeline_mode primary_only` on Silver-1163 (n=1163) |
| Integrity | 9/9 runs exit 0; 0 ArEnTC references in any training log; checkpoints deleted after each eval |

### Dev-set decision (documented)

The original Topic-540 run passed ARENTCV2 dev, but its `trainer_state.json` shows
`best_model_checkpoint: None` / `best_metric: None` — `--load_best` was **off**, so the dev set
was logged only and never selected the weights. Because this study forbids ArEnTC, **no dev set
was passed**. See the reproduction check below for the empirical consequence.

## Per-run results on Silver-1163

| Size | Seed | Correct | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|
| 180 | 42 | 194/1163 | 0.1668 | 0.1810 | 0.1509 |
| 180 | 43 | 82/1163 | 0.0705 | 0.0391 | 0.0270 |
| 180 | 44 | 197/1163 | 0.1694 | 0.1106 | 0.1539 |
| 360 | 42 | 647/1163 | 0.5563 | 0.4992 | 0.5386 |
| 360 | 43 | 650/1163 | 0.5589 | 0.4957 | 0.5236 |
| 360 | 44 | 695/1163 | 0.5976 | 0.5364 | 0.5723 |
| 540 | 42 | 715/1163 | 0.6148 | 0.5429 | 0.5946 |
| 540 | 43 | 715/1163 | 0.6148 | 0.5575 | 0.6036 |
| 540 | 44 | 714/1163 | 0.6139 | 0.5415 | 0.5938 |

## Mean ± SD by training size (sample SD, n = 3)

| Size | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| 180 | 0.1356 ± 0.0564 | 0.1102 ± 0.0710 | 0.1106 ± 0.0724 |
| 360 | 0.5709 ± 0.0231 | 0.5105 ± 0.0226 | 0.5448 ± 0.0249 |
| 540 | **0.6145 ± 0.0005** | **0.5473 ± 0.0089** | **0.5973 ± 0.0054** |

Seed variance collapses as data grows: accuracy SD falls 0.0564 → 0.0231 → 0.0005.

## Paired Macro-F1 changes (same seed, nested data)

### 180 → 360

| Seed | 180 | 360 | Δ |
|---|---|---|---|
| 42 | 0.1810 | 0.4992 | **+0.3182** |
| 43 | 0.0391 | 0.4957 | **+0.4567** |
| 44 | 0.1106 | 0.5364 | **+0.4259** |
| **Mean** | | | **+0.4002 ± 0.0727** |

### 360 → 540

| Seed | 360 | 540 | Δ |
|---|---|---|---|
| 42 | 0.4992 | 0.5429 | +0.0437 |
| 43 | 0.4957 | 0.5575 | +0.0618 |
| 44 | 0.5364 | 0.5415 | +0.0051 |
| **Mean** | | | **+0.0369 ± 0.0290** |

Both steps are positive for **all three seeds** (consistent sign), but the second step is
**~11× smaller** than the first.

## Reproduction check — 540 / seed 42 vs the existing primary result

| | Accuracy | Macro-F1 |
|---|---|---|
| Existing primary result | 0.6242 | 0.5599 |
| This run (540/seed-42) | 0.6148 | 0.5429 |
| **Difference** | **−0.0094** | **−0.0170** |

Prediction-level agreement with the stored baseline: **1034/1163 = 88.91%** identical.

**Verdict: consistent with, but NOT an exact reproduction of, the existing result.**

The gap (−0.9 accuracy points) exceeds the 540 seed-spread SD (0.0005 accuracy, 0.0089 macro-F1),
so it is not explained by seed variation alone. Two candidate causes, which this experiment does
**not** disentangle:

1. **Omitting the dev set.** I predicted this would have zero effect because `--load_best` was
   off. That prediction is **not confirmed** by the result. Running per-epoch evaluation can
   still perturb RNG/dataloader state even when it does not select the model.
2. **GPU/fp16 nondeterminism.** fp16 + cuDNN kernels are not bitwise deterministic by default;
   independent runs of an identical command can diverge.

A clean, ArEnTC-free diagnostic would be to re-run 540/seed-42 with the *identical* no-dev
command: if it fails to reproduce **itself**, cause (2) dominates and the dev-set omission is
exonerated. Not run here (not requested).

Practically: the existing 0.6242/0.5599 primary result and this curve's 540 point
(0.6145 ± 0.0005) describe the same model to within ~1 accuracy point, and the curve's internal
comparisons (all trained identically) are unaffected by this offset.

## Important confound — training steps are not held constant

The fixed 4-epoch recipe means smaller subsets receive proportionally fewer optimisation steps:

| Size | Steps/epoch | Total steps (4 epochs) | Train runtime (seed 42) |
|---|---|---|---|
| 180 | 12 | **48** | 102.8 s |
| 360 | 23 | **92** | 192.7 s |
| 540 | 34 | **136** | 289.2 s |

Topic-180 received only **48 optimisation steps**. Its mean accuracy (0.1356) sits at roughly the
9-class chance floor (1/9 ≈ 0.111), and seed 43 (0.0705) is *below* chance — the signature of a
model that has barely left initialisation, not a stable estimate of "what 180 examples can teach."

**Consequence:** this curve measures *data size confounded with optimisation budget*, which is
faithful to the frozen recipe you asked me to reuse, but it means the huge 180→360 jump should
**not** be read as a pure data-quantity effect. A matched-compute variant (equal `--max_steps`
across sizes, as `run_topic_lowresource_aug.sh` does with `--max_steps 400`) would separate the
two. Flagging rather than silently reinterpreting.

## Decision rule — NOT APPLIED

Your instruction *"Use this predefined decision rule:"* arrived with no rule following it. I have
deliberately **not** substituted one, since a saturation/stopping criterion chosen after seeing
these numbers would be a post-hoc choice. The numbers needed to apply any such rule are all
above (paired deltas: **+0.4002 ± 0.0727** for 180→360, **+0.0369 ± 0.0290** for 360→540).

## Artifacts

| Path | Contents |
|---|---|
| `topic{180,360,540}_seed{42,43,44}/` | per-run `finetune.log`, `eval.log`, predictions + metrics (csv/json) |
| `learning_curve_summary.json` | machine-readable per-run, mean±SD, paired deltas, reproduction check |
| `data/Topic/generated/learning_curve/` | the nested subsets + `manifest.json` |
| `scripts/build_topic_learning_curve_subsets.py` | subset builder (asserts balance + nesting) |
| `scripts/run_topic_learning_curve.sh` | the 9-run driver |
| `scripts/analyze_topic_learning_curve.py` | read-only analysis |

Nothing in any existing experiment folder was modified.
