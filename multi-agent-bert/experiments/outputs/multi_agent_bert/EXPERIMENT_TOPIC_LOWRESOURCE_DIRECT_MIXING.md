# Topic Low-Resource Direct Mixing (ARENTC + generated 540)

Does adding generated topic data directly to the training pool help in a
low-resource regime, as tested for sentiment? Three real-data ratios × two arms
(real-only baseline vs real + 540 generated), fresh `xlm-roberta-base` in both
arms. Evaluated `primary_only` on the full ARENTCV2 test set (21,134 samples).
Date: 2026-08-05.

## Recipe
Matched to the sentiment low-resource study: `max_steps 400`, `load_best`,
`eval_steps 50`, batch 4 × grad_accum 4, lr 2e-5, max_length 256, seed 42, fp16,
gradient checkpointing, adafactor, 9 topic labels.

**Two documented deviations from the sentiment runs:**
1. **Dev subsampled for model selection.** The full ARENTCV2 dev set is 10,562 rows
   (12.9× the sentiment dev of 818); with `eval_steps 50` / `max_steps 400` it is
   re-scored 8 times per run, adding ~28 min/run of pure overhead. A stratified
   999-row subsample (111 per label, seed 42, identical across all six arms) is used
   instead: `data/Topic/processed/ARENTCV2/dev_sub1000.jsonl`. Dev is used only to
   select the best checkpoint and is never reported; **test evaluation uses the full
   21,134-row set.**
2. **Checkpoints deleted after evaluation** (disk pressure; each is 2.2 GB). The
   `arentc50_only` cell initially failed with `os error 112` (disk full) at step
   350/400 and was re-run with its checkpoint written to `D:`.

## Results

| Real data | Real-only | +540 mixed | Δ accuracy | Δ macro F1 |
|---|---|---|---|---|
| 10 % (7,392) | 0.9819 / 0.9819 | 0.9832 / 0.9832 | +0.0013 | **+0.0013** |
| 25 % (18,484) | 0.9844 / 0.9844 | 0.9821 / 0.9821 | −0.0023 | **−0.0023** |
| 50 % (36,975) | 0.9850 / 0.9851 | 0.9825 / 0.9825 | −0.0025 | **−0.0026** |
| 100 % (73,956) | 0.9841 / 0.9841 | 0.9846 / 0.9846 | +0.0005 | **+0.0005** |

*(accuracy / macro F1. All four rows are matched-recipe cells. The pre-existing
`topic_arentcv2_xlmr` checkpoint — test macro F1 0.9947 — is NOT part of this grid:
its trainer args were not preserved and it was trained to convergence rather than
under the `max_steps 400` budget, so it is a converged-ceiling reference only.)*

**Note on the fixed-compute budget.** `max_steps 400` at effective batch 16 exposes
the model to ~6,400 samples regardless of pool size — 0.87 epochs at 10 % but only
0.086 epochs at 100 %. The baseline column therefore plateaus (and dips slightly at
100 %) rather than rising monotonically. This is the intended matched-compute design
and is valid for comparing the two arms **against each other at each ratio**; the
absolute values are not converged-model numbers.

## Comparison with the sentiment study (Δ macro F1 from direct mixing)

| Real data | Sentiment (+gen 960) | Topic (+gen 540) |
|---|---|---|
| 10 % | **−0.0291** | +0.0013 |
| 25 % | **−0.0213** | −0.0023 |
| 50 % | −0.0017 | −0.0026 |
| 100 % | **−0.0115** | +0.0005 |

## Finding

**Direct mixing has no measurable effect on topic classification.** The four deltas
alternate in sign (+, −, −, +) and span only 0.0039 in total — the signature of
run-to-run noise rather than a systematic effect. By contrast the sentiment deltas
were consistently negative and up to 22× larger.

The explanation is in the baseline column — **the topic task is already saturated
at 10 % of the real data**:

| | Sentiment | Topic |
|---|---|---|
| 10 % real-only macro F1 | 0.7586 | **0.9819** |
| Ladder range, 10 % → 100 % | 0.7586 → 0.8409 (**+0.082**) | 0.9819 → 0.9841 (**+0.002**) |
| Generated share of pool at 10 % | ~11 % (960 / 8.7 k) | 6.8 % (540 / 7.9 k) |
| Generated share of pool at 100 % | ~10 % | **0.72 %** (540 / 74.5 k) |
| Δ macro F1 from mixing at 10 % | −0.0291 | +0.0013 |

Topic sits at 0.982–0.985 macro F1 across the **entire** ladder, from 7,392 samples
to 73,956 — the task is saturated before augmentation can matter. Combined with a
generated set contributing 6.8 % of the pool at 10 % and only 0.72 % at 100 %, the
intervention is far too small relative to a near-ceiling task to register in either
direction.

**Defensible claim:** *the topic task saturates too early for low-resource
augmentation to be measurable* — not *"direct mixing harms topic classification."*
The direction is consistent with sentiment; the magnitude does not support a
standalone conclusion for this task.

## Status of the remaining arm
The **two-stage** arm (gen-540 pretrain → real fine-tune) was not run. On
sentiment this was the only augmentation strategy that helped (+0.02 F1 at 25 %,
50 % and 100 %). Given the saturation documented above, it is unlikely to be
measurable on topic either, but it is the arm that carried the positive result on
sentiment and would complete the grid.

## Reproduce
- Data: `data/Topic/processed/lowresource_arentc/arentc{10,25,50}_{only,plus540}.jsonl`
  (the `_plus540` files concatenate the real subset with
  `data/Topic/generated/merged/switchlingua_topic_train_540_60perlabel.jsonl`,
  normalised to `{text,label}`).
- Scripts: `scripts/run_topic_lowresource_aug.sh`,
  `scripts/rerun_topic_arentc50_only.sh`.
- Artifacts: `experiment_TopicLR_augmentation/arentc*/primary_only/*metrics.json`.
