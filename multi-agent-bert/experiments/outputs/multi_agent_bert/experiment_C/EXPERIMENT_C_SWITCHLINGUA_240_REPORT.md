# Experiment C — SwitchLingua generated-240 → real EESA transfer pilot

**Separate experiment.** Do not merge these tables/files with Experiment A or the
architecture / threshold-ablation reports. Date: 2026-06-13.

## Purpose
Test whether SwitchLingua-**generated** sentiment data transfers to **real EESA**.
- **Train:** 240 generated samples only
  (`data/Sentiment/generated/merged/switchlingua_sentiment_train_merged.jsonl`;
  80 pos / 80 neg / 80 neu; dataset card `DATASET_CARD_EXP_C.md`).
- **Dev:** `data/Sentiment/processed/eesa_sentiment_dev.jsonl` (818).
- **Test:** `data/Sentiment/processed/eesa_sentiment_test.jsonl` (818).
- **Model:** `xlm-roberta-base`, **fine-tuned** (continued training of the
  pretrained checkpoint — *not* trained from scratch).

## ⚠️ Critical caveats — read before interpreting
1. **Fine-tuning, not from scratch.** Experiment C still fine-tunes
   `xlm-roberta-base`; only the training *data* differs from Experiment A.
2. **Not size-matched.** EESA train = **2,464** samples; SwitchLingua generated
   train = **240**. This is a **generated-data transfer pilot, not a fair-size
   comparison.** A weaker result is expected partly from the 10× smaller train set.
3. **Optimizer differs from the Experiment A XLM-R run → pilot result.** Exp A used
   **AdamW**; Exp C uses **Adafactor** (see "Training environment" below). Because
   the optimizer differs, Exp C is **not** a controlled comparison against the Exp A
   numbers.
4. **For a fair comparison later:** re-run **EESA-2,464 *and* an EESA-240 subset**
   with the **same Adafactor setup** used here, so optimizer and train size are
   controlled. (The fine-tune script's `--optim adafactor` makes this one flag.)

## Training environment (why Adafactor)
GPU fine-tuning of XLM-R was attempted but is **not reliable** on this 4 GB Windows
GPU:
- **AdamW on GPU:** out-of-memory. Model (~1.1 GB) + AdamW optimizer state
  (~2.2 GB) + activations exceeds the ~3 GB usable after the Windows display
  reserve. Memory is set by model+optimizer+batch, **independent of the 240-row
  train size**.
- **Adafactor on GPU (first tries):** still OOM via VRAM **fragmentation** — the
  734 MB embedding-gradient block could not be allocated contiguously, and Windows
  torch does not support `expandable_segments`.
- **CPU + Adafactor:** **intermittent native segfault** (libiomp/MKL, exit 139) —
  crashed at step 3 in one run, epoch 3 in another. `KMP_DUPLICATE_LIB_OK=TRUE` and
  `MKL_THREADING_LAYER=SEQUENTIAL` reduced but did not eliminate it.
- **What finally worked:** **laptop restart → clean GPU → GPU + Adafactor**
  (batch 4 / grad_accum 4, fp16, gradient-checkpointing,
  `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`). A clean, unfragmented 4 GB GPU
  fit the run with ~zero spare VRAM. Adafactor was required to stay under the
  optimizer-memory floor.

Root cause of the earlier failures was **GPU memory state**, not the dataset:
Experiment A's XLM-R training "barely fit" a then-cleaner GPU; by Exp C the usable
VRAM was tighter (more resident apps + fragmentation from rapid retries), tipping
"just fits" → OOM.

## Training hyperparameters (final successful run)
| param | value | note |
|---|---|---|
| base | xlm-roberta-base | fine-tune |
| optimizer | **adafactor** | deviation from Exp A AdamW (memory) |
| lr | 2e-5 | same as Exp A |
| epochs | 4 | same |
| batch_size / grad_accum | 4 / 4 | effective batch **16** (same as Exp A) |
| max_length | 256 | same |
| fp16 / grad-checkpointing | on / on | same |
| seed | 42 | same |
| train_runtime | 202 s | 240 samples |
| final_train_loss | 0.8929 | |

## Dev metrics (EESA dev, per epoch)
| epoch | accuracy | macro F1 | weighted F1 |
|---|---|---|---|
| 1 | 0.4438 | 0.2049 | 0.2728 |
| 2 | 0.5465 | 0.4695 | 0.4988 |
| 3 | 0.5306 | 0.4320 | 0.4675 |
| **4 (final)** | **0.6284** | **0.6038** | **0.6258** |

(Consistency check: an earlier AdamW GPU run that trained but failed to save reached
dev acc 0.621 — Adafactor produced an equivalent model.)

## Primary_only result — EESA test (the core transfer measurement)
**accuracy 0.5905 · macro F1 0.5619 · weighted F1 0.5838** (n = 818)

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| positive | 0.657 | 0.755 | **0.703** | 363 |
| negative | 0.577 | 0.457 | **0.510** | 197 |
| neutral | 0.486 | 0.461 | **0.473** | 258 |

Confusion matrix (rows = true, cols = predicted; order pos / neg / neu):
```
            pred_pos  pred_neg  pred_neu
true_pos       274       22        67
true_neg        48       90        59
true_neu        95       44       119
```
Neutral and negative are weak; the main error is **neutral/positive confusion**
(95 true-neutral → predicted positive, 67 true-positive → predicted neutral).

## Full_agentic result — NOT COMPLETED (intentionally stopped)
Run at threshold 0.9 / w_primary 1.0 / signal OFF / gpt-4o-mini was **stopped after
~96 / 818 samples**. Reason: the weak primary is **poorly calibrated**, so **~95 %
of samples escalated** (vs ~23 % for the full-EESA model at 0.9). Completing it
projected to **~80 min and ~$0.45**, with full_agentic doing agent work on nearly
the entire test set rather than a hard slice. Partial spend ≈ $0.04. **No
full_agentic metric is reported.**

*Finding in itself:* at a fixed 0.9 threshold a weak primary barely shortcuts
anything — escalation rate is a function of primary confidence, and this model's
confidence is low. A meaningful full_agentic run here would need a **lower
threshold** (e.g. 0.6–0.7) or a calibrated confidence; recommended as a follow-up.

## Comparison vs prior results (SEPARATE — not merged)
Reference (full-EESA-trained XLM-R, 2,464 train) — *different experiment, shown for
context only:*
| setting | accuracy | macro F1 |
|---|---|---|
| Exp A reference, primary_only | 0.8240 | 0.8088 |
| Exp A reference, full_agentic best (th 0.9) | 0.8509 | 0.8401 |
| **Exp C, generated-240, primary_only** | **0.5905** | **0.5619** |

Δ primary_only = **−0.2335 acc / −0.2469 macro F1** vs full-EESA. Interpretation:
240 generated samples transfer to **~59 % accuracy on real EESA** — well above the
~33 % three-class chance baseline, but far below the 2,464-sample real-data model.
Given the **10× smaller train set *and* the AdamW→Adafactor change**, this gap is
**not** attributable to generated-vs-real data alone; it is a transfer *signal*, not
a clean effect size. Establishing that requires the size-/optimizer-matched runs in
caveat #4.

## Artifacts
- Checkpoint: `experiments/checkpoints/expC_switchlingua_xlmr_240/`
- Primary_only: `experiment_C/generated_240_xlmr/primary_only/`
- Logs: `experiment_C/finetune_expC.log`, `eval_primary_only.log`,
  `eval_full_agentic.log` (partial)
