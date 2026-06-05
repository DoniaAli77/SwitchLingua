# Experiment A — EESA Real-Data Reference Baseline

**Real results (not mock).** Fine-tuned mBERT on the real EESA sentiment train
split, validated on EESA dev, evaluated on EESA test through the Multi-Agent
BERT pipeline. Run date: 2026-06-06.

This is the **real-data reference baseline**: it shows what the primary
classifier achieves when *real* labeled code-switched training data is
available. Experiment C will later replace the training data with
SwitchLingua-generated sentiment data and evaluate on the **same EESA test set**.

---

## Dataset (EESA official splits, Arabic-English code-switched)

| Split | Samples | positive | negative | neutral |
|---|---|---|---|---|
| train | 2,464 | 1,092 | 594 | 778 |
| dev   | 818 | 363 | 197 | 258 |
| test  | 818 | 363 | 197 | 258 |

100% code-switched; labels exactly `{positive, negative, neutral}`. Source:
`data/Sentiment/processed/eesa_sentiment_{train,dev,test}.jsonl`.

---

## Fine-tuning settings

| | |
|---|---|
| Type | **fine-tuning** a pretrained checkpoint (not from scratch) |
| Base checkpoint | `bert-base-multilingual-cased` (mBERT) |
| Epochs | 4 |
| Batch size | 16 (per-device) |
| Learning rate | 2e-5 |
| Max length | 256 |
| Seed | 42 |
| Hardware | NVIDIA GTX 1650 Ti, 4 GB VRAM (CUDA) |
| Train runtime | 638 s (~10.6 min) |
| Final train loss | 0.437 |
| Script | `scripts/finetune_transformer_classifier.py` |
| **Checkpoint path** | **`experiments/checkpoints/eesa_mbert/`** |

Dev metrics at end of fine-tuning (`eesa_mbert/dev_metrics.json`):
accuracy **0.8068**, macro F1 **0.7896**, weighted F1 **0.8056**.

> Reproducibility note: transformers 5.x is incompatible with torch 2.6
> (`torch.float8_e8m0fnu`); this run used **transformers 4.57.6 + torch
> 2.6.0+cu124 + accelerate 1.13.0**. Pin `transformers<5` until torch is upgraded.

---

## Primary-only test metrics (real transformer, EESA test, 818 samples)

`--primary_model transformer --transformer_checkpoint experiments/checkpoints/eesa_mbert --pipeline_mode primary_only`

| Metric | Value |
|---|---|
| Accuracy | **0.7971** |
| Macro F1 | **0.7833** |
| Weighted F1 | **0.7973** |

Per-class:

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| positive | 0.8729 | 0.8512 | 0.8619 | 363 |
| negative | 0.7771 | 0.6904 | 0.7312 | 197 |
| neutral | 0.7163 | 0.8023 | 0.7569 | 258 |

Confusion matrix (rows = true, cols = predicted):

| true \ pred | positive | negative | neutral |
|---|---|---|---|
| **positive** | 309 | 20 | 34 |
| **negative** | 13 | 136 | 48 |
| **neutral** | 32 | 19 | 207 |

Observations: strongest on `positive` (the majority class); `negative` recall is
the weakest (0.69) — 48 negatives are misread as neutral, the main error mode.

---

## Mode comparison (same fine-tuned mBERT primary)

| Pipeline mode | Accuracy | Macro F1 | Escalation rate | Escalated acc |
|---|---|---|---|---|
| primary_only | **0.7971** | **0.7833** | 0.0000 | — |
| paper_style | 0.7958 | 0.7788 | 0.0575 | 0.4043 |
| full_agentic | 0.7958 | 0.7788 | 0.0575 | 0.4043 |

**Finding:** with a strong fine-tuned primary, the specialist agents **slightly
hurt** accuracy. Only 5.75% of cases escalate, and on that escalated subset the
agents are right just ~40% of the time — i.e. the keyword/regex/mock-LLM agents
are weaker than the fine-tuned mBERT they override. `paper_style` and
`full_agentic` are identical here because the escalation set is small and the
LLM/deliberation agents run on the **mock** LLM client (no real LLM was used).
The agent value proposition should be revisited when the primary is *weaker*
(e.g. Experiment C, generated-only training) or with real LLM agents.

Output files per mode under
`experiments/outputs/multi_agent_bert/experiment_A_eesa_reference/<mode>/`
(`*_metrics.{json,csv}`, `*_predictions.{json,csv}`).

---

## Notes

- **Experiment A = real-data reference baseline.** It is the upper-reference
  case: real, in-domain, labeled EESA training data. ~0.80 accuracy / 0.78 macro
  F1 on EESA test is the bar.
- **Experiment C (later, not run):** same `scripts/finetune_transformer_classifier.py`
  and the **same EESA test set**, but train on SwitchLingua-generated sentiment
  data instead of EESA train — to measure whether task-aware generated
  code-switched data transfers to real examples. A balanced generated training
  set must be produced first (pending approval). Do not run Experiment C yet.
- All numbers above are **real** (fine-tuned mBERT). Mock-primary numbers from
  earlier sanity runs are unrelated and must not be compared with these.
