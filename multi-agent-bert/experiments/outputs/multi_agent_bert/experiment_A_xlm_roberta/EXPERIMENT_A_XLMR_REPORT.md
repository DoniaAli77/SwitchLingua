# Experiment A — XLM-RoBERTa-base Reference Baseline

**Real results (not mock).** A second Experiment A baseline: fine-tuned
`xlm-roberta-base` on EESA train, validated on EESA dev, evaluated on EESA test
through the Multi-Agent BERT pipeline — identical data/labels/metrics/pipeline to
the mBERT run. Run date: 2026-06-06.

This is **fine-tuning a pretrained model**, not training from scratch.

---

## Setup (identical to mBERT Experiment A, except the base model + VRAM knobs)

| | mBERT run | **XLM-R run** |
|---|---|---|
| Base checkpoint | bert-base-multilingual-cased | **xlm-roberta-base** |
| train / dev / test | EESA 2,464 / 818 / 818 | same |
| Labels | positive/negative/neutral | same |
| Epochs | 4 | 4 |
| Per-device batch | 16 | **8** |
| grad_accum | 1 | **2** (→ effective batch 16) |
| fp16 | no | **yes** |
| gradient checkpointing | no | **yes** |
| LR / max_len / seed | 2e-5 / 256 / 42 | same |
| Checkpoint | experiments/checkpoints/eesa_mbert | **experiments/checkpoints/eesa_xlm_roberta_base** |

XLM-R is ~270M params (~2.5× mBERT). On the 4 GB GTX 1650 Ti, plain fp32
fine-tuning does not fit, so it required **fp16 + gradient accumulation +
gradient checkpointing**. Effective batch size (16) is kept equal to the mBERT
run for a fair comparison. (`--grad_accum` / `--gradient_checkpointing` were
added to `scripts/finetune_transformer_classifier.py`; defaults preserve the
mBERT run.)

---

## Headline comparison — primary_only on EESA test (818 samples)

| Metric | mBERT | **XLM-R** | Δ |
|---|---|---|---|
| Accuracy | 0.7971 | **0.8240** | **+0.0269** |
| Macro F1 | 0.7833 | **0.8088** | **+0.0255** |
| Weighted F1 | 0.7973 | **0.8232** | **+0.0259** |

**XLM-R wins on every aggregate metric.**

### Per-class F1 (test)

| Label | mBERT P / R / F1 | XLM-R P / R / F1 |
|---|---|---|
| positive | 0.873 / 0.851 / 0.862 | **0.885 / 0.909 / 0.897** |
| negative | 0.777 / 0.690 / 0.731 | **0.780 / 0.756 / 0.768** |
| neutral | 0.716 / 0.802 / 0.757 | **0.768 / 0.756 / 0.762** |

XLM-R improves **all three** classes. The largest gain is **negative recall
0.690 → 0.756** — the weakest class for mBERT — i.e. XLM-R recovers more of the
negatives that mBERT lost to neutral.

### Confusion matrices (rows = true, cols = predicted; order pos, neg, neu)

mBERT:
| true \ pred | pos | neg | neu |
|---|---|---|---|
| positive | 309 | 20 | 34 |
| negative | 13 | 136 | 48 |
| neutral | 32 | 19 | 207 |

XLM-R:
| true \ pred | pos | neg | neu |
|---|---|---|---|
| positive | 330 | 10 | 23 |
| negative | 12 | 149 | 36 |
| neutral | 31 | 32 | 195 |

XLM-R cuts the negative→neutral confusion (48 → 36) and the positive errors
(54 → 33). It trades a little neutral→negative (19 → 32), but net is clearly
better.

---

## Dev metrics (end of fine-tuning)

| | mBERT | XLM-R |
|---|---|---|
| Dev accuracy | 0.8068 | **0.8313** |
| Dev macro F1 | 0.7896 | **0.8187** |
| Dev weighted F1 | 0.8056 | **0.8307** |

---

## Mode comparison (same XLM-R primary)

| Pipeline mode | Accuracy | Macro F1 | Escalation rate |
|---|---|---|---|
| primary_only | **0.8240** | **0.8088** | 0.0000 |
| paper_style | 0.8142 | 0.7983 | 0.0501 |
| full_agentic | 0.8130 | 0.7973 | 0.0501 |

Same pattern as mBERT: with a strong fine-tuned primary the specialist agents
**slightly hurt** (≈5% escalate, agents weaker than the primary). paper_style ≈
full_agentic (small escalation set + mock LLM client).

---

## Runtime & GPU/memory notes

| | mBERT | XLM-R |
|---|---|---|
| Train runtime | 638 s (~10.6 min) | **2,457 s (~41 min)** |
| Final train loss | 0.437 | 0.594 |
| GPU | GTX 1650 Ti, 4 GB | same |
| OOM? | no (plain batch 16) | **no** (only via fp16 + grad_accum 2 + grad checkpointing) |

XLM-R is **~4× slower** here — larger model plus gradient checkpointing
(recompute in backward) trades compute for memory. Inference on the 818-row test
set is fast on GPU for both. Environment: torch 2.6.0+cu124, **transformers
4.57.6 (pin `<5`; 5.x is incompatible with torch 2.6)**, accelerate 1.13.0.

---

## Conclusion
- **XLM-RoBERTa-base is the stronger Experiment A reference baseline** on EESA
  (+~2.6 points accuracy / macro F1 over mBERT), at ~4× the training cost.
- Both confirm: the current non-trained specialist agents do not help on top of
  a strong primary; agent value should be revisited with a weaker primary
  (Experiment C) or real LLM agents.
- **Experiment C is not run.** It will reuse this fine-tune script and the same
  EESA test set with SwitchLingua-generated training data, once a balanced
  generated set is produced (pending approval).
