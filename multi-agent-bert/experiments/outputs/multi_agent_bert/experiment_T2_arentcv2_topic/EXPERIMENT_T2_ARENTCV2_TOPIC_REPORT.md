# Experiment T2 — ARENTCV2 Topic Classification (primary_only)

Topic classification (9 labels), **not sentiment**. Fresh `xlm-roberta-base`
fine-tune on ARENTCV2; no sentiment checkpoint reused; no Ahmed models. Kept
separate from V1. Date: 2026-06-20.

## Dataset (ARENTCV2 — "no fully-Arabic", 100% code-switched)
| split | rows | file |
|---|---|---|
| train | 73,956 | `data/Topic/processed/ARENTCV2/train.jsonl` |
| dev | 10,562 | `data/Topic/processed/ARENTCV2/dev.jsonl` |
| test | 21,134 | `data/Topic/processed/ARENTCV2/test.jsonl` |

Labels (balanced, ~8.2k/label train; ~2.3k/label test):
`business, education, finance, health, medical, shopping, social, sports, tech`.

## Training settings
| setting | value |
|---|---|
| base model | `xlm-roberta-base` (fine-tune, fresh) |
| epochs | 3 |
| batch size | 16 |
| grad accumulation | 1 (effective batch 16) |
| max_length | 64 (texts ≤ 36 tokens) |
| optimizer | adafactor |
| fp16 | yes |
| seed | 42 |
| train_runtime | ~5.0 h (13,869 steps @ ~1.3 s) |

**Dev (validation) by epoch:** 0.9928 → 0.9935 → **0.9944** (acc = macro F1 = weighted F1).
Best/last dev = epoch 3 = **0.9944**.

## Primary_only result — ARENTCV2 test (21,134)
**accuracy 0.9947 · macro F1 0.9947 · weighted F1 0.9947**

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| business | 0.9882 | 0.9962 | 0.9921 | 2344 |
| education | 1.0000 | 0.9996 | 0.9998 | 2353 |
| finance | 0.9948 | 0.9906 | 0.9927 | 2335 |
| health | 0.9910 | 0.9863 | 0.9887 | 2342 |
| medical | 0.9900 | 0.9924 | 0.9912 | 2383 |
| shopping | 0.9991 | 1.0000 | 0.9996 | 2327 |
| social | 0.9987 | 0.9966 | 0.9977 | 2365 |
| sports | 0.9983 | 0.9987 | 0.9985 | 2337 |
| tech | 0.9923 | 0.9919 | 0.9921 | 2348 |

Prediction distribution (well balanced, matches support):
business 2363 · education 2352 · finance 2325 · health 2331 · medical 2389 ·
shopping 2329 · social 2360 · sports 2338 · tech 2347.

Confusion matrix (rows = true, cols = predicted):
```
          busi educ fina heal medi shop soci spor tech
business  2335    0    2    0    0    0    1    0    6
education    1 2352    0    0    0    0    0    0    0
finance     11    0 2313    0    0    1    0    0   10
health       0    0    0 2310   24    0    2    4    2
medical      0    0    0   18 2365    0    0    0    0
shopping     0    0    0    0    0 2327    0    0    0
social       6    0    0    2    0    0 2357    0    0
sports       2    0    0    1    0    0    0 2334    0
tech         8    0   10    0    0    1    0    0 2329
```
Near-perfect diagonal. The only notable confusion is **health ↔ medical** (24 + 18) —
semantically adjacent classes; everything else is ≤ ~11 off-diagonal.

## Checkpoint
`experiments/checkpoints/topic_arentcv2_xlmr/` (sharded safetensors; config
`id2label` carries the 9 topic labels, business=0 … tech=8).

## Notes
- **Task confirmed: topic classification** (9 topic labels), not sentiment.
- This regime is far easier than the noisy real-EESA sentiment task: large (74k),
  clean, balanced, distinctly-separable topics, with dev/test sharing the train
  distribution → ~99.5% is expected, not surprising.
- primary_only only; **full_agentic not run** (per instruction). Given a ~99.5%
  primary, escalation headroom is tiny — agents would have very little to add here.
