# Experiment T1 — ARENTCV1 Topic Classification

Topic classification (9 labels). Fresh `xlm-roberta-base` fine-tune on ARENTCV1;
no sentiment checkpoint; no Ahmed models. Kept separate from V2. Date: 2026-06-20.

## Dataset (ARENTCV1 — full, ~99.96% code-switched, 30 fully-Arabic rows)
| split | rows | file |
|---|---|---|
| train | 73,976 | `data/Topic/processed/ARENTCV1/train.jsonl` |
| dev | 10,569 | `data/Topic/processed/ARENTCV1/dev.jsonl` |
| test | 21,137 | `data/Topic/processed/ARENTCV1/test.jsonl` |

Labels balanced (~8.2k/label train; ~2.3k/label test):
`business, education, finance, health, medical, shopping, social, sports, tech`.

## Training settings
`xlm-roberta-base` · 3 epochs · batch 16 · grad_accum 1 (eff 16) · max_length 64 ·
adafactor · fp16 · seed 42 · `--save_steps 1000` (periodic checkpoints for crash
safety — survived an internet drop mid-run with no loss). ~5 h on the 4 GB GPU.

**Dev by epoch:** 0.9911 → … → **0.9943** (final).

## Primary_only — ARENTCV1 test (21,137)
**accuracy 0.9946 · macro F1 0.9946 · weighted F1 0.9946**

| class | P | R | F1 | n |
|---|---|---|---|---|
| business | 0.9902 | 0.9949 | 0.9926 | 2344 |
| education | 0.9996 | 0.9996 | 0.9996 | 2353 |
| finance | 0.9940 | 0.9931 | 0.9936 | 2335 |
| health | 0.9889 | 0.9863 | 0.9876 | 2342 |
| medical | 0.9883 | 0.9912 | 0.9897 | 2383 |
| shopping | 0.9996 | 0.9996 | 0.9996 | 2330 |
| social | 0.9996 | 0.9975 | 0.9985 | 2365 |
| sports | 0.9979 | 0.9979 | 0.9979 | 2337 |
| tech | 0.9936 | 0.9915 | 0.9925 | 2348 |

Prediction distribution: business 2355 · education 2353 · finance 2333 · health 2336 ·
medical 2390 · shopping 2330 · social 2360 · sports 2337 · tech 2343 (well balanced).

Confusion matrix (rows = true, cols = predicted):
```
          busi educ fina heal medi shop soci spor tech
business  2332    1    3    0    0    0    1    1    6
education    1 2352    0    0    0    0    0    0    0
finance      7    0 2319    0    0    1    0    0    8
health       0    0    0 2310   27    0    0    4    1
medical      0    0    0   21 2362    0    0    0    0
shopping     1    0    0    0    0 2329    0    0    0
social       4    0    0    2    0    0 2359    0    0
sports       2    0    0    3    0    0    0 2332    0
tech         8    0   11    0    1    0    0    0 2328
```
Near-perfect diagonal; main confusion again **health ↔ medical** (27 + 21).

## Full_agentic — ARENTCV1 test (threshold 0.9, Fix-2 on, signal off, gpt-4o-mini)
Clean run (0 errors). Escalated **63 / 21,137 (0.3%)**, cost ~$0.03.

| | accuracy | macro F1 |
|---|---|---|
| primary_only | 0.9946 | 0.9946 |
| full_agentic | **0.9947** | **0.9948** |
| Δ | +0.0001 | +0.0002 |

Escalated-set (63): escalated-only acc **0.540** · agents changed 25 ·
**wrong→correct 14 · correct→wrong 11 · net +3**.

## Checkpoint
`experiments/checkpoints/topic_arentcv1_xlmr/` (sharded; `id2label` = 9 topic labels).

---

## V1 vs V2 comparison (same recipe, separate experiments)
| | T1 / ARENTCV1 | T2 / ARENTCV2 |
|---|---|---|
| train / dev / test | 73,976 / 10,569 / 21,137 | 73,956 / 10,562 / 21,134 |
| code-switching | ~99.96% (30 fully-AR) | 100% CS |
| final dev acc | 0.9943 | 0.9944 |
| **primary_only test acc** | **0.9946** | **0.9947** |
| primary_only macro F1 | 0.9946 | 0.9947 |
| full_agentic test acc | 0.9947 (net **+3**) | 0.9944 (net **−6**) |
| escalated @0.9 | 63 | 48 |
| escalated-only acc | 0.540 | 0.521 |

**Conclusions:**
- **V1 ≈ V2** on the primary (0.9946 vs 0.9947) — the ~30 fully-Arabic rows are
  immaterial; both XLM-R topic models are ~99.5% accurate.
- **Agents are noise-level on a near-perfect primary:** T1 net +3, T2 net −6 — the
  ~50–60 escalated samples are inherently ambiguous (escalated-only acc ~0.52–0.54),
  so the LLM panel is ≈ a coin flip and the overall effect oscillates around zero
  (±0.0003). **Recommendation: use primary_only** for these topic datasets; the
  agent panel adds cost without reliable benefit.
- Fits the cross-experiment strength curve: weak primary → agents rescue;
  strong (EESA sentiment 0.82) → small +2.7 pts; near-perfect (topic 0.99) → noise.
