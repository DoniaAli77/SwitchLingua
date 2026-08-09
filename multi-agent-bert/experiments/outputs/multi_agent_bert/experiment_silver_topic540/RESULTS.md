# Silver-1044 × Topic-540 primary_only Evaluation Results

Corpus: 1,044-row **high-confidence silver real-transcription test set**
(`silver_primary_1044.jsonl`), labels from `multi_llm_consensus_silver`.
**Not gold, not human-validated.** Frozen, `technology`→`tech` aliased, zero
overlap with Topic-540 train or any ArEnTC split (see HANDOVER.md).

Models: existing Topic-540 checkpoints (trained on 540 generated sentences
only), run `primary_only` — no retraining, no escalation/routing.

## Headline metrics

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| XLM-R (topic_gen540_xlmr) | 0.6303 | 0.5572 | 0.6135 |
| mBERT (topic_gen540_mbert) | 0.4904 | 0.4394 | 0.4825 |

## Reference — Topic-540 on its own (generated-only) test distribution

| Setting | Accuracy | Macro F1 |
|---|---|---|
| XLM-R primary-only (reference) | 0.7717 | 0.7552 |
| XLM-R full-agentic (reference) | 0.9232 | 0.9232 |
| mBERT primary-only (reference, ARENTC escalated subset) | 0.6530 | 0.6170 |
| mBERT full-agentic (reference, ARENTC escalated subset) | 0.9210 | 0.9210 |

Both checkpoints drop on the silver corpus relative to their own reference
primary-only numbers (XLM-R: 0.7717→0.6303 acc, −0.14; 0.7552→0.5572 macro-F1,
−0.20 / mBERT: 0.6530→0.4904 acc, −0.16; 0.6170→0.4394 macro-F1, −0.18) —
expected, since the silver corpus is real spontaneous transcription
(disfluent, ASR-derived, natural code-switching) vs. the clean generated
Topic-540 training/test distribution.

## Per-class Precision / Recall / F1 / Support

### XLM-R

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| business | 0.4912 | 0.2828 | 0.3590 | 99 |
| education | 0.7200 | 0.2143 | 0.3303 | 84 |
| health | 0.9459 | 0.3977 | 0.5600 | 88 |
| shopping | 0.5495 | 0.6854 | 0.6100 | 89 |
| medical | 0.4661 | 0.8594 | 0.6044 | 64 |
| sports | 0.4301 | 0.7547 | 0.5479 | 53 |
| tech | 0.6697 | 0.8185 | 0.7367 | 270 |
| finance | 0.7885 | 0.7593 | 0.7736 | 216 |
| social | 0.5538 | 0.4444 | 0.4932 | 81 |

### mBERT

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| business | 0.4630 | 0.2525 | 0.3268 | 99 |
| education | 1.0000 | 0.1190 | 0.2128 | 84 |
| health | 0.6875 | 0.1250 | 0.2115 | 88 |
| shopping | 0.2102 | 0.6966 | 0.3229 | 89 |
| medical | 0.4711 | 0.8906 | 0.6162 | 64 |
| sports | 0.6032 | 0.7170 | 0.6552 | 53 |
| tech | 0.7159 | 0.7000 | 0.7079 | 270 |
| finance | 0.7642 | 0.3750 | 0.5031 | 216 |
| social | 0.3391 | 0.4815 | 0.3980 | 81 |

## Confusion matrices (rows = true silver label, cols = predicted)

### XLM-R

| true\\pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|---|---|---|---|---|---|---|---|---|
| business | 28 | 0 | 0 | 11 | 0 | 6 | 26 | 24 | 4 |
| education | 2 | 18 | 0 | 2 | 24 | 5 | 25 | 2 | 6 |
| health | 2 | 0 | 35 | 4 | 19 | 11 | 13 | 1 | 3 |
| shopping | 0 | 0 | 1 | 61 | 7 | 5 | 8 | 3 | 4 |
| medical | 0 | 0 | 0 | 2 | 55 | 1 | 5 | 0 | 1 |
| sports | 1 | 0 | 0 | 1 | 0 | 40 | 8 | 2 | 1 |
| tech | 10 | 5 | 0 | 6 | 2 | 11 | 221 | 8 | 7 |
| finance | 9 | 0 | 0 | 18 | 3 | 7 | 12 | 164 | 3 |
| social | 5 | 2 | 1 | 6 | 8 | 7 | 12 | 4 | 36 |

### mBERT

| true\\pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|---|---|---|---|---|---|---|---|---|
| business | 25 | 0 | 0 | 31 | 1 | 2 | 17 | 16 | 7 |
| education | 4 | 10 | 2 | 9 | 22 | 1 | 18 | 2 | 16 |
| health | 0 | 0 | 11 | 24 | 28 | 5 | 6 | 2 | 12 |
| shopping | 1 | 0 | 2 | 62 | 1 | 6 | 3 | 1 | 13 |
| medical | 1 | 0 | 0 | 3 | 57 | 0 | 2 | 0 | 1 |
| sports | 1 | 0 | 0 | 6 | 1 | 38 | 2 | 1 | 4 |
| tech | 7 | 0 | 0 | 44 | 3 | 7 | 189 | 3 | 17 |
| finance | 13 | 0 | 0 | 92 | 3 | 1 | 20 | 81 | 6 |
| social | 2 | 0 | 1 | 24 | 5 | 3 | 7 | 0 | 39 |

## Predicted vs. silver label distribution

| Label | Silver n | XLM-R pred n | mBERT pred n |
|---|---|---|---|
| business | 99 | 57 | 54 |
| education | 84 | 25 | 10 |
| health | 88 | 37 | 16 |
| shopping | 89 | 111 | 295 |
| medical | 64 | 118 | 121 |
| sports | 53 | 93 | 63 |
| tech | 270 | 330 | 264 |
| finance | 216 | 208 | 106 |
| social | 81 | 65 | 115 |
| **Total** | **1044** | **1044** | **1044** |

Both models over-predict `medical` relative to silver share; mBERT sharply
over-predicts `shopping` (295 vs. 89 true) and under-predicts `education`,
`health`, and `finance`.

## Tag vs. intrasentential breakdown

| Model | cs_type | n | Accuracy | Macro F1 |
|---|---|---|---|---|
| XLM-R | tag | 458 | 0.5808 | 0.5327 |
| XLM-R | intrasentential | 586 | 0.6689 | 0.5709 |
| mBERT | tag | 458 | 0.4454 | 0.4115 |
| mBERT | intrasentential | 586 | 0.5256 | 0.4578 |

Both models perform worse on `tag`-type switches than `intrasentential`
switches, consistent across models (~9 pt accuracy gap for XLM-R, ~8 pt for
mBERT).

## Confidence distribution

| Model | Mean | Median | Std | Min | Max | % below 0.90 |
|---|---|---|---|---|---|---|
| XLM-R | 0.4832 | 0.4855 | 0.1578 | 0.1531 | 0.7964 | 100.00% (1044/1044) |
| mBERT | 0.3779 | 0.3531 | 0.1242 | 0.1342 | 0.8147 | 100.00% (1044/1044) |

Neither model ever reaches the routing threshold of 0.90 on this corpus — no
prediction from either checkpoint would be accepted without escalation under
the pipeline's normal (non-`primary_only`) confidence gate. Bucketed:

| Confidence bucket | XLM-R n (%) | mBERT n (%) |
|---|---|---|
| [0.0, 0.3) | 152 (14.6%) | 346 (33.1%) |
| [0.3, 0.5) | 402 (38.5%) | 511 (48.9%) |
| [0.5, 0.7) | 390 (37.4%) | 164 (15.7%) |
| [0.7, 0.9) | 100 (9.6%) | 23 (2.2%) |
| [0.9, 1.0] | 0 (0.0%) | 0 (0.0%) |

XLM-R's confidence mass sits meaningfully higher than mBERT's (mean 0.48 vs.
0.38), tracking its better accuracy, but both are far below the 0.90 gate —
consistent with a weak primary classifier (540 generated training sentences)
facing out-of-distribution real transcription.

## Summary

- Both existing Topic-540 primary classifiers generalize weakly to real,
  spontaneous Arabic-English code-switched speech: XLM-R drops ~14 accuracy
  points and mBERT ~16 points relative to their own primary-only reference
  numbers on the generated test set.
- XLM-R consistently outperforms mBERT here, matching the pattern seen on the
  ArEnTC escalated-subset reference numbers.
- The confidence gap (no sample from either model reaches 0.90) shows the
  primary classifiers alone are not fit to silently accept predictions on
  this real-transcription distribution — the router/escalation path (not
  evaluated here; out of scope for this task) exists precisely for this kind
  of domain shift.

## Artifacts

- `silver_primary_1044.jsonl` — frozen input corpus.
- `xlmr/silver1044_xlmr__full_pipeline_{predictions,metrics}.{json,csv}`
- `mbert/silver1044_mbert__full_pipeline_{predictions,metrics}.{json,csv}`
- `scripts/analyze_silver_topic540_results.py` — computes weighted-F1,
  confusion matrix, distribution, cs_type breakdown, confidence stats (not
  emitted by `evaluate_pipeline.py` itself).
