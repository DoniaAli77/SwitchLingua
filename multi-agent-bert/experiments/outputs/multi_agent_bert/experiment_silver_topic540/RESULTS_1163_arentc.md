# Full-ArEnTC Checkpoints × Silver-1163 primary_only Evaluation

Exploratory evaluation on an **automatically labeled real-transcription
corpus** (`label_source=multi_llm_consensus_silver`) — not gold, not
human-validated. `primary_only` inference only; same silver input,
preprocessing, normalization, evaluation code, and row order as the completed
Topic-540 evaluation (`RESULTS_1163.md`). No retraining, no agentic run, no
threshold changes, no label edits.

## 1. Checkpoint identity (confirmed from run artifacts, not guessed)

| Model | Checkpoint path | Reference metric source | Reference accuracy / macro-F1 |
|---|---|---|---|
| Full-ArEnTC XLM-R | `experiments/checkpoints/topic_arentcv2_xlmr` | `experiment_T2_arentcv2_topic/primary_only/T2_primary_only__full_pipeline_metrics.json` (ARENTCV2 **test**, n=21,134) | 0.994700 / 0.994706 |
| Full-ArEnTC mBERT | `experiments/checkpoints/topic_arentcv2_mbert` | `experiment_T2_arentcv2_mbert/primary_only/T2_mbert_1ep__full_pipeline_metrics.json` (ARENTCV2 **test**, n=21,134) | 0.992335 / 0.992344 |

Checkpoint path confirmed directly from the `evaluate_pipeline.py` run logs (`eval_primary_only.log` / `eval.log`) that produced those reference numbers: `"PrimaryTransformerClassifier: loading model from 'experiments/checkpoints/topic_arentcv2_xlmr'"` and `'...topic_arentcv2_mbert'` respectively. Both trained on ARENTCV2 train (73,956 rows).

**Label mapping — important finding:** the two full-ArEnTC checkpoints use **different internal class-index orderings** (both self-consistent between their own `config.json` and `label_map.json`):

| id | XLM-R (`topic_arentcv2_xlmr`) | mBERT (`topic_arentcv2_mbert`) |
|---|---|---|
| 0 | business | business |
| 1 | education | education |
| 2 | **finance** | **health** |
| 3 | **health** | **shopping** |
| 4 | medical | medical |
| 5 | **shopping** | **sports** |
| 6 | **social** | **tech** |
| 7 | **sports** | **finance** |
| 8 | **tech** | **social** |

This differs from the Topic-540 checkpoints (which share one ordering — see the earlier audit). Verified this is **not a bug**: `evaluate_pipeline.py` never passes an explicit `label_map` override to `PrimaryTransformerClassifier.from_pretrained()` (checked in code — `build_primary_classifier()` is called with no `label_map` kwarg), so the classifier always falls back to each checkpoint's own `config.id2label`. Each model's predictions are decoded through its own mapping, so the differing order does not cause label misalignment — but it does mean the two full-ArEnTC checkpoints are not drop-in interchangeable at the raw logit level, only through their respective label maps.

## 2. Input

Identical to the Topic-540 1,163-row evaluation: `silver_full1163_ordered.jsonl` (1,044 original frozen rows + 119 new rows, same order, same `technology→tech` normalization already verified). No sentences or labels altered.

## 3. Headline metrics — full 1,163-row silver corpus

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Full-ArEnTC XLM-R | 0.6346 | 0.5636 | 0.6173 |
| Full-ArEnTC mBERT | 0.4858 | 0.4470 | 0.4921 |

## 4. Per-class precision/recall/F1/support (1,163 full)

### Full-ArEnTC XLM-R

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| business | 0.6667 | 0.2727 | 0.3871 | 154 |
| education | 0.7692 | 0.2353 | 0.3604 | 85 |
| health | 0.4882 | 0.6889 | 0.5714 | 90 |
| shopping | 0.4818 | 0.5824 | 0.5274 | 91 |
| medical | 0.4857 | 0.7183 | 0.5795 | 71 |
| sports | 0.4757 | 0.7778 | 0.5904 | 63 |
| tech | 0.7372 | 0.8299 | 0.7808 | 294 |
| finance | 0.7573 | 0.7802 | 0.7686 | 232 |
| social | 0.6102 | 0.4337 | 0.5070 | 83 |

### Full-ArEnTC mBERT

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| business | 0.5862 | 0.2208 | 0.3208 | 154 |
| education | 0.7222 | 0.1529 | 0.2524 | 85 |
| health | 0.3268 | 0.5556 | 0.4115 | 90 |
| shopping | 0.1940 | 0.7143 | 0.3052 | 91 |
| medical | 0.5500 | 0.4648 | 0.5038 | 71 |
| sports | 0.6154 | 0.7619 | 0.6809 | 63 |
| tech | 0.6986 | 0.6701 | 0.6840 | 294 |
| finance | 0.7397 | 0.4655 | 0.5714 | 232 |
| social | 0.5152 | 0.2048 | 0.2931 | 83 |

## 5. Confusion matrices (1,163 full; rows=true, cols=predicted)

### Full-ArEnTC XLM-R

| true\\pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|---|---|---|---|---|---|---|---|---|
| business | 42 | 1 | 2 | 17 | 3 | 8 | 45 | 31 | 5 |
| education | 2 | 20 | 3 | 0 | 39 | 3 | 13 | 4 | 1 |
| health | 0 | 0 | 62 | 5 | 4 | 11 | 7 | 0 | 1 |
| shopping | 0 | 0 | 19 | 53 | 3 | 8 | 3 | 2 | 3 |
| medical | 0 | 0 | 8 | 1 | 51 | 2 | 7 | 2 | 0 |
| sports | 0 | 0 | 0 | 2 | 0 | 49 | 4 | 6 | 2 |
| tech | 9 | 5 | 3 | 8 | 5 | 8 | 244 | 8 | 4 |
| finance | 7 | 0 | 4 | 19 | 0 | 7 | 7 | 181 | 7 |
| social | 3 | 0 | 26 | 5 | 0 | 7 | 1 | 5 | 36 |

### Full-ArEnTC mBERT

| true\\pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|---|---|---|---|---|---|---|---|---|
| business | 34 | 1 | 8 | 41 | 0 | 5 | 46 | 16 | 3 |
| education | 4 | 13 | 9 | 15 | 22 | 2 | 16 | 4 | 0 |
| health | 3 | 0 | 50 | 23 | 1 | 6 | 3 | 2 | 2 |
| shopping | 2 | 0 | 16 | 65 | 0 | 2 | 2 | 3 | 1 |
| medical | 0 | 1 | 26 | 4 | 33 | 1 | 4 | 2 | 0 |
| sports | 0 | 0 | 2 | 8 | 0 | 48 | 0 | 5 | 0 |
| tech | 8 | 2 | 2 | 62 | 4 | 10 | 197 | 5 | 4 |
| finance | 6 | 1 | 12 | 85 | 0 | 2 | 12 | 108 | 6 |
| social | 1 | 0 | 28 | 32 | 0 | 2 | 2 | 1 | 17 |

## 6. Predicted vs. silver distribution (1,163 full)

| Label | True n | Full-ArEnTC XLM-R pred n | Full-ArEnTC mBERT pred n |
|---|---|---|---|
| business | 154 | 63 | 58 |
| education | 85 | 26 | 18 |
| health | 90 | 127 | 153 |
| shopping | 91 | 110 | 335 |
| medical | 71 | 105 | 60 |
| sports | 63 | 103 | 78 |
| tech | 294 | 331 | 282 |
| finance | 232 | 239 | 146 |
| social | 83 | 59 | 33 |

## 7. Confidence distribution (1,163 full) — a key finding

| Model | Mean | Median | Min | Max | Below 0.90 |
|---|---|---|---|---|---|
| Full-ArEnTC XLM-R | 0.9639 | 0.9999 | 0.2597 | 1.0000 | 126/1163 (10.83%) |
| Full-ArEnTC mBERT | 0.9391 | 0.9987 | 0.2716 | 0.9999 | 187/1163 (16.08%) |

This is qualitatively different from the Topic-540 checkpoints, whose confidence **never** cleared 0.90 on this same corpus (0% below → wait, 100% *below* 0.90, i.e. 0% would clear the gate). The full-ArEnTC models are confident on ~84–89% of the silver rows (would clear the 0.90 routing gate and be accepted without escalation) **while only being ~49–63% accurate overall** — i.e., **the full-ArEnTC models are considerably more overconfident/miscalibrated on this out-of-domain corpus than the weak Topic-540 checkpoints are.** Topic-540 models "know they don't know" (uniformly low confidence); full-ArEnTC models mostly don't.

## 8. Subset breakdown

| Subset | n | XLM-R acc | XLM-R macro-F1 | mBERT acc | mBERT macro-F1 |
|---|---|---|---|---|---|
| 1,044 lexical/phrase, standalone | 1044 | 0.6466 | 0.5659 | 0.4971 | 0.4558 |
| 105 named-entity-only | 105 | 0.4762 | 0.5967 | 0.3333 | 0.4163 |
| 12 acronym/model-only | 12 | 0.9167 | 0.1063* | 0.8333 | 0.1058* |
| 2 non-standalone (retained, not interpreted) | 2 | 1.0000 | 0.2222* | 0.5000 | 0.1111* |

`*` macro-F1 on the two tiny subsets (n=12, n=2) averages in many zero-support classes and is not a meaningful summary there — reported as-is per instructions, not treated as invalid or excluded.

Both named-entity-heavy and full-corpus subsets show the same qualitative pattern as Topic-540: `acronym_or_model_only` (almost entirely `tech`) scores high accuracy for both models; `named_entity_only` is harder than the general corpus for both models here (more so than it was for Topic-540).

## 9. Four-model comparison on the identical Silver-1163 set

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Topic-540 XLM-R | 0.6242 | 0.5599 | 0.6025 |
| Topic-540 mBERT | 0.4918 | 0.4473 | 0.4852 |
| **Full-ArEnTC XLM-R** | **0.6346** | **0.5636** | **0.6173** |
| **Full-ArEnTC mBERT** | **0.4858** | **0.4470** | **0.4921** |

### Same-architecture delta: Full-ArEnTC vs. Topic-540 (both on Silver-1163)

| Architecture | Δ Accuracy | Δ Macro F1 | Δ Weighted F1 |
|---|---|---|---|
| XLM-R | +0.0104 | +0.0037 | +0.0148 |
| mBERT | −0.0060 | −0.0003 | +0.0069 |

Going from 540 generated training sentences to the full 73,956-row ARENTCV2 training set (~137× more data) produces only a **marginal** XLM-R improvement (~1 point accuracy) and an essentially **flat-to-slightly-worse** mBERT result on this silver corpus.

## 10. Each full-ArEnTC model's own drop from its ArEnTC-test reference

| Model | ArEnTC-test accuracy | Silver-1163 accuracy | Δ accuracy | ArEnTC-test macro-F1 | Silver-1163 macro-F1 | Δ macro-F1 |
|---|---|---|---|---|---|---|
| Full-ArEnTC XLM-R | 0.9947 | 0.6346 | **−0.3601** | 0.9947 | 0.5636 | **−0.4311** |
| Full-ArEnTC mBERT | 0.9923 | 0.4858 | **−0.5065** | 0.9923 | 0.4470 | **−0.5453** |

**This is a cross-corpus comparison, not a same-distribution comparison**: the 0.9947/0.9923 reference numbers come from the ARENTCV2 **held-out test split**, which is drawn from the same generation process/distribution as ARENTCV2 train (both are `ArEnTC.xlsx`-derived, machine-translated/synthetic-style code-switched text). The Silver-1163 corpus is a separate, automatically-labeled, **real spontaneous transcription** corpus. The two accuracy numbers are not measuring generalization within one distribution — they bracket a distribution shift.

## 11. Observations (not conclusions)

- Both full-ArEnTC checkpoints lose 36–51 accuracy points and 43–55 macro-F1 points moving from their own held-out test to the silver corpus, despite near-perfect (0.99+) in-domain performance.
- More training data (73,956 vs. 540 rows, same architecture) does **not** meaningfully close this gap — the silver-corpus scores for Full-ArEnTC and Topic-540 are close to each other (within ~1 accuracy point for XLM-R, mBERT essentially flat), even though the two checkpoints differ by two orders of magnitude in training-set size and by ~35–50 points on their own respective test sets.
- The full-ArEnTC models are also markedly more confident (and correspondingly more miscalibrated) on this corpus than the Topic-540 models, despite similar accuracy — a separate axis from raw accuracy that a threshold-based router should care about.
- Per the task's framing: this pattern is *consistent with* the transfer gap being dominated by something other than raw training-set size (e.g., a train/test distribution mismatch between ArEnTC-style text and real transcription, and/or label-noise effects specific to the silver corpus's automatic labeling) — but **whether it is domain mismatch, label noise, or both is not established by this evaluation alone** and is explicitly not asserted here.

## 12. Artifacts

- `silver_full1163_ordered.jsonl` — the exact 1,163-row input used for both Full-ArEnTC and Topic-540 1,163-row runs (same order).
- `arentc_xlmr_1163/`, `arentc_mbert_1163/` — raw `evaluate_pipeline.py` outputs (predictions + metrics json/csv).
- `fullarentc_xlmr_1163_predictions.csv`, `fullarentc_mbert_1163_predictions.csv` — per-row predictions + confidence, tagged by `source_subset`.
- `silver1163_arentc_summary.json` — full machine-readable metrics (all subsets, both models).
- `scripts/analyze_silver1163_arentc_results.py` — read-only analysis script.
- Topic-540 1,163-row files (`RESULTS_1163.md`, `xlmr_combined_1163_predictions.csv`, etc.) — **unchanged**.

## 13. Run configuration (for reproducibility)

```
python evaluate_pipeline.py \
  --dataset experiments/outputs/multi_agent_bert/experiment_silver_topic540/silver_full1163_ordered.jsonl \
  --config src/config/default.yaml --active_task topic_classification \
  --pipeline_mode primary_only --mode full_pipeline \
  --primary_model transformer --transformer_checkpoint experiments/checkpoints/topic_arentcv2_{xlmr,mbert} \
  --transformer_device cuda \
  --output_dir experiments/outputs/multi_agent_bert/experiment_silver_topic540/arentc_{xlmr,mbert}_1163 \
  --run_id silver1163_arentc{xlmr,mbert}
```

Stopped after `primary_only` per instruction — no `full_agentic`, no retraining, no threshold tuning, no label correction, no dataset modification.
