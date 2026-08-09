# Handover — Silver Corpus × Topic-540 Evaluation

## Task
Evaluate the **existing weak Topic-540 primary classifiers** on a real-transcription
silver corpus. **No retraining. No further annotation.** Inference only.

## DONE (this session)

### 1. Frozen subset — 1,044 rows
`experiments/outputs/multi_agent_bert/experiment_silver_topic540/silver_primary_1044.jsonl`

Filtered from `data/Topic/transcription data/silver_corpus.csv` (1,163 rows) on:
`cs_verified=yes` AND `cs_category=lexical_or_phrase` AND `standalone=yes` AND
`route=accept_silver_primary` → **exactly 1,044 rows** (matches expectation).

Fields per row: `segment_id, text, label, silver_topic, cs_type, video_id,
num_tokens, cmi, ar_pct, en_pct`.

Distribution: tech 270, finance 216, business 99, shopping 89, health 88,
education 84, social 81, medical 64, sports 53.
`cs_type`: intrasentential 586, tag 458. 26 unique videos. 0 internal duplicates.

### 2. Label mapping — RESOLVED
The corpus uses **`technology`**; the Topic-540 checkpoints use **`tech`**.
Alias `{'technology': 'tech'}` is applied in the frozen file's `label` field
(original preserved in `silver_topic`). After aliasing, the silver label set is
**identical** to the ArEnTC train label set:
`business, education, finance, health, medical, shopping, social, sports, tech`.

### 3. Overlap verification — CLEAN
Normalised matching (NFKC, lowercase, strip Arabic diacritics/tatweel, fold
Arabic-Indic digits, remove punctuation, collapse whitespace); both exact equality
and token-set containment.

| Source | Rows | Exact | Contained |
|---|---|---|---|
| Topic-540 train (generated) | 540 | 0 | 0 |
| ARENTCV2 train / dev / test | 73,956 / 10,562 / 21,134 | 0 | 0 |
| ARENTCV1 train / dev / test | 73,976 / 10,569 / 21,137 | 0 | 0 |
| **TOTAL** | | **0** | **0** |

**No silver sentence overlaps Topic-540 training data or any ArEnTC split.**
Report: `overlap_report.json`. Script: `scripts/silver_freeze_and_overlap.py`.

## DONE — inference + full report (this session, continued)

`primary_only` inference run with both existing checkpoints (no retraining).
Full results (all requested breakdowns) written to `RESULTS.md` in this
directory. Headline: XLM-R acc 0.6303 / macro-F1 0.5572 / weighted-F1 0.6135;
mBERT acc 0.4904 / macro-F1 0.4394 / weighted-F1 0.4825. Both drop well below
their Topic-540 primary-only reference numbers (real-transcription domain
shift); neither model ever reaches 0.90 confidence on this corpus (100% below
threshold). See `RESULTS.md` for confusion matrices, predicted-vs-silver
distribution, tag-vs-intrasentential breakdown, confidence buckets.

Raw outputs: `xlmr/` and `mbert/` subdirectories (predictions + metrics
json/csv from `evaluate_pipeline.py`). Extra analysis (weighted-F1, confusion
matrix, distributions, confidence stats) computed by
`scripts/analyze_silver_topic540_results.py`, not by `evaluate_pipeline.py`
itself.

## REMAINING (superseded — kept for history)

~~Run `primary_only` inference with **both** existing checkpoints (do NOT retrain):~~
- `experiments/checkpoints/topic_gen540_xlmr`
- `experiments/checkpoints/topic_gen540_mbert`  (present and verified)

Both have the 9-label map ending in `tech`.

Then report, per classifier:
accuracy; macro-F1 and weighted-F1; per-class precision/recall/F1/support;
confusion matrix; predicted vs silver topic distribution; results split by
`cs_type` (tag vs intrasentential); confidence distribution and % below the
routing threshold (**topic threshold = 0.90**, per `src/config/default.yaml`).

### Reference numbers to compare against (Topic-540, generated-only training)
| Setting | Accuracy | Macro F1 |
|---|---|---|
| XLM-R primary-only | 0.7717 | 0.7552 |
| XLM-R full-agentic | 0.9232 | 0.9232 |

(Related in-repo figures on the ARENTC escalated subset: XLM-R primary 0.772/0.755
→ agentic 0.923/0.923; mBERT primary 0.653/0.617 → agentic 0.921/0.921.)

### Framing requirement
Describe the corpus as a **high-confidence silver real-transcription test set** —
labels come from `multi_llm_consensus_silver`. **Not gold, not human-validated.**

## Environment notes
- 4 GB GPU. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`.
- `evaluate_pipeline.py` runs one sample at a time (~15/sec) → 1,044 rows ≈ 1–2 min.
- Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- Disk: C: ~18 GB, D: ~22 GB free. Inference writes no checkpoints.
