# Full 1,163-row Silver Corpus × Topic-540 primary_only Evaluation

Exploratory evaluation on an **automatically labeled real-transcription silver
corpus** (`label_source=multi_llm_consensus_silver`) — **not a gold
benchmark, not human-validated**. `primary_only` inference only; no
retraining, no agentic pipeline, no threshold changes, no re-annotation.

## 0. Input validation (`silver_topic_test_1044.csv`)

| Check | Result |
|---|---|
| Row count | **1,163** ✓ |
| Column count | **2** (`text`, `label`) ✓ |
| Missing text | **0** |
| Missing label | **0** |
| Unique topic labels | **9** — business, education, health, shopping, medical, sports, tech, finance, social |
| `technology` → `tech` normalization | **Already applied in the supplied file.** Label column contains `tech` (294 rows), never `technology`. Verified by re-deriving from the raw `silver_corpus.csv`: every row's aliased `topic` (`technology`→`tech`) matches the supplied file's `label` exactly, for all 1,163 rows, with zero ambiguity (all 1,163 normalized texts are unique in the raw corpus). |

## 1. Reconciling with the original 1,044-row evaluation

Joined the new 1,163-row file back to `silver_corpus.csv` by exact normalized text (1:1 match, 0 unmatched, 0 collisions). Recomputing the original frozen-file filter (`cs_verified=yes AND cs_category=lexical_or_phrase AND standalone=yes AND route=accept_silver_primary`) from the raw CSV reproduces the exact same 1,044 `segment_id`s already frozen in `silver_primary_1044.jsonl` — **the original 1,044-row evaluation was reused unchanged**, not rerun.

The **119 additional rows** (1,163 − 1,044) break down exactly as expected:

| Category | n |
|---|---|
| `named_entity_only` | 105 |
| `acronym_or_model_only` | 12 |
| `lexical_or_phrase` but `standalone=no` | 2 |
| **Total new** | **119** |

Inference for these 119 rows only was run fresh (both checkpoints); the 1,044-row predictions were concatenated with them to form the 1,163-row combined result — the expensive part (1,044 rows) was not redone.

## 2. Headline metrics

| Model | Subset | n | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| XLM-R | 1,044 original (unchanged) | 1044 | 0.6303 | 0.5572 | 0.6135 |
| XLM-R | 119 new | 119 | 0.5714 | 0.5729 | 0.5259 |
| XLM-R | **1,163 combined** | **1163** | **0.6242** | **0.5599** | **0.6025** |
| mBERT | 1,044 original (unchanged) | 1044 | 0.4904 | 0.4394 | 0.4825 |
| mBERT | 119 new | 119 | 0.5042 | 0.5370 | 0.5255 |
| mBERT | **1,163 combined** | **1163** | **0.4918** | **0.4473** | **0.4852** |

### Delta: 1,163 combined vs. 1,044 original

| Model | Δ Accuracy | Δ Macro F1 | Δ Weighted F1 |
|---|---|---|---|
| XLM-R | −0.0061 | +0.0027 | −0.0110 |
| mBERT | +0.0014 | +0.0079 | +0.0027 |

The extra 119 rows (10.2% of the combined set) shift the aggregate numbers only marginally in either direction — the full-1,163 picture is consistent with the 1,044-row result, not materially different.

## 3. Per-class precision/recall/F1/support

### XLM-R — 1,163 combined

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| business | 0.5775 | 0.2662 | 0.3644 | 154 |
| education | 0.7308 | 0.2235 | 0.3423 | 85 |
| health | 0.9459 | 0.3889 | 0.5512 | 90 |
| shopping | 0.5294 | 0.6923 | 0.6000 | 91 |
| medical | 0.4841 | 0.8592 | 0.6193 | 71 |
| sports | 0.4667 | 0.7778 | 0.5833 | 63 |
| tech | 0.6505 | 0.8231 | 0.7267 | 294 |
| finance | 0.7458 | 0.7716 | 0.7585 | 232 |
| social | 0.5522 | 0.4458 | 0.4933 | 83 |

### mBERT — 1,163 combined

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| business | 0.5753 | 0.2727 | 0.3700 | 154 |
| education | 1.0000 | 0.1294 | 0.2292 | 85 |
| health | 0.7059 | 0.1333 | 0.2243 | 90 |
| shopping | 0.1963 | 0.6923 | 0.3058 | 91 |
| medical | 0.4884 | 0.8873 | 0.6300 | 71 |
| sports | 0.6389 | 0.7302 | 0.6815 | 63 |
| tech | 0.6796 | 0.7143 | 0.6965 | 294 |
| finance | 0.7414 | 0.3707 | 0.4943 | 232 |
| social | 0.3391 | 0.4699 | 0.3939 | 83 |

Full 1,044-only and 119-only per-class tables, confusion matrices, and true-vs-pred distributions are in `silver1163_summary.json` (machine-readable) — omitted here for length; see §6 for file list.

## 4. Confusion matrices — 1,163 combined (rows = true, cols = predicted)

### XLM-R

| true\\pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|---|---|---|---|---|---|---|---|---|
| business | 41 | 0 | 0 | 16 | 0 | 7 | 45 | 40 | 5 |
| education | 2 | 19 | 0 | 2 | 24 | 5 | 25 | 2 | 6 |
| health | 2 | 0 | 35 | 4 | 21 | 11 | 13 | 1 | 3 |
| shopping | 0 | 0 | 1 | 63 | 7 | 5 | 8 | 3 | 4 |
| medical | 0 | 0 | 0 | 2 | 61 | 1 | 6 | 0 | 1 |
| sports | 1 | 0 | 0 | 1 | 0 | 49 | 8 | 3 | 1 |
| tech | 11 | 5 | 0 | 6 | 2 | 13 | 242 | 8 | 7 |
| finance | 9 | 0 | 0 | 18 | 3 | 7 | 13 | 179 | 3 |
| social | 5 | 2 | 1 | 7 | 8 | 7 | 12 | 4 | 37 |

### mBERT

| true\\pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|---|---|---|---|---|---|---|---|---|
| business | 42 | 0 | 0 | 44 | 2 | 2 | 36 | 21 | 7 |
| education | 4 | 11 | 2 | 9 | 22 | 1 | 18 | 2 | 16 |
| health | 0 | 0 | 12 | 25 | 28 | 5 | 6 | 2 | 12 |
| shopping | 1 | 0 | 2 | 63 | 1 | 6 | 4 | 1 | 13 |
| medical | 1 | 0 | 0 | 3 | 63 | 1 | 2 | 0 | 1 |
| sports | 1 | 0 | 0 | 7 | 1 | 46 | 3 | 1 | 4 |
| tech | 7 | 0 | 0 | 47 | 3 | 7 | 210 | 3 | 17 |
| finance | 15 | 0 | 0 | 98 | 3 | 1 | 23 | 86 | 6 |
| social | 2 | 0 | 1 | 25 | 6 | 3 | 7 | 0 | 39 |

## 5. Predicted vs. true distribution — 1,163 combined

| Label | True n | XLM-R pred n | mBERT pred n |
|---|---|---|---|
| business | 154 | 71 | 73 |
| education | 85 | 26 | 11 |
| health | 90 | 37 | 17 |
| shopping | 91 | 119 | 321 |
| medical | 71 | 126 | 129 |
| sports | 63 | 105 | 72 |
| tech | 294 | 372 | 309 |
| finance | 232 | 240 | 116 |
| social | 83 | 67 | 115 |
| **Total** | **1163** | **1163** | **1163** |

## 6. Confidence distribution — 1,163 combined

| Model | Mean | Median | Min | Max | % below 0.90 |
|---|---|---|---|---|---|
| XLM-R | 0.4830 | 0.4857 | 0.1531 | 0.7964 | 100.00% (1163/1163) |
| mBERT | 0.3760 | 0.3510 | 0.1342 | 0.8147 | 100.00% (1163/1163) |

Same pattern as the 1,044-row set: neither model's confidence ever reaches 0.90 anywhere in the full corpus.

## 7. The 119-row diagnostic subset, by category

**Caveat:** `acronym_or_model_only` (n=12) and `standalone=no` (n=2) are small enough that macro-F1 over the full 9-class label space is not a meaningful summary (most classes have zero support, contributing F1=0 to the average by definition) — accuracy and the confusion matrix are more informative for these two. Reported as-is, not treated as invalid.

| Category | n | Model | Accuracy | Macro F1* | Notes |
|---|---|---|---|---|---|
| named_entity_only | 105 | XLM-R | 0.5333 | 0.5791 | Comparable to the 1,044-row set overall |
| named_entity_only | 105 | mBERT | 0.4667 | 0.5267 | Comparable to the 1,044-row set overall |
| acronym_or_model_only | 12 | XLM-R | 0.9167 | 0.2169* | 11/12 true label is `tech`; XLM-R gets 10/11 right |
| acronym_or_model_only | 12 | mBERT | 0.8333 | 0.1058* | Same skew; mBERT gets 10/11 `tech` right |
| standalone=no | 2 | XLM-R | 0.5000 | 0.1111* | n=2, not statistically meaningful on its own |
| standalone=no | 2 | mBERT | 0.5000 | 0.1111* | n=2, not statistically meaningful on its own |

`acronym_or_model_only` rows score *higher* accuracy than the general corpus for both models — expected, since this bucket is almost entirely single-topic (`tech`-heavy, e.g. segments naming a model/acronym), which both classifiers already handle relatively well. `named_entity_only` performs roughly in line with the general 1,044-row set, not obviously worse.

## 8. Summary

- The supplied two-column file validated cleanly: 1,163 rows, 2 columns, no missing text/labels, 9 topic labels, and the `technology`→`tech` normalization was already applied and verified consistent with the raw corpus.
- The full 1,163-row result is close to the already-reported 1,044-row result for both models (Δaccuracy ≤ 0.006, Δmacro-F1 ≤ 0.008) — adding back the excluded named-entity/acronym/non-standalone rows does not materially change the picture.
- The 119 previously-excluded rows are not uniformly harder or easier: `named_entity_only` tracks the main set, `acronym_or_model_only` scores noticeably higher (largely a `tech`-heavy bucket), and `standalone=no` is too small (n=2) to interpret on its own.
- No interpretation beyond the above is asserted here per the task's scope — this is a `primary_only` read on an automatically-labeled corpus, not a finalized domain-transfer conclusion.

## 9. Artifacts

- `silver_new119.jsonl` — the 119 newly-scored rows (text/label/metadata), frozen input for this run.
- `xlmr_new119/`, `mbert_new119/` — raw `evaluate_pipeline.py` outputs (predictions + metrics json/csv) for the 119 new rows only.
- `xlmr_combined_1163_predictions.csv`, `mbert_combined_1163_predictions.csv` — per-row predictions + confidence for all 1,163 rows per model, tagged with `source_subset` (`original_1044` / `named_entity_only` / `acronym_or_model_only` / `standalone_no`).
- `silver1163_summary.json` — full machine-readable metrics (all subsets, both models: accuracy, macro/weighted F1, per-class P/R/F1/support, confusion matrices, distributions, confidence stats).
- `scripts/analyze_silver1163_results.py` — read-only script that produced this report (reruns nothing already computed).
- Original `xlmr/`, `mbert/` (1,044-row) outputs and `RESULTS.md` — **unchanged**.
