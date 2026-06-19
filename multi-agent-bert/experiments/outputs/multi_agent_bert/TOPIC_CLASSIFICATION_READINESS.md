# Multi-Agent BERT — Topic Classification Readiness Report

Audit only. No fine-tuning, no full_agentic, nothing run. Date: 2026-06-19.

## Verdict
**Is the classification pipeline task-aware and ready for topic classification?
YES — at the architecture/config level.** The pipeline is fully config-driven and
already defaults to topic. The only gaps are **data and a trained checkpoint**, not
code or config.

---

## Item-by-item check

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | `active_task` selection in config | ✅ | `default.yaml` line 7: **`active_task: topic_classification`** (already the default). Overridable with `--active_task`. |
| 2 | Topic labels in config | ✅ | 9 labels: business, education, health, shopping, medical, sports, tech, finance, social. |
| 3 | Topic label descriptions | ✅ | Full bilingual (EN + AR) `label_descriptions` for all 9. |
| 4 | Topic keyword_map | ✅ | Config `label_knowledge` → `keywords_l1` (EN) + `keywords_l2` (AR) for all 9; built by `src/config/loader.py`. |
| 5 | Topic rule_map | ✅ | Config `regex_rules` for all 9; built by the loader (AR alternation auto-appended). |
| 6 | Primary classifier label_map | ✅ (conditional) | `build_primary_classifier` passes **`label_map=None`** → `PrimaryTransformerClassifier` falls back to the **checkpoint's `config.id2label`**. No hardcoded sentiment map. *Requires a topic-trained checkpoint that carries the correct 9-label id2label (our finetune script bakes this in).* |
| 7 | `evaluate_pipeline.py --active_task topic_classification` | ✅ | Flag supported (documented example). `task_type=classification` → normal path; only `sequence_labeling` (NER) hits the `NotImplementedError`. |
| 8 | Old synthesized train/dev/test labels match config labels | ⚠️ | **No real topic train/dev/test exists** — only a 30-row smoke dummy `data/dev_dummy.jsonl`, whose labels **exactly match** the 9. So no mismatch today, but no usable dataset either. |
| 9 | Sentiment-specific defaults that could affect topic | ✅ | None blocking. `active_task` already topic; `execution` defaults (threshold 0.6, full_agentic, signal off) are task-agnostic; the merged sentiment+topic keyword pool is **restricted to the active labels**, so sentiment keys are ignored for topic; label_map not hardcoded. |

---

## What is already task-aware
- **Config registry** (`tasks:` in `default.yaml`) — each task carries its own
  `task_type`, `labels`, `label_descriptions`, and `label_knowledge`. Topic entry
  is complete.
- **Loader** (`src/config/loader.py`, `load_task_bundle`) builds `keyword_map` /
  `rule_map` from the config; evaluate_pipeline uses these and only falls back to a
  hardcoded map if they're `None`.
- **Primary label mapping** is generic — taken from the checkpoint's `id2label`,
  not hardcoded to sentiment.
- **Router, consensus, explainability** are all label-agnostic (operate over
  `task_config.labels`).
- `--active_task` override and the `classification` execution path work for any
  classification task; topic is the default.

## What must change before a topic experiment (NOT architecture)
1. **A topic dataset.** There is no real topic `train/dev/test` — only a 30-sample
   dummy. Need a generated/curated topic set with labels **exactly** the 9 strings
   (lowercase, exact spelling), plus dev/test for evaluation.
2. **A topic-trained primary checkpoint.** We've only trained sentiment models.
   Fine-tune `xlm-roberta-base` on topic data (same recipe) so the checkpoint's
   `id2label` carries the 9 topic labels. *(Not now — you asked not to fine-tune.)*
3. **(Optional housekeeping)** There are **two** topic keyword/rule sources — the
   config `label_knowledge` (primary) and a hardcoded `_TOPIC_KNOWLEDGE` in
   `evaluate_pipeline.py` (fallback). Both use the same 9 labels, so no conflict,
   but for a single source of truth consider dropping the hardcoded copy.

## Final recommendations
- **Recommended `active_task` name:** **`topic_classification`** (already the config default — no change needed).
- **Final topic label set (9):** `business, education, health, shopping, medical, sports, tech, finance, social`.
- **Required config edits:** **None** — the config is complete and correct for topic.
  Optional only: remove the duplicate hardcoded `_TOPIC_KNOWLEDGE` in
  evaluate_pipeline.py to keep keyword/rule knowledge solely in `default.yaml`.

**Bottom line:** the pipeline is architecturally ready for topic classification
today. To actually run Experiment (topic), the missing pieces are a **topic dataset**
and a **topic-trained checkpoint** — both data tasks, no code/architecture changes
required.
