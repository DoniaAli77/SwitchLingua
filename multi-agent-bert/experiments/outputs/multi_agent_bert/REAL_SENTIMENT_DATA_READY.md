# Real Sentiment Data Readiness

Status date: 2026-06-05. **No training performed. No data generated.** Data path
preparation + sanity only.

---

## Task 1 — CSV → JSONL conversion (DONE)

Converter: [scripts/convert_eesa_csv_to_jsonl.py](../../../scripts/convert_eesa_csv_to_jsonl.py)
(reproducible; reads the normalized CSVs, never modifies them).

Output (in `data/Sentiment/processed/`):

| JSONL file | rows | labels (neg / neu / pos) | dropped | invalid |
|---|---|---|---|---|
| eesa_sentiment_train.jsonl | 2,464 | 594 / 778 / 1,092 | 0 | 0 |
| eesa_sentiment_dev.jsonl | 818 | 197 / 258 / 363 | 0 | 0 |
| eesa_sentiment_test.jsonl | 818 | 197 / 258 / 363 | 0 | 0 |

Schema: `{"text": "...", "label": "positive|negative|neutral"}`. Official splits
preserved 1:1, label set exact, counts identical to the source CSVs.

---

## Task 2 — Loader sanity (DONE — MOCK ONLY, NOT REAL RESULTS)

Confirmed Multi-Agent BERT reads EESA JSONL and runs all three modes without
training. Command pattern (per mode, on `eesa_sentiment_test.jsonl`):

```powershell
python evaluate_pipeline.py `
  --dataset data/Sentiment/processed/eesa_sentiment_test.jsonl `
  --config src/config/default.yaml --active_task sentiment_classification `
  --pipeline_mode <primary_only|paper_style|full_agentic> --mode full_pipeline `
  --primary_model mock
```

Result: all 3 modes loaded **818 samples, 0 skipped, 0 errors**.

| mode | accuracy | macro F1 |
|---|---|---|
| primary_only (mock) | 0.30 | 0.30 |
| paper_style | 0.45 | 0.34 |
| full_agentic (mock) | 0.42 | 0.27 |

**These are mock numbers — loader sanity only.** The mock primary is
non-deterministic; do not cite. They only prove the EESA JSONL pipeline is wired.

---

## Task 3 — Generated sentiment data readiness (NOT READY — plan only)

Searched modified SwitchLingua (`Modified_Version/output/`, read-only). Found
31 `task_generation_real_*.json` files; **10 contain a `sentiment` output**.

What's actually there:
- **Total: 20 generated sentiment sentences** (~1–2 per file).
- Labels are only available **indirectly** via the validator block
  (`validations.sentiment.label`, derived from `positive_hits`/`negative_hits`
  lexical counts — heuristic, not gold).
- Distribution is unusable: **15 positive, 5 with no label**, 0 negative,
  0 neutral.

**Conclusion: there is NO usable generated training set for Experiment C.** 20
skewed, heuristically-labeled sentences cannot train a classifier.

### Plan to produce Experiment C training data (DO NOT RUN without approval)
1. Use modified SwitchLingua task-generation (sentiment task) to generate a
   **balanced** code-switched set: target e.g. ~500–1,000 per class
   (positive/negative/neutral), prompted per target label so the label is
   the generation seed (not a post-hoc lexical guess).
2. Persist each instance as `{"text", "label"}` and concatenate to
   `data/Sentiment/processed/swlingua_sentiment_train.jsonl`
   (+ optional `_dev.jsonl`).
3. Keep generation seed/config recorded for reproducibility.
4. Experiment C then = train on `swlingua_sentiment_train.jsonl`, test on
   `eesa_sentiment_test.jsonl` (real benchmark, never used for training).

Blocked on your approval to run generation (and on the training loop in Task 4).

---

## Task 4 — Training plan (PROPOSAL ONLY — not implemented)

`PrimaryTransformerClassifier`
([src/models/primary_transformer_classifier.py](../../../src/models/primary_transformer_classifier.py))
is **inference-only**: `load` / `predict` / `run` / `from_pretrained`, no
training loop, no optimizer. So a real model must be fine-tuned by a **separate**
script and saved as a checkpoint that the existing classifier then loads.

**Recommendation: add a standalone `scripts/train_transformer_classifier.py`.**
Keep it fully separate from `evaluate_pipeline.py` and the agent architecture —
training is not a pipeline concern, and this avoids pulling torch into the
inference/test path.

Proposed contract:

- **Inputs:** `--train_jsonl` (+ optional `--dev_jsonl`), each `{"text","label"}`
  — works for both EESA (Exp A) and generated data (Exp C) unchanged.
- **Model:** `AutoModelForSequenceClassification` with `num_labels=3`, a fixed
  `label2id`/`id2label` = `{negative,neutral,positive}` written into the saved
  `config.json` so inference label mapping is automatic.
- **Loop:** HuggingFace `Trainer` (simplest) or a short manual torch loop;
  early stop / select on dev macro-F1.
- **Output checkpoint dir:** e.g.
  `experiments/checkpoints/eesa_mbert/` (Exp A) /
  `experiments/checkpoints/swlingua_mbert/` (Exp C) — a standard HF dir
  (`config.json`, `model.safetensors`, tokenizer files).
- **Loading into the pipeline (already supported, no new code):**
  ```powershell
  python evaluate_pipeline.py `
    --dataset data/Sentiment/processed/eesa_sentiment_test.jsonl `
    --config src/config/default.yaml --active_task sentiment_classification `
    --pipeline_mode primary_only --mode full_pipeline `
    --primary_model transformer `
    --transformer_checkpoint experiments/checkpoints/<run> `
    --transformer_device cpu
  ```
  Because `id2label` is baked into the checkpoint, the classifier maps and
  intersects with the task labels automatically.

### Recommended checkpoint for Arabic-English sentiment
- **Primary: `xlm-roberta-base`** (XLM-R). Strong on Arabic + handles the
  embedded English of code-switching well; generally ≥ mBERT on Arabic
  sentiment. ~270M params.
- **Lighter fallback: `bert-base-multilingual-cased`** (mBERT) — smaller, faster
  on CPU, the obvious baseline for a thesis comparison.
- Arabic-specialist models (MARBERT, CAMeLBERT) are excellent on Arabic but
  weaker on the English half of code-switched text — note as optional ablation,
  not the default.

### Dependencies
- Add to install for training: `torch`, `transformers`, plus `datasets` and
  `accelerate` (Trainer convenience), and `scikit-learn` (macro-F1 / report).
- `torch`/`transformers` already in `requirements.txt` (optional, uninstalled);
  `datasets`/`accelerate`/`scikit-learn` would be new — add only when training
  is approved.

### GPU / CPU considerations
- **Fine-tuning on CPU is impractical** for XLM-R/mBERT on 2,464 examples
  (hours/epoch). Recommend a GPU (Colab/Kaggle T4 is enough: minutes/epoch,
  3–4 epochs).
- **Inference** (the `--primary_model transformer` eval path) is fine on CPU for
  818 test rows.
- The corporate-network/SSL constraint applies to model download — may need the
  same proxy handling used elsewhere, or pre-download the checkpoint locally and
  point `--transformer_checkpoint` at the local path.

---

## What remains before real results
1. Decide experiment: **A** (EESA-train → EESA-test reference) vs **C**
   (generated → EESA-test transfer). A is unblocked on data; C needs generation.
2. (Approval) implement `scripts/train_transformer_classifier.py` + install
   training deps.
3. For C: (approval) run a balanced sentiment generation pass first.
4. Fine-tune → checkpoint → evaluate via the already-wired `--primary_model
   transformer` path.
