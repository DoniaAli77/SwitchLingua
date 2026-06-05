# Fine-Tuning Path — Sentiment (EESA reference baseline)

Status date: 2026-06-05. **Script implemented; no fine-tuning job run yet.**
This is **fine-tuning a pretrained checkpoint** (default
`bert-base-multilingual-cased`), not training from scratch.

Script: [scripts/finetune_transformer_classifier.py](../../../scripts/finetune_transformer_classifier.py)
— standalone, never imports `evaluate_pipeline.py`. The pipeline only *loads*
the resulting checkpoint via `--primary_model transformer`.

---

## 1. Exact fine-tuning command (EESA reference baseline)

```powershell
# one-time, only for fine-tuning:
pip install torch transformers accelerate numpy

python scripts/finetune_transformer_classifier.py `
  --train data/Sentiment/processed/eesa_sentiment_train.jsonl `
  --dev   data/Sentiment/processed/eesa_sentiment_dev.jsonl `
  --base_checkpoint bert-base-multilingual-cased `
  --output_dir experiments/checkpoints/eesa_mbert `
  --epochs 4 --batch_size 16 --lr 2e-5 --max_length 256 --seed 42
```

CSV inputs work too — point `--train`/`--dev` at the `.csv` files and (only if
the columns differ) pass `--text_col` / `--label_col`. EESA CSVs already use
`text`/`label`, so no column flags are needed.

Writes to `experiments/checkpoints/eesa_mbert/`:
- HF checkpoint (`config.json` with `id2label`/`label2id`, `model.safetensors`, tokenizer)
- `label_map.json`, `training_metrics.json`, `dev_metrics.json`
  (dev = accuracy, macro F1, weighted F1, per-class P/R/F1).

For **xlm-roberta-base** (the stronger Arabic-English option) just swap
`--base_checkpoint xlm-roberta-base --output_dir experiments/checkpoints/eesa_xlmr`.

---

## 2. Exact evaluation command (using the fine-tuned checkpoint)

No new pipeline code needed — the `--primary_model transformer` seam already
exists.

```powershell
python evaluate_pipeline.py `
  --dataset data/Sentiment/processed/eesa_sentiment_test.jsonl `
  --config src/config/default.yaml --active_task sentiment_classification `
  --pipeline_mode primary_only --mode full_pipeline `
  --primary_model transformer `
  --transformer_checkpoint experiments/checkpoints/eesa_mbert `
  --transformer_device cpu `
  --output_dir experiments/outputs/multi_agent_bert/sentiment --run_id eesa_mbert_primary
```

The checkpoint's baked-in `id2label` is used automatically and intersected with
the task's `positive/negative/neutral`. Swap `--pipeline_mode` to `paper_style`
/ `full_agentic` to measure what the agents add on top of the real primary.

---

## 3. Code changes needed

- **New file:** `scripts/finetune_transformer_classifier.py` (this script).
- **New tests:** `tests/test_finetune_transformer_classifier.py` — 13 offline
  tests (arg parsing, JSONL+CSV loading, label maps, metrics). No downloads.
- **No changes** to `evaluate_pipeline.py`, the agents, the orchestrator, or
  `PrimaryTransformerClassifier` — the transformer-loading seam was already
  wired in the previous step.
- Full suite after adding: **851 passed** (was 838 + 13 new).

---

## 4. Dependency check

| Package | In requirements.txt | Installed now | Needed for |
|---|---|---|---|
| torch | yes | **no** | fine-tune + transformer inference |
| transformers | yes | **no** | fine-tune + transformer inference |
| accelerate | no | no | `Trainer` (fine-tune only) |
| numpy | no | no | metrics during fine-tune |

- The script and its tests import **none** of these at module load (lazy import
  inside `main()`), so unit tests stay offline. Verified: `--help` runs without
  torch, and running `main` without torch fails fast with a clear install hint
  and writes nothing.
- Install only when you approve fine-tuning:
  `pip install torch transformers accelerate numpy`.
- `accelerate`/`numpy` are new vs. the previous report — add them to
  `requirements.txt` (or a `requirements-train.txt`) when fine-tuning is approved.

---

## 5. Estimated runtime / GPU needs

Dataset: EESA train = 2,464 rows, dev = 818, `max_length` 256.

- **GPU (recommended):** mBERT or XLM-R base, 4 epochs on a single T4
  (Colab/Kaggle free tier) ≈ **5–15 minutes** total. fp16 (`--fp16`) roughly
  halves it and lowers memory. XLM-R is ~2× the compute/memory of mBERT.
- **CPU:** impractical for fine-tuning — expect **~1+ hour per epoch** for
  mBERT on this set; XLM-R worse. Use only as a last resort with reduced
  `--epochs`/`--max_length`.
- **Inference** (the eval command in §2) on 818 test rows is fine on **CPU**
  (a couple of minutes).
- **Network/SSL:** the base checkpoint is downloaded from the HF Hub on first
  use. On the corporate network, either pre-download the model to a local dir
  and pass that path to `--base_checkpoint`, or apply the same proxy handling
  used elsewhere in the project.

---

## Next decision
1. Approve installing the training deps + running the EESA fine-tune (§1) →
   produces the **Experiment A** reference checkpoint, then evaluate via §2.
2. **Experiment C** reuses this exact script with
   `--train data/Sentiment/processed/swlingua_sentiment_train.jsonl` once a
   balanced generated set exists (still pending approval to generate).
