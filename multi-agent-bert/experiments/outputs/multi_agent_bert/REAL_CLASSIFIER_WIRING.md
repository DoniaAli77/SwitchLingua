# Real Primary Classifier Wiring

Status: **wired, default = mock, not yet trained/run with a real model.**
Date: 2026-06-05.

The pipeline can now select between the mock primary classifier (default) and
the existing real Hugging Face transformer classifier via an explicit CLI flag.
No architecture was changed; mock remains the default everywhere.

## What changed

All changes are in [evaluate_pipeline.py](../../../evaluate_pipeline.py) plus one
new test file. The multi-agent architecture, agents, orchestrator, and
`PrimaryTransformerClassifier` itself were **not** modified.

1. **New factory `build_primary_classifier(primary_model, *, transformer_checkpoint, device, label_map)`**
   — the single decision seam:
   - `"mock"` → `MockPrimaryClassifier(mode="heuristic")` (downloads nothing).
   - `"transformer"` → `PrimaryTransformerClassifier.from_pretrained(checkpoint, label_map, device)`.
     Raises `ValueError` if no checkpoint is given. This is the **only** code
     path that loads/downloads a model.
2. **`build_orchestrator(...)` gained an optional `primary_classifier=None` param.**
   When `None` (every existing caller and test) it builds the mock — so behaviour
   is unchanged. `main()` injects the chosen primary explicitly.
3. **New CLI flags:**
   - `--primary_model {mock,transformer}` (default `mock`)
   - `--transformer_checkpoint PATH_OR_ID` (required for transformer mode)
   - `--transformer_device DEVICE` (default `cpu`)
4. **Threaded through all three run paths** (standard classification, NER,
   ablation). The primary is built once in `main()`, so a load failure (e.g.
   torch missing) is reported before any dataset work. A warning is logged
   when a real model is used, so mock results are never mistaken for real.

Config-file support (`execution.primary_model`) was intentionally **not** added
to avoid touching the config loader / `TaskConfig` schema and its tests. CLI is
the selection mechanism for now.

## How to run — MOCK mode (default, current)

```powershell
python evaluate_pipeline.py `
  --dataset data/dev_dummy_sentiment.jsonl `
  --config src/config/default.yaml --active_task sentiment_classification `
  --pipeline_mode paper_style --mode full_pipeline `
  --primary_model mock `
  --output_dir experiments/outputs/multi_agent_bert/sentiment --run_id mock_run
```

`--primary_model mock` is the default, so it can be omitted. Mock results are
**not** real results (mock primary is non-deterministic).

## How to run — TRANSFORMER mode (later, after deps + a checkpoint exist)

```powershell
# 1. install heavy deps (only needed for transformer mode)
pip install torch transformers

# 2. run with a real / fine-tuned multilingual checkpoint
python evaluate_pipeline.py `
  --dataset <eval.jsonl> `
  --config src/config/default.yaml --active_task sentiment_classification `
  --pipeline_mode primary_only --mode full_pipeline `
  --primary_model transformer `
  --transformer_checkpoint <hf-id-or-local-path> `
  --transformer_device cpu `
  --output_dir experiments/outputs/multi_agent_bert/sentiment --run_id real_run
```

The model's `id2label` is used unless overridden; the classifier intersects its
labels with `task_config.labels` so only `positive/negative/neutral` are scored.

## Dependencies

- `torch` and `transformers` are already declared in
  [requirements.txt](../../../requirements.txt) but are **not installed** and are
  **not** in `pyproject.toml` dependencies — deliberately optional. The unit
  suite never imports them.
- Install on demand only for transformer mode: `pip install torch transformers`.

## Tests run

- New: [tests/test_primary_model_selection.py](../../../tests/test_primary_model_selection.py)
  — 8 tests:
  - default / explicit `mock` returns `MockPrimaryClassifier`;
  - `transformer` routes to `PrimaryTransformerClassifier.from_pretrained`
    (monkeypatched — **no download**), with checkpoint/device/label_map asserted;
  - `transformer` without checkpoint raises; unknown value raises;
  - `build_orchestrator` default uses mock; injected primary is honoured.
- Full suite: **838 passed** (was 830 + 8 new). No downloads, offline.
- CLI: `--help` shows the three flags; `--primary_model transformer` without a
  checkpoint fails fast with a clear error.
- Mock-mode sentiment dry run (paper_style, dummy data): ran clean, 0 errors.

## What remains before real training / real results

1. **Install `torch` + `transformers`** (transformer mode only).
2. **A real checkpoint** — either a public multilingual sentiment model or a
   fine-tuned mBERT on EESA. This wiring is **inference-only**; no training loop
   exists yet (the class has `load`/`predict`/`run`, no `train`). Fine-tuning
   mBERT on EESA train is a separate, not-yet-built step.
3. **Dataset loader gap:** EESA processed files are **CSV** (`text,label`), but
   the classification path in `evaluate_pipeline.py` loads **JSONL**. Before the
   real experiments we need a CSV→JSONL conversion (or a CSV loader). Flagged,
   not done.
4. Decide the experiment: **A** (EESA-train → EESA-test reference) vs **C**
   (SwitchLingua-generated train → EESA-test transfer). Both need steps 1–3.
