# Multi-Agent BERT — Experiments

Formal experiment outputs live here, separate from ad-hoc `results/`.

## Status: sanity baseline only (NOT thesis results)

These runs use **dummy data** (`data/dev_dummy_sentiment.jsonl`, 30 lines) and the
**mock** primary classifier + **mock** LLM client. They exist to prove the three
pipeline modes run end-to-end and produce the full metric set. They are **not**
comparable baselines and must not be cited as results.

### Why these numbers are not results
- The primary classifier is `MockPrimaryClassifier(mode="heuristic")`, which is
  **non-deterministic**: `primary_only` accuracy ranged **0.33–0.53** across
  identical runs on the same 30 rows.
- `full_agentic` uses a `label_echo`/`fixed` `MockLLMClient`, so its specialist
  agents inject noise rather than signal — expect it to score *below* `primary_only`.
- Only `paper_style` agents (keyword lexical, regex logic, TF-IDF contextual) run
  real algorithms, so its numbers are the only semi-meaningful ones — and even
  those sit on top of a mock primary.

## Sanity run — sentiment, 3 modes (2026-06-05)

Command (per mode, `$pm` in primary_only / paper_style / full_agentic):

```powershell
python evaluate_pipeline.py `
  --dataset data/dev_dummy_sentiment.jsonl `
  --config src/config/default.yaml --active_task sentiment_classification `
  --pipeline_mode $pm --mode full_pipeline `
  --output_dir experiments/outputs/multi_agent_bert/sentiment --run_id sanity_$pm
```

Representative output (single run — see non-determinism caveat above):

| pipeline_mode | accuracy | macro F1 | escalation rate |
|---|---|---|---|
| primary_only  | 0.33–0.53 | ~0.52 | 0.00 |
| paper_style   | 0.80 | 0.81 | 0.77 |
| full_agentic  | 0.37 | 0.28 | 0.80 |

Per-mode artifacts in `sentiment/`: `*_metrics.{json,csv}`, `*_predictions.{json,csv}`.

## What is required before this produces real results

1. **Real dataset** — a validated sentiment set (real, or validated export from
   SwitchLingua). Dummy data here is 30 hand-written rows.
2. **Real primary classifier** — `src/models/primary_transformer_classifier.py`
   exists but is (a) not wired into `build_orchestrator()` (which hardcodes the
   mock) and (b) needs `torch`/`transformers` installed (in `requirements.txt`,
   not currently importable). Wiring it in needs a small `--primary_model` flag.
3. (Optional) **Real LLM client** for `full_agentic`, replacing `MockLLMClient`.

Until 1–2 are done, treat `paper_style` as the only interpretable mode and all
numbers as smoke.
