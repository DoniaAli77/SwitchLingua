# SwitchLingua Pipeline Architecture

## Overview

A LangGraph-based async pipeline for generating Arabic-English code-switched sentences with task-aware quality control.  
Each scenario produces N sentences, all evaluated and optionally refined **per sentence** — not as a batch.

---

## Full Graph Flow

```
START
  │
  ▼
DataGenerationAgent          Generate N sentences for the scenario (topic / sentiment / NER)
  │
  ▼
TaskValidatorAgent           Validate each sentence against the task label (per-instance)
  │
  ├──► FluencyAgent          Score fluency        (0–10, per sentence)
  ├──► NaturalnessAgent      Score naturalness    (0–10, per sentence)
  ├──► CSRatioAgent          Score CS ratio       (0–10, per sentence, + deterministic stats)
  └──► SocialCulturalAgent   Score socio-cultural (0–10, per sentence)
             │  (fan-out, all 4 run in parallel)
             ▼
        SummarizeResult      Build sentence_records[], compute weighted scores, flag failing sentences
             │
             ▼
        meet_criteria?  ────── any failing_sentence_indices AND refine_count < MAX_REFINER_ITERATIONS?
             │                          │
            YES                        NO
             │                          │
             ▼                          ▼
        RefinerAgent               AcceptanceAgent ──► END
             │                     (write to output JSONL)
             │
             │  (re-run quality agents on updated sentences)
             ├──► FluencyAgent
             ├──► NaturalnessAgent
             ├──► CSRatioAgent
             └──► SocialCulturalAgent
                        │
                        ▼
                   SummarizeResult   (fresh scores for refined sentences)
                        │
                        ▼
                   meet_criteria?   (loop guard: refine_count < MAX_REFINER_ITERATIONS)
```

---

## Agent Details

### DataGenerationAgent
- Routes to task-specific generation prompt: `DATA_GENERATION_{TOPIC|SENTIMENT|NER}_PROMPT`
- Retries up to 4 times if no instances returned
- Returns: `data_generation_result: list[str]`

### TaskValidatorAgent
- Validates each sentence individually against the task label
- Routes to: `TASK_VALIDATION_{TOPIC|SENTIMENT|NER}_PROMPT`
- NER uses LLM-only validation (deterministic hybrid disabled but preserved in comments)
- Returns: `task_validation_result` (aggregate) + `task_validation_results_per_instances: list`

### Quality Agents (run in parallel)

| Agent | Prompt | Output key | Score field |
|---|---|---|---|
| FluencyAgent | `FLUENCY_PROMPT` | `fluency_results_per_instances` | `fluency_score` |
| NaturalnessAgent | `NATURALNESS_PROMPT` | `naturalness_results_per_instances` | `naturalness_score` |
| CSRatioAgent | `CS_RATIO_PROMPT` | `cs_ratio_results_per_instances` | `ratio_score` |
| SocialCulturalAgent | `SOCIAL_CULTURAL_PROMPT` | `social_cultural_results_per_instances` | `socio_cultural_score` |

All 4 run per-sentence in a loop (not batch).

### Weighting Formula

$$\text{weighted\_score}_i = 0.30 \times \text{fluency}_i + 0.25 \times \text{naturalness}_i + 0.20 \times \text{cs\_ratio}_i + 0.25 \times \text{socio\_cultural}_i$$

### SummarizeResult
- Computes `sentence_scores[]` using the weighting formula
- Flags `failing_sentence_indices` (score < 8.0)
- Builds `sentence_records[]` — one dict per sentence containing:
  - `weighted_score`, `fluency`, `naturalness`, `cs_ratio`, `socio_cultural`
  - `task_validation`, `refine_count`, `needs_refine`, `eligible_for_refine`
- Returns: `score` (aggregate), `sentence_scores`, `failing_sentence_indices`, `sentence_records`

### meet_criteria (conditional edge)
```python
def meet_criteria(state):
    has_failing = bool(state.get("failing_sentence_indices"))
    refine_count = int(state.get("refine_count", 0) or 0)
    if has_failing and refine_count < MAX_REFINER_ITERATIONS:
        return "RefinerAgent"
    return "AcceptanceAgent"
```
- Sentence-based: checks `failing_sentence_indices` (not aggregate score)
- `MAX_REFINER_ITERATIONS = 1` (graph-level loop cap, env-configurable)

### RefinerAgent (per-sentence, per-failure-type)

**Step 1 — Failure classification** (per sentence from `sentence_records`):
- `task_fail` → task validation failed (`passed=False`)
- `quality_fail` → task passed but `weighted_score < 8.0`

**Step 2 — Prompt routing**:
| Failure type | Task | Prompt used |
|---|---|---|
| `task_fail` | topic | `REFINER_TASK_TOPIC_PROMPT` |
| `task_fail` | sentiment | `REFINER_TASK_SENTIMENT_PROMPT` |
| `task_fail` | ner | `REFINER_TASK_NER_PROMPT` |
| `quality_fail` | any | `REFINER_PROMPT` (generic quality) |

**Step 3 — Accept/Reject Guardrail** (per candidate sentence):

```
quality_fail path:
  1. Re-validate task on candidate → task broke? → ROLLBACK
  2. Re-score quality (_rescore_single_sentence) → score < original? → ROLLBACK
  3. Both pass → ACCEPT candidate

task_fail path:
  1. Re-validate task on candidate → task still fails? → ROLLBACK
  2. Task now passes → ACCEPT (quality regression tolerated)
```

**Step 4 — `_rescore_single_sentence(text, state)`**:
- Calls all 4 quality agents on a single-sentence state
- Returns `weighted_score` using `compute_sentence_weighted_scores()[0]`
- Used only inside guardrail (not as a graph node)

**Eligibility guard**: sentence skipped if `refine_count >= MAX_SENTENCE_REFINES` (env var, default 1)

### AcceptanceAgent
- Strips transient state keys (`news_article`, `news_hash`, `news_dict`)
- Appends final state to `output/{first_language}.jsonl`

---

## State Keys (AgentRunningState)

| Key | Type | Set by |
|---|---|---|
| `task` | str | scenario config |
| `data_generation_result` | `list[str]` | DataGenerationAgent / RefinerAgent |
| `task_validation_result` | dict | TaskValidatorAgent |
| `task_validation_results_per_instances` | `list[dict]` | TaskValidatorAgent |
| `fluency_results_per_instances` | `list[dict]` | FluencyAgent |
| `naturalness_results_per_instances` | `list[dict]` | NaturalnessAgent |
| `cs_ratio_results_per_instances` | `list[dict]` | CSRatioAgent |
| `social_cultural_results_per_instances` | `list[dict]` | SocialCulturalAgent |
| `sentence_scores` | `list[float]` | SummarizeResult |
| `failing_sentence_indices` | `list[int]` | SummarizeResult |
| `sentence_records` | `list[dict]` | SummarizeResult |
| `instance_refine_counts` | `list[int]` | RefinerAgent |
| `refine_count` | int | RefinerAgent (graph-level counter) |
| `score` | float | SummarizeResult (aggregate) |

---

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `MAX_SENTENCE_REFINES` | `1` | Max times a single sentence can be refined |
| `MAX_REFINER_ITERATIONS` | `1` | Max graph-level refine loops (in agents.py) |
| `ENABLE_TASK_VALIDATOR` | `1` | Set to `0` to bypass task validation |
| `OPENAI_API_KEY` | — | LLM auth |
| `OPENAI_BASE_URL` | — | LLM base URL |

---

## Test Coverage

| File | Tests | What it covers |
|---|---|---|
| `test files/test_pipeline_full_mocked.py` | 3 | Full graph run (topic, sentiment, NER) with mocked LLM |
| `test files/test_refiner_guardrail.py` | 4 | Guardrail scenarios: quality accepted, task rollback, task fixed, quality regression rollback |

Run all tests:
```bash
cd Modified_Version/core
python "test files/test_pipeline_full_mocked.py"
python "test files/test_refiner_guardrail.py"
```
