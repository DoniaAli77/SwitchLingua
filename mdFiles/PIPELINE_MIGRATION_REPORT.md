# SwitchLingua Pipeline — Migration & Implementation Report

**Date:** April 16, 2026  
**Codebase:** `Modified_Version/core/`  
**Model:** `gpt-4o-mini` (temperature 0.1)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Problems Identified](#2-problems-identified)
3. [What Was Changed and Why](#3-what-was-changed-and-why)
4. [Full Pipeline Flow](#4-full-pipeline-flow)
5. [Agent-by-Agent Reference](#5-agent-by-agent-reference)
6. [Prompt Inventory](#6-prompt-inventory)
7. [Refiner Architecture (Deep Dive)](#7-refiner-architecture-deep-dive)
8. [Test Coverage](#8-test-coverage)
9. [State Schema Reference](#9-state-schema-reference)
10. [Current Status](#10-current-status)

---

## 1. Overview

The SwitchLingua pipeline generates Arabic–English code-switched sentences under three task types:

| Task | Goal |
|---|---|
| `topic` | Sentences must clearly belong to a given topic |
| `sentiment` | Sentences must express a specific sentiment label (`positive`/`negative`) |
| `ner` | Sentences must contain named entities of required types (`PER`, `ORG`, `LOC`, …) |

The pipeline evaluates each generated sentence across four quality dimensions (fluency, naturalness, CS-ratio, socio-cultural) and attempts to refine any sentence that fails either the task constraint or the quality threshold.

---

## 2. Problems Identified

### 2.1 Batch Evaluation (Root Cause)
Quality agents (`RunFluencyAgent`, `RunNaturalnessAgent`, `RunCSRatioAgent`, `RunSocialCulturalAgent`) were sending **all N sentences in one LLM call**. This caused:

- The LLM to score sentences **relative to each other**, not independently.
- Inconsistent scores across runs (order-sensitive LLM attention).
- Impossible to trace which score belongs to which sentence when there was a fallback path.

**Fix:** Migrated all 4 quality agents to **true per-sentence loops** — one LLM call per sentence.

### 2.2 CS-Ratio Prompt Variable Mismatch
The `CS_RATIO_PROMPT` expected a variable `{data_generation_result}` but the payload sent `{sentences_with_stats}`. The LLM received an empty/wrong input, causing consistently low CS-ratio scores.

**Fix:** Updated `CS_RATIO_PROMPT` to use `{sentence_with_stats}` (single sentence with deterministic stats injected from `compute_true_cs_stats()`).

### 2.3 Refiner Did Not Distinguish Task vs Quality Failures
The original `RunRefinerAgent` only checked `weighted_score < 8.0` and used a single generic prompt for all refinements. It could not:

- Fix a sentence that failed the task but had good quality scores.
- Prevent a quality-focused rewrite from flipping sentiment / removing NER entities.

**Fix:** Refiner now classifies each failure as `task_fail` or `quality_fail`, routes to a task-specific prompt or quality prompt accordingly, and applies an **accept/reject guardrail** after each rewrite.

### 2.4 Dead Code in CS-Ratio Agent
There was a commented-out legacy batch fallback block and duplicate return statements that created confusion.

**Fix:** Removed all dead code from `RunCSRatioAgent`.

---

## 3. What Was Changed and Why

### `Modified_Version/core/node_engine.py`

| Change | Reason |
|---|---|
| `RunFluencyAgent` — per-sentence loop | Remove batch evaluation bias |
| `RunNaturalnessAgent` — per-sentence loop | Same |
| `RunSocialCulturalAgent` — per-sentence loop | Same |
| `RunCSRatioAgent` — per-sentence loop + deterministic stats | Same + fix variable mismatch |
| `RunCSRatioAgent` — dead code removed | Clean up legacy fallback |
| `RunRefinerAgent` — `failure_reasons` dict | Classify each sentence as `task_fail` / `quality_fail` |
| `RunRefinerAgent` — task-specific prompt routing | Use correct prompt per task type |
| `RunRefinerAgent` — accept/reject guardrail | Prevent task regression after quality rewrite |
| Import `REFINER_TASK_TOPIC_PROMPT`, `REFINER_TASK_SENTIMENT_PROMPT`, `REFINER_TASK_NER_PROMPT` | New task-specific refiner prompts |

### `Modified_Version/core/prompt.py`

| Change | Reason |
|---|---|
| `FLUENCY_PROMPT` — `{sentences_for_batch}` → `{sentence}` | Per-sentence migration |
| `NATURALNESS_PROMPT` — `{sentences_for_batch}` → `{sentence}` | Same |
| `CS_RATIO_PROMPT` — `{sentences_with_stats}` → `{sentence_with_stats}` | Fix variable mismatch; single sentence |
| `SOCIAL_CULTURAL_PROMPT` — `{sentences_for_batch}` → `{sentence}` | Per-sentence migration |
| Added `REFINER_TASK_TOPIC_PROMPT` | Fix topic task failures |
| Added `REFINER_TASK_SENTIMENT_PROMPT` | Fix sentiment task failures (preserve polarity) |
| Added `REFINER_TASK_NER_PROMPT` | Fix NER task failures (preserve entity types) |

### Test files

| File | Change |
|---|---|
| `test_pipeline_full_mocked.py` | Added `RunRefinerAgent` call; patched 3 new task refiner prompts; added `refiner` / `refiner_task` LLM modes; added refine-count assertions |
| `test_refiner_guardrail.py` | **New file** — 3 focused guardrail scenario tests |

---

## 4. Full Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INPUT STATE                                      │
│  task, topic/label/annotations, cs_ratio, first_language,              │
│  second_language, tense, perspective, gender, age, …                   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1 — DataGeneration                                                │
│  RunDataGenerationAgent                                                 │
│  ├─ task=topic     → RunTopicDataGenerationAgent                        │
│  ├─ task=sentiment → RunSentimentDataGenerationAgent                    │
│  └─ task=ner       → RunNERDataGenerationAgent                          │
│                                                                         │
│  Output: data_generation_result = ["sentence_0", "sentence_1", …]      │
│  Prompt variables: topic/label/annotations, cs_ratio, cs_function, …   │
│  Retry: up to 4 times if instances list is empty                        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2 — TaskValidator (per-sentence, task-aware)                      │
│  RunTaskValidatorAgent                                                  │
│  ├─ task=topic     → RunTopicTaskValidatorAgent                         │
│  ├─ task=sentiment → RunSentimentTaskValidatorAgent                     │
│  └─ task=ner       → RunNERTaskValidatorAgent                           │
│                                                                         │
│  For each sentence[i]:                                                  │
│    LLM call → TaskValidationResult {passed, confidence, notes,          │
│                                     predicted_label, errors}            │
│                                                                         │
│  Output:                                                                │
│    task_validation_result (aggregate)                                   │
│    task_validation_results_per_instances [result_0, result_1, …]       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3 — Quality Evaluation (per-sentence, independent)                │
│                                                                         │
│  RunFluencyAgent                                                        │
│    For each sentence[i]:                                                │
│      stats = compute_true_cs_stats(sentence[i])                         │
│      LLM(FLUENCY_PROMPT, {sentence}) → {fluency_score, errors, summary} │
│    Output: fluency_results_per_instances, fluency_result (aggregate)    │
│                                                                         │
│  RunNaturalnessAgent                                                    │
│    For each sentence[i]:                                                │
│      LLM(NATURALNESS_PROMPT, {sentence}) → {naturalness_score, …}      │
│    Output: naturalness_results_per_instances, naturalness_result        │
│                                                                         │
│  RunCSRatioAgent                                                        │
│    For each sentence[i]:                                                │
│      stats = compute_true_cs_stats(sentence[i])   ← DETERMINISTIC       │
│      LLM(CS_RATIO_PROMPT, {sentence_with_stats, cs_ratio, …})           │
│           → {ratio_score, computed_ratio, notes}                        │
│    Output: cs_ratio_results_per_instances                               │
│                                                                         │
│  RunSocialCulturalAgent                                                 │
│    For each sentence[i]:                                                │
│      LLM(SOCIAL_CULTURAL_PROMPT, {sentence}) → {socio_cultural_score, …}│
│    Output: social_cultural_results_per_instances, social_cultural_result│
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4 — SummarizeResult                                               │
│                                                                         │
│  compute_sentence_weighted_scores(state)                                │
│    → weighted_score[i] = w_flu*fluency + w_nat*naturalness              │
│                         + w_cs*cs_ratio + w_soc*socio_cultural          │
│                                                                         │
│  build_sentence_records(state, …)                                       │
│    → sentence_records[i] = {                                            │
│        index, text, fluency, naturalness, cs_ratio, socio_cultural,     │
│        weighted_score, refine_count, status, task_passed,               │
│        task_validation                                                  │
│      }                                                                  │
│    status ∈ {pass, refined_pass, fail, budget_exhausted}                │
│                                                                         │
│  Output: score (final), summary, sentence_scores, sentence_records,     │
│          failing_sentence_indices, instance_refine_counts               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 5 — RunRefinerAgent (conditional, per-sentence)                   │
│                                                                         │
│  For each sentence[i] in sentence_records:                              │
│                                                                         │
│    ┌─── Classify failure ──────────────────────────────────────────┐   │
│    │  if task_validation[i].passed == False:                        │   │
│    │      failure_reason = "task_fail"                              │   │
│    │  elif weighted_score[i] < 8.0:                                 │   │
│    │      failure_reason = "quality_fail"                           │   │
│    │  else:                                                         │   │
│    │      skip (sentence is fine)                                   │   │
│    └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│    ┌─── Check refine budget ───────────────────────────────────────┐   │
│    │  if refine_count[i] >= MAX_SENTENCE_REFINES (default=1):       │   │
│    │      skip (budget exhausted)                                   │   │
│    └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│    ┌─── Select prompt ─────────────────────────────────────────────┐   │
│    │  task_fail + topic     → REFINER_TASK_TOPIC_PROMPT             │   │
│    │  task_fail + sentiment → REFINER_TASK_SENTIMENT_PROMPT         │   │
│    │  task_fail + ner       → REFINER_TASK_NER_PROMPT               │   │
│    │  quality_fail (any)    → REFINER_PROMPT (generic quality)      │   │
│    └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│    ┌─── LLM call ──────────────────────────────────────────────────┐   │
│    │  refiner.invoke({sentence, task_validation_feedback, …})       │   │
│    │  → candidate sentence                                          │   │
│    └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│    ┌─── Accept / Reject Guardrail ─────────────────────────────────┐   │
│    │                                                                │   │
│    │  Re-evaluate candidate with TaskValidator                      │   │
│    │                                                                │   │
│    │  quality_fail path:                                            │   │
│    │    task was passing → must still pass after refine             │   │
│    │    if candidate breaks task → ROLLBACK (keep original)         │   │
│    │                                                                │   │
│    │  task_fail path:                                               │   │
│    │    task was failing → must now pass after refine               │   │
│    │    if candidate still fails task → ROLLBACK (keep original)    │   │
│    │                                                                │   │
│    │  if accepted → update sentence, increment refine_count[i]     │   │
│    └────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Output: data_generation_result (updated), instance_refine_counts      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 6 — AcceptanceAgent                                               │
│  Writes final state to output/{language}.jsonl                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Agent-by-Agent Reference

| Agent | File | Task-Aware | Per-Sentence | Notes |
|---|---|---|---|---|
| `RunDataGenerationAgent` | `node_engine.py:492` | ✅ 3 variants | N/A (generates sentences) | Retry up to 4× if empty |
| `RunTopicTaskValidatorAgent` | `node_engine.py:251` | ✅ topic | ✅ | Uses `TASK_VALIDATION_TOPIC_PROMPT` |
| `RunSentimentTaskValidatorAgent` | `node_engine.py:259` | ✅ sentiment | ✅ | Uses `TASK_VALIDATION_SENTIMENT_PROMPT` |
| `RunNERTaskValidatorAgent` | `node_engine.py:448` | ✅ NER | ✅ | Deterministic entity check + LLM |
| `RunFluencyAgent` | `node_engine.py:539` | ❌ generic | ✅ | `{sentence}` per call |
| `RunNaturalnessAgent` | `node_engine.py:590` | ❌ generic | ✅ | `{sentence}` per call |
| `RunCSRatioAgent` | `node_engine.py:766` | ❌ generic | ✅ | Deterministic stats via `compute_true_cs_stats()` |
| `RunSocialCulturalAgent` | `node_engine.py:815` | ❌ generic | ✅ | `{sentence}` per call |
| `SummarizeResult` | `node_engine.py:861` | N/A | N/A | Builds `sentence_records`, computes weighted scores |
| `RunRefinerAgent` | `node_engine.py:935` | ✅ (task_fail path) | ✅ | Guardrail + rollback |
| `AcceptanceAgent` | `node_engine.py:920` | N/A | N/A | Writes to JSONL output |

> **Note:** Quality agents (Fluency, Naturalness, CSRatio, SocioCultural) are generic (not task-aware) because they measure linguistic quality, not task correctness. Task correctness is handled by TaskValidator and the task-specific refiner prompts.

---

## 6. Prompt Inventory

### Generation Prompts
| Prompt | Variables | Task |
|---|---|---|
| `DATA_GENERATION_TOPIC_PROMPT` | `topic`, `cs_ratio`, `first_language`, `second_language`, `cs_function`, `cs_type`, `tense`, `perspective`, `gender`, `age`, `education_level`, `mcp_result` | topic |
| `DATA_GENERATION_SENTIMENT_PROMPT` | Same + `label`, `intensity`, `ambiguity` | sentiment |
| `DATA_GENERATION_NER_PROMPT` | Same + `entity_types`, `min_entities`, `max_entities` | NER |

### Validation Prompts
| Prompt | Variables | Task |
|---|---|---|
| `TASK_VALIDATION_TOPIC_PROMPT` | `data_generation_result`, `topic` | topic |
| `TASK_VALIDATION_SENTIMENT_PROMPT` | `data_generation_result`, `label` | sentiment |
| `TASK_VALIDATION_NER_PROMPT` | `data_generation_result`, `entity_types` | NER |

### Quality Evaluation Prompts (generic, per-sentence)
| Prompt | Key Variable | Output |
|---|---|---|
| `FLUENCY_PROMPT` | `{sentence}` | `{fluency_score, errors, summary}` |
| `NATURALNESS_PROMPT` | `{sentence}` | `{naturalness_score, observations, summary}` |
| `CS_RATIO_PROMPT` | `{sentence_with_stats}`, `{cs_ratio}`, `{target_matrix_ratio}`, `{target_embedded_ratio}` | `{ratio_score, computed_ratio, notes}` |
| `SOCIAL_CULTURAL_PROMPT` | `{sentence}` | `{socio_cultural_score, issues, summary}` |

### Refiner Prompts
| Prompt | Used When | Key Variables |
|---|---|---|
| `REFINER_PROMPT` | `quality_fail` (any task) | `{summary}` (quality feedback) |
| `REFINER_TASK_TOPIC_PROMPT` | `task_fail` + `topic` | `{sentence}`, `{topic}`, `{task_validation_feedback}`, `{first_language}`, `{second_language}` |
| `REFINER_TASK_SENTIMENT_PROMPT` | `task_fail` + `sentiment` | `{sentence}`, `{label}`, `{task_validation_feedback}`, `{first_language}`, `{second_language}` |
| `REFINER_TASK_NER_PROMPT` | `task_fail` + `ner` | `{sentence}`, `{entity_types}`, `{task_validation_feedback}`, `{first_language}`, `{second_language}` |

---

## 7. Refiner Architecture (Deep Dive)

### Decision Logic

```
For each sentence[i]:

  1. Is task_passed[i] == False?
       YES → failure_reason = "task_fail"
       NO  → Is weighted_score[i] < 8.0?
               YES → failure_reason = "quality_fail"
               NO  → sentence is fine, SKIP

  2. Is refine_count[i] >= MAX_SENTENCE_REFINES?
       YES → SKIP (budget exhausted)

  3. Select prompt:
       task_fail  → task-specific prompt (topic/sentiment/ner)
       quality_fail → generic REFINER_PROMPT

  4. Call LLM → get candidate

  5. Guardrail: re-validate candidate with TaskValidator
       quality_fail + candidate breaks task → ROLLBACK
       task_fail + candidate still fails    → ROLLBACK
       otherwise                            → ACCEPT, increment refine_count[i]
```

### Guardrail Decision Table

| Failure Type | Before Refine | After Refine | Decision |
|---|---|---|---|
| `quality_fail` | task=PASS, score<8.0 | task=PASS | ✅ Accept |
| `quality_fail` | task=PASS, score<8.0 | task=FAIL | ❌ Rollback |
| `task_fail` | task=FAIL | task=PASS | ✅ Accept |
| `task_fail` | task=FAIL | task=still FAIL | ❌ Rollback |

### Why This Matters

Without the guardrail:
- A quality refiner might rephrase a positive sentence so naturally that the sentiment flips to negative.
- A task refiner might add the required topic keywords but destroy the NER entity structure.

With the guardrail, **every accepted rewrite is validated to be at least as correct on the task as before the refinement**.

---

## 8. Test Coverage

### `test files/test_pipeline_full_mocked.py`
Full end-to-end mocked pipeline for all 3 task types.

| Test | Covers |
|---|---|
| `topic` pipeline | DataGen → TaskVal → Fluency → Naturalness → CSRatio → SocioCultural → Summarize → Refiner |
| `sentiment` pipeline | Same flow |
| `ner` pipeline | Same flow |

**Assertions checked:**
- `data_generation_result` has N > 0 sentences
- All per-instance arrays have length N
- Aggregate score fields exist
- Final score matches `weighting_scheme(state)`
- `task_validation_result.passed == True`
- All `instance_refine_counts` == 1 (all sentences were below 8.0 threshold, refiner ran)

### `test files/test_refiner_guardrail.py`
Focused unit tests for the 3 guardrail scenarios.

| Test | Scenario | Verified |
|---|---|---|
| `test_refiner_quality_fail_accepted` | quality_fail → guardrail OK → accept | Updated sentence, refine_count=1 |
| `test_refiner_quality_fail_rollback` | quality_fail → guardrail: task broke → rollback | Original kept, refine_count=0 |
| `test_refiner_task_fail_accepted` | task_fail → task refiner → guardrail OK → accept | Fixed sentence, refine_count=1 |

**All 6 tests pass with exit code 0.**

---

## 9. State Schema Reference

Key fields in `AgentRunningState` after the full pipeline completes:

| Field | Type | Description |
|---|---|---|
| `data_generation_result` | `List[str]` | Final sentences (after any refinement) |
| `task_validation_result` | `dict` | Aggregate task validation result |
| `task_validation_results_per_instances` | `List[dict]` | Per-sentence task validation |
| `fluency_results_per_instances` | `List[dict]` | Per-sentence fluency scores |
| `naturalness_results_per_instances` | `List[dict]` | Per-sentence naturalness scores |
| `cs_ratio_results_per_instances` | `List[dict]` | Per-sentence CS-ratio scores |
| `social_cultural_results_per_instances` | `List[dict]` | Per-sentence socio-cultural scores |
| `sentence_records` | `List[dict]` | Canonical per-sentence records (all metrics + status) |
| `sentence_scores` | `List[float]` | Weighted score per sentence |
| `failing_sentence_indices` | `List[int]` | Indices of sentences below threshold |
| `instance_refine_counts` | `List[int]` | How many times each sentence was refined |
| `score` | `float` | Final pipeline score |
| `summary` | `str` | Human-readable summary string |
| `records_consistency` | `dict` | Consistency validation of sentence_records |

### `sentence_records[i]` structure:
```json
{
  "index": 0,
  "text": "أنا أحب learning because it is useful.",
  "fluency": {"fluency_score": 8.5, "errors": {}, "summary": "..."},
  "naturalness": {"naturalness_score": 7.8, "observations": {}, "summary": "..."},
  "cs_ratio": {"ratio_score": 7, "computed_ratio": "70%:30%", "notes": "..."},
  "socio_cultural": {"socio_cultural_score": 8.0, "issues": "", "summary": "..."},
  "weighted_score": 7.83,
  "refine_count": 1,
  "status": "refined_pass",
  "task_passed": true,
  "task_validation": {"passed": true, "confidence": 0.92, "notes": "...", "predicted_label": "positive", "errors": []}
}
```

### `sentence_records[i].status` values:
| Status | Meaning |
|---|---|
| `pass` | Score ≥ 8.0, never refined |
| `refined_pass` | Score ≥ 8.0 after at least one refinement |
| `fail` | Score < 8.0, still has refinement budget |
| `budget_exhausted` | Score < 8.0, refinement budget fully used |

---

## 10. Current Status

### Completed ✅
- Per-sentence migration of all 4 quality agents
- Prompt variable fixes (all 4 quality prompts updated)
- CS-ratio dead code removed
- Refiner failure classification (`task_fail` / `quality_fail`)
- Task-specific refiner prompts (3 variants)
- Prompt routing in refiner based on failure type + task
- Accept/reject guardrail with rollback
- Full pipeline test (6 tests, all passing)

### Remaining / Future Work ⚠️
- Quality agents (Fluency, Naturalness, SocioCultural) are still **generic** — not task-aware. They evaluate linguistic quality only. This is acceptable but could be improved with task-specific quality hints.
- `MAX_SENTENCE_REFINES` defaults to 1. Increasing this allows multiple refinement passes per sentence but increases LLM cost.
- Generation prompts do not enforce a fixed sentence count — the LLM decides how many to produce.
