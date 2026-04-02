# SwitchLingua Pipeline: Detailed Flow, State Shape, and Scoring Levels

## 1. High-Level Overview

The current runtime pipeline is orchestrated in `drive_code/core/run_french.py` using a LangGraph state machine, with node logic implemented in `drive_code/core/node_engine.py` and state schema in `drive_code/core/node_models.py`.

For each generated scenario:

1. Build initial state from config (`config2.yaml`) via `utils2.generate_scenarios(...)`.
2. Run `DataGenerationAgent`.
3. Run `TaskValidatorAgent` (or skip via passthrough if disabled).
4. Run quality agents in parallel:
   - `FluencyAgent`
   - `NaturalnessAgent`
   - `CSRatioAgent`
   - `SocialCulturalAgent`
5. Merge results in `SummarizeResult`.
6. Branch:
   - If quality is below threshold and refinements remain → `RefinerAgent` then back to `SummarizeResult`.
   - Else → `AcceptanceAgent` and write output.

---

## 2. Graph Flow (Actual Node Order)

Defined in `drive_code/core/run_french.py`.

```text
START
  → DataGenerationAgent
  → TaskValidatorAgent
  → FluencyAgent       ┐
  → NaturalnessAgent   │ (parallel fan-out)
  → CSRatioAgent       │
  → SocialCulturalAgent┘
  → SummarizeResult
  → (conditional)
      if score < 8 and refine_count < 1 → RefinerAgent → SummarizeResult
      else → AcceptanceAgent → END
```

**Notes:**
- `ENABLE_TASK_VALIDATOR` env var controls whether `TaskValidatorAgent` is active or replaced by a passthrough no-op.
- `MCPAgent` exists in the graph definition but is **not on the active path** (edges are commented out).
- `MAX_REFINER_ITERATIONS = 1` — at most one refinement pass per scenario.

---

## 3. Initial Scenario Construction (Input State)

Scenarios are built by `drive_code/core/utils2.py::generate_scenarios(pre_execute)`.

### 3.1 Shared fields — added for every scenario regardless of task

| Field | Source |
|---|---|
| `topic` | `pre_execute.shared.topic` |
| `tense` | `pre_execute.shared.tense` |
| `perspective` | `pre_execute.shared.perspective` |
| `gender` | `pre_execute.shared.character_setting.gender` |
| `age` | `pre_execute.shared.character_setting.age` |
| `education_level` | `pre_execute.shared.character_setting.education_level` |
| `cs_ratio` | `pre_execute.cs_ratio` |
| `use_tools` | `pre_execute.shared.use_tools` |
| `conversation_type` | `pre_execute.shared.conversation_type` |
| `first_language` | `pre_execute.shared.character_setting.nationality.first_language` |
| `second_language` | `pre_execute.shared.character_setting.nationality.second_language` |
| `cs_function` | `pre_execute.shared.cs_function` |
| `cs_type` | `pre_execute.shared.cs_type` |
| `data_generation_result` | initialized `[]` |
| `response` | initialized `""` |
| `refine_count` | initialized `0` |

### 3.2 Task-specific fields added per task

#### Topic task
| Field | Value |
|---|---|
| `task` | `"topic"` |
| `label` | one value from `pre_execute.topic.topics` |
| `topic` | overwritten to equal `label` (for contextual consistency) |

#### Sentiment task
| Field | Value |
|---|---|
| `task` | `"sentiment"` |
| `label` | `"positive"` / `"negative"` / `"neutral"` |
| `task_constraints` | `{ intensity, ambiguity }` |

#### NER task
| Field | Value |
|---|---|
| `task` | `"ner"` |
| `annotations` | `[]` (filled by generation) |
| `task_constraints` | `{ entity_types, min_entities, max_entities, must_include_types, allow_code_switched_entities }` |

---

## 4. State Schema (Canonical Shape)

State lives in `drive_code/core/node_models.py` as `AgentRunningState = BaseState`.

### 4.1 Task discriminator and payload

```python
task: "topic" | "sentiment" | "ner"
label: str                        # topic / sentiment
task_constraints: Dict[str, Any]  # sentiment / ner
annotations: list[NERSpan]        # ner
```

### 4.2 Scenario context (scenario-level, constant through graph)

```python
topic, tense, perspective, cs_ratio, gender, age, education_level
first_language, second_language, conversation_type, cs_function, cs_type
use_tools
```

### 4.3 Generation and quality outputs (written by nodes)

```python
data_generation_result: list[str]          # filled by DataGenerationAgent
task_validation_result: TaskValidationResult   # filled by TaskValidatorAgent
fluency_result: FluencyResponse            # filled by FluencyAgent
naturalness_result: NaturalnessResponse    # filled by NaturalnessAgent
cs_ratio_results_per_instances: List[CSRatioResponse]  # filled by CSRatioAgent (per sentence)
social_cultural_result: SocialCulturalResponse  # filled by SocialCulturalAgent
summary: str                               # filled by SummarizeResult
score: float                               # filled by SummarizeResult
refine_count: int                          # accumulated (add reducer)
```

### 4.4 Transient working fields (removed before JSONL write)

```python
news_article: str
news_hash: Set[str]
news_dict: Dict[str, Any]
mcp_result: str
```

---

## 5. Node-by-Node Behavior and State Mutations

### 5.1 `RunDataGenerationAgent`

- Dispatches by `task`:
  - `"topic"` → `RunTopicDataGenerationAgent` (uses `DATA_GENERATION_TOPIC_PROMPT`)
  - `"sentiment"` → `RunSentimentDataGenerationAgent` (uses `DATA_GENERATION_SENTIMENT_PROMPT`)
  - `"ner"` → `RunNERDataGenerationAgent` (uses `DATA_GENERATION_NER_PROMPT`)
- Retries up to 4 times if response contains no instances.
- **Writes:** `data_generation_result = response["instances"]` (list of generated text strings)
- If all retries fail: returns `data_generation_result = []`

---

### 5.2 `RunTaskValidatorAgent`

Dispatcher by `task`:

| Task | Validator |
|---|---|
| `"topic"` | `RunTopicTaskValidatorAgent` |
| `"sentiment"` | `RunSentimentTaskValidatorAgent` |
| `"ner"` | `RunNERTaskValidatorAgent` |

#### Topic / Sentiment validators

Both use `_validate_per_instance_with_retry(...)`:
- Validates **each generated sentence separately** (one LLM call per sentence).
- Aggregates into one `task_validation_result`:

```python
task_validation_result = {
    "passed": bool,                       # True only if ALL instances passed
    "confidence": float,                  # average across instances
    "predicted_label": str | "mixed",     # "mixed" if per-instance labels disagree
    "notes": str,
    "errors": list[str],                  # prefixed with "instance_N: ..."
    "per_instance_results": list[dict],   # full result per sentence
}
```

#### NER validator

`RunNERTaskValidatorAgent` currently uses **LLM-only** path (deterministic policy helper `_deterministic_ner_english_policy` is present but its merge is commented out).

```python
task_validation_result = {
    "passed": bool,
    "confidence": float,
    "notes": str,
    "llm_notes": str,
    "deterministic_notes": "",   # empty in current LLM-only mode
    "predicted_label": str | None,
    "errors": list[str],
}
```

**If `ENABLE_TASK_VALIDATOR=0`:** passthrough returns `{}` and `task_validation_result` stays unset.

---

### 5.3 Quality Agents (parallel fan-out)

All four run concurrently after `TaskValidatorAgent`.

#### `RunFluencyAgent`
- One LLM call for the whole scenario.
- **Writes:** `fluency_result = { fluency_score, errors, summary }`

#### `RunNaturalnessAgent`
- One LLM call for the whole scenario.
- **Writes:** `naturalness_result = { naturalness_score, observations, summary }`

#### `RunCSRatioAgent`
- Deterministically computes Arabic/English token ratios per sentence using `compute_true_cs_stats`.
- Sends all sentences + stats as a batch to LLM.
- Parses response as a JSON array; fallback entries are synthesized if the LLM response is short.
- **Writes:** `cs_ratio_results_per_instances = [ {ratio_score, computed_ratio, notes}, ... ]` — one entry per generated sentence.

#### `RunSocialCulturalAgent`
- One LLM call for the whole scenario.
- **Writes:** `social_cultural_result = { socio_cultural_score, issues, summary }`

---

### 5.4 `SummarizeResult`

- Builds human-readable `summary` string.
- Calls `weighting_scheme(state)` (from `drive_code/core/utils.py`) to compute the final scalar.
- **Writes:** `score`, `summary`

---

### 5.5 `RefinerAgent`

- Runs refiner prompt with full current state.
- **Writes:** `refiner_result`, `refine_count = 3` (forces exit from refine loop on next check)
- Graph returns to `SummarizeResult`, which recomputes `score`.

---

### 5.6 `AcceptanceAgent`

- Pops transient fields: `news_article`, `news_hash`, `news_dict`.
- Appends final state as one JSONL row to `drive_code/output/{first_language}.jsonl`.
- Creates output directory if it does not exist.

---

## 6. Scoring: Instance-Level vs Scenario-Level

### 6.1 Instance-level scores

| Field | Granularity |
|---|---|
| `cs_ratio_results_per_instances` | **Per sentence** |
| `task_validation_result.per_instance_results` | **Per sentence** (topic/sentiment only) |

### 6.2 Scenario-level scores (single value per scenario run)

| Field | Granularity |
|---|---|
| `fluency_result.fluency_score` | Whole scenario |
| `naturalness_result.naturalness_score` | Whole scenario |
| `social_cultural_result.socio_cultural_score` | Whole scenario |
| `score` | Whole scenario (final weighted) |

### 6.3 Final score formula (`drive_code/core/utils.py::weighting_scheme`)

$$
\text{score} = 0.30 \times \text{fluency} + 0.25 \times \text{naturalness} + 0.20 \times \overline{\text{cs\_ratio}} + 0.25 \times \text{socio\_cultural}
$$

Where $\overline{\text{cs\_ratio}}$ is the average of all `ratio_score` values in `cs_ratio_results_per_instances`.

> **Note:** `task_validation_result` is stored in state and saved to output but is **not included in the weighted score formula**. It is tracked as a separate quality signal.

---

## 7. Refinement Decision Logic

```python
def meet_criteria(state):
    if state["score"] < 8 and state["refine_count"] < MAX_REFINER_ITERATIONS:
        return "RefinerAgent"
    else:
        return "AcceptanceAgent"
```

- `MAX_REFINER_ITERATIONS = 1`
- `RefinerAgent` sets `refine_count = 3`, which permanently exceeds the limit, so only **one refinement pass** can occur regardless of score.

---

## 8. Minimal State Evolution Example (Sentiment Scenario)

```
Initial state (from utils2):
  task="sentiment", label="positive",
  task_constraints={intensity:"low", ambiguity:"low"},
  topic="shopping", cs_ratio="70%", first_language="Arabic", ...

After DataGenerationAgent:
  + data_generation_result = ["...", "...", "..."]

After TaskValidatorAgent:
  + task_validation_result = {
      passed: True,
      confidence: 0.92,
      predicted_label: "positive",
      per_instance_results: [{...}, {...}, {...}]
    }

After quality agents (parallel):
  + fluency_result = { fluency_score: 8.5, ... }
  + naturalness_result = { naturalness_score: 7.8, ... }
  + cs_ratio_results_per_instances = [
      { ratio_score: 8, computed_ratio: "72%:28%", notes: "..." },
      { ratio_score: 9, computed_ratio: "68%:32%", notes: "..." },
      ...
    ]
  + social_cultural_result = { socio_cultural_score: 8.0, ... }

After SummarizeResult:
  + summary = "..."
  + score = 8.23   → passes threshold → AcceptanceAgent

After AcceptanceAgent:
  - news_article, news_hash, news_dict removed
  → row appended to drive_code/output/Arabic.jsonl
```

---

## 9. Key Files Reference

| File | Purpose |
|---|---|
| `drive_code/core/run_french.py` | Graph construction, orchestration entrypoint |
| `drive_code/core/node_engine.py` | All node implementations |
| `drive_code/core/node_models.py` | State schema (`AgentRunningState = BaseState`) |
| `drive_code/core/utils2.py` | `generate_scenarios` used by `run_french.py` |
| `drive_code/core/utils.py` | `weighting_scheme`, `compute_true_cs_stats` used by `node_engine.py` |
| `drive_code/core/prompt.py` | All LLM prompt templates |
| `drive_code/config/config2.yaml` | Active run config |
| `drive_code/output/{language}.jsonl` | Final output rows |
