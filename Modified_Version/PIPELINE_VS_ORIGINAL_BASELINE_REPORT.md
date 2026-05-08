# Pipeline Comparison Report: Modified Version vs Original_baseLine

Generated: 2026-05-08
Scope: `Modified_Version/` pipeline vs `Original_baseLine/` pipeline only

---

## 1. Executive Summary

The `Modified_Version` pipeline is a task-aware evolution of `Original_baseLine`.

Core upgrades in `Modified_Version`:
- Multi-task operation (`topic`, `sentiment`, `ner`) in one run
- Task-aware prompts at generation, validation, and refinement stages
- Added `TaskValidatorAgent` with per-sentence validation outputs
- Scoring migrated to per-sentence granularity across quality dimensions
- Deterministic CS token-ratio calculation is computed first and fed into CS-ratio evaluation
- Per-sentence weighted scoring and failing-sentence targeting
- Refinement loop that re-runs quality agents after rewriting (fresh scores)
- Dual config format support (flat + nested `shared` format)
- UI-oriented runner (`run_french_ui.py`) for subprocess integration

Net effect: `Modified_Version` is more controllable and analytically richer than `Original_baseLine`, but also more complex and computationally heavier.

---

## 2. Side-by-Side Comparison

| Dimension | Original_baseLine | Modified_Version |
|---|---|---|
| Primary target | Single implicit pipeline (topic-like generation) | Task-aware pipeline for `topic`, `sentiment`, `ner` |
| Model setting in codebase | GPT-4o | GPT-4o-mini |
| Graph architecture | DataGen -> 4 quality agents -> Summarize -> optional Refiner | DataGen -> TaskValidator -> 4 quality agents -> Summarize -> optional Refiner -> re-evaluate quality agents |
| Task validation | Not present | `TaskValidatorAgent` present |
| Score granularity | Primarily scenario-level scoring | Per-sentence scoring representation across quality signals |
| CS-ratio counting method | Ratio count and score come from LLM evaluation | Deterministic ratio is computed first, then used in CS-ratio agent evaluation |
| CS-ratio evaluation | One `cs_ratio_result` for whole batch | `cs_ratio_results_per_instances` per sentence |
| Refinement strategy | Rewrites then goes directly to summarize | Rewrites only failing sentences, then re-runs quality agents |
| Criteria check | Uses single `state["score"]` | Uses per-sentence records/threshold logic |
| Prompt coverage | ~6 core prompts | 14+ prompts with task-specific variants |
| Configuration format | Flat `pre_execute` structure | Flat + nested `shared` structure (`config2.yaml`) |
| Output richness | Scenario-level fields | Scenario-level + per-instance validation and CS-ratio details |
| UI integration | CLI-centric (`run_french.py`) | UI-aware (`run_french_ui.py`, smoke test hooks) |

---

## 3. Architecture Flow Delta

### 3.1 Original_baseLine Flow

```text
START
  -> DataGenerationAgent
  -> (parallel) FluencyAgent, NaturalnessAgent, CSRatioAgent, SocialCulturalAgent
  -> SummarizeResult
  -> meet_criteria(score, refine_count)
      -> RefinerAgent (if needed)
      -> SummarizeResult
      -> AcceptanceAgent
  -> END
```

Characteristics:
- No explicit task branching by task type
- No task-label validator node
- Refinement does not trigger full re-evaluation of all quality signals

### 3.2 Modified_Version Flow

```text
START
  -> DataGenerationAgent (task-aware prompt)
  -> TaskValidatorAgent (task-aware validation prompt)
  -> (parallel) FluencyAgent, NaturalnessAgent, CSRatioAgent, SocialCulturalAgent
  -> SummarizeResult (per-sentence weighted scores)
  -> meet_criteria(failing_sentence_indices, refine_count)
      -> RefinerAgent (task-aware, failing sentences only)
      -> re-run Fluency/Naturalness/CSRatio/SocialCultural agents
      -> SummarizeResult
      -> AcceptanceAgent
  -> END
```

Characteristics:
- Explicit task-aware branching (`topic`/`sentiment`/`ner`)
- Per-sentence validation and per-sentence scoring records used in refinement decisions
- CS-ratio flow combines deterministic counting with agent-level scoring logic
- Post-refinement re-evaluation provides fresh quality scores

---

## 4. State Model Differences

### Original_baseLine

Key traits:
- Single scenario-level state emphasis
- One global `score`
- One `cs_ratio_result` for all generated sentences

Representative fields:
- `data_generation_result: list[str]`
- `fluency_result`
- `naturalness_result`
- `cs_ratio_result`
- `social_cultural_result`
- `score: float`
- `refine_count`

### Modified_Version

Key additions:
- Task identity fields and constraints
- Per-sentence validation outputs
- Per-sentence scoring records and selection for refinement
- Per-sentence CS-ratio outputs (with deterministic ratio input)

Representative added fields:
- `task`, `label`, `task_constraints`
- `task_validation_result` (with `per_instance_results`)
- `cs_ratio_results_per_instances`
- `sentence_records` (per-sentence score view used by `meet_criteria`)
- `failing_sentence_indices`

---

## 5. Prompting Strategy Differences

### Original_baseLine

Prompt set is compact and generic:
- Generation
- Fluency
- Naturalness
- CS ratio
- Social-cultural
- Refiner

### Modified_Version

Prompt set is expanded and task-sensitive:
- Task-specific data generation prompts
- Task-specific validator prompts
- Shared quality prompts (fluency/naturalness/CS/social-cultural)
- Task-specific refiner prompts

Practical impact:
- Better control over task conformance
- More explicit behavioral constraints
- Higher prompt maintenance overhead

---

## 6. Configuration Differences

### Original_baseLine

- Flat config schema under `pre_execute`
- No dedicated nested task blocks

### Modified_Version

- Supports both:
  - Flat (`config.yaml`)
  - Nested shared + task-specific (`config2.yaml`)
- Typical `config2.yaml` includes:
  - `task: [sentiment, ner, topic]`
  - `shared` settings for common constraints
  - Task blocks (`sentiment`, `ner`, `topic`) for specialized knobs

Practical impact:
- Higher flexibility and clearer task-level control
- More schema complexity to manage

---

## 7. Output and Analysis Differences

### Original_baseLine Output

- Primarily scenario-level summaries
- Limited per-sentence diagnostic granularity

### Modified_Version Output

- Maintains scenario-level outputs
- Adds per-sentence scoring records plus per-instance validation and CS-ratio diagnostics
- Better suited for detailed error analysis and UI exploration

---

## 8. Quality and Risk Trade-offs

Advantages of Modified_Version:
- Better alignment with multi-task research needs
- More robust observability and diagnostics
- Cleaner refinement targeting (failing items only)
- Reduced model cost relative to GPT-4o by using GPT-4o-mini

Risks/Costs introduced:
- More moving parts in graph and state
- More prompt variants to maintain
- Potentially higher total calls due to validator + re-evaluation loop
- Additional complexity in config compatibility and result interpretation

---

## 9. Bottom-Line Assessment

If your goal is a simple baseline generation pipeline, `Original_baseLine` is leaner.

If your goal is controlled, task-aware, analyzable generation with sentence-level diagnostics, `Modified_Version` is the stronger architecture and is the correct direction for research and UI-driven operation.

---

## 10. Key Files to Inspect

Original baseline:
- `Original_baseLine/core/agents.py`
- `Original_baseLine/core/node_engine.py`
- `Original_baseLine/core/node_models.py`
- `Original_baseLine/core/prompt.py`
- `Original_baseLine/core/utils.py`
- `Original_baseLine/core/run_french.py`

Modified pipeline:
- `Modified_Version/core/agents.py`
- `Modified_Version/core/node_engine.py`
- `Modified_Version/core/node_models.py`
- `Modified_Version/core/prompt.py`
- `Modified_Version/core/utils.py`
- `Modified_Version/core/run_french_ui.py`
- `Modified_Version/config/config2.yaml`
