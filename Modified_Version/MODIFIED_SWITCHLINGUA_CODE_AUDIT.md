# Modified SwitchLingua — Comprehensive Code Audit

**Scope:** every Python file under `Modified_Version/` (code + tests), verified by direct code reads and by
running the offline tests. **Purpose:** one authoritative, code-grounded reference for what the *modified*
(System B) pipeline actually does — so the thesis/diagrams describe the real behaviour. Date: 2026-07-05.

> **Headline finding:** the pipeline is correct and internally consistent. The **acceptance gate is
> quality-only**; task validation is a *separate* signal used for refinement routing + a rollback guardrail +
> a stored per-sentence label — it does **not** gate acceptance. Several docs/diagrams overstated the task's
> role at acceptance (a wording issue, not a code bug).

---

## 1. File inventory

### Active core (`Modified_Version/core/`)
| File | Lines | Role |
|---|--:|---|
| `node_engine.py` | 1185 | **All agent logic**: generation, task validation (incl. English-only NER policy), the 4 quality scorers, `SummarizeResult`, `RunRefinerAgent` (+ guardrail), `AcceptanceAgent`, helpers |
| `prompt.py` | 825 | 23 `*_PROMPT` constants (12 active + ~11 commented iteration variants) |
| `utils.py` | 465 | `load_config`, `generate_scenarios`, `compute_true_cs_stats`, `weighting_scheme`, `compute_sentence_weighted_scores`, `build_sentence_records` (status ladder), `validate_sentence_records_consistency` |
| `node_models.py` | 211 | `TypedDict` schemas; `AgentRunningState = BaseState` (line 211) |
| `run_french.py` | 206 | **The graph**: `CodeSwitchingAgent`, node/edge wiring, `meet_criteria` (accept gate), thresholds, `main()`/`arun()` |
| `run_french_ui.py` | 225 | UI variant — `meet_criteria` **identical** to run_french.py |
| `agents.py` | 163 | **Alternate/legacy** graph — different `meet_criteria` (uses `failing_sentence_indices`; Refiner→Fluency loop). **Not used for generation** (only `smoke_test_real_api.py` + `multi-agent-bert/legacy/`) |
| `mcp_tools.py` | 40 | Tool registry (MCP). `MCPAgent` exists but is **not wired** into the active graph |
| `smoke_test_real_api.py` | 125 | Real-API smoke (imports `agents.py`) |
| `from collections import Counter.py` | 48 | **Stray scratch file** — should be deleted |

### Legacy (excluded from active behaviour)
`core/old_files/{node_models_v2.py, utils_v2.py}`.

### Tests (`core/test files/` + `core/`)
| File | Lines | Covers | Status |
|---|--:|---|---|
| `test_refiner_guardrail.py` | 388 | 4 guardrail cases (accept / task-rollback / task-fix / quality-regression rollback) + loop-fix | ✅ 4/4 PASS |
| `test_ner_guidance.py` | 85 | config-driven NER entity guidance builder | ✅ 5/5 PASS |
| `test_per_instance_scoring.py` | 64 | per-instance weighting vs aggregate fallback | ✅ 2/2 PASS |
| `test_pipeline_full_mocked.py` | 210 | **full graph** run (topic/sentiment/NER), mocked LLM | ✅ 3/3 PASS |
| `test_task_generation_mock.py` | 70 | task→generation-prompt routing (no API) | ✅ PASS |
| `test_task_generation_real.py` | 389 | task generation w/ real API | ⚠ needs API (not run) |
| `test_pipeline_full_real.py` | 281 | full graph w/ real API | ⚠ needs API (not run) |
| `test_pipeline_output_reviewer.py` | 401 | CLI review utility (argparse) — **not a unit test** | n/a |
| `test_simple.py` | 122 | manual checker of a prior run's `output/Arabic.jsonl` — **not a unit test** | needs prior output |

> Note: `test_pipeline_full_mocked.py` and `test_task_generation_mock.py` don't self-add `sys.path`; run them
> with `PYTHONPATH=Modified_Version/core`. `test_refiner_guardrail/ner_guidance/per_instance_scoring` self-add it.

---

## 2. The active pipeline (LangGraph, `run_french.py`)

**Entry:** `CodeSwitchingAgent(scenario).run()` (used by every generation runner — verified: `manage_sentiment_data.py`, `run_gen_sensitivity_pilot.py`, `run_full_pipeline_generation.py`, `step1_generate.py`, `run_ablation/model_control/refinement_strategy/llm_ratio_repeats`, `_ner_realpipeline_smoke`).

**Nodes** (each = a `node_engine` function): DataGenerationAgent · TaskValidatorAgent · FluencyAgent · NaturalnessAgent · CSRatioAgent · SocialCulturalAgent · SummarizeResult · RefinerAgent · AcceptanceAgent. (`MCPAgent` defined but commented out of the graph.)

**Edges** (run_french.py:106-125):
```
START → DataGenerationAgent → TaskValidatorAgent
        → {FluencyAgent, NaturalnessAgent, CSRatioAgent, SocialCulturalAgent}   (parallel fan-out)
        → SummarizeResult
        → meet_criteria (conditional):  RefinerAgent  |  AcceptanceAgent
   RefinerAgent → TaskValidatorAgent   (loop: task is RE-validated after each refine)
   AcceptanceAgent → END
```

---

## 3. The two decisions (the core clarification)

**(a) Acceptance gate — `meet_criteria` (run_french.py:45-63): QUALITY-ONLY.**
```python
# per sentence: eligible for refinement iff
float(score) < SENTENCE_SCORE_THRESHOLD  and  refine_count < MAX_SENTENCE_REFINES
# any eligible → RefinerAgent ; else → AcceptanceAgent
```
`task_passed` is **not** in this condition. A high-quality, task-wrong sentence is **accepted without refinement**.

**(b) Refiner guardrail — "Accept/Reject Guardrail" (node_engine.py:1124-1155): keep-rewrite-vs-original.**
Runs **only inside `RunRefinerAgent`** on a below-threshold sentence. It decides whether to keep the *rewrite*
or revert to the *original* — **not** whether the sentence enters the corpus. Asymmetric:
- `quality_fail` rewrite → kept only if **task still passes AND quality does not regress**; else rollback.
- `task_fail` rewrite → kept only if **task now passes** (quality regression **tolerated**); else rollback.

Neither branch requires the rewrite to exceed the threshold — only "not worse". `refine_count` is incremented
on **every** attempt (node_engine.py:1165), including rollbacks (fixes the earlier NER infinite-loop).

---

## 4. Scoring, thresholds, status (verified)

- **Quality score** (`utils.py:214` & `:245`): `weighted_score = 0.30·fluency + 0.25·naturalness + 0.20·cs_ratio + 0.25·socio_cultural`. **Task is not a term** (by design — task-correctness ≠ quality).
- **`failing_sentence_indices`** (`node_engine.py:919-921`): indices with `score < 8.0` (quality only).
- **Per-sentence `status`** (`utils.py:319-326`) — the "gradual accept" ladder:
  | status | condition |
  |---|---|
  | `pass` | score ≥ 8, refine_count = 0 |
  | `refined_pass` | score ≥ 8, refine_count > 0 |
  | `budget_exhausted` | score < 8, refine_count ≥ max (**accepted below threshold**) |
  | `fail` | score < 8, budget left (transient loop state) |
  With `MAX_SENTENCE_REFINES=1`, a below-8 sentence gets one refine attempt then is accepted as `refined_pass` or `budget_exhausted`.
- **Constants:** `SENTENCE_SCORE_THRESHOLD=8.0` (run_french.py:41); `MAX_SENTENCE_REFINES=1` (env); `MAX_REFINER_ITERATIONS=1` (used only in the non-sentence-records fallback / agents.py path); `MODEL="gpt-4o-mini"`; `OUTPUT_DIR=Modified_Version/output`; `ENABLE_TASK_VALIDATOR` (env, default on).

---

## 5. Refiner classification (`RunRefinerAgent`, node_engine.py:1006-1172)
For each sentence routed in (i.e. already `score < 8`): `task_fail` if `not task_passed`, else `quality_fail`.
Prompt routing: `task_fail` → `REFINER_TASK_{TOPIC|SENTIMENT|NER}_PROMPT`; `quality_fail` → generic `REFINER_PROMPT`.
`_rescore_single_sentence` re-runs the 4 quality agents on the candidate for the regression check.

---

## 6. Acceptance / output (`AcceptanceAgent`, node_engine.py:976-987)
Strips transient keys (`news_article/news_hash/news_dict`) and **appends the whole `state`** (the entire
batch, with per-sentence `sentence_records`) to `OUTPUT_DIR/{first_language}.jsonl`. **No per-sentence
filtering** — `budget_exhausted` (below-threshold) and task-failing sentences are all written, each carrying
its `weighted_score`, `status`, `task_passed`, `refine_count`.

---

## 7. Scenario generation & CS counter (`utils.py`)
- `generate_scenarios` (`:19`): full **Cartesian product** of base dims × task-specific dims (sentiment: labels×intensity×ambiguity). `on_execute.round` and `shared.style` are **not read** (inert). No `target_total`.
- `compute_true_cs_stats` (`:153`): Arabic-script vs Latin token counts → `cs_ar_ratio`, `cs_en_ratio`, `is_code_switched` (ar>0 AND en>0). **Deterministic, 0 variance.** This feeds CSRatioAgent's deterministic stats and every downstream CS-validity filter.

## 8. NER task validation (English-only policy)
`_deterministic_ner_english_policy` + `_extract_english_ner_counts` + `build_ner_entity_guidance`
(`DEFAULT_ENTITY_GUIDANCE`, `{ner_entity_guidance}` placeholder). Required entities must be English/Latin-script;
Arabic-script names are context, not counted. LLM validation for topic/sentiment; NER uses the deterministic
English-script check (hybrid, LLM path preserved in comments).

## 9. Prompts (`prompt.py`)
Active: `DATA_GENERATION_{,TOPIC,SENTIMENT,NER}`, `TASK_VALIDATION_{TOPIC,SENTIMENT,NER}`, `FLUENCY`,
`NATURALNESS`, `CS_RATIO`, `SOCIAL_CULTURAL`, `REFINER`, `REFINER_TASK_{TOPIC,SENTIMENT,NER}`. ~11
commented-out variants (mostly CS_RATIO iterations) remain as history — cosmetic noise.

## 10. Schemas (`node_models.py`)
`TypedDict`s for each agent response (`GenerationResponse`, `FluencyResponse`, `NaturalnessResponse`,
`SocialCulturalResponse`, `CSRatioResponse`, `TaskValidationResult`), plus `SentenceRecord`, `BaseState`
(= `AgentRunningState`), and task states `TopicState`/`SentimentState`/`NERState`.

---

## 11. Discrepancies & accuracy notes (doc vs. code)

| # | Issue | Reality |
|---|---|---|
| D1 | Docs/methodology say acceptance requires **task + quality** | Acceptance is **quality-only**; task feeds refiner routing + guardrail + stored label |
| D2 | `ARCHITECTURE.md` (mine) gate reads "score ≥ bar **AND task_passed**" | Should be "score < threshold → refine; else accept" |
| D3 | `PIPELINE_ARCHITECTURE.md` documents **`agents.py`** (different `meet_criteria`, Refiner→Fluency loop) | The runner is **`run_french.py`** (Refiner→**TaskValidator** loop, per-record `weighted_score` gate) |
| D4 | `PIPELINE_ARCHITECTURE.md` lists record fields `needs_refine`, `eligible_for_refine` | Those **don't exist**; the real field is **`status`** (pass/refined_pass/budget_exhausted/fail) |
| D5 | "Accept/Reject Guardrail" naming | Candidate-level (keep rewrite vs original), **not** corpus acceptance |
| D6 | Three `meet_criteria` variants exist | run_french + run_french_ui (identical, quality-only) vs agents.py (quality-only via `failing_sentence_indices`). **All quality-only; none task-gated.** |
| D7 | Test 2 "TaskValidator reduces task-wrong accepts 25→9" | A **simulated post-hoc policy** (`run_task_validator_necessity.py`), **not** the deployed pipeline |
| D8 | Stray `from collections import Counter.py`; commented prompt variants; unwired `MCPAgent` | Cosmetic — safe to clean |

**Git note:** nothing lost (clean tree, 0 dangling commits, no deleted-tracked, no stash). But `*.yaml`,
`*.jsonl`, `*.csv`, and checkpoints are **gitignored** → generation configs and datasets are **local-only**
(reproducibility exposure; back them up outside git and consider un-ignoring the small configs).

---

## 12. Recommendations
1. **Docs to match code (no code change):** fix D1–D4 in `ARCHITECTURE.md`, `PIPELINE_ARCHITECTURE.md`, the PNG diagram, and the 4 methodology sentences → acceptance = quality-only; task = refiner routing + guardrail + label.
2. Frame Test 2 (D7) explicitly as a **simulated acceptance policy**, not pipeline behaviour.
3. Housekeeping (optional): delete the stray scratch file, prune commented prompt variants, note `agents.py` as legacy.
4. **Reproducibility:** un-ignore & commit the generation `*.yaml` configs; back up datasets/checkpoints off-git.

**Bottom line:** the modified pipeline is correct and verified (all offline tests pass; every dataset was
generated by `run_french.py`). The only fixes needed are in the **documentation**, to describe quality-gated
acceptance with task validation as a separate refinement/guardrail/label signal.
