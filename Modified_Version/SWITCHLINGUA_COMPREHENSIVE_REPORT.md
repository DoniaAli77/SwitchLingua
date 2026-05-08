# SwitchLingua — Comprehensive System Report

**Generated:** 2026-05-06
**Scope:** Original_baseLine · Modified_Version · Control Center UI

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Workspace Layout](#2-workspace-layout)
3. [Architecture Overview](#3-architecture-overview)
4. [Original Baseline Pipeline](#4-original-baseline-pipeline)
   - 4.1 Purpose
   - 4.2 Key Files
   - 4.3 State Model
   - 4.4 Graph Flow (LangGraph)
   - 4.5 Agent Descriptions
   - 4.6 Configuration
   - 4.7 Scenario Generation
   - 4.8 Scoring & Weighting
   - 4.9 Output Format
5. [Modified Version Pipeline](#5-modified-version-pipeline)
   - 5.1 Purpose & Goals
   - 5.2 Key Files
   - 5.3 State Model — Additions
   - 5.4 Graph Flow — Changes
   - 5.5 New & Changed Agents
   - 5.6 Configuration System (Dual Format)
   - 5.7 Scenario Generation — Task-Aware
   - 5.8 Prompts
   - 5.9 Scoring & Weighting — Changes
   - 5.10 Output Format (JSONL)
6. [Baseline vs Modified — Side-by-Side Comparison](#6-baseline-vs-modified--side-by-side-comparison)
7. [Control Center UI](#7-control-center-ui)
   - 7.1 Purpose & Stack
   - 7.2 Key Files
   - 7.3 Page Layout & Tabs
   - 7.4 Config Editor
   - 7.5 Run Controls
   - 7.6 Output Viewer
   - 7.7 Theme & Sidebar
8. [Data Flow End-to-End](#8-data-flow-end-to-end)
9. [Configuration Reference](#9-configuration-reference)
10. [Current Metrics & Status](#10-current-metrics--status)
11. [Known Limitations & Risks](#11-known-limitations--risks)

---

## 1. Project Overview

**SwitchLingua** (accepted at NeurIPS 2025) is an AI-powered pipeline that generates, validates,
and evaluates high-quality Arabic-English code-switched text for NLP research.
The system covers three NLP tasks: **Sentiment Analysis**, **NER**, and **Topic Classification**.

| Component | Folder | Role |
|---|---|---|
| **Original Baseline** | `Original_baseLine/` | Reference implementation — single-task, flat config, GPT-4o |
| **Modified Version** | `Modified_Version/` | Extended version — multi-task, nested config, task validation, per-sentence scoring, GPT-4o-mini |
| **Control Center UI** | `switch-lingua-ui/` | Streamlit web app — config editor, pipeline launcher, output explorer |

---

## 2. Workspace Layout

```
SwitchLingua/
│
├── Original_baseLine/              ← Reference pipeline
│   ├── core/
│   │   ├── agents.py               ← CodeSwitchingAgent + LangGraph builder
│   │   ├── node_engine.py          ← Node functions (RunXxxAgent)
│   │   ├── node_models.py          ← AgentRunningState TypedDict
│   │   ├── prompt.py               ← LLM prompt templates (6 prompts)
│   │   ├── utils.py                ← Config loading, scenario generation, weighting
│   │   ├── mcp_tools.py            ← External tool integration
│   │   └── run_french.py           ← Main entry point (batches of 40, 7200s timeout)
│   ├── output/                     ← JSONL output files
│   ├── logs/
│   ├── Sample/
│   └── requirements.txt
│
├── Modified_Version/               ← Extended research pipeline
│   ├── config/
│   │   ├── config.yaml             ← Flat config format (legacy)
│   │   └── config2.yaml            ← Nested shared config (recommended)
│   ├── core/
│   │   ├── agents.py               ← Extended CodeSwitchingAgent
│   │   ├── node_engine.py          ← Task-aware node functions
│   │   ├── node_models.py          ← Extended state schema
│   │   ├── prompt.py               ← 14+ prompt templates (per task × agent)
│   │   ├── utils.py                ← Task-aware scenario gen; CS stats helpers
│   │   ├── mcp_tools.py            ← External tool integration
│   │   ├── run_french_ui.py        ← UI-facing entry point (subprocess target)
│   │   ├── run_french.py           ← Direct CLI entry point
│   │   └── smoke_test_real_api.py  ← Quick sanity runner (≤N scenarios)
│   ├── output/
│   │   ├── Arabic.jsonl            ← Latest full run (18 scenarios, 293 KB)
│   │   └── pipeline_full_real_*.json
│   ├── logs/
│   └── requirements.txt
│
├── switch-lingua-ui/               ← Control Center UI
│   ├── app.py                      ← Streamlit application (~1350 lines)
│   ├── .streamlit/config.toml      ← Dark theme settings
│   ├── saved_configs/              ← User-saved YAML snapshots
│   └── requirements.txt
│
├── mdFiles/                        ← Design & analysis documents
└── SWITCHLINGUA_COMPREHENSIVE_REPORT.md
```

---

## 3. Architecture Overview

Both pipelines share the same conceptual shape — a LangGraph `StateGraph` driven by
GPT-4o (Baseline) or GPT-4o-mini (Modified). The Modified Version adds a `TaskValidatorAgent`,
per-sentence scoring, and task-aware prompts that branch by task type at every agent.

### 3.1 High-Level System Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Control Center UI  (Streamlit)                    │
│                                                                      │
│  ┌─────────────────┐   ┌──────────────┐   ┌───────────────────────┐ │
│  │  Config Editor  │──▶│  Run Control │──▶│    Output Viewer      │ │
│  │  (config2.yaml) │   │  subprocess  │   │  parse_jsonl_to_rows  │ │
│  └─────────────────┘   └──────┬───────┘   └───────────────────────┘ │
└─────────────────────────────── │ ────────────────────────────────────┘
                                 │ SWITCHLINGUA_CONFIG_PATH env var
                                 ▼
                    run_french_ui.py  (Modified_Version/core/)
                         │
                         ├─ load_config(path)
                         │
                         └─ generate_scenarios(pre_execute)
                              │
                              │  config2.yaml  ─────────────────────────────┐
                              │  task: [sentiment, ner, topic]              │
                              │  shared: {topic, tense, cs_ratio, ...}      │
                              │  sentiment: {labels, intensity, ambiguity}  │
                              │  ner: {entity_types, min/max, ...}          │
                              │  topic: {topics}                            │
                              └────────────────────────────────────────────┘
                                   │
                                   │  itertools.product  →  18 scenarios
                                   │  each has: task + label + shared params
                                   ▼
                    ┌──────────────────────────────────────────────┐
                    │  for each scenario  (task, label, params)    │
                    │  CodeSwitchingAgent(scenario).ainvoke()      │
                    └─────────────────┬────────────────────────────┘
                                      │
                                      ▼
                         [LangGraph StateGraph — see §3.2]
                                      │
                                      ▼
                         Modified_Version/output/Arabic.jsonl
```

---

### 3.2 Task-Aware LangGraph Pipeline (Modified Version)

Each scenario carries `task ∈ {topic, sentiment, ner}` and `label` through every node.
Agents select their prompt based on the task at runtime.

```
                          START
                            │
                            ▼
              ┌─────────────────────────────────────┐
              │         DataGenerationAgent          │
              │                                     │
              │  task=topic    → DATA_GEN_TOPIC      │
              │  task=sentiment→ DATA_GEN_SENTIMENT  │
              │  task=ner      → DATA_GEN_NER        │
              │                                     │
              │  output: data_generation_result      │
              │          list[str]  (5 sentences)    │
              └─────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────────────┐
              │        TaskValidatorAgent            │
              │                                     │
              │  task=topic    → TV_TOPIC_PROMPT     │
              │  task=sentiment→ TV_SENTIMENT_PROMPT │
              │  task=ner      → TV_NER_PROMPT       │
              │                                     │
              │  output: task_validation_result      │
              │    .per_instance_results[i]          │
              │      passed / confidence /           │
              │      predicted_label / notes         │
              └─────────────────────────────────────┘
                            │
              ┌─────────────┴───────────────────────┐
              │   fan-out: 4 parallel quality agents │
              ▼             ▼            ▼           ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────────┐
   │ FluencyAgent │ │Naturalness   │ │CSRatio   │ │SocialCultural    │
   │              │ │Agent         │ │Agent     │ │Agent             │
   │ FLUENCY_     │ │NATURALNESS_  │ │CS_RATIO_ │ │SOCIAL_CULTURAL_  │
   │ PROMPT       │ │PROMPT        │ │PROMPT    │ │PROMPT            │
   │              │ │              │ │          │ │                  │
   │ fluency_     │ │naturalness_  │ │cs_ratio_ │ │social_cultural_  │
   │ result       │ │result        │ │results_  │ │result            │
   │ (score 0-10) │ │(score 0-10)  │ │per_inst  │ │(score 0-10)      │
   │ errors[]     │ │observations[]│ │[i].score │ │issues[]          │
   │ summary      │ │summary       │ │[i].ratio │ │summary           │
   └──────┬───────┘ └──────┬───────┘ └────┬─────┘ └────────┬─────────┘
          └─────────────────┴──────────────┴────────────────┘
                            │  fan-in
                            ▼
              ┌─────────────────────────────────────┐
              │           SummarizeResult            │
              │                                     │
              │  for each sentence i:               │
              │    weighted_score[i] =              │
              │      w_flu  × fluency_score         │
              │    + w_nat  × naturalness_score      │
              │    + w_cs   × cs_ratio_score[i]      │
              │    + w_soc  × soc_cult_score         │
              │    + w_tv   × tv_confidence[i]       │
              │                                     │
              │  failing_sentence_indices =          │
              │    [i for i if score[i] < 8.0]       │
              └─────────────────────────────────────┘
                            │
                            ▼
                    meet_criteria()
                  /                 \
    failing sentences?              all sentences ≥ 8.0
    refine_count < MAX?             OR refine_count == MAX
                │                               │
                ▼                               ▼
  ┌─────────────────────────────┐   ┌────────────────────────┐
  │       RefinerAgent          │   │    AcceptanceAgent      │
  │                             │   │                        │
  │  task=topic    → REF_TOPIC  │   │  write to Arabic.jsonl │
  │  task=sentiment→ REF_SENT   │   │  (full state dump)     │
  │  task=ner      → REF_NER    │   └────────────────────────┘
  │                             │              │
  │  rewrites ONLY failing      │             END
  │  sentence indices           │
  │  (data_generation_result    │
  │   updated in-place)         │
  └─────────────────────────────┘
                │
                │  re-run quality agents (fresh scores)
                ▼
   ┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────────┐
   │ FluencyAgent │ │Naturalness   │ │CSRatio   │ │SocialCultural    │
   │  (re-eval)   │ │Agent(re-eval)│ │(re-eval) │ │Agent  (re-eval)  │
   └──────┬───────┘ └──────┬───────┘ └────┬─────┘ └────────┬─────────┘
          └─────────────────┴──────────────┴────────────────┘
                            │
                            ▼
                     SummarizeResult
                     (loop back; refine_count++)
```

---

### 3.3 Task Branching at Each Agent

The table below shows which prompt template each agent uses depending on the active task:

```
Agent                │ task=topic               │ task=sentiment             │ task=ner
─────────────────────┼──────────────────────────┼────────────────────────────┼─────────────────────────
DataGenerationAgent  │ DATA_GENERATION_TOPIC     │ DATA_GENERATION_SENTIMENT  │ DATA_GENERATION_NER
                     │ (topic domain, style)     │ (intensity, ambiguity)     │ (entity types, counts)
TaskValidatorAgent   │ TASK_VALIDATION_TOPIC     │ TASK_VALIDATION_SENTIMENT  │ TASK_VALIDATION_NER
                     │ (classify topic label)    │ (label + intensity)        │ (entity presence/types)
FluencyAgent         │ FLUENCY_PROMPT (shared)   │ FLUENCY_PROMPT (shared)    │ FLUENCY_PROMPT (shared)
NaturalnessAgent     │ NATURALNESS_PROMPT        │ NATURALNESS_PROMPT         │ NATURALNESS_PROMPT
CSRatioAgent         │ CS_RATIO_PROMPT           │ CS_RATIO_PROMPT            │ CS_RATIO_PROMPT
SocialCulturalAgent  │ SOCIAL_CULTURAL_PROMPT    │ SOCIAL_CULTURAL_PROMPT     │ SOCIAL_CULTURAL_PROMPT
RefinerAgent         │ REFINER_TASK_TOPIC        │ REFINER_TASK_SENTIMENT     │ REFINER_TASK_NER
                     │ (preserve topic label)    │ (preserve sentiment)       │ (preserve entities)
```

---

### 3.4 Per-Sentence State Flow

For each scenario, 5 sentences are generated and tracked individually through
`sentence_records`:

```
sentence_records[i]:
  ├─ index            int          position in data_generation_result
  ├─ text             str          the generated sentence
  ├─ fluency_score    float        from FluencyAgent (scenario-level, shared)
  ├─ naturalness_score float       from NaturalnessAgent (scenario-level, shared)
  ├─ cs_ratio_score   float        from cs_ratio_results_per_instances[i]  ← per-sentence
  ├─ soc_cult_score   float        from SocialCulturalAgent (scenario-level, shared)
  ├─ tv_passed        bool         from task_validation_result.per_instance_results[i]
  ├─ tv_confidence    float        from task_validation_result.per_instance_results[i]
  ├─ weighted_score   float        computed by SummarizeResult
  ├─ refine_count     int          how many times this sentence was rewritten
  └─ status           str          "pass" | "fail" | "refined"
```

---

## 4. Original Baseline Pipeline

### 4.1 Purpose

The reference implementation (NeurIPS 2025). Generates code-switched sentences for a
**single implicit task** (no task validation), uses a flat configuration, scores with four
quality agents, and refines once if the overall score is below 8.

---

### 4.2 Key Files

| File | Role |
|---|---|
| `core/agents.py` | `CodeSwitchingAgent`; `_construct_graph_with_data_generation()` |
| `core/node_engine.py` | All node functions: RunDataGenerationAgent, RunFluencyAgent, RunNaturalnessAgent, RunCSRatioAgent, RunSocialCulturalAgent, RunRefinerAgent, SummarizeResult, AcceptanceAgent |
| `core/node_models.py` | `AgentRunningState` TypedDict; response models |
| `core/prompt.py` | 6 prompts: generation, fluency, naturalness, CS ratio, social-cultural, refiner |
| `core/utils.py` | `load_config()`, `generate_scenarios()`, `weighting_scheme()` |
| `core/mcp_tools.py` | `get_all_tools()` for optional news fetch |
| `core/run_french.py` | Async entry point; batches of 40 scenarios, 7200s timeout |

---

### 4.3 State Model

```python
class AgentRunningState(TypedDict):
    # Scenario inputs
    topic: str
    tense: str
    perspective: str
    cs_ratio: str
    gender: str
    age: str
    education_level: str
    first_language: str
    second_language: str
    conversation_type: str
    cs_function: str
    cs_type: str
    news_article: Optional[str]

    # Generation output
    response: str
    data_generation_result: list[str]    # N generated sentences
    news_generation_result: list[str]

    # Quality agent results (scenario-level — one score covers all sentences)
    fluency_result:          FluencyResponse
    naturalness_result:      NaturalnessResponse
    cs_ratio_result:         CSRatioResponse    # single result for whole batch
    social_cultural_result:  SocialCulturalResponse

    # Scoring
    summary: str
    score:   float                       # single overall score

    # Lifecycle
    refine_count: Annotated[int, add]
    news_hash: set
    news_dict: dict
```

**Key limitation:** `cs_ratio_result` is a **single** object for the batch, not per-sentence.
`score` is a single float covering all generated sentences together.

---

### 4.4 Graph Flow (LangGraph)

```
START
  │
  ▼
DataGenerationAgent
  │
  ├──────────────────────────────────┐
  ▼          ▼           ▼           ▼
FluencyAgent NaturalnessAgent CSRatioAgent SocialCulturalAgent
  └──────────┴───────────┴───────────┘
                    │
                    ▼
             SummarizeResult
             (score = weighting_scheme(flu, nat, cs, soc))
                    │
             meet_criteria()
             score < 8 AND refine_count < 1?
                  /      \
               Yes         No
                │           │
          RefinerAgent   AcceptanceAgent
                │              │
                ▼              ▼
          SummarizeResult    END
          (loop once — stale scores from before refinement reused)
```

**Critical:** `RefinerAgent` → `SummarizeResult` directly — quality agents do **NOT** re-run.
The scores evaluated before refinement are reused as-is.

---

### 4.5 Agent Descriptions

| Agent | Description |
|---|---|
| **DataGenerationAgent** | Calls GPT-4o with the base generation prompt. Returns `instances: list[str]` — N code-switched sentences. |
| **FluencyAgent** | Rates linguistic fluency 0–10. Returns `fluency_score`, `errors`, `summary`. Covers all sentences as one batch. |
| **NaturalnessAgent** | Rates code-switching naturalness 0–10. Returns `naturalness_score`, `observations`, `summary`. |
| **CSRatioAgent** | Checks if actual language ratio matches configured `cs_ratio`. Returns a **single** `ratio_score`, `computed_ratio`, `notes` for the whole batch. |
| **SocialCulturalAgent** | Checks cultural appropriateness 0–10. Returns `socio_cultural_score`, `issues`, `summary`. |
| **SummarizeResult** | Runs `weighting_scheme()` to produce `score`. |
| **RefinerAgent** | Rewrites all sentences using the base refiner prompt. |
| **AcceptanceAgent** | Finalises; writes to output JSONL. |

---

### 4.6 Configuration

Flat YAML — single structure, no task nesting:

```yaml
pre_execute:
  topics: [tech, finance]
  task: [topic]
  tense: [Present]
  perspective: [First Person]
  cs_ratio: ["70%"]
  cs_function: [Expressive]
  cs_type: [Intrasentential]
  conversation_type: [single_turn]
  use_tools: false
  character_setting:
    nationality:
      first_language: Arabic
      second_language: English
    age: ["18-25"]
    gender: [Male]
    education_level: [College]
```

`generate_scenarios(config)` reads all fields from the flat dict via `itertools.product`.
No task-specific sub-sections; no `shared` block.

---

### 4.7 Scenario Generation

`generate_scenarios(config)` produces:

```
topics × tenses × perspectives × genders × ages × education_levels
× cs_ratios × conversation_types × cs_functions × cs_types
```

All scenarios are structurally identical — no `task` or `label` fields. Each scenario
carries only a `topic` string.

---

### 4.8 Scoring & Weighting

`weighting_scheme(fluency, naturalness, cs_ratio, soc_cult)` in `utils.py` produces a
single weighted float.

`meet_criteria(state)`:
```python
if state["score"] < 8 and state["refine_count"] < MAX_REFINER_ITERATIONS:
    return "RefinerAgent"
else:
    return "AcceptanceAgent"
```

`MAX_REFINER_ITERATIONS = 1` — at most one refine loop.

---

### 4.9 Output Format

Each JSONL line is the raw `AgentRunningState` dump — flat fields + scenario-level quality
agent results. No per-sentence breakdown.

---

## 5. Modified Version Pipeline

### 5.1 Purpose & Goals

Extends the baseline to:

1. Support **three NLP tasks** (topic, sentiment, NER) in one run
2. Add a **TaskValidatorAgent** to validate label conformance per sentence
3. Compute **per-sentence** CS ratio scores (not one batch score)
4. Rewrite only the **failing sentences** during refinement
5. **Re-evaluate quality agents after refinement** (fresh scores — baseline does not)
6. Support a **nested config format** (`config2.yaml`) with task-specific sections
7. Switch to **GPT-4o-mini** (cost reduction from baseline GPT-4o)
8. Provide a **UI-facing entry point** (`run_french_ui.py`) callable via subprocess
9. Include a **smoke test** runner for quick validation

---

### 5.2 Key Files

| File | Role |
|---|---|
| `core/agents.py` | Extended graph; `TaskValidatorAgent` inserted; re-evaluation edges after `RefinerAgent` |
| `core/node_engine.py` | Task-aware node functions; per-sentence CS scoring; selective refinement |
| `core/node_models.py` | Extended state: `task`, `label`, `task_validation_result`, `cs_ratio_results_per_instances`, `sentence_records`, `failing_sentence_indices` |
| `core/prompt.py` | 14+ prompt templates — per task × agent |
| `core/utils.py` | `generate_scenarios` task-aware; `compute_true_cs_stats()`; `build_sentence_records()`; `compute_sentence_weighted_scores()` |
| `core/mcp_tools.py` | Unchanged from baseline |
| `core/run_french_ui.py` | UI subprocess target; reads `SWITCHLINGUA_CONFIG_PATH` env var; `--max-scenarios` flag |
| `core/smoke_test_real_api.py` | Runs ≤ N scenarios; `sys.exit(0/1)` for CI use |
| `config/config2.yaml` | Primary nested config (recommended) |
| `config/config.yaml` | Legacy flat config |
| `output/Arabic.jsonl` | Latest full run (18 scenarios, 293 KB) |

---

### 5.3 State Model — Additions

New fields added to `AgentRunningState`:

```python
# Task identity (NEW)
task: Literal["topic", "sentiment", "ner"]
label: str                               # e.g. "positive", "tech", "PER+ORG"
task_constraints: Dict[str, Any]

# Task validation (NEW)
task_validation_result: TaskValidationResult
    # .passed, .confidence, .predicted_label, .notes
    # .per_instance_results: list[TaskValidationResult]  ← one per sentence

# Per-sentence CS ratio (NEW — was a single result in the baseline)
cs_ratio_results_per_instances: list[CSRatioResponse]

# Per-sentence lifecycle (NEW)
sentence_records: list[SentenceRecord]
    # .index, .text, .fluency, .naturalness, .cs_ratio, .socio_cultural
    # .weighted_score, .refine_count, .status, .task_validation
failing_sentence_indices: list[int]       # sentences below threshold
```

---

### 5.4 Graph Flow — Changes

```
START
  │
  ▼
DataGenerationAgent
  │
  ▼
TaskValidatorAgent                    ← NEW
  │
  ├──────────────────────────────────┐
  ▼          ▼           ▼           ▼
FluencyAgent NaturalnessAgent CSRatioAgent SocialCulturalAgent
  └──────────┴───────────┴───────────┘
                    │
                    ▼
             SummarizeResult
             (per-sentence weighted scores → failing_sentence_indices)
                    │
             meet_criteria()
             any sentence score < 8.0 AND refine_count < MAX?
                  /      \
               Yes         No
                │           │
          RefinerAgent   AcceptanceAgent → END
                │
                ▼
         ┌──────────────────────────────────┐
         │  FluencyAgent  NaturalnessAgent   │  ← RE-EVALUATE (NEW)
         │  CSRatioAgent  SocialCulturalAgent│    fresh scores after refine
         └──────────────────────────────────┘
                    │
                    ▼
             SummarizeResult
             (loop up to MAX_REFINER_ITERATIONS=1)
```

**Critical difference:** After `RefinerAgent`, the four quality agents run again producing
**fresh scores**. The baseline skips this — stale pre-refine scores are reused.

---

### 5.5 New & Changed Agents

| Agent | Baseline | Modified Version |
|---|---|---|
| **DataGenerationAgent** | Generic single prompt | Task-specific prompt (topic / sentiment / NER) |
| **TaskValidatorAgent** | Not present | NEW — BERT or LLM classifier; `per_instance_results` list |
| **FluencyAgent** | Scenario-level | Same; feeds `sentence_records` |
| **NaturalnessAgent** | Scenario-level | Same; feeds `sentence_records` |
| **CSRatioAgent** | Single batch result | Per-sentence list: `cs_ratio_results_per_instances` |
| **SocialCulturalAgent** | Scenario-level | Same; feeds `sentence_records` |
| **SummarizeResult** | Single `score` | Per-sentence `weighted_score`; populates `failing_sentence_indices` |
| **RefinerAgent** | Rewrites all sentences; generic prompt | Rewrites only **failing** sentences; task-specific prompt |
| **AcceptanceAgent** | Writes JSONL | Same |

---

### 5.6 Configuration System (Dual Format)

**Format 1 — `config.yaml` (flat, same as baseline):**
```yaml
pre_execute:
  topics: [tech, finance]
  task: [topic]
  tense: [Present]
  cs_ratio: ["70%"]
  ...
```

**Format 2 — `config2.yaml` (nested, recommended):**
```yaml
pre_execute:
  task: [sentiment, ner, topic]    # multi-task
  cs_ratio: ["70%"]
  shared:                          # applied to all tasks
    topic: [tech, finance]
    tense: [Present]
    cs_function: [Expressive]
    cs_type: [Intrasentential]
    conversation_type: [single_turn]
    character_setting:
      nationality:
        first_language: Arabic
        second_language: English
      age: ["18-25"]
      gender: [Male]
      education_level: [College]
  sentiment:
    labels: [positive, negative, neutral]
    intensity: [low, medium]
    ambiguity: [low]
  ner:
    entity_types: [PER, ORG, LOC]
    min_entities: [2]
    max_entities: [3]
    must_include_types: [PER, ORG]
    allow_code_switched_entities: [true]
  topic:
    topics: [business, education]
```

Both formats are handled transparently by helper wrappers in `utils.py` and `app.py`:
`_get_topics()`, `_set_topics()`, `_get_shared()`, `_set_shared()`, `_get_char()`, `_set_char()`.

---

### 5.7 Scenario Generation — Task-Aware

`generate_scenarios(pre_execute)` now:

1. Iterates over `pre_execute["task"]` — each task runs in the same pass
2. For **sentiment**: one scenario per `labels × base combinations`
3. For **ner**: one scenario per entity type combination × base combinations
4. For **topic**: one scenario per topic sub-domain × base combinations
5. Populates `task`, `label`, `task_constraints` on each scenario dict

With the default `config2.yaml`: **18 scenarios** (3 tasks × 2 topics × shared params).

---

### 5.8 Prompts

14+ `ChatPromptTemplate` objects in `core/prompt.py`:

| Prompt | Task | Purpose |
|---|---|---|
| `DATA_GENERATION_PROMPT` | any | Base generation |
| `DATA_GENERATION_TOPIC_PROMPT` | topic | Topic-specific generation |
| `DATA_GENERATION_SENTIMENT_PROMPT` | sentiment | Includes intensity/ambiguity |
| `DATA_GENERATION_NER_PROMPT` | ner | Entity types, counts, CS entity names |
| `TASK_VALIDATION_TOPIC_PROMPT` | topic | Validates topic classification |
| `TASK_VALIDATION_SENTIMENT_PROMPT` | sentiment | Validates sentiment label, intensity, ambiguity |
| `TASK_VALIDATION_NER_PROMPT` | ner | Validates entity presence and type |
| `FLUENCY_PROMPT` | all | Rates fluency, lists errors |
| `NATURALNESS_PROMPT` | all | Rates naturalness |
| `CS_RATIO_PROMPT` | all | Per-instance ratio scoring |
| `SOCIAL_CULTURAL_PROMPT` | all | Cultural appropriateness |
| `REFINER_PROMPT` | all | Base refiner |
| `REFINER_TASK_TOPIC_PROMPT` | topic | Topic-aware refiner |
| `REFINER_TASK_SENTIMENT_PROMPT` | sentiment | Sentiment-aware refiner |
| `REFINER_TASK_NER_PROMPT` | ner | NER-aware refiner |

---

### 5.9 Scoring & Weighting — Changes

```
per-sentence weighted_score =
    w_fluency      × fluency_score
  + w_naturalness  × naturalness_score
  + w_cs_ratio     × cs_ratio_score          ← now per-sentence
  + w_soc_cult     × social_cultural_score
  + w_task_val     × task_validation_confidence  ← new term
```

`SENTENCE_SCORE_THRESHOLD = 8.0` — sentences below this are added to `failing_sentence_indices`.

`compute_true_cs_stats(text)` provides **deterministic** token-level language detection
as a ground-truth complement to the LLM CS ratio assessment.

`meet_criteria(state)` reads `sentence_records` to check individual sentence scores,
not `state["score"]` (the single float used by the baseline).

---

### 5.10 Output Format (JSONL)

Each line of `Arabic.jsonl`:

```json
{
  "task": "sentiment",
  "label": "positive",
  "topic": "tech",
  "tense": "Present",
  "perspective": "First Person",
  "cs_ratio": "70%",
  "gender": "Male",
  "age": "18-25",
  "education_level": "College",
  "conversation_type": "single_turn",
  "data_generation_result": ["sentence1", "sentence2", "..."],
  "task_validation_result": {
    "passed": true,
    "confidence": 0.9,
    "predicted_label": "positive",
    "notes": "...",
    "per_instance_results": [
      {"passed": true, "confidence": 0.9, "predicted_label": "positive", "notes": "..."}
    ]
  },
  "fluency_result":           {"fluency_score": 8.5, "errors": [], "summary": "..."},
  "naturalness_result":       {"naturalness_score": 8.0, "observations": [...], "summary": "..."},
  "social_cultural_result":   {"socio_cultural_score": 9.0, "issues": [], "summary": "..."},
  "cs_ratio_results_per_instances": [
    {"ratio_score": 4, "computed_ratio": "40%:60%", "notes": "diff=10%"}
  ],
  "score": 7.27,
  "refine_count": 1
}
```

---

## 6. Baseline vs Modified — Side-by-Side Comparison

| Feature | Original Baseline | Modified Version |
|---|---|---|
| **Tasks** | Single implicit task (topic only) | Three tasks: topic, sentiment, NER |
| **Task validation** | None | `TaskValidatorAgent` per sentence |
| **Config format** | Flat dict only | Flat **or** nested `shared` block |
| **CS ratio scoring** | Single result for whole batch | Per-sentence list |
| **Refinement target** | All sentences rewritten | Only failing sentences |
| **Re-evaluation after refine** | No — stale scores reused | Yes — quality agents re-run |
| **`meet_criteria` input** | `state["score"]` (single float) | `sentence_records` (per sentence) |
| **LLM model** | GPT-4o | GPT-4o-mini |
| **Prompt count** | 6 | 14+ |
| **State schema** | Flat TypedDict, no task field | Extended with task, label, sentence_records, failing_sentence_indices |
| **Deterministic CS counting** | Not present | `compute_true_cs_stats()` |
| **Entry point** | `run_french.py` (CLI batch) | `run_french_ui.py` (UI subprocess) + `run_french.py` |
| **Smoke test** | Not present | `smoke_test_real_api.py` |
| **Max refine iterations** | 1 | 1 (configurable via `MAX_REFINER_ITERATIONS`) |

---

## 7. Control Center UI

### 7.1 Purpose & Stack

A Streamlit web application providing a GUI to operate the **Modified Version** pipeline —
edit configs, launch runs, and explore outputs — without touching files or the terminal directly.

**Stack:** Python 3.12 · Streamlit · pandas · PyArrow · openpyxl · PyYAML

---

### 7.2 Key Files

| File | Role |
|---|---|
| `app.py` | Main application (~1350 lines) |
| `.streamlit/config.toml` | Dark theme: `base="dark"`, custom palette |
| `saved_configs/` | User-saved YAML snapshots |

---

### 7.3 Page Layout & Tabs

```
┌──────────────────────────────────────────────────────────────────┐
│              Switch Lingua Control Center                        │
├──────────────────┬───────────────────────────────────────────────┤
│                  │                                               │
│  SIDEBAR         │  [Config Editor] [Run] [Outputs]             │
│                  │                                               │
│  Config file     │  Tab 1 — Config Editor                       │
│  selector        │    Full YAML form (all pre_execute fields)   │
│                  │    Save / Load snapshot                       │
│  Recent runs     │                                               │
│  history         │  Tab 2 — Run                                  │
│                  │    Smoke test (≤ N scenarios, live output)    │
│  Dark/light      │    Full pipeline run (all scenarios)          │
│  toggle          │    Auto-detects newly created output files    │
│                  │                                               │
│                  │  Tab 3 — Outputs                              │
│                  │    File browser (JSONL / JSON / XLSX)         │
│                  │    Metrics bar, charts, scenario + sentence   │
│                  │    tables, export buttons                     │
└──────────────────┴───────────────────────────────────────────────┘
```

---

### 7.4 Config Editor

Renders a complete form for every field in `pre_execute`, supporting both config formats:

| Section | Fields |
|---|---|
| **General** | Tasks (multiselect), CS Ratio, Output Format, Use Tools |
| **Shared** | Topics, Tense, Perspective, CS Function, CS Type, Conversation Type |
| **Character** | First/Second Language, Age group, Gender, Education level |
| **Sentiment** | Labels, Intensity, Ambiguity |
| **NER** | Entity types, Min/Max entities, Must-include types, Allow CS entities |
| **Topic** | Topic sub-domains |
| **on_execute** | Max rounds, Verbose flag |

Changes can be saved to `saved_configs/` as named YAML snapshots and reloaded.

---

### 7.5 Run Controls

**Smoke Test:**
- Invokes `smoke_test_real_api.py --max-scenarios N` via `subprocess`
- Sets `SWITCHLINGUA_CONFIG_PATH` and `PYTHONIOENCODING=utf-8` in the subprocess env
- Shows live stdout/stderr; detects newly created output files

**Full Pipeline Run:**
- Invokes `run_french_ui.py` with optional `--max-scenarios` cap
- 4-hour timeout
- Auto-highlights output files created/modified during the run
- Appends a record to `st.session_state.recent_runs` (shown in sidebar)

---

### 7.6 Output Viewer

**JSONL (`Arabic.jsonl`):**

Parsed by `parse_jsonl_to_rows()` → `(scenario_rows, sentence_rows)`.

- **5-column metrics bar:** Scenarios · Avg Overall · Avg Fluency · Avg Naturalness · Avg Socio-Cultural
- **3 export buttons:**
  - Scenarios CSV
  - XLSX with 5 sheets (Scenarios, Sentences, Fluency stats, Naturalness stats, SocioCultural stats)
  - Sentences CSV
- **Score Charts expander:** Histograms for Overall, Fluency, Naturalness, Socio-Cultural
- **Scenario Summary table** (one row per JSONL record):
  `overall_score`, `fluency_score`, `naturalness_score`, `social_cultural_score`,
  `task_val_passed`, `task_val_confidence`, `task_val_predicted`, `refine_count`,
  `sentence_count`, summaries, notes
- **Per-Sentence Details table** (one row per generated sentence):
  All scenario-level scores + per-instance `tv_passed/confidence/predicted/notes`
  + per-instance `cs_ratio_score/computed/notes`
  Filterable by task, label, and task validation pass/fail

**Per-sentence column order:**
```
line, sentence_index, task, label, topic, sentence, overall_score,
fluency_score, fluency_errors, fluency_summary,
naturalness_score, naturalness_observations, naturalness_summary,
social_cultural_score, social_cultural_issues, social_cultural_summary,
tv_passed, tv_confidence, tv_predicted, tv_notes,
cs_ratio_score, cs_ratio_computed, cs_ratio_notes,
refine_count
```

**JSON (`pipeline_full_real_*.json`):**
- Parsed by `parse_pipeline_json()` → tabbed view: Sentences | Scenarios

**XLSX:** Multi-sheet preview with sheet selector.

---

### 7.7 Theme & Sidebar

**Dark mode (default)** via `.streamlit/config.toml`:
```toml
[theme]
base                     = "dark"
primaryColor             = "#3aa6b9"
backgroundColor          = "#1a1f2e"
secondaryBackgroundColor = "#161b27"
textColor                = "#e8eaf6"
```

**Sidebar persistence fix:** JavaScript injected via `st.markdown` clears all `localStorage`
keys containing `"sidebar"` on every page load — prevents Streamlit's hydration from
auto-collapsing the sidebar. Combined with `initial_sidebar_state="expanded"`.

**Expand button:** `[data-testid="stSidebarCollapsedControl"]` is fixed at `left:0; top:2.5rem`
with an explicit background colour so it remains visible when the sidebar is collapsed.

---

## 8. Data Flow End-to-End

```
User selects / edits config in UI
          │
          ▼
Modified_Version/config/config2.yaml
          │
          ▼  subprocess  (SWITCHLINGUA_CONFIG_PATH=... PYTHONIOENCODING=utf-8)
run_french_ui.py
          │
          ├─ load_config(path)
          │
          ├─ generate_scenarios(pre_execute)
          │       └─ itertools.product → list[AgentRunningState]
          │           (task-aware: one per task × label × domain × shared params)
          │
          └─ for each scenario:
                 CodeSwitchingAgent(scenario).run()
                       │
                       ▼
                 LangGraph StateGraph (async)
                       │
                 DataGenerationAgent
                 TaskValidatorAgent         ← validates label per sentence
                 Flu + Nat + CSRatio + Soc  ← parallel quality agents
                 SummarizeResult            ← per-sentence weighted scores
                 RefinerAgent?              ← rewrite failing sentences only
                 Re-evaluate quality        ← fresh scores after refine
                 AcceptanceAgent
                       │
                       ▼
             Modified_Version/output/Arabic.jsonl
                       │
                       ▼
       UI Output Viewer reads Arabic.jsonl
                       │
               parse_jsonl_to_rows()
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
     scenario_rows           sentence_rows
   (one per JSONL line)    (one per sentence)
           │                       │
    Metrics bar              Filter controls
    Score charts             Per-sentence table
    Scenario table           (all 5 agent scores + TV + CS visible)
    Export buttons
```

---

## 9. Configuration Reference

### config2.yaml — Fully annotated

```yaml
pre_execute:
  cs_ratio:                           # Target matrix:embedded ratio
    - "70%"                           # 70% Arabic, 30% English

  task:                               # Tasks to generate (all three in one pass)
    - sentiment
    - ner
    - topic

  shared:                             # Applied to all tasks
    topic:
      - tech
      - finance
    style:
      - casual
    tense:                            # Present | Past | Future
      - Present
    perspective:                      # First Person | Second Person | Third Person
      - First Person
    cs_function:                      # Expressive | Referential | Directive | Phatic | Metalinguistic
      - Expressive
    cs_type:                          # Intrasentential | Intersentential | Tag-switching
      - Intrasentential
    conversation_type:                # single_turn | multi_turn
      - single_turn
    output_format: json
    use_tools: false
    character_setting:
      nationality:
        first_language: Arabic
        second_language: English
      age:
        - "18-25"
      gender:
        - Male
      education_level:
        - College

  sentiment:
    labels: [positive, negative, neutral]
    intensity: [low, medium]
    ambiguity: [low]

  ner:
    entity_types: [PER, ORG, LOC]
    min_entities: [2]
    max_entities: [3]
    must_include_types: [PER, ORG]
    allow_code_switched_entities: [true]

  topic:
    topics: [business, education]
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | API key for GPT-4o-mini |
| `OPENAI_BASE_URL` | — | Custom API base URL |
| `ENABLE_TASK_VALIDATOR` | `"1"` | Set `"0"` to bypass TaskValidatorAgent |
| `MAX_SENTENCE_REFINES` | `"1"` | Max refine attempts per individual sentence |
| `SWITCHLINGUA_CONFIG_PATH` | — | Set by UI via subprocess env |
| `PYTHONIOENCODING` | — | Set to `utf-8` by UI to prevent Windows encoding errors |

---

## 10. Current Metrics & Status

From `Arabic.jsonl` (full run 2026-04-01, 18 scenarios):

| Metric | Value | Target | Status |
|---|---|---|---|
| Scenarios processed | 18 / 18 | 18 | OK |
| Average Overall Score | 7.27 / 10 | > 8.0 | Below target |
| Average Fluency | ~8.2 / 10 | > 8.0 | OK |
| Average Naturalness | ~8.0 / 10 | > 8.0 | OK |
| CS Ratio Accuracy | ~50% avg | 70% | Below target |
| Task Validation Pass Rate | ~40% | > 80% | Below target |
| Sentences per scenario | 5 | 5 | OK |
| Max refine iterations used | 1 | 1 | OK |

---

## 11. Known Limitations & Risks

| Area | Issue | Severity |
|---|---|---|
| **CS Ratio** | GPT-4o-mini doesn't reliably hit exact token ratio targets. Deterministic counting confirms mismatch. Refinement rarely corrects ratio alone. | Medium |
| **Task Validation** | ~40–50% pass rate. Intensity/ambiguity (sentiment) and entity type constraints (NER) frequently violated by the LLM. | High |
| **Score asymmetry** | Fluency/naturalness/social-cultural are scenario-level; task validation and CS ratio are per-sentence. SummarizeResult must reconcile these. | Medium |
| **Refinement cap** | `MAX_REFINER_ITERATIONS=1` — sentences still failing after one refine pass are accepted. | Medium |
| **Baseline re-evaluation gap** | Baseline reuses stale quality scores after refinement. Fixed in Modified Version at higher API cost. | Low (fixed) |
| **Windows encoding** | Arabic/Unicode in subprocess pipes requires `PYTHONIOENCODING=utf-8`. Mitigated in the UI runner. | Low |
| **LLM cost** | GPT-4o-mini is called for every agent per scenario. 100+ scenario runs accumulate significant cost. | Medium |
| **Flat config support** | `run_french_ui.py` and `generate_scenarios()` assume the nested `shared` format. Flat `config.yaml` may not populate task-specific fields without the helper wrappers in `app.py`. | Low |
