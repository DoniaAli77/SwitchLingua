# SwitchLingua — Multi-Agent BERT Pipeline
## Comprehensive Project Report

**Date:** 2026-05-05  
**Test suite:** 830 passed, 0 failed (3.26 s)  
**Python:** 3.13.5 · venv at `C:\Users\Eng.Donia\Documents\matser\SwitchLingua\.venv\`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Repository Layout](#2-repository-layout)
3. [Architecture Overview](#3-architecture-overview)
4. [Supported Tasks](#4-supported-tasks)
5. [Pipeline Modes](#5-pipeline-modes)
6. [Pipeline Flow — Classification](#6-pipeline-flow--classification)
7. [Pipeline Flow — NER (Sequence Labeling)](#7-pipeline-flow--ner-sequence-labeling)
8. [Component Reference](#8-component-reference)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Configuration System](#10-configuration-system)
11. [Data Assets](#11-data-assets)
12. [Test Suite](#12-test-suite)
13. [Entry Points](#13-entry-points)
14. [Current Limitations](#14-current-limitations)
15. [Next Steps](#15-next-steps)

---

## 1. Executive Summary

SwitchLingua is a research-grade, multi-agent NLP pipeline for Arabic-English code-switched text. It supports two fundamentally different task types:

- **Classification** — assigns a single label per input (topic, sentiment).
- **Sequence Labeling** — assigns one tag per token (Named Entity Recognition, BIO scheme).

Both task types share the same `PipelineState` / `TaskConfig` contract and are dispatched by a single `PipelineOrchestrator`. Three pipeline modes (`primary_only`, `paper_style`, `full_agentic`) are supported for both task types. All components are unit-tested; the framework runs with zero external API calls via mock LLM and mock primary-classifier stubs.

**Status:** All classification and NER smoke pipelines are working end-to-end. The primary classifier is currently a heuristic mock; a transformer stub (`PrimaryTransformerClassifier`) is present but not yet wired to real weights.

---

## 2. Repository Layout

```
multi-agent-bert/
├── src/
│   ├── agents/           # Classification + NER specialist agents
│   ├── config/           # YAML loader + TaskConfig / TaskBundle dataclasses
│   ├── evaluation/       # Evaluator, NER Evaluator, Ablation Study
│   ├── llm/              # LLM client abstractions (mock + base)
│   ├── models/           # Primary classifiers (mock + transformer stub)
│   ├── pipeline/         # PipelineOrchestrator + Router
│   ├── prompts/          # Structured prompt builders (LLM agents)
│   └── state/            # PipelineState schema (shared contract)
├── tests/                # 22 test files, 830 tests total
├── scripts/              # Data generation, debug, inspection utilities
├── data/                 # Smoke datasets (JSONL)
├── results/              # Evaluation output directory
├── legacy/               # Isolated original codebase (not imported)
├── evaluate_pipeline.py  # Main CLI evaluation entry point
├── run_demo.py           # Quick smoke-run demo
├── pyproject.toml
└── requirements.txt
```

---

## 3. Architecture Overview

```
┌───────────────────────────────────────────────────────────────────┐
│                        evaluate_pipeline.py                       │
│  CLI: --config · --active_task · --dataset · --mode               │
│       --pipeline_mode · --deliberation · --output_dir             │
└─────────────────────────────┬─────────────────────────────────────┘
                              │  build_orchestrator()
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                     PipelineOrchestrator                          │
│  · Receives PipelineState (text + TaskConfig)                     │
│  · Dispatches to classification path or NER path                  │
│  · Writes state.history entries at every stage                    │
└──────────────┬─────────────────────────────┬──────────────────────┘
               │ task_type=classification    │ task_type=sequence_labeling
               ▼                             ▼
  Classification Path                   NER Path
  (see §6)                              (see §7)
               │                             │
               └──────────────┬──────────────┘
                              ▼
                    PipelineState (final_output populated)
                              │
                    ┌─────────┴──────────┐
                    │ Evaluator /        │
                    │ NEREvaluator       │
                    └─────────┬──────────┘
                              ▼
                    Metrics + JSON/CSV output files
```

**Shared data contract** — every component reads and writes `PipelineState`:

| Field | Type | Description |
|---|---|---|
| `input_text` | `str` | Raw input sentence |
| `task_config` | `TaskConfig` | Active task definition |
| `primary_output` | `AgentOutput` | Primary classifier result |
| `routing_info` | `RoutingInfo` | accept / escalate decision |
| `lexical_output` | `LexicalOutput` | Lexical/NER lexical agent result |
| `logic_output` | `LogicOutput` | Logic/NER logic agent result |
| `contextual_output` | `ContextualOutput` | Contextual/NER contextual result |
| `deliberation_output` | `DeliberationOutput` | (classification only, optional) |
| `consensus_output` | `ConsensusOutput` | Weighted voting result |
| `final_output` | `FinalOutput` | Label (classification) or payload (NER) |
| `explanation_output` | `ExplanationOutput` | Human-readable explanation |
| `history` | `List[HistoryEntry]` | Ordered execution trace |
| `extras` | `Dict[str, Any]` | Arbitrary pass-through context |

---

## 4. Supported Tasks

| Task name | Task type | Labels |
|---|---|---|
| `topic_classification` | `classification` | business, education, health, shopping, medical, sports, tech, finance, social |
| `sentiment_classification` | `classification` | positive, negative, neutral |
| `ner` | `sequence_labeling` | O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC |

Tasks are defined in `src/config/default.yaml`. Switching tasks requires no code change — pass `--active_task <name>` to any entry point. The loader reads `label_knowledge` (keywords + regex rules) to build `keyword_map` and `rule_map` for classification tasks. For `sequence_labeling`, these maps are intentionally empty (not applicable at token level).

**Active default task:** `topic_classification`

---

## 5. Pipeline Modes

| Mode | Description |
|---|---|
| `primary_only` | Runs only the primary classifier. Router and all specialist agents are skipped. Fastest; no deliberation. |
| `paper_style` | Primary → Router → deterministic specialist agents (Lexical, Logic, TransformerContextual) → ConsensusAgent → ExplainabilityAgent. No LLM calls. |
| `full_agentic` | Primary → Router → LLM-backed agents (LLMLexical, LLMLogic, ContextualAgent) → optional DeliberationAgent → ConsensusAgent → LLMExplainabilityAgent. All agents use `MockLLMClient` in tests. |

All three modes are fully supported for both classification and NER tasks.

---

## 6. Pipeline Flow — Classification

```
Input text + TaskConfig (task_type = "classification")
        │
        ▼
┌────────────────────────────┐
│     PrimaryClassifier      │
│  (MockPrimaryClassifier)   │
│  → label, confidence       │
└────────────┬───────────────┘
             │  confidence ≥ threshold (0.6)?
     ┌───────┴────────┐
     │ YES (accept)   │ NO (escalate)
     ▼                ▼
 final_output       Router
 label = primary    decision = "escalate"
 (primary_only)          │
                    ┌────┴────────────────────────────────────────┐
                    │  paper_style mode    │  full_agentic mode   │
                    │                      │                      │
                    │  LexicalAgent        │  LLMLexicalAgent     │
                    │  LogicAgent          │  LLMLogicAgent       │
                    │  TransformerContext. │  ContextualAgent     │
                    │                      │  DeliberationAgent   │
                    │                      │  (if --deliberation) │
                    │  ConsensusAgent      │  ConsensusAgent      │
                    │  ExplainabilityAgent │  LLMExplainability   │
                    └──────────┬───────────┘──────────────────────┘
                               ▼
                    final_output.label  (winning label)
                    final_output.confidence
                    explanation_output.text
```

**ConsensusAgent** weights: `lexical=1.0`, `logic=1.0`, `contextual=1.0`, `deliberation=1.5` (configurable via `deliberation_weight`).

**Router threshold:** 0.6 (from `execution.threshold` in `default.yaml`). In tests, `MockPrimaryClassifier` is tuned to stay below threshold → all specialist agents always run.

---

## 7. Pipeline Flow — NER (Sequence Labeling)

```
Input tokens + TaskConfig (task_type = "sequence_labeling")
        │
        ▼
┌────────────────────────────┐
│     PrimaryClassifier      │
│   (stub; not NER-aware)    │
└────────────┬───────────────┘
             │
     ┌───────┴──────────────────────────────────────────┐
     │ primary_only mode       │ paper_style / full_agentic │
     ▼                         ▼                            │
 stub FinalOutput          NERLexicalAgent                  │
 (no tags written)         ← gazetteer B/I tagger           │
 (NER agents skipped)      NERLogicAgent                    │
                           ← regex-rule B/I tagger          │
                           NERContextualAgent               │
                           ← capitalisation + known_entities│
                           NERConsensusAgent                │
                           ← weighted token-level voting    │
                           ExplainabilityAgent              │
                                │                           │
                                ▼                           │
                    final_output.label = None               │
                    final_output.payload["sequence_output"] │
                      → [{"token": str,                     │
                           "tag":   str,   (BIO tag)        │
                           "confidence": float}, ...]       │
                    final_output.payload["token_count"] = N │
```

**NERConsensusAgent** weights: `lexical=1.0`, `logic=1.0`, `contextual=1.0` (configurable). Voting is per-token: highest weighted-vote tag wins; `"O"` is the tiebreaker fallback.

**Token resolution order:** `state.extras["tokens"]` → first available `sequence_output` tokens → `input_text.split()`.

---

## 8. Component Reference

### `src/config/`

| File | Purpose |
|---|---|
| `default.yaml` | Task registry (all three tasks), language pair, execution defaults. Single source of truth for labels, keywords, regex rules, BIO tag descriptions. |
| `loader.py` | `load_task_bundle(config_path, active_task, overrides) → TaskBundle`. Parses YAML, resolves active task, builds `keyword_map` / `rule_map` from `label_knowledge`. |
| `task_config.py` | `TaskConfig` dataclass. Fields: `task_name`, `task_type`, `labels`, `label_descriptions`, `threshold`, `pipeline_mode`, `enable_deliberation`, `contextual_use_prior_outputs`. |

### `src/state/`

| File | Purpose |
|---|---|
| `schema.py` | All shared data structures: `PipelineState`, `AgentOutput`, `FinalOutput`, `RoutingInfo`, `LexicalOutput`, `LogicOutput`, `ContextualOutput`, `ConsensusOutput`, `DeliberationOutput`, `ExplanationOutput`, `StateMetadata`, `HistoryEntry`, `TokenTag`, `SequenceLabelingOutput`. **Do not modify** — all agents depend on this contract. |

### `src/pipeline/`

| File | Purpose |
|---|---|
| `orchestrator.py` | `PipelineOrchestrator` — controls full execution flow per mode. Dispatches to classification or NER path based on `task_config.task_type`. |
| `router.py` | `Router` — compares primary classifier confidence against `threshold`. Returns `RoutingInfo(decision="accept"\|"escalate")`. |

### `src/agents/` — Classification agents

| File | Agent | Mode | Description |
|---|---|---|---|
| `base_agent.py` | `BaseAgent` | all | Abstract base. Manages `self.name`, calls `state.append_history(...)`. |
| `lexical_agent.py` | `LexicalAgent` | paper_style | Keyword lookup over `keyword_map`. |
| `llm_lexical_agent.py` | `LLMLexicalAgent` | full_agentic | LLM-backed lexical analysis via `LLMLexicalPrompt`. |
| `logic_agent.py` | `LogicAgent` | paper_style | Regex rule matching over `rule_map`. |
| `llm_logic_agent.py` | `LLMLogicAgent` | full_agentic | LLM-backed rule analysis via structured prompt. |
| `transformer_contextual_agent.py` | `TransformerContextualAgent` | paper_style | TF-IDF cosine similarity against `label_descriptions`. No LLM call. |
| `contextual_agent.py` | `ContextualAgent` | full_agentic | LLM contextual classification via `ContextualPrompt`. |
| `deliberation_agent.py` | `DeliberationAgent` | full_agentic (opt.) | LLM deliberation over all specialist outputs. Enabled by `--deliberation`. |
| `consensus_agent.py` | `ConsensusAgent` | paper_style, full_agentic | Weighted voting over all outputs. Produces `final_output.label` + confidence. |
| `explainability_agent.py` | `ExplainabilityAgent` | paper_style | Template-based explanation (no LLM). |
| `llm_explainability_agent.py` | `LLMExplainabilityAgent` | full_agentic | LLM-generated explanation. |

### `src/agents/` — NER agents

| File | Agent | Description |
|---|---|---|
| `ner_lexical_agent.py` | `NERLexicalAgent` | Gazetteer-based BIO tagger. Case-insensitive dict lookup; emits B/I continuation tags. Constructor: `NERLexicalAgent(gazetteer: Dict[str, List[str]] = None)`. |
| `ner_logic_agent.py` | `NERLogicAgent` | Regex-rule BIO tagger. Per-token `re.fullmatch` / word-boundary search; invalid patterns are skipped with a warning. Constructor: `NERLogicAgent(rule_map: Dict[str, List[str]] = None)`. |
| `ner_contextual_agent.py` | `NERContextualAgent` | Heuristic capitalisation + `known_entities` override tagger. Latin capitalized tokens → `B-PER` by default. Constructor: `NERContextualAgent(known_entities: Dict[str, str] = None)`. |
| `ner_consensus_agent.py` | `NERConsensusAgent` | Weighted token-level voting across lexical/logic/contextual outputs. Writes `final_output.payload["sequence_output"]` and `final_output.payload["token_count"]`. Exported helpers: `_extract_seq_output`, `_vote_token`. |

### `src/models/`

| File | Class | Status |
|---|---|---|
| `mock_primary_classifier.py` | `MockPrimaryClassifier` | **Active** — heuristic keyword scoring. Used in all tests and smoke evaluations. |
| `primary_transformer_classifier.py` | `PrimaryTransformerClassifier` | **Stub** — HuggingFace checkpoint interface ready; not yet wired to real weights or `build_orchestrator()`. |

### `src/llm/`

| File | Purpose |
|---|---|
| `base_client.py` | `BaseLLMClient` abstract class. Defines `complete(prompt) → str`. |
| `mock_client.py` | `MockLLMClient` — three modes: `label_echo`, `fixed`, `random`. Zero external calls. |

### `src/prompts/`

| File | Purpose |
|---|---|
| `contextual_prompt.py` | Few-shot contextual classification prompt for `ContextualAgent`. |
| `deliberation_prompt.py` | Structured deliberation prompt (expects JSON response). |
| `llm_explainability_prompt.py` | Explanation generation prompt for `LLMExplainabilityAgent`. |
| `llm_lexical_prompt.py` | Keyword-evidence prompt for `LLMLexicalAgent`. |
| `llm_logic_prompt.py` | Rule-evidence prompt for `LLMLogicAgent`. |

### `src/evaluation/`

| File | Purpose |
|---|---|
| `evaluator.py` | `Evaluator` — classification evaluation. Runs orchestrator over full dataset, computes accuracy / macro-F1 / per-label P-R-F1. Writes `predictions.json`, `predictions.csv`, `metrics.json`, `metrics.csv`. |
| `ner_evaluator.py` | `NEREvaluator` — NER evaluation. Computes **token accuracy**, **macro F1**, **per-tag P/R/F1/support**. Writes `*_ner_predictions.json`, `*_ner_predictions.csv`, `*_ner_metrics.json`, `*_ner_metrics.csv`. Key classes: `NERSampleResult`, `NERReport`. Helper: `_align_tags(gold, pred, n)`. |
| `ablation.py` | `AblationStudy` / `AblationConfig` / `AblationReport` — multi-run ablation from YAML/JSON config. Can toggle individual agents or vary consensus weights. |

---

## 9. Evaluation Framework

### Classification evaluation (`Evaluator`)

```python
from src.evaluation.evaluator import Evaluator
evaluator = Evaluator(task_config, orchestrator, run_id="my_run", logger=logger)
report = evaluator.evaluate(dataset)      # dataset: List[Dict]
paths  = evaluator.save(report, output_dir="results/")
```

**Output metrics:** accuracy, macro-F1, per-label: precision, recall, F1, support.

**Output files:**
- `{run_id}_predictions.json` / `.csv` — per-sample: `id`, `text`, `gold`, `predicted`, `error`
- `{run_id}_metrics.json` / `.csv` — aggregate + per-label

### NER evaluation (`NEREvaluator`)

```python
from src.evaluation.ner_evaluator import NEREvaluator
evaluator = NEREvaluator(task_config, orchestrator, run_id="ner_run", logger=logger)
report = evaluator.evaluate(dataset)
paths  = evaluator.save(report, output_dir="results/")
```

**Output metrics:** token accuracy, macro token-F1, per-tag: precision, recall, F1, support.

**Output files:**
- `{run_id}_ner_predictions.json` / `.csv` — per-sample: `id`, `text`, `tokens`, `gold_tags`, `predicted_tags`, `token_count`, `correct_count`, `pipeline_error`
- `{run_id}_ner_metrics.json` / `.csv` — aggregate (token_accuracy, macro_f1) + per-tag; `__summary__` row in CSV

**Tag alignment:** `_align_tags(gold, pred, n)` pads or truncates to `n` tokens using `"O"` fill, ensuring gold/predicted sequences always have equal length before metric computation.

### Ablation study (`AblationStudy`)

Run multiple orchestrator configurations in one pass. Each run in the config can:
- toggle individual agents on/off (`disable_agents: [lexical, logic]`)
- override consensus weights (`consensus_weights: {lexical: 0.5, contextual: 2.0}`)
- override pipeline mode

Results are collected into an `AblationReport` with per-run metrics for side-by-side comparison.

---

## 10. Configuration System

`src/config/default.yaml` is the single configuration file. The key sections are:

```yaml
active_task: topic_classification          # default task; override via CLI

tasks:
  ner:
    task_type: sequence_labeling
    labels: [O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC]
    label_descriptions: { ... }
    label_knowledge: {}                    # empty for NER — not applicable

  sentiment_classification:
    task_type: classification
    labels: [positive, negative, neutral]
    label_knowledge:
      positive: { keywords_l1: [...], keywords_l2: [...], regex_rules: [...] }
      ...

  topic_classification:
    task_type: classification
    labels: [business, education, health, shopping, medical, sports, tech, finance, social]
    label_knowledge: { ... }

language_pair:
  pair_name: en-ar
  l1: en
  l2: ar
  code_switching_allowed: true

execution:
  threshold: 0.6
  pipeline_mode: full_agentic
  enable_deliberation: false
  deliberation_weight: 1.5
  contextual_use_prior_outputs: false
  verbose: false
  output_format: jsonl
```

**Known fixed bug (2026-05-05):** `enable_deliberation` and `contextual_use_prior_outputs` were each defined twice in `execution:`. Duplicate keys removed; only the intended values are now present.

---

## 11. Data Assets

| File | Task | Size | Notes |
|---|---|---|---|
| `data/dev_dummy.jsonl` | `topic_classification` | 30 examples | ~3-4 per label; Arabic-English code-switched |
| `data/dev_dummy_sentiment.jsonl` | `sentiment_classification` | 30 examples | 10 positive / 10 negative / 10 neutral |
| `data/dev_dummy_ner.jsonl` | `ner` | 30 examples | BIO-tagged; O:152, B-PER:16, I-PER:8, B-ORG:22, I-ORG:10, B-LOC:29, I-LOC:7 tokens |

**Format — classification dataset:**
```jsonl
{"id": "s001", "text": "شركتنا launched a new product", "label": "business"}
```

**Format — NER dataset:**
```jsonl
{"id": "n001", "tokens": ["Ahmed", "works", "at", "Google"], "tags": ["B-PER", "O", "O", "B-ORG"]}
```

Scripts to regenerate:
```bash
python scripts/generate_dummy_data.py
python scripts/generate_dummy_sentiment_data.py
python scripts/generate_dummy_ner_data.py
```

---

## 12. Test Suite

**Total: 830 passed, 0 failed** (run: `python -m pytest` from `multi-agent-bert/`)

### Per-file breakdown

| Test file | Tests | Coverage area |
|---|---|---|
| `test_pipeline_modes_smoke.py` | 67 | End-to-end all three modes: correct agents in history, wrong-mode agents absent, final output structure |
| `test_task_aware_config_modes.py` | 35 | Config loading + pipeline runs for both classification tasks; cross-task label contamination guards |
| `test_config_loader.py` | 37 | `load_task_bundle`: label loading, keyword/rule map construction, override precedence, error handling |
| `test_evaluator.py` | 103 | `Evaluator` metric computation, file output, per-label P/R/F1, error counting |
| `test_ablation.py` | 46 | `AblationStudy` multi-run orchestration, report aggregation, per-run metrics |
| `test_ner_agents.py` | 54 | `NERLexicalAgent`, `NERLogicAgent`, `NERContextualAgent`: BIO tagging, B/I continuation, edge cases, skip guard for non-NER tasks |
| `test_ner_consensus_agent.py` | 44 | `NERConsensusAgent`: weighted voting, tie-breaking, `_vote_token`, `_extract_seq_output`, payload structure |
| `test_ner_config.py` | 15 | NER config loading, `evaluate_pipeline.main()` error handling for wrong dataset format |
| `test_ner_evaluator.py` | 53 | `NEREvaluator`: token accuracy, per-tag metrics, macro F1, prediction extraction, error handling, all 4 file outputs, `evaluate_pipeline.main()` NER routing |
| `test_orchestrator_ner_path.py` | 32 | NER orchestrator for all three modes, custom agent injection, error capture, classification-path regression |
| `test_dataset_loading.py` | 27 | `load_classification_dataset`, `load_sequence_labeling_dataset` (token/tag length validation, tag membership), `load_dataset` alias |
| `test_orchestrator_flow.py` | 3 | Orchestrator routing logic, escalation/accept paths |
| `test_consensus_agent.py` | 40 | `ConsensusAgent` weighted voting, tie-breaking, deliberation weight |
| `test_contextual_agent.py` | 32 | `ContextualAgent` LLM prompt construction and output parsing |
| `test_transformer_contextual_agent.py` | 32 | `TransformerContextualAgent` TF-IDF similarity, cosine scoring |
| `test_deliberation_agent.py` | 31 | `DeliberationAgent` prompt, JSON response parsing, fallback behaviour |
| `test_explainability_agent.py` | 37 | `ExplainabilityAgent` template output structure and content |
| `test_llm_explainability_agent.py` | 21 | `LLMExplainabilityAgent` prompt and output parsing |
| `test_llm_specialist_agents.py` | 42 | `LLMLexicalAgent` + `LLMLogicAgent` prompt construction, output parsing |
| `test_lexical_agent.py` | 22 | `LexicalAgent` keyword scoring, empty text, unknown labels |
| `test_logic_agent.py` | 27 | `LogicAgent` regex rule matching, pattern edge cases |
| `test_primary_transformer_classifier.py` | 30 | `PrimaryTransformerClassifier` interface contract (mock weights) |

### Running the tests

```bash
# All tests
python -m pytest

# Single file
python -m pytest tests/test_ner_evaluator.py -v

# NER-related tests only
python -m pytest tests/test_ner_agents.py tests/test_ner_consensus_agent.py \
    tests/test_ner_config.py tests/test_ner_evaluator.py \
    tests/test_orchestrator_ner_path.py -v

# With coverage report
python -m pytest --cov=src --cov-report=term-missing
```

---

## 13. Entry Points

### `evaluate_pipeline.py` — Main CLI

```
usage: evaluate_pipeline.py [-h]
    --config CONFIG
    --active_task {topic_classification,sentiment_classification,ner}
    --dataset DATASET
    --mode {full_pipeline,ablation}
    --pipeline_mode {primary_only,paper_style,full_agentic}
    [--deliberation]
    [--output_dir OUTPUT_DIR]
    [--run_id RUN_ID]
    [--verbose]
```

**Routing inside `main()`:**
1. Load `TaskConfig` from config + `--active_task`
2. If `task_type == "classification"` → `load_classification_dataset()` → `_run_classification_evaluation()` (or ablation)
3. If `task_type == "sequence_labeling"` → `load_sequence_labeling_dataset()` → `_run_ner_evaluation()`
4. Ablation mode is rejected for NER (returns exit code 1 with a clear message)

**Example — topic classification:**
```bash
python evaluate_pipeline.py \
    --config src/config/default.yaml \
    --active_task topic_classification \
    --dataset data/dev_dummy.jsonl \
    --mode full_pipeline \
    --pipeline_mode paper_style \
    --output_dir results/topic_paper_style
```

**Example — NER:**
```bash
python evaluate_pipeline.py \
    --config src/config/default.yaml \
    --active_task ner \
    --dataset data/dev_dummy_ner.jsonl \
    --mode full_pipeline \
    --pipeline_mode paper_style \
    --output_dir results/ner_smoke
```

### `run_demo.py` — Quick smoke demo

Runs all three pipeline modes on a single hard-coded sentence and prints the result for each mode. No arguments required.

### `scripts/`

| Script | Purpose |
|---|---|
| `generate_dummy_data.py` | Generates `data/dev_dummy.jsonl` (30 topic examples) |
| `generate_dummy_sentiment_data.py` | Generates `data/dev_dummy_sentiment.jsonl` (30 sentiment examples) |
| `generate_dummy_ner_data.py` | Generates `data/dev_dummy_ner.jsonl` (30 NER BIO examples); includes `_validate()` assertions |
| `debug_run_modes.py` | Runs pipeline in all three modes and prints per-sample trace. Accepts `--config`, `--active_task`, `--dataset`. |
| `inspect_pipeline_modes.py` | Instantiates an orchestrator per mode and prints the concrete class wired to each role. |
| `build_keyword_map.py` | Data-driven keyword map builder; extracts high-PMI tokens per label from a labelled JSONL. |

---

## 14. Current Limitations

| # | Area | Limitation |
|---|---|---|
| 1 | **Primary classifier** | `MockPrimaryClassifier` uses keyword heuristics. `PrimaryTransformerClassifier` exists in `src/models/` but is not wired to real mBERT/XLM-R weights, and not yet used by `build_orchestrator()`. |
| 2 | **NER agents** | All three NER specialist agents (Lexical, Logic, Contextual) are deterministic heuristics — no real transformer-based NER model. Results on real data will be low-quality. |
| 3 | **NER evaluation** | `NEREvaluator` computes **token-level** metrics only. Span-level (entity-boundary-aware) CoNLL-style F1 is not yet implemented. |
| 4 | **LLM client** | Only `MockLLMClient` is implemented. No real OpenAI/Claude/Ollama integration; `full_agentic` mode runs deterministically against mock responses. |
| 5 | **Deliberation (NER)** | `DeliberationAgent` is a classification-only component; no NER deliberation step exists yet. |
| 6 | **`legacy/`** | Isolated original codebase; not imported by any active code. Kept for reference only. |

---

## 15. Next Steps

### Immediate priority
1. **Wire `PrimaryTransformerClassifier` into `build_orchestrator()`** in `evaluate_pipeline.py`. Load a real mBERT or XLM-R checkpoint. Re-evaluate on `dev_dummy.jsonl` and `dev_dummy_sentiment.jsonl`.

```bash
python evaluate_pipeline.py \
    --config src/config/default.yaml \
    --active_task topic_classification \
    --dataset data/dev_dummy.jsonl \
    --mode full_pipeline \
    --pipeline_mode full_agentic \
    --deliberation \
    --output_dir results/real_mbert_topic
```

### Near-term
2. **Span-level NER F1** — implement entity-boundary-aware evaluation in `NEREvaluator`. Use IOB2 span extraction for CoNLL-style precision/recall/F1.
3. **Real LLM client** — implement `OpenAIClient` or `OllamaClient` extending `BaseLLMClient`. Inject via `build_orchestrator()` kwargs.
4. **NER primary model** — integrate a real transformer NER model (e.g., `CAMeL-Lab/bert-base-arabic-camelbert-mix-ner`) as `NERPrimaryClassifier`; wire into NER path's `primary_only` mode.
5. **Ablation for NER** — enable ablation mode in `evaluate_pipeline.main()` for `sequence_labeling` tasks.

---

*Generated automatically from codebase analysis — 2026-05-05*
