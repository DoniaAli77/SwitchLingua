# Development Status

**Date:** 2026-05-05  
**Test suite:** 830 passed, 0 failed  
**Status:** Level 2 NER smoke support complete — ready for real primary transformer integration

---

## Supported Tasks

| Task | Task Type | Labels |
|---|---|---|
| `topic_classification` | `classification` | business, education, health, shopping, medical, sports, tech, finance, social |
| `sentiment_classification` | `classification` | positive, negative, neutral |
| `ner` | `sequence_labeling` | O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC |

All tasks are defined in `src/config/default.yaml` under `tasks:` and loaded at runtime by `src/config/loader.py`. Switching tasks requires no code change — pass `--active_task <name>` to any entry point.

---

## Supported Pipeline Modes

| Mode | What runs |
|---|---|
| `primary_only` | Primary classifier only. Router and all specialist agents are skipped. |
| `paper_style` | Primary → Router → specialist agents (see per-path details below) → ConsensusAgent → ExplainabilityAgent |
| `full_agentic` | Primary → Router → LLM-backed agents → (optional DeliberationAgent) → ConsensusAgent → LLMExplainabilityAgent |

---

## Pipeline Paths by Task Type

### Classification path (`task_type == "classification"`)

```
Input text + TaskConfig
        │
        ▼
┌─────────────────────────┐
│   PrimaryClassifier     │  (MockPrimaryClassifier / PrimaryTransformerClassifier)
│   confidence, label     │
└────────────┬────────────┘
             │  confidence ≥ threshold?
     ┌───────┴────────┐
     │ YES            │ NO  (escalate)
     ▼                ▼
 final_output      Router
 (primary_only)       │
                      ▼
          ┌───────────────────────────────────┐
          │  paper_style mode                 │  full_agentic mode
          │                                   │
          │  LexicalAgent                     │  LLMLexicalAgent
          │  LogicAgent                       │  LLMLogicAgent
          │  TransformerContextualAgent        │  ContextualAgent
          │                                   │  (DeliberationAgent) ← optional
          │  ConsensusAgent                   │  ConsensusAgent
          │  ExplainabilityAgent              │  LLMExplainabilityAgent
          └────────────────┬──────────────────┘
                           ▼
                    final_output.label
                    explanation_output
```

### NER path (`task_type == "sequence_labeling"`)

```
Input tokens + TaskConfig
        │
        ▼
┌─────────────────────────┐
│   PrimaryClassifier     │  (primary_only mode only; runs stub)
└────────────┬────────────┘
             │
     ┌───────┴────────────────────┐
     │ primary_only               │ paper_style / full_agentic
     ▼                            ▼
 stub FinalOutput          NERLexicalAgent   ← gazetteer B/I tagger
 (no NER agents)           NERLogicAgent     ← regex-rule B/I tagger
                           NERContextualAgent← capitalisation + known_entities
                           NERConsensusAgent ← weighted token-level voting
                           ExplainabilityAgent
                                │
                                ▼
                 final_output.payload["sequence_output"]
                   → [{token, tag, confidence}, ...]
                 final_output.payload["token_count"]
                 final_output.label = None
```

Each agent appends an entry to `state.history`. The orchestrator adds `"orchestrator"` events at start and finish of each path.

---

## Current Primary Classifier

`MockPrimaryClassifier` (heuristic mode) — assigns labels by keyword frequency over the input text. Confidence is intentionally kept below the routing threshold (0.99 in tests) so that `paper_style` and `full_agentic` modes always escalate and all specialist agents run.

**Replacement path:** swap `MockPrimaryClassifier` for `PrimaryTransformerClassifier` (`src/models/primary_transformer_classifier.py`) in `build_orchestrator()` inside `evaluate_pipeline.py`. No other changes required.

---

## Smoke Datasets

| File | Task | Size | Balance |
|---|---|---|---|
| `data/dev_dummy.jsonl` | `topic_classification` | 30 examples | ~3-4 per label across 9 labels |
| `data/dev_dummy_sentiment.jsonl` | `sentiment_classification` | 30 examples | 10 positive / 10 negative / 10 neutral |
| `data/dev_dummy_ner.jsonl` | `ner` | 30 examples | O:152, B-PER:16, I-PER:8, B-ORG:22, I-ORG:10, B-LOC:29, I-LOC:7 tokens |

All datasets contain short Arabic-English code-switched sentences. Used for framework smoke testing only — not for accuracy benchmarking.

---

## Current Limitations

- **Primary classifier is still mock.** `MockPrimaryClassifier` uses keyword heuristics; `PrimaryTransformerClassifier` exists in `src/models/` but is not yet wired into `build_orchestrator()`.
- **NER is operational smoke support only.** The three NER specialist agents (Lexical, Logic, Contextual) are deterministic heuristics — no real transformer-based NER model is integrated yet.
- **Entity-level NER F1 is not yet implemented.** `NEREvaluator` reports token-level accuracy and token-level per-tag P/R/F1. Span-level (entity-boundary-aware) CoNLL-style F1 is deferred to a later milestone.
- **`legacy/` is isolated.** Not imported by any active code; kept for reference only.

---

## File Purposes

### Entry Points

| File | Purpose |
|---|---|
| `evaluate_pipeline.py` | Main CLI evaluation script. Loads config, builds orchestrator, runs Evaluator over a JSONL dataset, writes predictions + metrics to `results/`. Supports `--config`, `--active_task`, `--dataset`, `--pipeline_mode`, `--threshold`, `--deliberation`, `--output_dir`. |
| `run_demo.py` | Interactive single-sample demo. Builds orchestrator, runs one hardcoded sentence through all three modes, prints formatted output. |

### `src/config/`

| File | Purpose |
|---|---|
| `default.yaml` | Single source of truth for task definitions. Declares labels, bilingual `label_descriptions`, and `label_knowledge` (keywords + regex rules) for every task. Also sets execution defaults (`pipeline_mode`, `threshold`, etc.). |
| `loader.py` | Reads `default.yaml`, resolves the active task, merges CLI overrides, builds `keyword_map` and `rule_map` from `label_knowledge`, returns a `TaskBundle` dataclass. |
| `task_config.py` | Thin dataclass helper (imported by `loader.py`). |

### `src/state/`

| File | Purpose |
|---|---|
| `schema.py` | Defines all shared data structures: `PipelineState`, `TaskConfig`, `AgentOutput`, `RoutingInfo`, `LexicalOutput`, `LogicOutput`, `ContextualOutput`, `ConsensusOutput`, `DeliberationOutput`, `ExplanationOutput`, `StateMetadata`, `HistoryEntry`. **Do not modify** — all agents and tests depend on this contract. |
| `example_state.py` | Standalone usage example for `PipelineState`. |

### `src/pipeline/`

| File | Purpose |
|---|---|
| `orchestrator.py` | `PipelineOrchestrator` — controls the full execution flow per mode. Calls the primary classifier, optionally calls the router, then dispatches to the correct set of specialist agents based on `task_config.pipeline_mode`. Writes `state.history` entries at each stage. |
| `router.py` | `Router` — compares primary classifier confidence against `task_config.threshold`. Returns `RoutingInfo(decision="accept")` or `RoutingInfo(decision="escalate")`. |

### `src/agents/`

| File | Agent | Mode | Description |
|---|---|---|---|
| `base_agent.py` | `BaseAgent` | all | Abstract base. Sets `self.name = name or self.__class__.__name__`, calls `state.append_history(component=self.name, ...)`. |
| `lexical_agent.py` | `LexicalAgent` | paper_style | Keyword lookup over `keyword_map`. Returns per-label match counts and a ranked label. |
| `llm_lexical_agent.py` | `LLMLexicalAgent` | full_agentic | LLM-backed lexical analysis. Uses `LLMLexicalPrompt` to ask the LLM for keyword-level evidence. |
| `logic_agent.py` | `LogicAgent` | paper_style | Regex rule matching over `rule_map`. Returns per-label rule hit counts and a ranked label. |
| `llm_logic_agent.py` | `LLMLogicAgent` | full_agentic | LLM-backed logic/rule analysis via structured prompt. |
| `transformer_contextual_agent.py` | `TransformerContextualAgent` | paper_style | TF-IDF cosine similarity against `label_descriptions`. No LLM call. |
| `contextual_agent.py` | `ContextualAgent` | full_agentic | LLM-backed contextual classification using `ContextualPrompt`. |
| `deliberation_agent.py` | `DeliberationAgent` | full_agentic (optional) | LLM deliberation step. Receives all specialist outputs and returns a `recommended_label` with justification. Enabled by `--deliberation` flag. |
| `consensus_agent.py` | `ConsensusAgent` | paper_style, full_agentic | Weighted voting over all specialist outputs (+ optional deliberation). Produces the final `label` and `confidence`. |
| `explainability_agent.py` | `ExplainabilityAgent` | paper_style | Template-based explanation. Summarises agent votes without an LLM call. |
| `llm_explainability_agent.py` | `LLMExplainabilityAgent` | full_agentic | LLM-generated natural-language explanation of the final decision. |
| `ner_lexical_agent.py` | `NERLexicalAgent` | NER path | Gazetteer-based BIO tagger. Case-insensitive dict lookup; emits B/I continuation tags. |
| `ner_logic_agent.py` | `NERLogicAgent` | NER path | Regex-rule BIO tagger. Per-token `re.fullmatch` / word-boundary search; bad patterns skipped with warning. |
| `ner_contextual_agent.py` | `NERContextualAgent` | NER path | Heuristic contextual BIO tagger. `known_entities` overrides → capitalisation heuristic for Latin tokens (→ B-PER). |
| `ner_consensus_agent.py` | `NERConsensusAgent` | NER path | Weighted token-level voting across lexical/logic/contextual outputs. Stores tags in `final_output.payload["sequence_output"]`; `final_output.label = None`. |

### `src/models/`

| File | Purpose |
|---|---|
| `mock_primary_classifier.py` | `MockPrimaryClassifier` — heuristic keyword scorer used in all tests and smoke evals. Writes `component="primary_classifier"` to history. |
| `primary_transformer_classifier.py` | `PrimaryTransformerClassifier` — production stub ready for real mBERT/XLM-R weights. Loads a HuggingFace checkpoint, runs tokenisation + forward pass, returns a `ClassifierOutput`. **Not yet wired into `build_orchestrator()`**. |

### `src/llm/`

| File | Purpose |
|---|---|
| `base_client.py` | `BaseLLMClient` abstract class. Defines `complete(prompt) → str`. |
| `mock_client.py` | `MockLLMClient` — three modes: `label_echo` (extract a valid label from prompt text), `fixed` (return a preset JSON string, used by `DeliberationAgent`), `random`. Zero external calls. |

### `src/prompts/`

| File | Purpose |
|---|---|
| `contextual_prompt.py` | Builds the few-shot contextual classification prompt for `ContextualAgent`. |
| `deliberation_prompt.py` | Builds the structured deliberation prompt for `DeliberationAgent`. Expects JSON response. |
| `llm_explainability_prompt.py` | Builds the explanation generation prompt for `LLMExplainabilityAgent`. |
| `llm_lexical_prompt.py` | Builds the keyword-evidence prompt for `LLMLexicalAgent`. |
| `llm_logic_prompt.py` | Builds the rule-evidence prompt for `LLMLogicAgent`. |

### `src/evaluation/`

| File | Purpose |
|---|---|
| `evaluator.py` | `Evaluator` — runs the orchestrator over a full dataset, collects per-sample predictions, computes accuracy / macro-F1 / per-label precision-recall-F1, writes `predictions.json`, `predictions.csv`, `metrics.json`, `metrics.csv` to the output directory. |
| `ner_evaluator.py` | `NEREvaluator` — NER-specific evaluator. Computes token accuracy, macro F1, and per-tag precision/recall/F1/support. Writes `*_ner_predictions.json`, `*_ner_predictions.csv`, `*_ner_metrics.json`, `*_ner_metrics.csv`. Prediction rows include `id`, `text`, `tokens`, `gold_tags`, `predicted_tags`. |
| `ablation.py` | `AblationStudy` / `AblationConfig` / `AblationReport` — drives multi-run ablation experiments from a YAML/JSON config. Each run can toggle individual agents on/off or vary consensus weights. |

### `scripts/`

| File | Purpose |
|---|---|
| `generate_dummy_data.py` | Generates `data/dev_dummy.jsonl` (30 topic_classification examples). |
| `generate_dummy_sentiment_data.py` | Generates `data/dev_dummy_sentiment.jsonl` (30 sentiment_classification examples). |
| `generate_dummy_ner_data.py` | Generates `data/dev_dummy_ner.jsonl` (30 NER examples with BIO tags). Includes internal `_validate()` assertions. |
| `debug_run_modes.py` | Runs the pipeline in all three modes on a dataset and prints per-sample mode/label/routing/history. Accepts `--config`, `--active_task`, `--dataset`. Default: topic_classification / dev_dummy.jsonl. |
| `inspect_pipeline_modes.py` | Instantiates an orchestrator for each mode, prints the concrete class wired to each role. Useful for verifying agent wiring without running data. |
| `build_keyword_map.py` | Data-driven keyword map builder. Reads a labelled JSONL file and extracts high-PMI tokens per label. Output replaces the hand-crafted maps in `default.yaml`. |

### `tests/`

| File | Tests | What it covers |
|---|---|---|
| `test_pipeline_modes_smoke.py` | 67 | End-to-end integrity for all three modes: correct outputs, correct agents in history, wrong-mode agents absent. |
| `test_task_aware_config_modes.py` | 35 | Config loading and pipeline runs for both classification tasks; cross-task label contamination checks. |
| `test_config_loader.py` | 37 | `load_task_bundle`: label loading, keyword/rule map construction, override precedence, error handling. |
| `test_dataset_loading.py` | 27 | `load_classification_dataset`, `load_sequence_labeling_dataset` (validation of tokens/tags length and tag membership), `load_dataset` alias. |
| `test_ner_agents.py` | 54 | `NERLexicalAgent`, `NERLogicAgent`, `NERContextualAgent`: tagging behaviour, B/I continuation, unknown tokens, skip when task_type != sequence_labeling. |
| `test_ner_consensus_agent.py` | 44 | `NERConsensusAgent`: weighted voting, tie-breaking, `_vote_token`, `_extract_seq_output`, payload structure. |
| `test_ner_config.py` | 15 | NER config loading via `load_task_bundle`; evaluate_pipeline dataset-format guard. |
| `test_orchestrator_ner_path.py` | 32 | NER orchestrator path for all three modes; custom agent injection; error capture; classification-path regression. |
| `test_ner_evaluator.py` | 53 | `NEREvaluator`: token accuracy, per-tag metrics, macro F1, prediction extraction, error handling, all 4 file outputs, `evaluate_pipeline.main()` NER routing. |
| `test_orchestrator_flow.py` | — | Orchestrator routing logic, escalation paths, primary-accept paths. |
| `test_lexical_agent.py` | — | `LexicalAgent` keyword scoring, empty text, unknown labels. |
| `test_logic_agent.py` | — | `LogicAgent` regex rule matching, pattern edge cases. |
| `test_contextual_agent.py` | — | `ContextualAgent` LLM prompt construction and output parsing. |
| `test_transformer_contextual_agent.py` | — | `TransformerContextualAgent` TF-IDF similarity logic. |
| `test_consensus_agent.py` | — | `ConsensusAgent` weighted voting, tie-breaking, deliberation weight. |
| `test_deliberation_agent.py` | — | `DeliberationAgent` prompt, JSON response parsing, fallback behaviour. |
| `test_explainability_agent.py` | — | `ExplainabilityAgent` template output structure. |
| `test_llm_explainability_agent.py` | — | `LLMExplainabilityAgent` prompt and output parsing. |
| `test_llm_specialist_agents.py` | — | `LLMLexicalAgent` and `LLMLogicAgent` prompt and output parsing. |
| `test_primary_transformer_classifier.py` | — | `PrimaryTransformerClassifier` interface contract (mock weights). |
| `test_evaluator.py` | — | `Evaluator` metric computation, file output, error counting. |
| `test_ablation.py` | — | `AblationStudy` multi-run orchestration, report aggregation. |

### `legacy/`

Isolated copy of the original flat-layout codebase. **Not imported by any active code.** Kept for reference only. Contains `agents/`, `pipeline/`, `prompts/`, `models/`, `utils/`, `config/`, and two legacy test files under `legacy/tests/`. Safe to delete once real transformer integration is complete.

---

## Next Step

Replace `MockPrimaryClassifier` with `PrimaryTransformerClassifier` loaded from a real mBERT or XLM-R checkpoint, then re-run:

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