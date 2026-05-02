# Multi-Agent BERT Pipeline — Full Technical Report

**Project:** `multi-agent-bert`  
**Date:** April 21, 2026  
**Test suite:** 199 passed, 0 failed  

---

## Table of Contents

1. [Overview](#1-overview)
2. [Directory Structure](#2-directory-structure)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Execution Flows](#4-execution-flows)
   - 4.1 [Fast Path (accept_primary)](#41-fast-path-accept_primary)
   - 4.2 [Escalation Path (standard)](#42-escalation-path-standard)
   - 4.3 [Escalation Path (with deliberation)](#43-escalation-path-with-deliberation)
5. [Shared State Schema](#5-shared-state-schema)
6. [Components](#6-components)
   - 6.1 [Primary Classifier](#61-primary-classifier)
   - 6.2 [Router](#62-router)
   - 6.3 [LexicalAgent](#63-lexicalagent)
   - 6.4 [LogicAgent](#64-logicagent)
   - 6.5 [ContextualAgent](#65-contextualagent)
   - 6.6 [DeliberationAgent](#66-deliberationagent)
   - 6.7 [ConsensusAgent](#67-consensusagent)
   - 6.8 [ExplainabilityAgent](#68-explainabilityagent)
   - 6.9 [PipelineOrchestrator](#69-pipelineorchestrator)
7. [LLM Client Interface](#7-llm-client-interface)
8. [Prompt System](#8-prompt-system)
9. [Configuration](#9-configuration)
10. [Feature Flags](#10-feature-flags)
11. [Error Handling](#11-error-handling)
12. [Execution History](#12-execution-history)
13. [Test Coverage](#13-test-coverage)
14. [Design Decisions & Tradeoffs](#14-design-decisions--tradeoffs)
15. [Extension Guide](#15-extension-guide)

---

## 1. Overview

This is a **stateful, multi-agent text-classification pipeline** designed for code-switched text (e.g., Arabic–English). A single input text passes through a sequence of agents, each writing its results into a shared `PipelineState` object. The pipeline supports two task types:

- **`classification`** — assigns one label to the whole input (e.g., sentiment, topic)
- **`sequence_labeling`** — assigns one tag per token (e.g., NER with BIO tags)

The system is designed to be **backend-agnostic** (any LLM can be plugged in), **fully testable without a network** (via `MockLLMClient`), and **observable** (every stage appends a structured history event to state).

---

## 2. Directory Structure

```
multi-agent-bert/
├── config/
│   └── default.yaml              # Task registry + execution flags
├── src/
│   ├── agents/
│   │   ├── base_agent.py         # Abstract BaseAgent[StateT]
│   │   ├── lexical_agent.py      # Keyword-based scorer
│   │   ├── logic_agent.py        # Regex-rule scorer
│   │   ├── contextual_agent.py   # LLM-backed classifier
│   │   ├── deliberation_agent.py # LLM-backed vote reviewer (optional)
│   │   ├── consensus_agent.py    # Weighted-voting aggregator
│   │   └── explainability_agent.py # Template-based explainer
│   ├── llm/
│   │   ├── base_client.py        # LLMClient ABC + LLMClientError
│   │   └── mock_client.py        # MockLLMClient (fixed / label_echo / raise)
│   ├── models/
│   │   └── mock_primary_classifier.py  # MockPrimaryClassifier
│   ├── pipeline/
│   │   ├── orchestrator.py       # PipelineOrchestrator — wires everything
│   │   └── router.py             # Router — confidence-threshold gate
│   ├── prompts/
│   │   ├── contextual_prompt.py  # ContextualAgent prompt templates
│   │   └── deliberation_prompt.py # DeliberationAgent prompt templates
│   └── state/
│       └── schema.py             # All shared dataclasses (PipelineState, etc.)
├── tests/
│   ├── test_agents_interfaces.py
│   ├── test_consensus_agent.py
│   ├── test_contextual_agent.py
│   ├── test_deliberation_agent.py
│   ├── test_explainability_agent.py
│   ├── test_lexical_agent.py
│   ├── test_logic_agent.py
│   ├── test_orchestrator_flow.py
│   └── test_state_models.py
├── run_demo.py                   # End-to-end demo script
├── requirements.txt
└── pyproject.toml
```

---

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PipelineOrchestrator                           │
│                                                                         │
│  Input text + TaskConfig                                                │
│        │                                                                │
│        ▼                                                                │
│  ┌─────────────────┐                                                    │
│  │ PrimaryClassifier│  → state.primary_model_output                    │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌────────────────┐                                                     │
│  │     Router     │  → state.routing_info                              │
│  └──────┬─────┬──┘                                                     │
│         │     │                                                         │
│  conf ≥ │     │ conf < threshold                                        │
│ threshold│    │ (escalate)                                              │
│         │     │                                                         │
│         ▼     ▼                                                         │
│   ┌──────┐  ┌─────────────────────────────────────────────┐            │
│   │Explain│  │  Escalation Path                            │            │
│   │Agent │  │                                             │            │
│   │(fast)│  │  ┌─────────────┐   state.lexical_output    │            │
│   └──────┘  │  │ LexicalAgent│──────────────────────────▶│            │
│             │  └─────────────┘                            │            │
│             │  ┌────────────┐    state.logic_output       │            │
│             │  │ LogicAgent │───────────────────────────▶│            │
│             │  └────────────┘                             │            │
│             │  ┌───────────────┐  state.contextual_output │            │
│             │  │ContextualAgent│─────────────────────────▶│            │
│             │  └───────────────┘  (optionally reads       │            │
│             │                      lexical+logic outputs) │            │
│             │                                             │            │
│             │  ┌──────────────────┐ [enable_deliberation] │            │
│             │  │ DeliberationAgent│ state.deliberation_   │            │
│             │  │   (optional)     │ output                │            │
│             │  └──────────────────┘                       │            │
│             │  ┌──────────────┐   state.consensus_output  │            │
│             │  │ ConsensusAgent│  state.final_output      │            │
│             │  └──────────────┘                           │            │
│             │  ┌──────────────┐   state.explanation_output│            │
│             │  │  Explain     │                           │            │
│             │  │  Agent(full) │                           │            │
│             │  └──────────────┘                           │            │
│             └─────────────────────────────────────────────┘            │
│                                                                         │
│  Output: state.final_output  (label + confidence)                      │
│          state.history       (full audit trail)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Execution Flows

### 4.1 Fast Path (`accept_primary`)

Triggered when the primary model's confidence is **≥ `task_config.threshold`**.

```
PrimaryClassifier
      │
      ▼
   Router  →  decision = "accept_primary"
      │
      ▼
 state.final_output ← primary label + confidence
      │
      ▼
 ExplainabilityAgent  (short explanation: "accepted primary, no escalation")
      │
      ▼
  DONE
```

**State fields written:**
| Field | Writer |
|---|---|
| `primary_model_output` | PrimaryClassifier |
| `routing_info` | Router |
| `final_output` | Router |
| `explanation_output` | ExplainabilityAgent |

---

### 4.2 Escalation Path (standard)

Triggered when primary confidence is **< `task_config.threshold`**.  
`enable_deliberation = False` (default).

```
PrimaryClassifier
      │
      ▼
   Router  →  decision = "escalate"
      │
      ├──▶ LexicalAgent       → state.lexical_output
      │
      ├──▶ LogicAgent         → state.logic_output
      │
      ├──▶ ContextualAgent    → state.contextual_output
      │       (optionally reads primary/lexical/logic when
      │        contextual_use_prior_outputs = true)
      │
      ├──▶ ConsensusAgent     → state.consensus_output
      │                          state.final_output
      │
      └──▶ ExplainabilityAgent → state.explanation_output
```

**State fields written:**
| Field | Writer |
|---|---|
| `primary_model_output` | PrimaryClassifier |
| `routing_info` | Router |
| `lexical_output` | LexicalAgent |
| `logic_output` | LogicAgent |
| `contextual_output` | ContextualAgent |
| `consensus_output` | ConsensusAgent |
| `final_output` | ConsensusAgent |
| `explanation_output` | ExplainabilityAgent |

---

### 4.3 Escalation Path (with deliberation)

Triggered when `enable_deliberation = True` **and** a `DeliberationAgent` instance is wired into the orchestrator.

```
... (same as 4.2 through ContextualAgent)
      │
      ├──▶ DeliberationAgent  → state.deliberation_output
      │       (reads lexical + contextual + logic outputs,
      │        produces recommended_label + justification)
      │
      ├──▶ ConsensusAgent     → state.consensus_output
      │       (includes deliberation as an extra weighted vote
      │        when ConsensusAgent.weights["deliberation"] > 0)
      │
      └──▶ ExplainabilityAgent → state.explanation_output
```

**Additional state fields:**
| Field | Writer |
|---|---|
| `deliberation_output` | DeliberationAgent |

---

## 5. Shared State Schema

All components communicate exclusively through `PipelineState`. No agent holds inter-call state.

```python
@dataclass(slots=True)
class PipelineState:
    # --- Required at construction ---
    metadata: StateMetadata          # sample_id, timestamp
    input_text: str                  # raw input to classify
    task_config: TaskConfig          # labels, threshold, flags

    # --- Written by pipeline stages ---
    primary_model_output: ModelOutput          = ModelOutput()
    routing_info:         Optional[RoutingInfo]          = None
    lexical_output:       Optional[AgentOutput]          = None
    contextual_output:    Optional[AgentOutput]          = None
    logic_output:         Optional[AgentOutput]          = None
    deliberation_output:  Optional[DeliberationOutput]   = None
    consensus_output:     Optional[ConsensusOutput]      = None
    explanation_output:   Optional[ExplanationOutput]    = None
    final_output:         Optional[FinalOutput]          = None

    # --- Cross-cutting concerns ---
    extras:  Dict[str, Any]        = {}   # pipeline_error stored here on failure
    history: List[HistoryEvent]    = []   # structured audit trail
```

### Key Dataclasses

| Class | Purpose | Key Fields |
|---|---|---|
| `StateMetadata` | Sample identity | `sample_id`, `timestamp` |
| `TaskConfig` | Task parameters & flags | `task_name`, `task_type`, `labels`, `label_descriptions`, `threshold`, `contextual_use_prior_outputs`, `enable_deliberation` |
| `ModelOutput` | Generic model prediction | `label`, `confidence`, `probabilities`, `entities`, `raw_text` |
| `AgentOutput` | Specialist agent wrapper | `agent_name`, `model_output`, `sequence_output`, `notes`, `features` |
| `RoutingInfo` | Router decision | `threshold`, `decision` |
| `DeliberationOutput` | Deliberation result | `recommended_label`, `confidence`, `justification`, `mode` |
| `ConsensusOutput` | Merged vote | `label`, `confidence`, `votes`, `rationale` |
| `ExplanationOutput` | Human-readable explanation | `summary`, `evidence`, `caveats` |
| `FinalOutput` | User-facing result | `label`, `confidence`, `payload` |
| `HistoryEvent` | Single audit entry | `component`, `timestamp`, `summary`, `outputs` |

---

## 6. Components

### 6.1 Primary Classifier

**File:** `src/models/mock_primary_classifier.py`  
**Class:** `MockPrimaryClassifier`  

The first stage. Runs the primary model and writes `state.primary_model_output`. In the mock implementation, three modes are supported:

| Mode | Behaviour |
|---|---|
| `"fixed"` | Always returns a configured label + confidence |
| `"random"` | Randomly picks a label with random confidence |
| `"heuristic"` | Keyword-assisted prediction from `keyword_label_map` |

The real production implementation replaces this with any model whose `run(state) -> state` signature is compatible (e.g. a fine-tuned BERT).

**State writes:** `state.primary_model_output`  
**Validation:** Label checked against `task_config.labels`; confidence must be `[0, 1]`.

---

### 6.2 Router

**File:** `src/pipeline/router.py`  
**Class:** `Router`  

Compares `primary_model_output.confidence` against `task_config.threshold`.

| Condition | Decision | Effect |
|---|---|---|
| `confidence ≥ threshold` | `"accept_primary"` | `state.final_output` written immediately |
| `confidence < threshold` | `"escalate"` | Specialist agents are invoked |

**State writes:** `state.routing_info`, conditionally `state.final_output`  
**Raises:** `ValueError` if primary output is missing or label is invalid.

---

### 6.3 LexicalAgent

**File:** `src/agents/lexical_agent.py`  
**Class:** `LexicalAgent`  

Keyword-based label scorer. For each label in `task_config.labels`, counts how many configured keywords appear in the input text. Normalises raw hit counts into a probability distribution.

**Keyword matching rules:**
- **ASCII keywords** → whole-word, case-insensitive regex (`\bkeyword\b`)
- **Arabic / non-ASCII / multi-word** → substring search (handles connected script)

**No-match fallback:** When zero keywords match, returns uniform probabilities (`1 / |labels|`) and assigns `labels[0]` with low confidence.

**Constructor parameter:** `keyword_map: Dict[str, List[str]]`  
**State writes:** `state.lexical_output: AgentOutput`  

---

### 6.4 LogicAgent

**File:** `src/agents/logic_agent.py`  
**Class:** `LogicAgent`  

Regex-rule based label scorer. Each rule that matches contributes one vote to its label. Patterns are compiled once at construction with `re.IGNORECASE | re.UNICODE`.

**Invalid pattern handling:** Patterns that fail to compile are skipped with a warning rather than crashing.

**No-match fallback:** Same uniform distribution strategy as `LexicalAgent`.

**Constructor parameter:** `rule_map: Dict[str, List[str]]` (patterns, not keywords)  
**State writes:** `state.logic_output: AgentOutput`  

---

### 6.5 ContextualAgent

**File:** `src/agents/contextual_agent.py`  
**Class:** `ContextualAgent`  

LLM-backed classifier. Uses `string.Template`-based prompt construction from `src/prompts/contextual_prompt.py`.

**LLM Response contract** (strict JSON, no fences):
```json
{
  "label":      "<one of the allowed labels>",
  "confidence": 0.87,
  "reasoning":  "<one sentence>",
  "evidence":   ["<short phrase>"]
}
```

**Inter-agent awareness (optional):** When `task_config.contextual_use_prior_outputs = True`, the agent calls `_build_prior_summaries(state)` and injects compact summaries of the primary, lexical, and logic results into the prompt as weak hints. Notes are sanitized (whitespace normalized, truncated at 160 chars) to prevent prompt injection.

**Error handling:**
- `ContextualParseError` → caught, low-confidence fallback written, history event appended
- `LLMClientError` → re-raised for the orchestrator to handle

**Constructor parameter:** `llm_client: LLMClient`  
**State writes:** `state.contextual_output: AgentOutput`  

---

### 6.6 DeliberationAgent

**File:** `src/agents/deliberation_agent.py`  
**Class:** `DeliberationAgent`  

Optional LLM-backed deliberation stage. Reads all three specialist agent outputs and asks the LLM to review the votes and produce a reconciled judgment.

**LLM Response contract** (strict JSON, no fences):
```json
{
  "recommended_label": "<one of the allowed labels>",
  "confidence":        0.82,
  "justification":     "<one or two sentences>",
  "mode":              "recommendation"
}
```

`mode` must be one of:
- `"recommendation"` — agent clearly favours one label
- `"justification"` — agent explains why the majority view holds

**Vote collection:** Only `(slot_name, label, confidence, notes)` are forwarded; raw input text and full LLM responses are never passed. Notes are sanitized and truncated at 120 chars.

**Error handling:** `DeliberationParseError` is caught internally; on failure `state.deliberation_output` remains `None` and a history event is appended. The pipeline continues to consensus.

**Activation gate:** The orchestrator checks `state.task_config.enable_deliberation and self._deliberation is not None` before running this stage.

**Constructor parameter:** `llm_client: LLMClient`  
**State writes:** `state.deliberation_output: DeliberationOutput` (or `None` on parse error)  

---

### 6.7 ConsensusAgent

**File:** `src/agents/consensus_agent.py`  
**Class:** `ConsensusAgent`  

Weighted-voting aggregator. For each active specialist slot, computes:

```
score[label] += weight[slot] × confidence[slot]
```

The label with the highest accumulated score wins. Ties are broken deterministically by the order labels appear in `task_config.labels`.

**Slots and default weights:**

| Slot | Default Weight | Notes |
|---|---|---|
| `lexical` | `1.0` | Always active |
| `contextual` | `1.0` | Always active |
| `logic` | `1.0` | Always active |
| `deliberation` | `0.0` | **Off by default** — set > 0 to include |

**Deliberation integration:** When `weights["deliberation"] > 0` and `state.deliberation_output` contains a valid label + confidence, the deliberation vote is injected into the score accumulation exactly like a specialist agent. It appears in the `rationale` string.

**No-vote fallback:** When no agent produces a usable vote (all `None` or all weight `0`), assigns the first label with uniform confidence (`1 / |labels|`).

**Constructor parameter:** `weights: Optional[Dict[str, float]]`  
**State writes:** `state.consensus_output: ConsensusOutput`, `state.final_output: FinalOutput`  

---

### 6.8 ExplainabilityAgent

**File:** `src/agents/explainability_agent.py`  
**Class:** `ExplainabilityAgent`  

Template-based (no LLM). Generates a human-readable explanation of how the final label was reached. Two code paths:

**Fast path (primary accepted):**
> "Primary model predicted 'positive' with 85.0% confidence (threshold: 60.0%). No specialist agents were consulted."

**Escalation path:**
- Lists each specialist agent that voted for the winning label with its confidence and notes
- Flags dissenting agents in `caveats`
- Produces a multi-line `evidence` list

**State writes:** `state.explanation_output: ExplanationOutput`  

---

### 6.9 PipelineOrchestrator

**File:** `src/pipeline/orchestrator.py`  
**Class:** `PipelineOrchestrator`  

Wires all components and executes them in order. Each stage is wrapped by `_run_stage()` which:
- Logs start/end
- Catches any unhandled exception
- On exception: writes `state.extras["pipeline_error"]`, appends a history event, returns `(state, False)`
- Returns `(state, ok)` — if `ok=False`, the orchestrator returns immediately, preserving all upstream results

**Constructor parameters:**

| Parameter | Type | Required |
|---|---|---|
| `primary_classifier` | `MockPrimaryClassifier` | Yes |
| `router` | `Router` | Yes |
| `lexical_agent` | `LexicalAgent` | Yes |
| `contextual_agent` | `ContextualAgent` | Yes |
| `logic_agent` | `LogicAgent` | Yes |
| `consensus_agent` | `ConsensusAgent` | Yes |
| `explainability_agent` | `ExplainabilityAgent` | Yes |
| `deliberation_agent` | `Optional[DeliberationAgent]` | No (`None`) |
| `logger` | `Optional[logging.Logger]` | No |

---

## 7. LLM Client Interface

**File:** `src/llm/base_client.py`

```python
class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...

class LLMClientError(RuntimeError): ...
```

Any backend (OpenAI, Anthropic, Ollama, etc.) implements `LLMClient`. The pipeline never imports a concrete backend directly — all agents depend only on the abstract interface.

### MockLLMClient

**File:** `src/llm/mock_client.py`  
**Modes:**

| Mode | Behaviour |
|---|---|
| `"fixed"` | Returns the same hard-coded JSON string every call |
| `"label_echo"` | Detects which label appears first in the prompt, returns valid JSON for it |
| `"raise_on_call"` | Raises `LLMClientError` on every call (tests error-handling paths) |

Optional `call_log: List[str]` parameter captures every prompt for assertions.

---

## 8. Prompt System

Both LLM-backed agents keep their prompts in dedicated modules separate from agent logic.

### ContextualAgent Prompt (`src/prompts/contextual_prompt.py`)

Built with Python `string.Template`. Key design properties:
- **Label-locked:** allowed labels embedded verbatim
- **Schema-first:** exact expected JSON schema shown in both system and user prompt
- **Prior context block:** when `contextual_use_prior_outputs = True`, a `PRIOR AGENT SUMMARIES` block is injected as "weak hints" — explicitly framed as informational, not authoritative

### DeliberationAgent Prompt (`src/prompts/deliberation_prompt.py`)

Built with Python `string.Template`. Key design properties:
- **Vote-aware:** each specialist agent's label/confidence/notes is listed in an `AGENT VOTES` block
- **Mode-aware:** system prompt instructs the model to use `"recommendation"` vs `"justification"` to signal certainty
- Empty vote list renders `"(no agent votes available)"` as a safe fallback text

---

## 9. Configuration

**File:** `config/default.yaml`

```yaml
pipeline:
  name: stateful_multi_agent_classifier
  version: 0.1.0

active_task: sentiment_classification

tasks:
  sentiment_classification:
    task_type: classification
    labels: [positive, negative, neutral]
    label_descriptions: { ... }
    label_knowledge:         # keyword/regex lists per label, per language
      positive:
        keywords_l1: []      # English keywords
        keywords_l2: []      # Arabic keywords
        regex_rules: []

  topic_classification:
    task_type: classification
    labels: [tech, sports, politics, health, other]
    ...

  ner:
    task_type: sequence_labeling
    labels: [O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC]
    ...

language_pair:
  pair_name: en-ar
  l1: en
  l2: ar
  code_switching_allowed: true

execution:
  threshold: 0.6
  verbose: false
  output_format: jsonl
  enable_deliberation: false
  deliberation_weight: 1.5
  contextual_use_prior_outputs: false
```

---

## 10. Feature Flags

| Flag | Location | Default | Effect when `true` |
|---|---|---|---|
| `contextual_use_prior_outputs` | `TaskConfig` / `execution:` YAML | `false` | `ContextualAgent` reads primary model, lexical, and logic outputs and injects compact summaries into its prompt as weak hints |
| `enable_deliberation` | `TaskConfig` / `execution:` YAML | `false` | Orchestrator runs `DeliberationAgent` after contextual and before consensus (only if a `deliberation_agent` instance is provided) |

Both flags default to `false` — **zero behavioral change** on existing pipelines unless explicitly enabled.

---

## 11. Error Handling

### Stage-level (orchestrator)

Every stage runs inside `_run_stage()`. On any unhandled exception:
1. Error + traceback logged at `ERROR` level
2. `state.extras["pipeline_error"] = {"stage": ..., "message": ..., "traceback": ...}`
3. History event appended with `error_type` and `error` keys
4. Pipeline returns immediately — all prior stage results are preserved

### Agent-level (parse errors)

| Agent | Error Type | Behaviour |
|---|---|---|
| `ContextualAgent` | `ContextualParseError` | Writes low-confidence fallback to `state.contextual_output`; pipeline continues |
| `DeliberationAgent` | `DeliberationParseError` | Leaves `state.deliberation_output = None`; pipeline continues to consensus |
| `LexicalAgent` | — | No parse; fallback to uniform distribution on zero matches |
| `LogicAgent` | — | No parse; fallback to uniform distribution on zero matches |
| `ConsensusAgent` | — | No parse; fallback to `labels[0]` with uniform confidence on zero usable votes |

### LLMClientError

All LLM-backed agents re-raise `LLMClientError` (network, auth, quota failures) to the orchestrator, which catches it via the `_run_stage` wrapper and halts the pipeline.

---

## 12. Execution History

Every component appends a structured `HistoryEvent` to `state.history`. The complete history is JSON-serializable via `state.get_history()`.

### History event structure

```json
{
  "component": "consensus_agent",
  "timestamp": "2026-04-21T10:30:00.000000+00:00",
  "summary": "Label 'positive' (confidence=0.812).",
  "outputs": {
    "label": "positive",
    "confidence": 0.812,
    "votes": {"positive": 2.1, "negative": 0.3, "neutral": 0.5},
    "fallback": false
  }
}
```

### Components that append history

| Component | Events appended |
|---|---|
| Orchestrator | `"Pipeline started"`, `"Pipeline finished"` |
| PrimaryClassifier | Prediction + mode |
| Router | Decision + confidence vs threshold |
| LexicalAgent | Label + evidence count |
| LogicAgent | Label + rules fired |
| ContextualAgent | Label + reasoning (or parse error) |
| DeliberationAgent | Recommendation + mode (or parse error) |
| ConsensusAgent | Final label + vote breakdown (or fallback) |
| ExplainabilityAgent | Explanation written |
| Any failing stage | Error + traceback type |

---

## 13. Test Coverage

**Total: 199 tests, 0 failures**

| Test file | Tests | What it covers |
|---|---|---|
| `test_lexical_agent.py` | ~25 | Keyword matching, Unicode/Arabic, fallback, probabilities |
| `test_logic_agent.py` | ~25 | Regex scoring, compile errors, fallback, Unicode |
| `test_contextual_agent.py` | ~40 | Happy path, parse errors, label validation, inter-agent awareness, call log |
| `test_deliberation_agent.py` | 31 | Happy path, all parse error branches, LLM error, vote collection, prompt content, consensus integration |
| `test_consensus_agent.py` | ~25 | Weights, voting math, tiebreak, deliberation slot, fallback |
| `test_explainability_agent.py` | ~20 | Fast path, escalation path, disagreement caveats |
| `test_orchestrator_flow.py` | ~20 | Fast path, escalation, deliberation gate, error propagation |
| `test_agents_interfaces.py` | ~8 | BaseAgent contract, validate hooks |
| `test_state_models.py` | ~5 | Dataclass construction, `append_history`, `get_history` |

---

## 14. Design Decisions & Tradeoffs

### Single shared mutable state
All agents write to the same `PipelineState` object rather than returning new states. This simplifies wiring (no result threading) and makes the audit history a first-class citizen. The tradeoff is that agents must not read outputs they haven't been told to depend on.

### Strict JSON response contracts for LLM agents
Every LLM-backed agent requires exact JSON with a fixed key set. This makes parsing deterministic and easy to test, at the cost of requiring the model to follow a strict format (which can fail). Fallback paths handle parse failures gracefully.

### `slots=True` on all dataclasses
Enables attribute access validation at class definition time and slightly reduces per-instance memory. The tradeoff is that adding fields post-hoc requires editing the class definition — no dynamic attributes.

### Deliberation off by default (`weight = 0`)
The `"deliberation"` slot in `ConsensusAgent.weights` defaults to `0.0`, not `1.5`. This means even if a `DeliberationAgent` runs and writes output, it has zero effect on consensus unless the caller explicitly sets the weight. This makes the default behavior identical to the pre-deliberation baseline, preventing surprises.

### Orchestrator-level gate for deliberation
The `enable_deliberation` flag check lives in the orchestrator, not inside `DeliberationAgent.run()`. This keeps the agent itself unconditional and easier to unit test — the agent always executes when called; the orchestrator decides whether to call it.

### Prompt injection safety
Notes injected into LLM prompts (for inter-agent awareness and deliberation) are:
- Whitespace-normalized (multiple spaces/newlines collapsed)
- Truncated at a fixed character limit (160 chars for contextual, 120 chars for deliberation)
- Framed as "weak hints" / "context only" to minimize over-reliance

---

## 15. Extension Guide

### Add a new task

1. Add a new entry under `tasks:` in `config/default.yaml` with `task_type`, `labels`, `label_descriptions`, and optionally `label_knowledge`.
2. No code changes required — task config is consumed generically.

### Add a real LLM backend

1. Implement `src/llm/base_client.py:LLMClient`:
   ```python
   class OpenAIClient(LLMClient):
       def generate(self, prompt: str) -> str:
           response = openai.chat.completions.create(...)
           return response.choices[0].message.content
   ```
2. Inject it into `ContextualAgent` and/or `DeliberationAgent` at construction time.

### Add a new specialist agent

1. Subclass `BaseAgent[PipelineState]` and implement `run(state) -> state`.
2. Add a new `Optional[AgentOutput]` field to `PipelineState`.
3. Add the new agent slot to `ConsensusAgent._DEFAULT_WEIGHTS`.
4. Wire the agent into `PipelineOrchestrator.__init__` and add its stage to the escalation loop.

### Enable deliberation end-to-end

```python
from src.agents.deliberation_agent import DeliberationAgent
from src.llm.mock_client import MockLLMClient

delib_agent = DeliberationAgent(llm_client=MockLLMClient(mode="fixed", fixed_response=...))
consensus = ConsensusAgent(weights={"deliberation": 1.5})

orchestrator = PipelineOrchestrator(
    ...,
    deliberation_agent=delib_agent,
    consensus_agent=consensus,
)

state.task_config.enable_deliberation = True
```

### Enable inter-agent awareness for ContextualAgent

```python
state.task_config.contextual_use_prior_outputs = True
# No other changes needed — ContextualAgent reads the flag from state at runtime.
```
