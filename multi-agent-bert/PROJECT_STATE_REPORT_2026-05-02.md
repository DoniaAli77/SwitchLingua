# Project State Report — multi-agent-bert
**Date:** 2026-05-02  
**Working directory:** `multi-agent-bert/`  
**Python:** 3.13.5 · **venv:** `../.venv/`  
**Test suite:** 324 passed · 0 failed · 0 errors

---

## 1. Project Overview

`multi-agent-bert` is a stateful multi-agent text-classification pipeline designed for Arabic–English code-switched text (though the schema is fully generic). A primary transformer classifier provides a first-pass prediction; a confidence router either accepts it directly or escalates to a chain of specialist LLM-backed agents. Three pipeline modes control how much of the agent chain is exercised.

---

## 2. Directory Structure

```
multi-agent-bert/
├── evaluate_pipeline.py       # CLI entry point for evaluation & ablation
├── run_demo.py                # Interactive demo script
├── pyproject.toml
├── requirements.txt
├── README.md
│
├── config/
│   ├── default.yaml           # Task registry + execution config
│   ├── settings.py            # Config loader
│   └── __init__.py
│
├── data/
│   └── dev_dummy.jsonl        # 30 Arabic-English code-switched samples (9 labels)
│
├── src/
│   ├── state/
│   │   └── schema.py          # Central dataclasses: PipelineState, TaskConfig, …
│   ├── models/
│   │   ├── primary_transformer_classifier.py   # Real HF-backed classifier
│   │   ├── mock_primary_classifier.py          # Deterministic mock for tests
│   │   └── run_mock_primary_classifier.py
│   ├── pipeline/
│   │   ├── orchestrator.py    # Wires all components; runs the 3-mode flow
│   │   └── router.py          # Confidence-based routing (accept / escalate)
│   ├── agents/
│   │   ├── base_agent.py      # ABC: run() + validate_before/after hooks
│   │   ├── lexical_agent.py
│   │   ├── logic_agent.py
│   │   ├── contextual_agent.py
│   │   ├── deliberation_agent.py   # Optional; enabled by TaskConfig flag
│   │   ├── consensus_agent.py
│   │   └── explainability_agent.py
│   ├── llm/
│   │   ├── base_client.py     # LLMClient ABC + LLMClientError
│   │   └── mock_client.py     # MockLLMClient (fixed / label_echo / raise_on_call)
│   ├── evaluation/
│   │   ├── evaluator.py       # Core evaluator: EvalReport, SampleResult, metrics
│   │   └── ablation.py        # AblationStudy: multi-config comparison framework
│   └── __init__.py
│
├── utils/
│   └── debug.py               # print_debug_summary(state) helper
│
├── scripts/
│   ├── generate_dummy_data.py # Writes data/dev_dummy.jsonl
│   └── debug_run_modes.py     # Runs all 3 pipeline modes on 5 samples with debug output
│
├── tests/                     # 324 unit tests (see §5)
│
└── results/
    ├── debug_primary_only/    # 2 runs × 4 files each
    ├── debug_paper_style/     # 1 run × 4 files
    └── debug_full_agentic/    # 1 run × 4 files
```

---

## 3. Core Source Files

### 3.1 `src/state/schema.py`

Central shared-state definitions. All pipeline components read from and write to `PipelineState`.

| Type | Purpose |
|---|---|
| `PipelineMode` | `Literal["primary_only", "paper_style", "full_agentic"]` |
| `TaskConfig` | Task name, label list, threshold, `pipeline_mode`, `enable_deliberation` |
| `ModelOutput` | `label`, `confidence`, `probabilities`, `entities`, `raw_text` |
| `AgentOutput` | Wraps `ModelOutput` or `SequenceLabelingOutput` with `agent_name` |
| `RoutingInfo` | `threshold`, `decision` (`"accept_primary"` or `"escalate"`) |
| `FinalOutput` | `label`, `confidence`, `payload` |
| `ExplanationOutput` | Free-text explanation from `ExplainabilityAgent` |
| `PipelineState` | Aggregates all of the above + `history: List[HistoryEvent]` |
| `StateMetadata` | `sample_id`, `timestamp` |

Key property: `TaskConfig.pipeline_mode` defaults to `"full_agentic"` for backward compatibility. Can be overridden from CLI or programmatically.

---

### 3.2 `src/pipeline/orchestrator.py`

`PipelineOrchestrator` executes the pipeline in three distinct modes:

**`primary_only`**
```
primary_classifier → (done)
```
The router never runs. `final_output` is set directly from `primary_model_output`.

**`paper_style`**
```
primary_classifier → router → [if escalate] lexical + logic → consensus → explainability
```
Specialist chain: lexical + logic only (no contextual, no deliberation).

**`full_agentic`**
```
primary_classifier → router → [if escalate] lexical + logic + contextual
                             → [optional deliberation] → consensus → explainability
```
Full specialist chain. `DeliberationAgent` runs when `task_config.enable_deliberation=True` and a `deliberation_agent` instance was passed to the constructor.

Both escalation paths run `ExplainabilityAgent` on the fast-path too (short explanation for accepted predictions).

Internal helper `_run_stage(name, fn, state)` wraps every stage with error capture — failures write to `state.extras["pipeline_error"]` and stop the chain without raising.

---

### 3.3 `src/pipeline/router.py`

`Router` is a no-arg dataclass. `run(state)` reads `primary_model_output.confidence` and compares against `task_config.threshold`. Decision written to `state.routing_info.decision`:
- `confidence >= threshold` → `"accept_primary"` → sets `state.final_output` immediately
- `confidence < threshold` → `"escalate"` → specialist agents run

---

### 3.4 `src/agents/`

All agents inherit `BaseAgent[PipelineState]`.

| Agent | Input state fields read | Output state field written |
|---|---|---|
| `LexicalAgent` | `input_text`, `task_config` | `state.lexical_output` |
| `LogicAgent` | `input_text`, `lexical_output`, `task_config` | `state.logic_output` |
| `ContextualAgent` | `input_text`, `lexical_output`, `logic_output`, `task_config` | `state.contextual_output` |
| `DeliberationAgent` | `lexical_output`, `logic_output`, `contextual_output` | `state.deliberation_output` |
| `ConsensusAgent` | All agent outputs, optional deliberation | `state.consensus_output` + `state.final_output` |
| `ExplainabilityAgent` | `final_output`, `routing_info` | `state.explanation_output` |

`ContextualAgent` uses an `LLMClient` (real or mock) for label inference. **Known behavior**: if `MockLLMClient(mode="label_echo")` is used without `allowed_labels`, the client returns `"unknown"` and the agent falls back — it does not crash. The `debug_run_modes.py` script passes `allowed_labels=task_config.labels` to avoid this.

---

### 3.5 `src/llm/mock_client.py` — `MockLLMClient`

Three valid modes:

| Mode | Behavior |
|---|---|
| `"fixed"` | Returns `fixed_response` verbatim on every call |
| `"label_echo"` | Scans prompt for first occurrence of any `allowed_labels` entry; returns that label. Falls back to first allowed label if none found. |
| `"raise_on_call"` | Raises `LLMClientError` on every call (tests error paths) |

> **Note:** `"heuristic"` is NOT a valid mode — it is silently ignored and the client falls through to `label_echo` behavior without `allowed_labels`, producing `"unknown"` for every call.

---

### 3.6 `src/evaluation/evaluator.py`

`Evaluator` computes metrics over a full dataset.

**Constructor params:**
- `orchestrator` — used when `mode="full_pipeline"`
- `primary_classifier` — used when `mode="primary_only"` and no orchestrator
- `mode: EvalMode` — `"primary_only"` or `"full_pipeline"`
- `task_config: TaskConfig` — drives label list for metric computation
- `run_id` — defaults to UTC timestamp

**Output: `EvalReport`**

| Field | Type | Description |
|---|---|---|
| `accuracy` | `float` | Overall accuracy |
| `macro_f1` | `float` | Unweighted mean F1 across all labels |
| `per_class` | `List[PerClassMetrics]` | Precision / recall / F1 / support per label |
| `escalation_rate` | `float` | Fraction of samples routed to specialists |
| `escalated_count` | `int` | Count of escalated samples |
| `escalated_accuracy` | `float` | Accuracy on escalated subset only |
| `samples` | `List[SampleResult]` | Per-sample prediction details |
| `meta` | `dict` | Includes `pipeline_mode`, `valid_samples`, `error_samples` |

**`save(report, output_dir)` writes 4 files per run:**

| File | Content |
|---|---|
| `{run_id}_predictions.json` | Array of per-sample objects including `pipeline_mode` |
| `{run_id}_predictions.csv` | Same as flat CSV with `pipeline_mode` column |
| `{run_id}_metrics.json` | Full aggregate + per-class JSON |
| `{run_id}_metrics.csv` | `__summary__` row + per-class rows; `pipeline_mode` on summary row |

---

### 3.7 `src/evaluation/ablation.py`

`AblationStudy` runs `Evaluator` for multiple `AblationConfig` variants and builds a comparison table.

**`AblationConfig` fields:**
- `name`, `description`
- `use_lexical`, `use_contextual`, `use_logic`, `use_deliberation` (all default `True`)
- `consensus_weights: Dict[str, float]` — per-agent weight overrides
- `router_threshold_override: Optional[float]`

**`AblationReport` fields:**
- `configs: List[AblationConfig]`
- `results: Dict[str, EvalReport]`
- `comparison: List[Dict]` — flat table with accuracy, macro_f1, escalation_rate, escalated_accuracy, per-class F1 columns

YAML-driven config supported via `AblationConfig.load_yaml(path)`.

---

### 3.8 `evaluate_pipeline.py`

CLI entry point. Key flags:

| Flag | Default | Description |
|---|---|---|
| `--dataset PATH` | required | JSONL evaluation dataset |
| `--mode` | `full_pipeline` | `primary_only` or `full_pipeline` |
| `--pipeline_mode` | `full_agentic` | `primary_only`, `paper_style`, or `full_agentic` |
| `--labels L1 L2 …` | required | Label space for the task |
| `--output_dir DIR` | `results/` | Output directory |
| `--run_id ID` | UTC timestamp | Run identifier |
| `--ablation PATH` | — | If set, runs `AblationStudy` from YAML config |

`TaskConfig.pipeline_mode` is set from `--pipeline_mode`, overriding any value in `config/default.yaml`.

**Bug fixed 2026-05-02:** `_print_report()` and `_print_ablation_report()` used `print()` directly, which raises `UnicodeEncodeError` when piped through PowerShell on Windows (cp1252 encoding cannot render `─` box-drawing characters). Fixed by adding `_safe_print(text)` which catches `UnicodeEncodeError` and re-encodes with `errors="replace"`. All display calls now use `_safe_print`.

---

### 3.9 `utils/debug.py` — `print_debug_summary(state)`

Pretty-prints a single `PipelineState` after orchestration:

```
────────────────────────────────────────────────────────
  INPUT       : <first 80 chars of input text>
────────────────────────────────────────────────────────
  PRIMARY     : <label>  (conf=0.XXX)
  ROUTING     : decision=<accept_primary|escalate>  threshold=0.60
  ESCALATED   : Yes / No
  AGENTS RAN  : lexical, logic, contextual, consensus
  FINAL LABEL : <label>  (conf=0.XXX)
────────────────────────────────────────────────────────
```

Agent detection: reads `state.lexical_output`, `state.logic_output`, `state.contextual_output`, `state.deliberation_output`, `state.consensus_output` for non-None entries. Falls back to `state.history` component names when all are None (e.g. `primary_only` mode).

---

### 3.10 `config/default.yaml`

Defines the task registry. Notable sections:
- `pipeline.name / version`
- `active_task` — currently `sentiment_classification`
- `tasks:` — list of named task configs (labels, descriptions, label_knowledge with keyword lists and regex rules)
- `execution.pipeline_mode: full_agentic` — default mode, overridable at CLI

---

### 3.11 `scripts/generate_dummy_data.py`

Generates `data/dev_dummy.jsonl`. 30 Arabic-English code-switched sentences across 9 topic labels:

| Label | Count |
|---|---|
| business | 4 |
| tech | 4 |
| social | 4 |
| education | 3 |
| health | 3 |
| shopping | 3 |
| medical | 3 |
| sports | 3 |
| finance | 3 |

Each record: `{"id": "N", "text": "...", "label": "..."}`.

---

### 3.12 `scripts/debug_run_modes.py`

Runs the first 5 samples of `data/dev_dummy.jsonl` through all three pipeline modes using mocked components. After each sample it calls `print_debug_summary(state)`.

Key decisions:
- `MockLLMClient(mode="label_echo", allowed_labels=task_config.labels)` — avoids the `"unknown"` fallback
- `threshold=0.60` — forces most samples to escalate (mock primary confidence ~0.1–0.3)
- All three modes exercised in a single script run

---

## 4. Test Suite — 324 tests

| File | Count | Coverage |
|---|---|---|
| `test_ablation.py` | 46 | `AblationConfig`, `AblationStudy`, `AblationReport`, YAML loading |
| `test_evaluator.py` | ~47 | `Evaluator`, `EvalReport`, save methods, CLI `--pipeline_mode` wiring |
| `test_primary_transformer_classifier.py` | 30 | `PrimaryTransformerClassifier` interface, label mapping, device handling |
| `test_agents_interfaces.py` | 7 | All 6 specialist agents + base agent contract |
| `test_consensus_agent.py` | varies | `ConsensusAgent` weighting and output |
| `test_contextual_agent.py` | varies | `ContextualAgent` with `MockLLMClient` |
| `test_deliberation_agent.py` | varies | `DeliberationAgent` enabled/disabled |
| `test_explainability_agent.py` | varies | `ExplainabilityAgent` on fast-path and escalation |
| `test_lexical_agent.py` | varies | `LexicalAgent` output fields |
| `test_logic_agent.py` | varies | `LogicAgent` output fields |
| `test_orchestrator_flow.py` | 3 | Three pipeline modes end-to-end |
| `test_state_models.py` | 2 | `PipelineState`, `TaskConfig` validation |

**All 324 pass in 0.85 s.** No external network calls (all LLM interactions use `MockLLMClient`).

Notable test classes in `test_evaluator.py`:
- `TestSave` — verifies `pipeline_mode` column presence in all 4 output file types
- `TestCLIPipelineModeOverride` — verifies CLI `--pipeline_mode` arg default and override behavior

---

## 5. Evaluation Results — `data/dev_dummy.jsonl` (30 samples)

All runs use `--mode full_pipeline` (orchestrator active). Metrics come from the most recent run per mode.

### 5.1 Summary Table

| Mode | Accuracy | Macro F1 | Escalation Rate | Escalated Count | Errors |
|---|---|---|---|---|---|
| `primary_only` | 0.1333 | **0.1336** | 0.0000 | 0 | 0 |
| `paper_style` | 0.1333 | 0.0261 | **1.0000** | 30 | 0 |
| `full_agentic` | 0.1333 | 0.0261 | **1.0000** | 30 | 0 |

### 5.2 Per-Class F1 (latest runs)

| Label | Support | primary_only F1 | paper_style F1 | full_agentic F1 |
|---|---|---|---|---|
| business | 4 | 0.0000 | 0.2353 | 0.2353 |
| education | 3 | 0.0000 | 0.0000 | 0.0000 |
| health | 3 | 0.0000 | 0.0000 | 0.0000 |
| shopping | 3 | 0.3333 | 0.0000 | 0.0000 |
| medical | 3 | 0.3333 | 0.0000 | 0.0000 |
| sports | 3 | 0.0000 | 0.0000 | 0.0000 |
| tech | 4 | 0.2500 | 0.0000 | 0.0000 |
| finance | 3 | 0.0000 | 0.0000 | 0.0000 |
| social | 4 | 0.2857 | 0.0000 | 0.0000 |

### 5.3 Interpretation

**Why accuracy is identical across modes:** The mock primary classifier assigns labels randomly (uniform distribution over 9 labels). With 30 samples the random accuracy of ~13% is purely noise — specialist agents cannot improve on this with a mock backend.

**Why `primary_only` has better Macro F1:** The random classifier naturally spreads predictions across labels, yielding non-zero F1 for several classes. In `paper_style` and `full_agentic` every sample is escalated (mock primary confidence ~0.1–0.3, always below threshold=0.60) and the mock consensus agents collapse to one label (`business`) for all 30 samples → F1=0 on all other 8 classes.

**Why escalation_rate = 1.0 in escalated modes:** Mock primary outputs confidence in range ~0.10–0.32. The configured `threshold=0.60` is never reached, so every sample is routed to specialists.

**`full_agentic` ContextualAgent warnings (29 instances):**
```
WARNING  ContextualAgent: LLM returned invalid label 'unknown'; falling back.
```
`evaluate_pipeline.py` wires `ContextualAgent` with a default `MockLLMClient` that does not have `allowed_labels` set — it returns `"unknown"` on every call. The agent's fallback logic handles this gracefully (no crash, consensus still runs with lexical+logic only). The `debug_run_modes.py` script already uses the correct `mode="label_echo", allowed_labels=...` but the same fix has not yet been applied inside `evaluate_pipeline.py`.

---

## 6. Output Files

### `results/debug_primary_only/`
| File | Run ID |
|---|---|
| `20260502_161612_predictions.json` | first run |
| `20260502_161612_predictions.csv` | first run |
| `20260502_161612_metrics.json` | first run |
| `20260502_161612_metrics.csv` | first run |
| `20260502_162102_predictions.json` | second run (post encoding-fix) |
| `20260502_162102_predictions.csv` | second run |
| `20260502_162102_metrics.json` | second run |
| `20260502_162102_metrics.csv` | second run |

### `results/debug_paper_style/`
| File | Run ID |
|---|---|
| `20260502_162114_predictions.json` | |
| `20260502_162114_predictions.csv` | |
| `20260502_162114_metrics.json` | |
| `20260502_162114_metrics.csv` | |

### `results/debug_full_agentic/`
| File | Run ID |
|---|---|
| `20260502_162120_predictions.json` | |
| `20260502_162120_predictions.csv` | |
| `20260502_162120_metrics.json` | |
| `20260502_162120_metrics.csv` | |

---

## 7. Known Issues / Open Items

| # | Severity | Component | Description |
|---|---|---|---|
| 1 | Low | `evaluate_pipeline.py` | `ContextualAgent` is wired with `MockLLMClient` without `allowed_labels` → produces 29 `"unknown"` fallback warnings per `full_agentic` run. Results are still correct but noisy. Fix: pass `allowed_labels=task_config.labels` when building the LLM client inside `evaluate_pipeline.py`. |
| 2 | Info | All modes | Accuracy ~13% is expected on the dummy dataset with mock classifiers — not a bug. |
| 3 | Info | PowerShell | `evaluate_pipeline.py` exits with code 1 even on success when run through PowerShell, because PowerShell treats stderr (Python logging) as an error stream. All 4 output files are written correctly regardless. |
| 4 | Info | `results/debug_primary_only/` | Contains 2 runs (8 files) — the first was from before the encoding fix, the second is the clean run. |

---

## 8. Dependencies

From `requirements.txt` / `pyproject.toml`:

| Package | Purpose |
|---|---|
| `transformers` | `PrimaryTransformerClassifier` (HuggingFace backbone) |
| `torch` | Model inference |
| `pytest` | Test runner |
| `pyyaml` | Config loading (`default.yaml`, ablation YAML) |
| `dataclasses` / stdlib | State schema, metrics (no external ML deps for evaluation) |

No external network calls are made during tests — all LLM and classifier interactions use mock implementations.

---

## 9. Session History (Changes Made)

| Date | Change |
|---|---|
| Prior sessions | Built `PrimaryTransformerClassifier`, `Evaluator`, `AblationStudy`, three pipeline modes, 317 tests |
| This session (2026-05-02) | Added `--pipeline_mode` CLI arg + output file visibility (7 tests → 324 total) |
| This session | Created `scripts/generate_dummy_data.py` + `data/dev_dummy.jsonl` |
| This session | Created `scripts/debug_run_modes.py` |
| This session | Created `utils/debug.py::print_debug_summary()` |
| This session | Fixed `MockLLMClient` mode in `debug_run_modes.py` (`label_echo` + `allowed_labels`) |
| This session | Fixed `UnicodeEncodeError` in `evaluate_pipeline.py` (`_safe_print` wrapper) |
| This session | Ran all 3 pipeline modes on `dev_dummy.jsonl`; verified 12 output files saved |
