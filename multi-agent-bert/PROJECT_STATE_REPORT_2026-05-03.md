# SwitchLingua Multi-Agent Pipeline — Project State Report
**Date:** 2026-05-03  
**Working directory:** `multi-agent-bert/`  
**Test suite:** 412 passed · 0 failed · 0 skipped

---

## 1. Architecture Overview

The pipeline implements the SwitchLingua paper's multi-agent Arabic-English
code-switched text classification system.  Three execution modes are supported:

| Mode | Agents Executed |
|---|---|
| `primary_only` | Primary classifier only; router and specialists skipped |
| `paper_style` | Primary → Router → (if escalated) Lexical + Logic + Contextual → Consensus → Explainability |
| `full_agentic` | Same as `paper_style` + optional DeliberationAgent before Consensus |

---

## 2. Files Changed or Created This Session

### 2.1 `evaluate_pipeline.py` — Bilingual Regex Rules Upgrade

**Change:** Replaced single-keyword alternation rules in `_TOPIC_KNOWLEDGE["regex_rules"]`
with 3 combinatorial `A.*B` bilingual patterns per topic label.

**Motivation:** Single-keyword rules fire on any occurrence of a word in
isolation.  Combinatorial pair rules require both a subject term and a
related object term to co-occur, making them more precise and better
suited to Arabic-English code-switching where a sentence mixes both scripts.

**Labels updated:** `business`, `education`, `health`, `shopping`, `medical`,
`sports`, `tech`, `finance`, `social` (9 total).

**Rule format (example — finance):**
```python
"regex_rules": [
    r"(bank|بنك).*(loan|قرض|interest|فائدة)",
    r"(investment|استثمار|portfolio|محفظة).*(profit|ربح|risk|مخاطرة)",
    r"(dollar|دولار|currency|عملة).*(inflation|تضخم|price|سعر)",
],
```

Each rule follows the same bilingual structure:
- Left side: English term **or** Arabic equivalent
- Right side: contextually related English term **or** Arabic equivalent
- Compiled by `LogicAgent` with `re.IGNORECASE | re.UNICODE`
- `_TOPIC_RULE_MAP` also auto-appends one Arabic-alternation rule per label
  derived from `keywords_ar`

**Comment block added:** All `regex_rules` entries are annotated with
`# Manually curated seed rules — can be refined from training/dev data.`

---

### 2.2 `src/agents/transformer_contextual_agent.py` — **New file**

A non-LLM contextual agent for `paper_style` mode, approximating the BERT /
RoBERTa / XLNet contextual layer described in the paper without fine-tuning or
LLM API calls.

**Class:** `TransformerContextualAgent(BaseAgent[PipelineState])`

**Operating modes:**

| Mode | Description | Dependencies |
|---|---|---|
| `tfidf` | TF-IDF cosine similarity between `input_text` and each `label_description`; smooth IDF, L2 cosine | stdlib only |
| `embedding` | Mean-pooled HuggingFace transformer embeddings; silently falls back to `tfidf` if `transformers`/`torch` absent or inference fails | `transformers`, `torch` |

**Input:**
- `state.input_text`
- `state.task_config.labels`
- `state.task_config.label_descriptions` (falls back to label name if missing)

**Output** — writes `state.contextual_output` (`AgentOutput`):
- `model_output.label` — most similar label
- `model_output.confidence` — normalized similarity score for best label
- `model_output.probabilities` — softmax-like distribution over all labels
- `notes` — mode used, best match, fallback note if applicable
- `features["similarity_scores"]` — raw per-label cosine scores
- `features["effective_mode"]` — `"tfidf"` or `"embedding"`

**History:** Calls `state.append_history(component=self.name, ...)` with
`label`, `confidence`, `probabilities`, `effective_mode`, `similarity_scores`.

**Key design decisions:**
- No `llm_client` attribute — cannot call any LLM
- Zero-vocabulary-overlap → uniform probability fallback (`1/|labels|`)
- Missing descriptions → label name used as description
- `embedding` mode failure is non-fatal; `tfidf` is always the safety net
- Default model (embedding mode): `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

---

### 2.3 `src/pipeline/orchestrator.py` — Dual Contextual Agent Routing

**Change:** Added optional `paper_contextual_agent: Optional[Any] = None`
constructor parameter to `PipelineOrchestrator`.

**Routing logic in the escalation path:**
```python
contextual_for_mode = (
    self._paper_contextual
    if (pipeline_mode == _PAPER_STYLE and self._paper_contextual is not None)
    else self._contextual
)
```

| Mode | `paper_contextual_agent` provided | Agent used |
|---|---|---|
| `paper_style` | Yes | `paper_contextual_agent` (e.g. `TransformerContextualAgent`) |
| `paper_style` | No | `contextual_agent` (backward compatible) |
| `full_agentic` | Either | `contextual_agent` (LLM-backed) always |
| `primary_only` | Either | Neither (specialists skipped) |

**Backward compatibility:** All existing orchestrator call sites that omit
`paper_contextual_agent` continue to work unchanged.

---

### 2.4 `tests/test_evaluator.py` — Bilingual Regex Rule Tests

**New class:** `TestTopicSeedRegexRules` — 27 tests

Tests verify that sample Arabic-English code-switched sentences trigger the
expected topic rules using the same semantics as `LogicAgent`
(`re.IGNORECASE | re.UNICODE`).  Three sentences per label:

| Label | English sentence | Arabic sentence | Code-switched sentence |
|---|---|---|---|
| business | startup → market | شركة → أرباح | startup → سوق |
| education | student → deadline | امتحان → جامعة | exam → جامعة |
| health | exercise → health | تمارين → لياقة | exercise → الصحة |
| shopping | bought → store | اشتريت → متجر | discount → سعر |
| medical | diagnosis → treatment | دكتور → مريض | دكتور → surgery |
| sports | match → team | مباراة → فريق | الفريق → goal |
| tech | software → update | برنامج → تحديث | app → تحديث |
| finance | bank → loan | بنك → تضخم | دولار → تضخم |
| social | post → Instagram | بوست → لايكات | Instagram → لايكات |

Also updated: `test_topic_rule_map_contains_english_pattern` assertion changed
from `"money"` to `"bank"` to match the new finance rule set.

---

### 2.5 `tests/test_transformer_contextual_agent.py` — **New file** (32 tests)

| Class | Count | What is tested |
|---|---|---|
| `TestTransformerContextualAgentTfidf` | 18 | valid label, tech ranking, output written, confidence ∈ [0,1], probs sum to 1, all labels in probs, similarity scores, effective_mode field, notes text, agent name, history event (5 sub-assertions), no LLM attribute |
| `TestTransformerContextualAgentEmbedding` | 4 | no crash, effective_mode present, output written, probs sum to 1 |
| `TestOrchestratorPaperContextualAgentRouting` | 10 | paper_style calls paper agent; paper_style skips LLM; fallback to contextual_agent when None; full_agentic uses LLM; both modes write `contextual_output`; label validity across 5 inputs |

---

## 3. Test Suite Summary

| File | Tests |
|---|---|
| `test_ablation.py` | 46 |
| `test_agents_interfaces.py` | 7 |
| `test_consensus_agent.py` | 40 |
| `test_contextual_agent.py` | 32 |
| `test_deliberation_agent.py` | 31 |
| `test_evaluator.py` | 103 |
| `test_explainability_agent.py` | 37 |
| `test_lexical_agent.py` | 22 |
| `test_logic_agent.py` | 27 |
| `test_orchestrator_flow.py` | 3 |
| `test_primary_transformer_classifier.py` | 30 |
| `test_state_models.py` | 2 |
| `test_transformer_contextual_agent.py` | 32 |
| **Total** | **412** |

---

## 4. Key Design Principles Applied

- **No fine-tuning required:** `TransformerContextualAgent` uses only pretrained
  embeddings or pure TF-IDF, matching the paper's description of contextual
  architecture without specifying fine-tuning details.
- **Graceful degradation:** Every dependency on optional libraries (`transformers`,
  `torch`) has a deterministic stdlib fallback.  Tests never require downloads.
- **Backward compatibility:** All existing orchestrator instantiations without
  `paper_contextual_agent` continue to work unchanged.
- **Knowledge engineering:** Topic regex rules are bilingual `A.*B` pair patterns
  stored in `_TOPIC_KNOWLEDGE["regex_rules"]` alongside seed keyword maps.  They
  are annotated as manually curated seeds replaceable by data-driven extraction
  (`scripts/build_keyword_map.py`).
- **No PipelineState / AgentOutput changes:** All new features are implemented
  by extending agent behaviour and orchestrator wiring only.

---

## 5. How to Use the New Components

### Wire TransformerContextualAgent into paper_style mode
```python
from src.agents.transformer_contextual_agent import TransformerContextualAgent
from src.pipeline.orchestrator import PipelineOrchestrator

orch = PipelineOrchestrator(
    primary_classifier=...,
    router=...,
    lexical_agent=...,
    logic_agent=...,
    contextual_agent=llm_contextual_agent,      # used in full_agentic
    consensus_agent=...,
    explainability_agent=...,
    paper_contextual_agent=TransformerContextualAgent(mode="tfidf"),  # used in paper_style
)
```

### Run with topic labels and descriptions
```python
from src.state.schema import PipelineState, StateMetadata, TaskConfig

state = PipelineState(
    metadata=StateMetadata(sample_id="s1"),
    input_text="الشركة حققت profit كبير في السوق",
    task_config=TaskConfig(
        task_name="topic",
        labels=["business", "tech", "sports"],
        label_descriptions={
            "business": "company market profit merger CEO",
            "tech": "software app AI programming",
            "sports": "match team goal football",
        },
        pipeline_mode="paper_style",
        threshold=0.7,
    ),
)
result = orch.run(state)
print(result.final_output.label)           # e.g. "business"
print(result.contextual_output.notes)      # "Mode: tfidf. Best match: 'business' ..."
```

### Build topic knowledge maps (for evaluate_pipeline CLI)
```python
import evaluate_pipeline
keyword_map, rule_map = evaluate_pipeline.build_agent_knowledge_maps(
    ["business", "tech", "sports", "health", "finance"]
)
```
