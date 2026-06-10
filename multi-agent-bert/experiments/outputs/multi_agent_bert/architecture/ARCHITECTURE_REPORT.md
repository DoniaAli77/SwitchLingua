# Multi-Agent BERT — Architecture Report

Inspection-only. Traced from code; no changes made. Date: 2026-06-09.

## Entry point and dispatch

```
evaluate_pipeline.py :: main()
  ├─ load task_config  (--active_task → tasks.<name> in src/config/default.yaml)
  ├─ build_primary_classifier()   → mock | transformer        (shared slot)
  ├─ build_llm_client()           → mock | openai             (shared slot)
  ├─ build_orchestrator(...)      → ONE PipelineOrchestrator wiring BOTH
  │                                  the classification agents AND the NER agents
  └─ dispatch by task_config.task_type:
        task_type == "classification"     → Evaluator      (sentiment, topic)
        task_type == "sequence_labeling"  → NEREvaluator   (NER)
```

The **task router** is `PipelineOrchestrator.run()`
([orchestrator.py:243](../../../src/pipeline/orchestrator.py#L243)):

```python
if state.task_config.task_type == "sequence_labeling":
    return self._run_ner_path(state, pipeline_mode)
# else: classification path (sentiment AND topic, identical)
```

Selection is purely by `task_type` from config. **Sentiment vs topic is never
branched in code** — they run the byte-identical classification path; the only
difference is the config (`labels`, `label_descriptions`, `keyword_map`,
`rule_map`, `task_name`).

---

## Diagram

```
                          ┌───────────────────────────┐
                          │ PipelineOrchestrator.run() │
                          │   task router (task_type)  │
                          └────────────┬──────────────┘
              task_type=classification │ task_type=sequence_labeling
        (sentiment, topic — SAME path) │ (NER)
                                       │
   ┌───────────────────────────────────┐   ┌──────────────────────────────────┐
   │ CLASSIFICATION PATH                │   │ NER PATH (_run_ner_path)          │
   │                                    │   │                                   │
   │ 1 Primary classifier (shared slot) │   │ primary_only: primary stub only   │
   │ 2 Router  ── confidence ≥ thr ? ──┐│   │  (transformer is class-only →     │
   │      │ accept_primary             ││   │   empty FinalOutput)              │
   │      │ → FINAL = primary          ││   │                                   │
   │      │ escalate                   ││   │ paper_style / full_agentic:       │
   │      ▼                            ││   │   NERLexicalAgent                 │
   │   specialist agents (by mode):    ││   │   NERLogicAgent                   │
   │   paper_style:                    ││   │   NERContextualAgent              │
   │     LexicalAgent (keyword)        ││   │   NERConsensusAgent  → FINAL tags │
   │     LogicAgent (regex)            ││   │   ExplainabilityAgent (SHARED) ⚠  │
   │     TransformerContextualAgent    ││   │                                   │
   │   full_agentic:                   ││   │   NO Router / NO escalation       │
   │     LLMLexicalAgent               ││   │   NO primary vote                 │
   │     LLMLogicAgent                 ││   └──────────────────────────────────┘
   │     ContextualAgent (LLM)         ││
   │   (DeliberationAgent — optional)  ││
   │   ▼                               ││
   │   ConsensusAgent (weighted vote)  ││   FINAL OUTPUT
   │   ▼  → FINAL = consensus label    ││   - classification: state.final_output (label+conf)
   │   ExplainabilityAgent /           ││   - NER: state.final_output.payload (token tags)
   │   LLMExplainabilityAgent (SHARED) ◄┘
   └────────────────────────────────────┘
```

---

## Answers

### 1. Which agents are shared between sentiment and topic?
**All of them.** Sentiment and topic use the identical classification path with
zero task-name branching. Shared, config-driven components:

| Slot | paper_style | full_agentic |
|---|---|---|
| primary | MockPrimaryClassifier / PrimaryTransformerClassifier | same |
| escalation router | Router | Router |
| lexical | LexicalAgent (keyword_map) | LLMLexicalAgent |
| logic | LogicAgent (rule_map) | LLMLogicAgent |
| contextual | TransformerContextualAgent (tfidf) | ContextualAgent (LLM) |
| deliberation | — | DeliberationAgent (optional) |
| consensus | ConsensusAgent | ConsensusAgent |
| explainability | ExplainabilityAgent | LLMExplainabilityAgent |

They differ between sentiment and topic only through `task_config`
(`labels`, `label_descriptions`, `keyword_map`, `rule_map`, `task_name`).

### 2. Which agents are NER-specific?
`NERLexicalAgent`, `NERLogicAgent`, `NERContextualAgent`, `NERConsensusAgent`
(plus the NER dataset loader and `NEREvaluator`). These run **only** in
`_run_ner_path` and produce token/entity tags, not a single class label.

### 3. Any classification agent accidentally sentiment- or topic-specific?
- **No agent is hardcoded to a specific label set** — labels always come from
  config; nothing special-cases positive/negative/neutral.
- **But two prompts lean topic, not sentiment:** `LLMLexicalAgent` and
  `LLMLogicAgent` system prompts say "identify the most likely **topic** label"
  and reason about "**domain**" ([llm_lexical_prompt.py](../../../src/prompts/llm_lexical_prompt.py),
  [llm_logic_prompt.py](../../../src/prompts/llm_logic_prompt.py)). This is a
  generality bug (already logged as M1 in the agent audit): the wording is
  topic-flavored, so it mildly mis-frames sentiment. It leans **topic**, never
  sentiment. Contextual / consensus / explainability / deliberation are generic.

### 4. Does NER incorrectly reuse classification agents?
**Almost no — with one shared component.** NER uses its own NER* lexical/logic/
contextual/consensus. The **explainability agent is shared**: `_run_ner_path`
ends by calling the same `ExplainabilityAgent` / `LLMExplainabilityAgent` used by
the classification path ([orchestrator.py:474-480](../../../src/pipeline/orchestrator.py#L474)).
That agent's prompt is built around a single `final_label` + agent votes, which
doesn't cleanly fit token-level NER output — a mild mismatch (the LLM
explainability would try to explain a "final label" NER doesn't produce). The
**primary classifier slot** is also shared, but NER `primary_only` correctly
treats it as a stub (the transformer is classification-only). No NER stage reuses
the classification lexical/logic/contextual/consensus.

### 5. Does the architecture match the intended design?
**Yes, substantially — with three caveats.**

✅ Sentiment and topic share generic, config-driven classification agents.
✅ NER is a separate sequence-labeling flow with NER-specific agents, consensus,
and evaluator producing token/entity tags.

Caveats / deviations to be aware of (none are blockers):
1. **Shared explainability into NER** (Q4) — the only place NER touches a
   classification agent; the LLM explainability prompt is class-label-shaped.
2. **Topic-flavored wording** in the LLM lexical/logic prompts (Q3 / audit M1) —
   leaks "topic"/"domain" into what should be a task-agnostic classification
   agent. Fix is the generic-prompt change already proposed.
3. **Structural asymmetry:** the classification path has a confidence **Router +
   escalation** and a (currently primary-discarding) **ConsensusAgent**, while
   the NER path has **no router, no escalation, no primary vote** — both NER
   modes always run all four NER agents. Intended-design-wise this is fine (NER
   tagging isn't a single-confidence decision), but it means "paper_style vs
   full_agentic" is currently a no-op distinction for NER (same deterministic
   agents).

### Net
The core intent holds: **one shared, config-driven classification flow
(sentiment = topic by config only) + a separate NER sequence-labeling flow.** The
generic classification agents are genuinely task-config driven; the only true
task leakage is the topic-flavored *wording* in the lexical/logic prompts, and
the only cross-flow reuse is the explainability agent. Both are the targets the
agent audit already flagged for the task-config-driven refactor.
```
