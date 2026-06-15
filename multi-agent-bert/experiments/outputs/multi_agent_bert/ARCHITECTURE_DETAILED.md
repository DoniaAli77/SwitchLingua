# Multi-Agent BERT — Detailed Architecture

Faithful to the code in `src/`. Date: 2026-06-13.
Entry point: `PipelineOrchestrator.run(state)` in
[`src/pipeline/orchestrator.py`](../../../src/pipeline/orchestrator.py).

---

## 1. End-to-end data flow (classification path)

Every stage reads and writes fields on a single shared **`PipelineState`** object
and appends to an immutable **audit `history`**. Field names below are the actual
`state.*` attributes.

```
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │ PipelineState  (one per input sample, threaded through every stage)               │
 │  input_text · metadata(sample_id) · task_config(labels, threshold, pipeline_mode, │
 │  task_type, enable_deliberation) · primary_model_output · routing_info ·          │
 │  lexical_output · logic_output · contextual_output · deliberation_output ·        │
 │  consensus_output · final_output · explanation_output · history[] · extras{}      │
 └──────────────────────────────────────────────────────────────────────────────────┘

   INPUT TEXT
       │
       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 1 — PRIMARY CLASSIFIER                                  │
 │ src/models/primary_transformer_classifier.py (XLM-R / mBERT) │
 │   or mock_primary_classifier.py (tests)                       │
 │ writes → primary_model_output {label, confidence,             │
 │          probabilities{label:p}, raw_text}                    │
 └───────────────────────────────┬─────────────────────────────┘
                                 │
            mode == primary_only?  ──────────► YES ─────────────────────────┐
                                 │ NO                                        │
                                 ▼                                           │
 ┌─────────────────────────────────────────────────────────────┐           │
 │ STAGE 2 — ROUTER   src/pipeline/router.py                    │           │
 │   threshold = task_config.threshold                          │           │
 │   decision = "accept_primary" if confidence >= threshold      │           │
 │              else "escalate"                                  │           │
 │ writes → routing_info {threshold, decision}                  │           │
 │   if accept_primary: final_output = primary (source=         │           │
 │   "primary_model")                                           │           │
 └───────────┬───────────────────────────────┬─────────────────┘           │
   accept_primary                         escalate                          │
   (FAST PATH)                            (SLOW PATH)                        │
        │                                     │                             │
        ▼                                     ▼                             │
 ┌────────────────┐         ┌─────────────────────────────────────────┐     │
 │ Explainability │         │ STAGE 3 — SPECIALIST PANEL (per-mode)    │     │
 │ (template,     │         │  3a Lexical  → lexical_output            │     │
 │  short)        │         │  3b Logic    → logic_output              │     │
 │ final already  │         │  3c Contextual → contextual_output       │     │
 │ set by router  │         │  3d Deliberation (optional, full_agentic │     │
 └───────┬────────┘         │     & enable_deliberation) → deliberation│     │
         │                  │     _output                              │     │
         │                  │  3e CONSENSUS → consensus_output +       │     │
         │                  │     final_output                          │     │
         │                  │  3f Explainability (full)                │     │
         │                  └───────────────────┬─────────────────────┘     │
         └──────────────────────────────────────┤                           │
                                                ▼                           ▼
                              ┌──────────────────────────────────────────────────┐
                              │ FINAL OUTPUT  final_output{label, confidence,     │
                              │ payload} + explanation_output{summary, evidence,  │
                              │ caveats} + full history[] (every stage recorded)  │
                              └──────────────────────────────────────────────────┘

 Error handling: any stage exception is caught in _run_stage(), recorded to
 extras["pipeline_error"]={stage,message,traceback}, and the chain stops early
 with all upstream results preserved.
```

---

## 2. Stage 3 — specialist panel, expanded (escalation path)

```
                        escalate
                            │
   ┌────────────────────────┼─────────────────────────────────────────────┐
   │  PER-MODE AGENT RESOLUTION (orchestrator picks the concrete classes)   │
   │                                                                        │
   │  paper_style                          full_agentic                     │
   │  ───────────                          ────────────                     │
   │  lexical   = LexicalAgent             lexical   = LLMLexicalAgent      │
   │              (keyword/regex)                      (→ LexicalAgent if   │
   │                                                    not provided)        │
   │  logic     = LogicAgent               logic     = LLMLogicAgent        │
   │              (rules/negation)                     (→ LogicAgent …)      │
   │  contextual= paper_contextual         contextual= ContextualAgent      │
   │              (TransformerContextual               (LLM-backed)          │
   │               → ContextualAgent …)                                      │
   │  deliberation = OFF                   deliberation = optional           │
   └────────────────────────┬───────────────────────────────────────────────┘
                            ▼
   ┌─────────────┐   ┌─────────────┐   ┌───────────────┐
   │  LEXICAL    │   │   LOGIC     │   │  CONTEXTUAL   │   each writes an
   │  agent      │   │   agent     │   │   agent       │   AgentOutput:
   │  surface    │   │  rules /    │   │  meaning in   │   {label, confidence,
   │  cues       │   │  negation   │   │  context      │    rationale, evidence}
   └──────┬──────┘   └──────┬──────┘   └───────┬───────┘
          │ lexical_output  │ logic_output     │ contextual_output
          └─────────────────┴──────┬───────────┘
                                   │
                  (full_agentic only, optional)
                   ┌───────────────▼───────────────┐
                   │     DELIBERATION AGENT         │  cross-reads the three
                   │  enable_deliberation == True   │  agent outputs, may
                   │  → deliberation_output         │  recommend a label
                   │  {recommended_label,confidence}│  (default weight 0)
                   └───────────────┬───────────────┘
                                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ CONSENSUS AGENT   src/agents/consensus_agent.py                        │
   │                                                                        │
   │ weighted vote:  score[label] += weight[slot] * agent_confidence        │
   │   default weights: lexical 1.0 · contextual 1.0 · logic 1.0 ·          │
   │                    deliberation 0.0 · primary 1.0                       │
   │                                                                        │
   │ ── Fix #2 PRIMARY-AWARE PRIOR (audit C1) ───────────────────────────   │
   │   if w_primary > 0 and primary usable:                                 │
   │       score[primary.label] += w_primary * primary.confidence           │
   │   (confidence-scaled: a near-threshold primary anchors more)           │
   │                                                                        │
   │ ── ABSTAIN FALLBACK (audit) ────────────────────────────────────────   │
   │   if no agent voted (active_weight_sum == 0):                          │
   │       defer to primary if usable, else label = None (no_decision)      │
   │       — NEVER silently picks labels[0]                                  │
   │                                                                        │
   │ ── NON-POSITIONAL TIE-BREAK ────────────────────────────────────────   │
   │   among score-tied labels: (1) primary's label · (2) most voting       │
   │   agents · (3) highest single contribution · (4) alphabetical          │
   │   (never task_config.labels order)                                     │
   │                                                                        │
   │ final_confidence = score[winner] / active_weight_sum                   │
   │ writes → consensus_output{label,confidence,votes,rationale}            │
   │          final_output{label,confidence}                                │
   └───────────────────────────────┬──────────────────────────────────────┘
                                   ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ EXPLAINABILITY  (LLMExplainabilityAgent in full_agentic if provided,   │
   │ else template ExplainabilityAgent) → explanation_output               │
   └──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Fix #3 — optional primary-signal prompt block (default OFF)

```
   agents_use_primary_signal == True  (ablation seam, default False)
            │
            ▼
   src/prompts/_primary_block.py injects the primary's predicted label (+ its
   confidence) INTO each specialist agent's prompt BEFORE they vote.
   ── Effect (measured): raises agent↔primary agreement (anchoring) +3–7 pts,
      with NO accuracy gain → kept OFF for strong-agent sentiment.
```

Note the difference: **Fix #2** lets the primary vote *at consensus time* (after
agents decide independently); **Fix #3** would show agents the primary *before*
they decide (anchoring risk). They are independent seams — the 2×2 ablation
crossed them.

---

## 4. Run modes (concrete wiring)

| Mode | Router | Specialists | Contextual class | Deliberation | Explain |
|---|---|---|---|---|---|
| **primary_only** | skipped | none | — | — | template (short) |
| **paper_style** | yes | LexicalAgent · LogicAgent | TransformerContextualAgent → ContextualAgent | off | template |
| **full_agentic** | yes | LLMLexicalAgent · LLMLogicAgent | ContextualAgent (LLM) | optional | LLMExplainabilityAgent → template |

`→` = backward-compat fallback when the preferred agent is not wired.

---

## 5. NER / sequence-labeling path (separate)

Dispatched when `task_config.task_type == "sequence_labeling"`:

```
 primary_only : primary stage (or clean empty FinalOutput stub)
 paper_style / full_agentic :
     NERLexicalAgent (gazetteer) → NERLogicAgent (regex rules) →
     NERContextualAgent (heuristic) → NERConsensusAgent → Explainability
```
Same deterministic NER agents in both non-primary modes; the split is kept so
LLM-backed NER agents can slot into `full_agentic` later. (Known issue: fixed-mode
refinement loop guardrail/counter interaction.)

---

## 6. Supporting layers

| Layer | Files | Notes |
|---|---|---|
| **State** | `src/state/schema.py`, `example_state.py` | `PipelineState`, `ModelOutput`, `AgentOutput`, `ConsensusOutput`, `FinalOutput`, `RoutingInfo`, `TaskConfig` |
| **LLM clients** | `src/llm/{base_client,openai_client,mock_client}.py` | `openai_client` → gpt-4o-mini; `mock_client` for offline tests |
| **Prompts** | `src/prompts/*` incl. `_primary_block.py`, `_abstain.py`, `contextual_prompt.py`, `llm_*_prompt.py`, `deliberation_prompt.py` | task-config-driven, generic (Fix #1) |
| **Config** | `src/config/{task_config,loader}.py`, `config/default.yaml` | labels, threshold, mode, weights, enable_deliberation |
| **Evaluation** | `src/evaluation/{evaluator,ner_evaluator,ablation}.py`, `evaluate_pipeline.py` | metrics, confusion, LLM usage/cost, per-cell ablation |

---

## 7. Key control knobs (CLI seams, no default change)
| Flag | Default | Controls |
|---|---|---|
| `--pipeline_mode` | full_agentic | primary_only / paper_style / full_agentic |
| `--threshold` | 0.6 (cfg) | router escalation cutoff |
| `--consensus_primary_weight` | 1.0 | Fix #2 `w_primary` (0 = legacy agents-only) |
| `--agents_use_primary_signal` | off | Fix #3 primary-signal prompt block |
| `--primary_model` / `--transformer_checkpoint` | — | which primary + weights |
| `--llm_client` / `--llm_model` | — | openai / mock; gpt-4o-mini |

This is the architecture the ablations (§4 of the project status report) and
Experiments A/C were run against.
```
