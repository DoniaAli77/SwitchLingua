# Agent Call Audit — real LLM vs deterministic

Inspection-only; no code changed. Date: 2026-06-09. Traced from
`build_orchestrator()` wiring + `PipelineOrchestrator` execution paths.

## TL;DR

- **Exactly four** agents make real OpenAI calls, and **only** in
  `full_agentic` + `--llm_client openai`:
  **LLMLexicalAgent, LLMLogicAgent, ContextualAgent, LLMExplainabilityAgent**.
- `paper_style` makes **zero** LLM calls in any configuration (even with
  `--llm_client openai`).
- `ConsensusAgent` and `Router` are **deterministic** (no LLM, ever).
- `DeliberationAgent` is wired to a **separate fixed mock**, never the injected
  client, and is **off by default** → 0 calls.
- NER agents are **deterministic**; the NER path's only LLM exposure is the
  **shared** explainability agent (full_agentic).

---

## Which agents receive the injected LLM client

In `build_orchestrator()`:
- The built client (`mock` or `openai`) is injected into **four** agents:
  `ContextualAgent`, `LLMLexicalAgent`, `LLMLogicAgent`, `LLMExplainabilityAgent`.
- `DeliberationAgent` (only if `enable_deliberation`) gets a **separate**
  `MockLLMClient(mode="fixed")` — [evaluate_pipeline.py:640-644](../../../evaluate_pipeline.py#L640)
  — so it never calls OpenAI regardless of `--llm_client`.

`generate()` callers in code: `contextual_agent`, `llm_lexical_agent`,
`llm_logic_agent`, `llm_explainability_agent`, `deliberation_agent` (the last
bound to the fixed mock).

---

## Master table

Legend — **LLM?** = calls `BaseLLMClient.generate()`; **OpenAI?** = makes a real
OpenAI call when `--llm_client openai`; **Engine** = deterministic mechanism.

| Agent | File / class | Used in mode(s) | LLM? | OpenAI when `openai`? | Deterministic engine | Uses labels / descriptions | Task |
|---|---|---|---|---|---|---|---|
| Primary classifier | `models/primary_transformer_classifier.py` (or mock) | all | No | No | local HF transformer inference (no API) | labels (intersect) | sent/topic (+NER stub) |
| **Router** | `pipeline/router.py::Router` | classification (paper_style, full_agentic) | **No** | No | **threshold compare** `conf ≥ thr` | labels (validate); not descriptions | sent/topic |
| LexicalAgent | `agents/lexical_agent.py` | **paper_style** | No | No | **keyword match** (keyword_map) | labels ✓ / desc ✗ | sent/topic |
| LogicAgent | `agents/logic_agent.py` | **paper_style** | No | No | **regex** (rule_map) | labels ✓ / desc ✗ | sent/topic |
| TransformerContextualAgent | `agents/transformer_contextual_agent.py` | **paper_style** | No | No | **TF-IDF** cosine vs label descriptions (local, no download) | labels ✓ / desc ✓ | sent/topic |
| **LLMLexicalAgent** | `agents/llm_lexical_agent.py` | **full_agentic** | **Yes** | **Yes** | — | labels ✓ / desc ✓ | sent/topic |
| **LLMLogicAgent** | `agents/llm_logic_agent.py` | **full_agentic** | **Yes** | **Yes** | — | labels ✓ / desc ✓ | sent/topic |
| **ContextualAgent** | `agents/contextual_agent.py` | **full_agentic** | **Yes** | **Yes** | — | labels ✓ / desc ✓ | sent/topic |
| DeliberationAgent | `agents/deliberation_agent.py` | full_agentic (optional, **off by default**) | Yes* | **No** (fixed mock) | — | labels ✓ / desc ✓ | sent/topic |
| **ConsensusAgent** | `agents/consensus_agent.py` | classification (paper_style, full_agentic) | **No** | No | **weighted vote** (sum of weight·confidence) | labels ✓ / desc ✗ | sent/topic |
| ExplainabilityAgent (template) | `agents/explainability_agent.py` | paper_style; NER (non-full) | No | No | **string template** | task_name; not label list | sent/topic/NER |
| **LLMExplainabilityAgent** | `agents/llm_explainability_agent.py` | **full_agentic** (classification **and** NER) | **Yes** | **Yes** | — | task_name + final_label + agent summaries; not label list/desc | sent/topic/NER |
| NERLexicalAgent | `agents/ner_lexical_agent.py` | NER (paper_style, full_agentic) | No | No | **gazetteer/keyword** (deterministic) | tag labels ✓ | NER |
| NERLogicAgent | `agents/ner_logic_agent.py` | NER | No | No | **regex/rule** (deterministic) | tag labels ✓ | NER |
| NERContextualAgent | `agents/ner_contextual_agent.py` | NER | No | No | **deterministic** (no client) | tag labels ✓ | NER |
| NERConsensusAgent | `agents/ner_consensus_agent.py` | NER | No | No | **deterministic token-vote** | tag labels ✓ | NER |

\* DeliberationAgent *can* call `generate()`, but `build_orchestrator` binds it
to `MockLLMClient(fixed)`, so it never reaches OpenAI; and it is disabled by
default (`enable_deliberation: false`).

---

## Per-mode call profile

### 1. paper_style (classification)
On escalation: LexicalAgent → LogicAgent → TransformerContextualAgent →
ConsensusAgent → ExplainabilityAgent(template). **Zero `BaseLLMClient` calls;
zero OpenAI calls even with `--llm_client openai`.** Fully deterministic/local.

### 2. full_agentic + `--llm_client mock`
On escalation: LLMLexicalAgent, LLMLogicAgent, ContextualAgent, then
ConsensusAgent, then LLMExplainabilityAgent. All four LLM agents call
`MockLLMClient(label_echo)` — **no real API**, deterministic mock JSON.

### 3. full_agentic + `--llm_client openai`
Same four agents call **OpenAIClient → real GPT-4o-mini**. ConsensusAgent and
Router stay deterministic. **4 OpenAI calls per escalated sample** (see below).

### 4. NER path
NERLexical → NERLogic → NERContextual → NERConsensus (all deterministic, no
client) → Explainability. Explainability is the **shared** agent: template in
paper_style, **LLMExplainabilityAgent** in full_agentic — so NER `full_agentic` +
`--llm_client openai` would make **1 OpenAI call/sample** (explainability only).
No NER lexical/logic/contextual/consensus ever calls an LLM.

---

## Specific confirmations requested

| Question | Answer |
|---|---|
| LLMLexicalAgent — real LLM call? | **Yes**, when `--llm_client openai` (else mock). |
| LLMLogicAgent — real LLM call? | **Yes**, when openai (else mock). |
| ContextualAgent — real LLM call? | **Yes**, when openai (full_agentic only; paper_style uses TransformerContextualAgent instead). |
| LLMExplainabilityAgent — real LLM call? | **Yes**, when openai (full_agentic). |
| DeliberationAgent — real LLM call? | **No** — bound to a fixed mock; and off by default. |
| ConsensusAgent — LLM or deterministic? | **Deterministic** weighted-vote combiner. No LLM. |
| Router — LLM or deterministic? | **Deterministic** threshold (`conf ≥ task_config.threshold`). No LLM. |
| LexicalAgent / LogicAgent / TransformerContextualAgent | **Local/deterministic** — keyword / regex / TF-IDF. No API. |
| NERLexical/Logic/Contextual/Consensus | **All deterministic.** No LLM client. |

---

## Real GPT-4o-mini pilot — call accounting

- **LLM calls per escalated classification sample: 4.**
- **Produced by:** LLMLexicalAgent (1) + LLMLogicAgent (1) + ContextualAgent (1)
  — run during escalation — and **LLMExplainabilityAgent (1)** — run after
  ConsensusAgent. ConsensusAgent and Router add **0** calls.
- Non-escalated (accepted-primary) samples make **0** LLM calls.
- Pilot evidence: mBERT 47 escalated → 186 calls ≈ 4×47; XLM-R 41 → 164 ≈ 4×41.

**Does ExplainabilityAgent affect the final label?** **No.** ConsensusAgent
writes `state.final_output` (the label); the explainability agent runs *after*
and writes only `state.explanation_output`
([explainability_agent.py:215](../../../src/agents/explainability_agent.py#L215),
[llm_explainability_agent.py:204](../../../src/agents/llm_explainability_agent.py#L204)).
It is **explanation-only** — 1 of the 4 calls (~25% of pilot cost) has zero
effect on accuracy/F1.

**DeliberationAgent disabled → zero calls?** **Confirmed.** With
`enable_deliberation: false` (default), `build_orchestrator` sets
`deliberation_agent = None`, the orchestrator skips the deliberation stage
entirely, and even if enabled it would use the fixed mock — never OpenAI.

---

## Cost-relevant takeaways (no change made)
- Only 3 of the 4 paid calls per escalated sample influence the label
  (lexical/logic/contextual → consensus); the 4th (explainability) is cosmetic.
- Making explainability template-based (or optional) in full_agentic would cut
  ~25% of API cost with no accuracy impact — a candidate for the later refactor
  (not done here).
