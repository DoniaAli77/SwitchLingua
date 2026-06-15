# Detailed Architecture — SwitchLingua + Multi-Agent BERT (Exp C)

Three views: (1) the SwitchLingua generation pipeline (LangGraph), (2) the Multi-Agent BERT
classification architecture, (3) the Experiment-C bridge that connects them. Diagrams reflect the
actual code (`Modified_Version/core/run_french.py`, `multi-agent-bert/src/pipeline/orchestrator.py`).

================================================================================================
## 1. SwitchLingua generation pipeline — System B (the contribution)
LangGraph `StateGraph(AgentRunningState)`; one graph run = one scenario (a batch of N sentences).
================================================================================================
```
   CONFIG (pre_execute) ──► generate_scenarios()  [Cartesian product of dims]
        cs_ratio · task · topic · cs_type · cs_function · tense · perspective
        · gender · age · education · sentiment{labels,intensity,ambiguity}
                              │  one scenario dict {task,label,cs_ratio,cs_type,task_constraints,…}
                              ▼
        ┌─────────────────────────────────────────────────────────────────────────────────────┐
        │ START                                                                                 │
        │   │                                                                                   │
        │   ▼                                                                                   │
        │ ① DataGenerationAgent ── LLM(gpt-4o-mini) → instances = [s1,s2,…,sN]                  │
        │        (DATA_GENERATION_*_PROMPT per task; sentiment/topic/NER)                       │
        │   │                                                                                   │
        │   ▼                                                                                   │
        │ ② TaskValidatorAgent ── per sentence: does it satisfy the task?                       │
        │        topic→on-topic · sentiment→label match · NER→required entities (English script)│
        │        writes task_validation_results_per_instances[i] {passed,predicted_label,…}     │
        │   │                                                                                   │
        │   ├──────────────┬──────────────┬───────────────┬───────────────────┐  (PARALLEL fan-out)
        │   ▼              ▼              ▼               ▼                   │                  │
        │ ③ Fluency    ④ Naturalness  ⑤ CSRatio       ⑥ SocialCultural       │                  │
        │   /10           /10           /10 ◄── deterministic CS counter      │                  │
        │                               compute_true_cs_stats(text)          │                  │
        │                               (Arabic vs Latin tokens; 0 variance) │                  │
        │   └──────────────┴──────────────┴───────────────┴───────────────────┘                │
        │                              ▼  (fan-in)                                              │
        │ ⑦ SummarizeResult ── PER-SENTENCE weighted_score = w·{flu,nat,cs,socio};              │
        │        builds sentence_records[i] {text,weighted_score,task_passed,…}                 │
        │   │                                                                                   │
        │   ▼  conditional: meet_criteria(state)   ◄═══════ THE CONTRIBUTION ═══════            │
        │   │      per sentence i:  PASS  iff  weighted_score[i] ≥ bar  AND  task_passed[i]      │
        │   │      (System C instead tests the SCENARIO MEAN → "masking" of one weak sentence)  │
        │   │                                                                                   │
        │   ├───────────────── all sentences pass / refine-budget spent ──────────► ⑨ Acceptance│
        │   │                                                                          │  →JSONL │
        │   └── some sentence fails (and budget left) ─► ⑧ RefinerAgent                │        │
        │                                                  │  targeted + task-aware rewrite     │
        │                                                  │  of ONLY the failing sentences     │
        │                                                  │  GUARDRAIL: re-validate + re-score; │
        │                                                  │  rollback if task breaks or score↓ │
        │                                                  │  (refine_count[i]++ every attempt) │
        │                                                  ▼                                    │
        │                                            back to ② TaskValidatorAgent (loop)        │
        │                                                                                       │
        │ ⑨ AcceptanceAgent ──► END   (writes per-sentence records to OUTPUT_DIR/Arabic.jsonl)  │
        └─────────────────────────────────────────────────────────────────────────────────────┘

 EDGES (verbatim): START→Data→TaskValidator→{Fluency,Naturalness,CSRatio,SocialCultural}→Summarize
                   Summarize ─conditional(meet_criteria)→ Refiner | Acceptance ;  Refiner→TaskValidator ;  Acceptance→END
 SYSTEM C (control): same nodes, but the accept/refine decision uses the SCENARIO-AGGREGATE score
                     + a generic refiner, topic task only.  This is the head-to-head baseline.
```

================================================================================================
## 2. Multi-Agent BERT — classification architecture
`PipelineOrchestrator.run(state)`; primary classifier + router + specialist agents, 3 modes.
================================================================================================
```
                          INPUT: code-switched sentence  ──►  PipelineState
                                          │
                                          ▼
        ┌──────────────────────────── PRIMARY CLASSIFIER ────────────────────────────┐
        │  --primary_model:                                                           │
        │    • mock        → MockPrimaryClassifier (non-deterministic; smoke only)    │
        │    • transformer → PrimaryTransformerClassifier (xlm-roberta-base)          │
        │                      ← THIS is the Exp-C fine-tuned checkpoint              │
        │  output: predicted_label + probabilities {positive,negative,neutral}        │
        └───────────────────────────────────┬────────────────────────────────────────┘
                                             │ state.primary_model_output
                ┌────────────────────────────┴──────────── MODE = primary_only ──► (skip everything below) ──► FINAL
                ▼
        ┌──────────────────────────── ROUTER ────────────────────────────────────────┐
        │  confidence vs threshold →  decision ∈ {accept_primary , escalate}          │
        └───────┬───────────────────────────────────────────────┬────────────────────┘
                │ accept_primary (FAST path)                      │ escalate (SLOW path)
                ▼                                                 ▼
        ┌───────────────────────┐          ┌──────────────────── SPECIALIST AGENTS (parallel) ──────────────────┐
        │ ExplainabilityAgent   │          │  MODE = paper_style (non-LLM, reference framework):                │
        │  short "why accepted" │          │     LexicalAgent · LogicAgent · ContextualAgent(paper)            │
        │  → final_output       │          │  MODE = full_agentic (LLM-backed):                                │
        └───────────┬───────────┘          │     llm_LexicalAgent · llm_LogicAgent · ContextualAgent           │
                    │                       │        └─ (optional) DeliberationAgent  [if enable_deliberation]  │
                    │                       └───────────────────────────────┬───────────────────────────────────┘
                    │                                                       ▼
                    │                                              ┌────────────────────┐
                    │                                              │ ConsensusAgent     │  combine specialists
                    │                                              │ → final_output     │  → escalated label
                    │                                              └─────────┬──────────┘
                    │                                                        ▼
                    │                                              ┌────────────────────┐
                    │                                              │ ExplainabilityAgent│  full escalated explanation
                    │                                              └─────────┬──────────┘
                    └──────────────────────────────┬─────────────────────────┘
                                                    ▼
                                          FINAL: label + explanation + routing_info
                                          (NER task has parallel ner_{lexical,logic,contextual,consensus} agents)

 MODES:  primary_only  = classifier only (router + agents skipped)   ← Exp C currently evaluated here
         paper_style   = + router + non-LLM specialists on escalation
         full_agentic  = + router + LLM specialists (+ optional deliberation)  ← not run while primary is weak
 ERROR handling: any stage exception → state.extras["pipeline_error"] {stage,message}, state returned.
```

================================================================================================
## 3. Experiment-C bridge — how the two systems connect
SwitchLingua *generates the training data*; the fine-tuned model *becomes the BERT primary classifier*.
================================================================================================
```
  ┌─ SwitchLingua System B (gpt-4o-mini) ─┐     ┌──────────── FILTER (every example) ───────────┐
  │  generate_scenarios → graph run        │ raw │ non-empty                                     │
  │  (config_sentiment_expC_v3.yaml:       │────►│  → TaskValidator passed                       │
  │   cs_ratio[50,60], Intrasentential)    │~4.7 │  → deterministic CS-valid (is_code_switched)  │
  └────────────────────────────────────────┘ /scn│  → quality ≥ 7.0  → de-dup (normalized text)  │
                                                  └───────────────────────┬───────────────────────┘
                  manage_sentiment_data.py        kept                     ▼
        ┌───────────── ACCUMULATE (resume-safe, append-only) ─────────────────────────────┐
        │  pilot_v1/ (frozen)  +  daily_runs/run_YYYYMMDD_*  +  completed_scenarios_v3.json│
        │  merge → cross-dedup → balance to N/label                                        │
        └───────────────────────────────────────┬─────────────────────────────────────────┘
                                                 ▼  switchlingua_sentiment_train_*.jsonl (text,label,+meta)
                              fine-tune  xlm-roberta-base  (HF Trainer, Adafactor)
                                                 ▼
                              experiments/checkpoints/expC_switchlingua_xlmr_240/
                                                 ▼  loaded as PrimaryTransformerClassifier
                              evaluate_pipeline.py --primary_model transformer --pipeline_mode primary_only
                                                 ▼
                              EESA test (818, REAL)  →  acc 0.590 / macro-F1 0.562
                              (vs Exp A train-on-real-EESA: 0.831 / 0.819)
```

### Legend / notes
- **Per-sentence vs aggregate** (View 1, node ⑦→⑧) is the entire SwitchLingua thesis lever.
- The **deterministic CS counter** is reused in two places: inside generation (CSRatioAgent) and as the
  **filter** in the Exp-C bridge (View 3) — same `compute_true_cs_stats`.
- Exp C currently exercises only the **primary_only** path of View 2 (the agentic paths are built but
  intentionally unused while the primary classifier is weak).
```
```
