# Architecture Spec — Parallel Design G vs Sequential V1 vs Sequential V2

Reproducible, code-verified definitions for the thesis. Every claim below is
confirmed from source or recorded run artifacts (not inferred from names or
results). Task: Arabic–English code-switched sentiment, Ahmed frozen primary,
EESA test (818). Date compiled: 2026-07-19.

---

## 1. Exact agent order

**Parallel Design G** — variant `lexical_polarity_contextual_intent_gate`
(`evaluate_pipeline.py:787-796`). Execution order (`orchestrator.py:383-419`):

```
primary → router → lexical_agent (LLMLexicalAgent)
                 → logic_agent   (PolarityAgent → writes logic_output)
                 → contextual_agent (ContextualAgent, LLM)
                 → polarity_agent (IntentAgent name="IntentGate", weight 0.0 — NON-VOTING)
                 → consensus_agent (ConsensusAgent)
                 → intent_gate    (IntentGateAgent — post-consensus guard)
                 → explainability
```
The four specialists run one-after-another but are **independent** (none reads
another's output) — "parallel" in the design sense.

**Sequential V1** — variant `sequential_sentiment_v1`
(`evaluate_pipeline.py:741-745`; orchestrator `328-354`):
```
seq_intent_stage   (SeqIntentAgent)
→ seq_polarity_stage (SeqPolarityAgent)
→ seq_pragmatic_stage (SeqPragmaticAgent)
→ sequential_controller (SequentialController, no LLM)
→ explainability
```

**Sequential V2** — variant `sequential_sentiment_v2`
(`evaluate_pipeline.py:754-758`):
```
seq_intent_stage             (SeqV2IntentAgent)
→ seq_pragmatic_features_stage (SeqV2PragmaticFeaturesAgent)
→ seq_polarity_resolver_stage  (SeqV2PolarityResolverAgent)
→ sequential_controller        (SequentialControllerV2, no LLM)
→ explainability
```

---

## 2. Information passed between agents

| Agent | Sentence | Primary label/conf | Preceding outputs | Emits |
|---|---|---|---|---|
| G — Lexical / Polarity / Contextual / IntentGate(LLM) | Yes | **No** | **None (independent)** | label, confidence, reasoning, evidence |
| V1 S1 Intent | Yes | No | — | opinion_expressed, target, speech_act, use_vs_mention, confidence, evidence |
| V1 S2 Polarity | Yes | No | **S1 full JSON** | label, confidence, mixed, reasoning, evidence |
| V1 S3 Pragmatic | Yes | No | **S1 + S2 full JSON** | keep_or_revise, final_label, confidence, reasoning, evidence |
| V2 S1 Intent (lean) | Yes | No | — | opinion_expressed, target, confidence, evidence |
| V2 S2 Pragmatic Features | Yes | No | **S1 full JSON** | structured features, **NO label** |
| V2 S3 Polarity Resolver | Yes | No | **S1 + S2 full JSON** | label, confidence, used_features, reasoning, evidence |

Verified notes:
- Sequential stages receive **all** preceding stage outputs as **complete raw
  JSON** (labels + reasoning + confidence + evidence), via `json.dumps` in the
  prompt builders — not just the immediate predecessor.
- **No LLM in any of the three receives the primary label/confidence.** G gates
  it behind `agents_use_primary_signal`, which was **off** (default; no override
  in run logs; `llm_lexical_agent.py:102-104`). V1/V2 use the primary **only in
  the deterministic controller** as a fallback (`sequential_sentiment.py:531-538`).
- G's consensus consumes **only `(label, confidence)`** (`_extract_vote` discards
  reasoning/evidence/probabilities). The IntentGate guard consumes **labels only**.

---

## 3. Exact V1 → V2 difference (not "stronger reasoning")

| | V1 | V2 |
|---|---|---|
| Stage 1 | Intent, 6 fields (`_INTENT_REQUIRED`) incl. speech_act, use_vs_mention | **Lean** intent, 4 fields (`_INTENT_V2_REQUIRED`); "do NOT analyze sarcasm here" |
| Stage 2 | **Polarity resolver — OUTPUTS a label** | **Pragmatic feature extractor — NO label** (sarcasm_or_irony, implicit_stance, use_vs_mention, description_vs_evaluation, target_attribution, stance_strength) |
| Stage 3 | **Pragmatic verifier — shown the proposed label**, decides keep/revise | **Polarity resolver — decides label once, fresh**; "NOT reviewing or ratifying a previous label" |
| Controller | 4 branches incl. pragmatic_revision / polarity_kept (`sequential_sentiment.py:494-528`) | keep/revise **removed**; no-opinion gate → feature-aware polarity → fallback (`sequential_sentiment_v2.py:298-334`) |
| Neutral gate | Rule 1 on **intent alone** + escape hatch | Rule 1 **cross-checked**: opinion False AND intent_conf≥τ AND implicit_stance==none AND (use_vs_mention≠use OR description==description) |

One line: **V2 deletes the label-ratification stage, inserts a no-label
structured-feature stage before the decision, makes the single label decision
last, and replaces keep/revise with a two-stage-agreement neutral gate.**

---

## 4. Consensus placement

- **G:** consensus runs **after all specialists** (`orchestrator.py:411`).
  Voters = lexical + logic(Polarity) + contextual (3) + primary (w=1.0);
  polarity_output (IntentGate LLM) present but **weight 0.0 → not counted**.
  Fusion = `ConsensusAgent` weighted hard-vote.
- **V1 & V2:** **no ConsensusAgent runs.** The sequential branch returns before
  the consensus stage; the deterministic controller writes `consensus_output` +
  `final_output` directly (`sequential_sentiment.py:457-471`; `v2:263-275`).
  Specialists produce **no separate votes** — a precedence rule picks the label.

---

## 5. IntentGate placement

- **G:** post-consensus `IntentGateAgent` guard runs **only after consensus**
  (`orchestrator.py:418`). Inputs = **labels only**: consensus label, primary
  label, gate (polarity_output) label (`intent_gate_agent.py:52-71`). Blocks a
  consensus override of the primary iff the gate sides with the primary.
  ("IntentGate" in G = a pre-consensus non-voting LLM agent **plus** this guard.)
- **V1/V2:** **no IntentGateAgent.** Intent reasoning appears **earlier**, as
  Stage 1; no-opinion→neutral is folded into controller Rule 1, described in code
  as "the IntentGate, promoted to a first-class branch"
  (`sequential_sentiment.py:386-390`).

---

## 6. Controlled settings

| Setting | Fixed across all three? | Evidence |
|---|---|---|
| Primary predictions | Yes — Ahmed precomputed, 818 | seqv1/seqv2 log L3; G same escalated count |
| Routing threshold | Yes — 0.70 | seqv1/seqv2 log L2; G escalation_rate 0.1027 = 84/818 |
| Specialist LLM | Yes — gpt-4o-mini | seqv1/seqv2 log L5; G llm_usage.json |
| agents_use_primary_signal | Yes — OFF | no override line in logs |
| Specialist prompts | **No — differ by construction** | G = parallel semantic_v1 prompts; V1/V2 own stage prompts; V1≠V2 (§3) |
| Consensus rule | **No — replaced** | G = ConsensusAgent; V1/V2 = deterministic controller |
| IntentGate prompt | **Not comparable** | G = full IntentAgent gate; V1/V2 = none |

Honest framing: **primary, threshold, LLM, and primary-signal-off are constant;
prompts, fusion mechanism, and control logic are the independent variable and
cannot be equal because the architectures differ.**

---

## 7. Run identification (from *_full_pipeline_metrics.json)

| Reported F1 | Run dir | run_id | macro_f1 / acc | escalated |
|---|---|---|---|---|
| G = 0.9242 | experiment_ahmed_designG_intent_gate | ahmed_designG__full_pipeline | 0.9242 / 0.9279 | 84/818 |
| V1 = 0.9195 | experiment_seqv1_ahmed | seqv1_ahmed__full_pipeline | 0.9195 / 0.9242 | 84/818 |
| V2 = 0.9076 | experiment_seqv2_ahmed | seqv2_ahmed__full_pipeline | 0.9076 / 0.9120 | 84/818 |

---

## 8. Compact comparison

| Dimension | Parallel G | Sequential V1 | Sequential V2 |
|---|---|---|---|
| Specialists | Lexical, Polarity, Contextual (+ non-voting IntentGate) | Intent → Polarity → Pragmatic-verifier | Intent(lean) → Pragmatic-features → Polarity-resolver |
| Topology | Independent (parallel) | Staged, sees all prior | Staged, sees all prior |
| Stage-2 emits label? | n/a | Yes | **No (features only)** |
| Final label decided | by voting | S2, possibly revised by S3 | **S3 once**, feature-aware |
| Fusion | ConsensusAgent (weighted hard-vote) + primary vote | Deterministic controller (keep/revise) | Deterministic controller (no keep/revise) |
| Neutral gate | post-consensus guard (labels only) | controller Rule 1 (intent-keyed) | controller Rule 1 (cross-checked) |
| Primary role | 4th voter (w 1.0) | router + fallback | router + fallback |
| Macro-F1 (818) | **0.9242** | 0.9195 | 0.9076 |

---

## 9. Thesis-ready methodological paragraph

All three configurations share an identical front end: the external Ahmed
classifier as a frozen primary (precomputed predictions on the 818-sample EESA
test set), a confidence router at threshold 0.70 (escalating 84 samples), and
gpt-4o-mini specialists with the primary signal withheld from every agent. They
differ only in how escalated samples are reasoned about and fused. **Parallel
Design G** runs three independent specialists — a lexical-cue agent, a polarity
decider, and a contextual agent — whose (label, confidence) votes are combined by
a confidence-weighted consensus in which the primary participates as a fourth
vote (weight 1.0); a non-voting IntentGate then runs post-consensus and blocks a
consensus override of the primary only when an intent agent independently sides
with the primary. **Sequential V1** replaces voting with a three-stage cascade —
intent (opinion-existence) → polarity (label) → pragmatic verification
(keep/revise the proposed label) — resolved by a deterministic controller; the
primary is used only as a fallback. **Sequential V2** removes the
confirmation-anchored review: pragmatics is extracted upstream as label-free
structured features, the sentiment label is decided exactly once by a final
feature-aware resolver that is never shown a prior label, and the controller
drops keep/revise in favour of a cross-checked no-opinion gate requiring intent
and pragmatic features to agree. On the escalated EESA subset the parallel voting
design attained the highest macro-F1 (0.9242), ahead of Sequential V1 (0.9195)
and Sequential V2 (0.9076).

---

## 10. Warnings — inaccurate/unsupported statements to avoid

1. **Not a clean "parallel vs sequential" ablation.** Prompts, fusion, and
   control logic change together; attributing the F1 gap solely to topology is
   unsupported.
2. **Sequential pipelines perform no voting/consensus.** Calling their output a
   "consensus/majority vote" is wrong — it is a deterministic controller.
3. **V2 is not merely "stronger forward reasoning."** State the concrete change
   (§3).
4. **Two different "IntentGates."** In G it is a pre-consensus non-voting LLM
   agent plus a post-consensus guard; the sequential Stage-1 intent is different;
   the guard does not exist in V1/V2.
5. **Confidence numbers not comparable across families** (weighted vote-share vs
   stage confidence).
6. **Single-run, temperature-0.** The 0.9242 / 0.9195 / 0.9076 spread is ~14
   samples on 818; the G→V1 gap (~4 samples) is within ±1–2-sample LLM
   non-determinism. Report as single runs unless repeated.
