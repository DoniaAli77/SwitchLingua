# Design G2 — The Selective IntentGate: Complete Technical Report

Consolidated, code-verified reference for the Selective IntentGate used in the
final sentiment-classification architecture. Every claim below is traced to the
executed implementation or to recorded run artifacts; inferences from naming or
from prior prose have been excluded. Compiled 2026-08-04.

---

## 1. Executive summary

| Property | Value |
|---|---|
| Variant string | `lexical_polarity_contextual_selective_gate` |
| Voting agents | Lexical, Polarity (logic slot), Contextual — 3 voters, weight 1.0 each |
| Gate vote weight | **0.0 — never votes** |
| Gate role | Conditional, one-directional veto over consensus overrides |
| Can restore | **Only** the primary's label — never a third label, never its own |
| Fires when | consensus overrode the primary **and** the gate agrees with the primary |
| Accuracy (n=5) | **0.9306** (range 0.9291–0.9315), macro F1 **0.9266** |
| Gain over no-gate (Design C) | **+3 escalated samples** (64 → 67 of 84) |
| Gate firings | 6 of 84 escalated samples |

**Headline finding:** the gate's contribution is **architectural, not linguistic**.
Its criteria produce no benefit when placed inside a voting agent (Design H) and a
statistically significant benefit when applied as a post-consensus veto (§6.1).

---

## 2. Architecture — two distinct components

The name "IntentGate" denotes **two separate objects**. Conflating them is the most
common source of error in describing this design.

| Component | Class | Role | Votes? |
|---|---|---|---|
| **Judgement producer** | `IntentAgent(name="IntentGate", system_variant="selective")` | Calls the LLM, writes label to `state.polarity_output` | **No** — weight 0.0 |
| **Decision applier** | `IntentGateAgent` | Post-consensus guard; applies the restoration rule | n/a — a rule, not a voter |

Construction (`evaluate_pipeline.py`, G2 branch):
```python
llm_lexical_agent = LLMLexicalAgent(llm_client=llm_client)
llm_logic_agent   = PolarityAgent(llm_client=llm_client)
polarity_agent    = IntentAgent(llm_client=llm_client, output_attr="polarity_output",
                                name="IntentGate", system_variant="selective")
_consensus_polarity_weight = 0.0        # gate does NOT vote
_intent_gate_agent = IntentGateAgent()  # separate post-consensus guard stage
```

**Note on naming:** the object is an `IntentAgent` whose *display name* is set to
`"IntentGate"`. It occupies the 4th specialist slot structurally but contributes
nothing to the weighted sum. It should **not** be called "the fourth voter".

---

## 3. What makes it *selective*

G and G2 share an identical guard mechanism. The **only** difference between them
is the system prompt given to the judgement producer.

| Design | Prompt | Neutral criterion |
|---|---|---|
| G | `SYSTEM_PROMPT` (default) | Broad — any text lacking clear authorial evaluation |
| **G2** | `SYSTEM_PROMPT_SELECTIVE` | **Narrow** — genuine platform/meta/mention/reference only |

The selective prompt opens by explicitly disclaiming generality:

> *"You are a SELECTIVE authorial-intent gate… You are NOT a sentiment classifier
> and NOT a general opinion detector."*

**Neutral is returned only for:**
- platform/meta-comments (likes, dislikes, comments, shares, subscribers, view counts, other users' reactions)
- clip / video / song / lyric / episode references, plot or scene descriptions
- quotations, named-entity / brand / logo / media *spotting*
- questions or remarks **about other people's actions**

**A polar label is returned when a stance is expressed, even implicitly:**
- implicit insult, mockery, sarcasm, put-down → negative
- excited fan reaction, cheering, hype, affection → positive
- praise or criticism in slang / informal / misspelled form
- strong affective wording, exclamation, emotional emphasis

> *"RULE OF THUMB: absence of an explicit sentiment word is NOT enough for neutral."*

This narrowing is what stops G's over-blocking of implicit-opinion rescues while
retaining its meta/mention protections.

---

## 4. The restoration rule

Let \(y_p\) = primary prediction, \(y_c\) = consensus prediction, \(y_g\) = gate judgement.

\[
y_{\text{final}} =
\begin{cases}
y_p, & \text{if } y_c \neq y_p \text{ and } y_g = y_p \\
y_c, & \text{otherwise}
\end{cases}
\]

Implemented in `src/agents/intent_gate_agent.py` as three sequential early-returns:

| Guard condition | Line | Effect |
|---|---|---|
| consensus missing / label None | 53 | return unchanged |
| primary None **or** \(y_c = y_p\) | 61 | return unchanged (no override to guard) |
| \(y_g\) None **or** \(y_g \neq y_p\) | 71 | return unchanged (gate does not support primary) |
| all passed | 74+ | **restore** \(y_p\); append `intent_gate=BLOCKED override 'X'->'Y'` |

**Directional asymmetry:** the gate can only reverse a move *away from* the
primary. It cannot introduce a third label, and it cannot impose its own label
when that differs from the primary's. It is a brake, never a steering input.

**Confidence recomputation:** on firing, the reported confidence is recomputed as
`votes[primary] / active_weight_sum` (lines 80–84). Because only the primary voted
for the restored label, this number is typically very low (e.g. 0.142) and should
be described as a weighted vote share, not a probability.

---

## 5. Information flow

| Component | Receives | Does **not** receive |
|---|---|---|
| Gate judgement producer | raw text, allowed labels, label descriptions | consensus output; primary prediction* |
| Post-consensus guard | `consensus.label`, `primary.label`, `polarity_output.label` — **labels only** | text, reasoning, confidences (for the decision) |

\* `agents_use_primary_signal` was **off** in every reported run, so the primary
block rendered empty.

The gate therefore forms its judgement **blind** to both the consensus and the
primary, exactly like the three voting specialists; its label is only *compared*
against them afterwards.

---

## 6. Empirical results

### Main comparison (Ahmed frozen primary, EESA test n=818, 84 escalated)

| Config | Model | Accuracy | Macro F1 | Escalated |
|---|---|---|---|---|
| primary_only | — | 0.9254 | 0.9207 | 63/84 |
| Design C (no gate) | 4.1-mini | 0.9266 | 0.9216 | 64/84 |
| Design G (full gate) | 4o-mini | 0.9279 | 0.9257 | 65/84 |
| Design G (full gate) | 4.1-mini | 0.9291 | 0.9248 | 66/84 |
| **Design G2 (selective gate)** | 4.1-mini | **0.9303** | **0.9262** | **67/84** |

### Run-to-run variation (identical G2 config, n=5)

| Run | Accuracy | Macro F1 | Escalated |
|---|---|---|---|
| 1 (canonical) | 0.9303 | 0.9262 | 67/84 |
| 2 | 0.9315 | 0.9277 | 68/84 |
| 3 | 0.9291 | 0.9251 | 66/84 |
| 4 | 0.9315 | 0.9277 | 68/84 |
| 5 | 0.9303 | 0.9262 | 67/84 |
| **Mean** | **0.9306** | **0.9266** | **67.2/84** |
| **Range / SD** | 0.9291–0.9315 / 0.0010 | 0.9251–0.9277 / 0.0011 | 66–68 |

**Noise band: ±2 escalated samples ≈ ±0.0026 macro F1.** All 734 non-escalated
predictions are identical across runs (frozen primary); all variance arises in the
84 escalated samples where the LLM agents run.

### Gate firing behaviour

| Metric | Value |
|---|---|
| Escalated samples | 84 |
| Consensus overrode the primary | 29 |
| Gate fired (blocked an override) | **6** |
| — restored a **correct** primary label | 5 |
| — restored an **incorrect** primary label | 1 |
| Net accuracy effect vs Design C | **+3** escalated samples |

Firings: `00097`, `00193`, `00203`, `00330`, `00363` (correct restorations);
`00045` (incorrect restoration).

---

## 7. Ablations isolating the mechanism

### 7.1 Design H — prompt content vs. architectural position

The selective gate's criteria were merged **verbatim** into the Polarity agent's
prompt, with the separate gate removed (3 voters, no guard).

| Config | Accuracy | Macro F1 | Escalated |
|---|---|---|---|
| Design C (no gate) | 0.9266 | 0.9216 | 64/84 |
| **Design H (criteria inside a vote)** | 0.9267 | 0.9217 | **64/84** |
| Design G2 (criteria as a veto) | 0.9306 (mean) | 0.9266 | 67.2/84 |

**Design H is indistinguishable from having no gate at all** (ΔF1 = 0.0001,
identical 64/84), and sits 3.8 σ below the G2 mean — outside the measured noise
band. The identical criteria are worth **+3 samples as a veto and 0 inside a vote**.

*Mechanism:* in confidence-weighted voting one voter cannot outvote two agreeing
voters, however well-informed its prompt:
```
Polarity neutral @0.95 + primary neutral @0.50 = 1.45
Lexical negative @0.85 + Contextual negative @0.85 = 1.70   ← wins
```
The veto escapes this arithmetic by acting *after* the weighted sum.

### 7.2 Designs E / F — intent reasoning as a genuine voter

Both E (`lexical_intent_polarity_contextual`) and F (`intent_polarity_contextual`)
place `IntentAgent` as a **voting** 4th agent (`_consensus_polarity_weight = 1.0`).
Both were outperformed by the non-voting gate designs — independent corroboration
of §7.1.

### 7.3 G2-lazy — deferring the gate's LLM call

Because the gate's prompt never sees consensus or the primary, its call may be
deferred until the guard needs it (i.e. only on overrides).

| Config | Accuracy | Macro F1 | Escalated | Gate LLM calls |
|---|---|---|---|---|
| G2 (eager) mean, n=5 | 0.9306 | 0.9266 | 67.2/84 | 84 (always) |
| **G2-lazy** | 0.9303 | 0.9267 | 67/84 | **29 (65 % saved)** |

Decision-equivalent (identical 6 blocked overrides), inside the noise band, and
65 % cheaper on gate calls. Under this wiring the gate is genuinely a
post-consensus component.

---

## 8. Implemented vs. used vs. abandoned

| Aspect | Implemented in code | Used in reported G2 runs | Abandoned / unused |
|---|---|---|---|
| Gate prompt | 3 variants | `SYSTEM_PROMPT_SELECTIVE` | default (G), `SYSTEM_PROMPT_DISAMBIG` (`semantic_v2_disambig`, degraded results) |
| Gate vote weight | configurable | **0.0** | 1.0 (Designs E/F) |
| Guard placement | standalone post-consensus stage | active | legacy in-consensus implementation (behaviour identical) |
| Primary signal to gate | optional flag | **off** | on |
| Gate invocation timing | eager or lazy | **eager** (reported runs) | lazy (validated, decision-equivalent) |

---

## 9. Thesis wording guidance

**Accurate description:**
> The final architecture augments the three voting specialists with a fourth
> LLM-based component that casts no vote. Rather than judging sentiment, it decides
> whether a message is a platform-level or referential utterance — a remark about
> likes or other users' reactions, a reference to a clip or lyric, a quotation or
> brand mention, a question about someone else's behaviour — or an utterance in
> which the author expresses an evaluative stance. The absence of explicit sentiment
> vocabulary is treated as insufficient grounds for a neutral reading: implicit
> insults, mockery, informal praise and affective emphasis are all classified as
> stance-bearing. This judgement is consulted only after the weighted consensus has
> been computed, and only when the consensus has overturned the primary
> classifier's prediction; if the component independently agrees with the primary,
> the override is withheld and the primary's label restored. In all other cases the
> consensus stands. Because it never introduces a label of its own and never acts
> when consensus and primary already agree, it functions as a one-directional veto
> rather than an additional classifier or a general verification stage.

**Avoid these formulations — none is supported by the implementation:**

| Do not write | Why |
|---|---|
| "a general verification step" | it verifies nothing in general; it answers one binary meta/mention question |
| "a general opinion detector" | the prompt explicitly disclaims this |
| "the fourth voter" | vote weight is 0.0 |
| "validates the consensus" | it cannot endorse or alter consensus except by restoring the primary |
| "the gate decides the final label" | it can only choose between \(y_c\) and \(y_p\) |
| (in the eager wiring) "after consensus the gate determines…" | the LLM call happens *before* consensus; only the *consultation* is post-consensus. Accurate under G2-lazy. |

**Report accuracy as** 0.9306 (range 0.9291–0.9315, n = 5), and state the noise
band once so that ±1–2-sample differences elsewhere are not over-interpreted.

---

## 10. Reproduction

| Item | Location |
|---|---|
| Variant selection | `evaluate_pipeline.py` — G2 branch; `src/agents/_sentiment_agent_variant.py` |
| Judgement producer | `src/agents/intent_agent.py`; prompt `src/prompts/intent_prompt.py` (`SYSTEM_PROMPT_SELECTIVE`, `get_system_prompt("selective")`) |
| Post-consensus guard | `src/agents/intent_gate_agent.py` |
| Stage ordering | `src/pipeline/orchestrator.py` (specialists → consensus → intent_gate) |
| Canonical run | `experiment_G2_ahmed_41mini/` |
| Noise band (n=5) | `experiment_ahmed_g2_repeats/` · `EXPERIMENT_G2_NOISE_BAND.md` |
| Design H ablation | `experiment_ahmed_designH_merged_gate/` · `EXPERIMENT_DESIGNH_GATE_PROMPT_VS_POSITION.md` |
| G2-lazy | `experiment_ahmed_g2_lazy_gate/` · `EXPERIMENT_G2_LAZY_GATE.md` |
| Gate ablation (C / G / G2) | `EXPERIMENT_GPT41_GATE_ABLATION.md` |
| Selective-gate design rationale | `EXPERIMENT_SENTIMENT_SELECTIVE_INTENT_GATE_G2.md` |

Fixed configuration across all reported G2 runs: Ahmed precomputed primary
(char-CNN + BiLSTM over AraBERT features, external), EESA test 818 samples,
threshold 0.70 (84 escalations), `semantic_v1` prompts, consensus `w_primary` 1.0,
gpt-4.1-mini at temperature 0, `agents_use_primary_signal` off, no training and no
data generation.
