# Sentiment Polarity-Agent Variant (`lexical_polarity_contextual`) — Implementation Changelog

Design C from `EXPERIMENT_SENTIMENT_POLARITY_AGENT_REDESIGN_PROPOSAL.md`: an **opt-in**
sentiment agent-architecture variant that replaces the weak **Logic** agent with a
sentiment **Polarity** agent, keeping **Lexical** and **Contextual**. General sentiment
reasoning — **no dataset/benchmark named**, task-aware not dataset-aware. **Default
behaviour unchanged.** No training, no generation. **No paid LLM run** — only unit/smoke
tests so far. Date: 2026-06-30.

**Scope (NOT touched):** router, consensus, primary model, semantic_v1 prompts,
deliberation, NER path. The Polarity agent writes the **same `state.logic_output` slot**
the Logic agent uses, so it is a drop-in at the orchestrator's logic stage — no consensus
or router change was required.

---

## 1. Files changed

| file | change |
|---|---|
| `src/prompts/polarity_prompt.py` | **NEW.** Polarity system prompt (decision-framed, 6-step reasoning order) + `build_user_prompt` + `get_system_prompt`. Same 4-key JSON contract as the other agents. No dataset names. |
| `src/agents/polarity_agent.py` | **NEW.** `PolarityAgent` — mirrors `LLMLogicAgent` (same parse/validate/abstain logic, same JSON schema) but uses the polarity prompt and writes `state.logic_output`. Name = `"PolarityAgent"`. |
| `src/agents/_sentiment_agent_variant.py` | **NEW.** Resolver `active_agent_variant()` — reads env `SENTIMENT_AGENT_VARIANT` (default `"default"`), validates `{default, lexical_polarity_contextual}`, raises on unknown. |
| `evaluate_pipeline.py` | Import `PolarityAgent` + `active_agent_variant`; `build_orchestrator(..., sentiment_agent_variant=None)` swaps Logic→Polarity at the logic slot when the variant is active; new CLI flag `--sentiment_agent_variant {default,lexical_polarity_contextual}` (sets the env var, passed to the classification `build_orchestrator` call). |
| `tests/test_polarity_agent_variant.py` | **NEW.** 12 tests (agent-list wiring, env gating, prompt cleanliness, JSON contract, bad-value raise, agent behaviour). |

**No changes** to: `consensus_agent.py`, `router.py`, `orchestrator.py`, the primary
classifier, the semantic_v1 prompt modules, or the Lexical/Contextual agents.

---

## 2. How to enable the variant (opt-in only)

```bash
# experimental trio: Lexical + Polarity + Contextual
python evaluate_pipeline.py ... --sentiment_agent_variant lexical_polarity_contextual

# combine with the semantic_v1 prompts (intended for the validation run):
python evaluate_pipeline.py ... \
  --sentiment_prompt_variant semantic_v1 \
  --sentiment_agent_variant lexical_polarity_contextual
```
or set `SENTIMENT_AGENT_VARIANT=lexical_polarity_contextual` directly. Omitting the flag
(or `default`) keeps **Lexical + Logic + Contextual**. The two flags are **independent**:
`--sentiment_prompt_variant` governs Lexical/Contextual wording; `--sentiment_agent_variant`
governs which trio runs. An unknown value raises `ValueError`.

**Mechanism:** `build_orchestrator` resolves the variant and, when active, passes a
`PolarityAgent` into the orchestrator's `llm_logic_agent` slot. The orchestrator's
full_agentic logic stage runs whatever sits in that slot and writes `state.logic_output`;
consensus reads lexical/logic/contextual outputs unchanged. Default = `LLMLogicAgent`.

---

## 3. Exact Polarity Agent role

Single question: **"Is the author expressing an evaluative attitude? If yes, what
polarity?"** Reasoning order baked into the system prompt:
1. Decide whether the author expresses an evaluative opinion **at all**.
2. If yes, decide polarity: positive / negative / neutral.
3. If the text only **mentions** sentiment words, platform actions, plot events, lyrics,
   clips, emojis, slogans, or **other people's reactions** without the author's own
   stance → neutral or low confidence.
4. Account for explicit sentiment words, **negation, intensifiers, mixed polarity,
   emojis, weak cues, short informal comments**.
5. Separate **expressing** polarity from merely **mentioning** it.
6. **Lower confidence** when polarity is weak, artifact-based, target-ambiguous, or only
   surface-implied.

Explicit boundaries in the prompt: it does **not** perform full pragmatic/social
interpretation (Contextual's job) and does **not** merely list lexical cues (Lexical's
job). Output is the identical 4-key JSON (`label`, `confidence`, `reasoning`, `evidence`)
with the same `OUTPUT FORMAT` contract as every other classifier agent.

---

## 4. How this differs from the Logic agent

| | Logic agent (default) | **Polarity agent (variant)** |
|---|---|---|
| core question | structural/relational reasoning (entity-action-object, co-occurrence, discourse cues) | "is an evaluation expressed, and what polarity?" |
| output | a label from structural patterns | a **polarity decision** (expression-gated) |
| mention vs express | not its focus | **central** — gate polarity on the author actually expressing it |
| artifacts (UI words/emoji/plot) | no explicit handling | explicit weak-cue / mention handling → neutral/low-conf |
| confidence discipline | generic | **lower confidence when weak/ambiguous/target-unclear** |
| measured weakness it targets | Logic was the weakest agent (0.690) and ~0.89 correlated with Lexical | replace it with a distinct decider to decorrelate + own the neutral↔polar boundary |
| state slot written | `state.logic_output` | **`state.logic_output`** (same — drop-in) |

In short: Logic asked *"what structure does the text have?"*; Polarity asks *"does the
author actually evaluate, and how?"* — the question the B1 error analysis showed was the
real failure axis (mention-vs-express, target attribution, neutral↔polar boundary).

---

## 5. Expected impact
- **Decorrelation:** a Polarity decider reasons differently from the Lexical reporter and
  the Contextual interpreter → expect all-3 agreement to fall further (was 91.7%→84.5%
  with semantic_v1) and per-agent accuracy at the logic slot to rise above Logic's 0.690.
- **Fewer literalism breaks:** the explicit mention-vs-express + artifact rules target the
  remaining neutral→polar over-reads (the dominant break type).
- **Strong-primary regime (Ahmed):** realistic target is **harm-reduction** — match or beat
  semantic_v1's net −2; not expected to beat primary_only 0.9254 (agent-ceiling, per the
  consensus-simulation finding).
- **Weak-primary regime (C3):** keeping the 3-voter structure should **preserve** the
  +0.059 rescue (validated only if/after the Ahmed run is safe).

## 6. Risks
1. **Lexical↔Polarity re-correlation.** Both read sentiment vocabulary; if they collapse
   into a new redundant pair, the decorrelation gain is lost. **The validation must measure
   their pairwise agreement** — the trigger to fall back to Design B (merge them).
2. **Over-conservatism / neutral bias.** The repeated "lower confidence / neutral when
   weak" guidance could push toward neutral and lose genuine polar wins (the one
   semantic_v1 regression was this pattern). Watch W→C and the neutral rate.
3. **Loss of structural signal.** Removing Logic drops explicit negation/contrast/structure
   reasoning; the Polarity prompt re-includes negation/intensifiers, but a structural case
   Logic caught could regress.
4. **C3 erosion.** Fewer-distinct-but-equal-count voters could change the rescue dynamics
   on a weak primary; only the B-stage C3 check confirms the +0.059 holds.
5. **Cross-task leakage.** The Polarity prompt is sentiment-specific ("(polarity)"-framed);
   the variant is intended for sentiment only. Default-off makes this safe by construction.

## 7. Validation plan

### Done now — unit/smoke (no LLM calls)
- **`tests/test_polarity_agent_variant.py` — 12 tests pass.** Verifies: default trio =
  Lexical+Logic+Contextual; variant = Lexical+**Polarity**+Contextual (via param **and**
  env var); explicit `default` overrides env; bad value raises; Polarity prompt has **no**
  `eesa/arensa/ahmed/twitter/arsentd/tweet`; Polarity prompt preserves the JSON `OUTPUT
  FORMAT` contract + 4 keys; PolarityAgent writes a valid `state.logic_output` and abstains
  (None label, never labels[0]) on an invalid label.
- **Full suite: 909 passed** (897 prior + 12 new) — no regressions.
- **Mock end-to-end smoke:** with both flags set, `build_orchestrator` puts `PolarityAgent`
  in the logic slot and a full_agentic run completes, `logic_output` authored by
  `PolarityAgent`.

### Pending your approval — paid A/B (NOT run)

**A. Strong-primary safety/improvement — Ahmed frozen-primary, threshold 0.7.**
Run with `--sentiment_prompt_variant semantic_v1 --sentiment_agent_variant
lexical_polarity_contextual`, all else identical to the prior Ahmed runs.

Compare against:
| baseline | accuracy | macro F1 | net (escalated) |
|---|---|---|---|
| Ahmed primary_only | 0.9254 | 0.9207 | — |
| old full_agentic | 0.9205 | 0.9153 | −4 |
| semantic_v1 full_agentic | 0.9230 | 0.9183 | −2 |

Measure: accuracy, macro F1, escalated accuracy, wrong→correct, correct→wrong, net, **all-3
agent agreement**, **Lexical↔Polarity agreement** (the C→B trigger), **Polarity accuracy on
the escalated subset** (does it beat Logic's 0.690?), neutral→negative / neutral→positive
breaks, and whether the remaining artifact/literalism failures decrease (old↔new per-sample
diff, via a deterministic per-agent capture as in B1).

**B. Weak-primary gain guard — C3 generated-primary full_agentic.**
Only **if A improves or stays safe**, later run C3 with the same flags to confirm the prior
**+0.059** gain is not destroyed.

**Do not run any paid evaluation until approved.** The default (no flags) reproduces the
existing behaviour exactly — the gate is safe.

---

## Appendix — reproduction command (when approved)
```bash
python evaluate_pipeline.py \
  --dataset data/Sentiment/external/ahmed/ahmed_eesa_test_dataset.jsonl \
  --config src/config/default.yaml --active_task sentiment_classification \
  --mode full_pipeline --pipeline_mode full_agentic --threshold 0.7 \
  --primary_model precomputed \
  --precomputed_predictions data/Sentiment/external/ahmed/ahmed_eesa_test_predictions_aligned.csv \
  --llm_client openai --llm_model gpt-4o-mini --consensus_primary_weight 1.0 \
  --sentiment_prompt_variant semantic_v1 \
  --sentiment_agent_variant lexical_polarity_contextual \
  --output_dir experiments/outputs/multi_agent_bert/experiment_ahmed_polarity/full_agentic_th07_polarity \
  --run_id ahmed_full_agentic_th07_polarity
```
