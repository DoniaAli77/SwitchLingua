# Sentiment Prompt Variant `semantic_v1` — Implementation Changelog

Prompt-only experimental variant for the three sentiment LLM specialist agents
(Lexical / Logic / Contextual), implementing the approved role redesign from
`EXPERIMENT_SENTIMENT_AGENT_ROLE_REDESIGN.md`. **General sentiment-reasoning
guidance only — no dataset/benchmark named, no dataset tailoring.** Date: 2026-06-30.

**Scope (what was NOT touched):** architecture, consensus, router, training,
data generation. Only system-prompt text + a gating flag. The default behaviour
is **byte-identical** to before (verified — see Validation §A). No paid LLM
evaluation was run; that is gated on your approval of this changelog.

---

## 1. Exact files changed

| file | change |
|---|---|
| `src/prompts/_sentiment_variant.py` | **NEW.** Variant resolver — reads `SENTIMENT_PROMPT_VARIANT` env var (default `"default"`), validates against `{default, semantic_v1}`, raises on unknown value. |
| `src/prompts/llm_lexical_prompt.py` | Added `_SEMANTIC_V1_ADDENDUM`, `SYSTEM_PROMPT_SEMANTIC_V1`, `get_system_prompt()`. **`SYSTEM_PROMPT` unchanged.** |
| `src/prompts/llm_logic_prompt.py` | Same pattern (logic addendum). `SYSTEM_PROMPT` unchanged. |
| `src/prompts/contextual_prompt.py` | Same pattern (contextual addendum). `SYSTEM_PROMPT` unchanged. |
| `src/agents/llm_lexical_agent.py` | Import `get_system_prompt` instead of `SYSTEM_PROMPT`; call `get_system_prompt()` in the one `generate(...)` line. |
| `src/agents/llm_logic_agent.py` | Same one-line swap. |
| `src/agents/contextual_agent.py` | Same one-line swap. |
| `evaluate_pipeline.py` | Added `import os`; added `--sentiment_prompt_variant {default,semantic_v1}` (default `default`); sets `os.environ["SENTIMENT_PROMPT_VARIANT"]` before the run. |

**No changes** to: `consensus_agent.py`, the router/orchestrator, `_primary_block.py`,
`build_user_prompt(...)` (the user-prompt text is untouched in both variants), any
training/data code.

---

## 2. How the variant is enabled

Two equivalent ways; **default is off** (`default`):

```bash
# via CLI flag (sets the env var for the run)
python evaluate_pipeline.py ... --sentiment_prompt_variant semantic_v1

# or via environment variable directly
SENTIMENT_PROMPT_VARIANT=semantic_v1 python evaluate_pipeline.py ...
```

Mechanism: each agent now calls `get_system_prompt()` at request time, which calls
`active_variant()` → reads the env var → returns `SYSTEM_PROMPT_SEMANTIC_V1` when
`semantic_v1`, else the original `SYSTEM_PROMPT`. Resolution is **at call time**, so
import order is irrelevant. An unknown value raises `ValueError` (fails loudly, never
silently falls back). The `semantic_v1` system prompt = the original prompt with a
role-specific guidance block inserted **immediately before the `OUTPUT FORMAT` block**,
so the JSON output contract remains the final instruction.

---

## 3. Exact prompt changes (verbatim addenda)

The guidance is appended to each agent's **system prompt only** (role lives there);
the user-prompt template is unchanged. Each block is inserted right before
`OUTPUT FORMAT (copy this structure exactly):`.

### Lexical Agent — `_SEMANTIC_V1_ADDENDUM`
```
LEXICAL EVIDENCE GUIDANCE (sentiment) — weigh vocabulary cues carefully:
- Identify the explicit positive, negative, and neutral lexical cues actually present.
- Do NOT assign strong sentiment from isolated words alone — a single token is weak,
  defeasible evidence, not a decision on its own.
- Distinguish a sentiment word being MENTIONED or REFERENCED from the AUTHOR
  EXPRESSING that sentiment (e.g. naming a feeling is not the same as feeling it).
- Treat platform / interface words (like, dislike, unlike, comment, share, clip,
  lyrics, video, button, subscribe) as WEAK cues unless the author clearly states
  their own opinion.
- Treat emojis, slogans, and repeated punctuation as weak SUPPORTING cues only,
  never decisive evidence by themselves.
- If the lexical evidence is weak, conflicting, or only artifact-based, return
  LOWER confidence.
- Your job is to report the lexical evidence and its strength — not to resolve the
  full pragmatic meaning (target attribution and overall intent are other agents' roles).
```

### Logic Agent — `_SEMANTIC_V1_ADDENDUM`
```
STRUCTURE & TARGET GUIDANCE (sentiment) — resolve structure before polarity:
- FIRST identify the sentiment TARGET: what or who is the author evaluating?
- Distinguish the AUTHOR'S OWN sentiment from discussion of other people's
  actions, reactions, or opinions.
- Do NOT classify the text as negative merely because it MENTIONS negative words,
  dislike counts, plot events, death, failure, or other emotionally loaded content.
- Decide whether the text EXPRESSES an evaluation, or merely DESCRIBES / MENTIONS
  something.
- Handle negation, contrast, sarcasm, rhetorical questions, and implicit insults
  or praise, including polarity flips in their scope.
- If the text discusses platform behavior or other users without the author's own
  clear evaluation, prefer neutral or low confidence.
```

### Contextual Agent — `_SEMANTIC_V1_ADDENDUM`
```
WHOLE-MESSAGE INTERPRETATION GUIDANCE (sentiment) — judge overall intent:
- Interpret the overall communicative intent of the ENTIRE message.
- Decide whether the text is an opinion, a meta-comment, a joke, a quote, a
  plot / content description, or a platform interaction.
- Do NOT overrule a neutral reading just because emotional words or emojis appear.
- Use context to detect implicit sarcasm, mockery, praise, or insult.
- If surface cues conflict with the overall message, PRIORITIZE the overall message.
- If the author's stance is genuinely unclear, prefer neutral or lower confidence.
```

---

## 4. Why each change maps to a failure mode

From the linguistic error analysis, the genuine (non-convention) errors were:
surface-cue over-reading, sentiment-target confusion, description-vs-evaluation
confusion, and missed implicit sentiment — amplified by the 92%-correlated agent bloc.

| failure mode (observed) | which agent now owns it | specific lines that address it |
|---|---|---|
| **UI/platform word over-read** (a "dislike/unlike" token → negative) | Lexical (weak-cue rule) + Logic (target) | Lexical: platform/interface words = WEAK cues. Logic: distinguish author's own sentiment from others' actions. |
| **Emoji / punctuation as sentiment** | Lexical + Contextual | Lexical: emojis/slogans/punctuation = weak supporting cues only. Contextual: do NOT overrule neutral just because emojis appear. |
| **Plot / topic / loaded words → negative** | Logic | "Do NOT classify negative merely because it mentions … plot events, death, failure …"; "EXPRESSES vs DESCRIBES/MENTIONS". |
| **Sentiment-target confusion** (others' actions read as author's) | Logic | "FIRST identify the sentiment TARGET"; "distinguish the author's own sentiment from discussion of other people's actions". |
| **Mention vs expression** | Lexical (token) + Logic (clause) | Lexical: MENTIONED/REFERENCED ≠ author EXPRESSING. Logic: EXPRESSES vs DESCRIBES/MENTIONS. |
| **Missed implicit sarcasm/insult/praise** | Contextual | "Use context to detect implicit sarcasm, mockery, praise, or insult"; "if surface cues conflict with the overall message, prioritize the overall message". |
| **Over-confidence on weak evidence** (bloc votes decisively) | all three | each addendum ends with an explicit "prefer lower confidence / neutral when evidence is weak/unclear" instruction. |

**Decorrelation (the systemic problem):** the three addenda assign *different*
reasoning modes — Lexical = *what evaluative words are present + strength*; Logic =
*who/what is the target + is it expressed*; Contextual = *whole-message intent +
sarcasm*. The goal is **productive disagreement**, not uniformity, so the consensus
adds value over a single model.

---

## 5. Risks

1. **Over-conservatism / neutral bias.** Multiple "prefer neutral / lower confidence"
   instructions could push the agents toward neutral, losing genuine wins where a
   strong explicit cue was the correct signal. *Watch:* neutral-rate and W→C count on
   the escalated subset.
2. **Reduced decisiveness on weak primaries.** Lower agent confidence ⇒ less override
   power. For C3/EESA (weak primary, agents *should* override) this could **shrink the
   previous gains** (+0.059 / +0.027). This is the explicit C3 regression check in §6.
3. **Contextual over-reading.** "Detect implicit sarcasm" can hallucinate sarcasm in
   terse text. Mitigated by the paired "if stance unclear, prefer neutral" clause, but
   monitor neutral→polar breaks.
4. **Cross-task leakage.** The addenda are sentiment-specific ("(sentiment)"-tagged).
   The variant is intended for **sentiment runs only**; it should not be enabled for
   topic/NER. Default off makes this safe by construction.
5. **Convention-disagreement cases unchanged.** ~1/3 of "breaks" were label-convention
   disagreements (linguistically defensible). The redesign deliberately does **not**
   target these (that would be dataset tailoring), so they will likely persist.
6. **No semantic guarantee.** Prompt wording changes are non-deterministic in effect;
   the only proof is the A/B run in §6.

---

## 6. Validation plan

### A. Format/gating validation — DONE (no LLM calls)
- `tests/test_lexical_agent.py test_logic_agent.py test_contextual_agent.py
  test_llm_specialist_agents.py test_primary_signal_block.py` → **135 passed.**
- Dedicated gating script (scratchpad `validate_semantic_v1.py`), **all checks pass**:
  default `get_system_prompt()` is byte-identical to the original `SYSTEM_PROMPT`;
  `semantic_v1` differs, contains the addendum, keeps the JSON `OUTPUT FORMAT` contract
  last, retains all four JSON keys; env-var gating flips correctly and bad values raise;
  `build_user_prompt` output is identical across variants; all three agents import; and
  **none** of `eesa / arensa / ahmed / twitter / arsentd / tweet` appear in any prompt.

### B. Paid A/B evaluation — PENDING YOUR APPROVAL (do not run yet)

**B1 — Primary regression guard (Ahmed frozen-primary, threshold 0.7).**
Re-run the Ahmed frozen-primary full_agentic exactly as before but with
`--sentiment_prompt_variant semantic_v1`. Same primary (frozen aligned predictions),
same threshold 0.7, same consensus (Fix-2 ON, w_primary=1.0), GPT-4o-mini.

Compare against the recorded old-prompt result:

| metric | primary_only | old full_agentic | semantic_v1 (target) |
|---|---|---|---|
| full-test accuracy | **0.9254** | **0.9205** | ≥ 0.9205 (ideally → 0.9254) |
| net (W→C − C→W) on 84 escalated | — | **−4** | ≥ −4 (ideally ≥ 0) |

Measure and report, on the 84 escalated samples:
correct→wrong, wrong→correct, **net**, escalated accuracy, **agent agreement %**
(target: **down** from 92% — the decorrelation goal), and the break matrix
(neutral→positive / neutral→negative / positive→neutral / negative→neutral).
**Success = fewer harmful overrides (C→W) and lower agent agreement without losing W→C.**

**B2 — Weak-primary gain guard (C3 generated-primary full_agentic).**
Re-run C3 (generated-960 primary, threshold 0.9) full_agentic with `semantic_v1` and
confirm the previous **+0.059** gain is **not reduced** (the over-conservatism risk #2).
Target: net on escalated ≥ the old +48 / full-test Δ ≥ +0.059 (small regression
tolerable only if Ahmed harm is clearly reduced).

**Order & cost control:** B1 first (smaller, 84 escalated at th 0.7 — the cheap,
decisive test). Only if B1 looks good, run B2. **No paid run starts until you approve.**

---

## Appendix — reproduction commands (for when approved)

```bash
# B1: Ahmed frozen-primary, semantic_v1
python evaluate_pipeline.py \
  --primary_model precomputed \
  --precomputed_predictions data/Sentiment/external/ahmed/ahmed_eesa_test_predictions_aligned.csv \
  --mode full_agentic --escalation_threshold 0.7 \
  --sentiment_prompt_variant semantic_v1 \
  ...   # (same dataset / consensus / output flags as the recorded old run)
```
(The default run — omit `--sentiment_prompt_variant` or pass `default` — reproduces the
old 0.9205 result, confirming the gate.)
