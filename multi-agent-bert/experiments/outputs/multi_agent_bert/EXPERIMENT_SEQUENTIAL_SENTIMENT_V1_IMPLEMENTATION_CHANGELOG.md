# sequential_sentiment_v1 — Implementation Changelog

Implementation of the approved staged-reasoning sentiment pipeline
(`EXPERIMENT_SEQUENTIAL_SENTIMENT_V1_PROMPT_CONTROLLER_DESIGN.md`). **Experimental,
opt-in.** The default pipeline is unchanged: none of the new components are
constructed unless the `sequential_sentiment_v1` variant is explicitly selected. No
training, no generation, **no paid evaluation run yet** — only unit/smoke tests.
Date: 2026-07-01.

## What it does
Replaces the parallel *trio + consensus* on the escalation path with a **staged
pipeline**: Intent → Polarity → Pragmatic → deterministic controller. Each stage
outputs its own JSON, is conditioned on the previous stage(s), and the controller
composes them with fixed rules (no voting). Only reached when `pipeline_mode ==
full_agentic`, on **escalated** samples, when the variant is on.

```
text ─▶ Stage 1 Intent ─▶ Stage 2 Polarity ─▶ Stage 3 Pragmatic ─▶ Stage 4 Controller ─▶ final
           (JSON)          (text+intent)       (text+intent+pol)      (deterministic)
```

## Files changed

### New
| file | purpose |
|---|---|
| `src/prompts/sequential_sentiment_prompts.py` | 3 stage system prompts + user-prompt builders. JSON-only, general, no dataset named. |
| `src/agents/sequential_sentiment.py` | `SeqIntentAgent`, `SeqPolarityAgent`, `SeqPragmaticAgent` (LLM stages, retry+coerce), `SequentialController` (deterministic Stage 4). Persists trace under `state.extras["sequential_sentiment"]`. |
| `tests/test_sequential_sentiment_v1.py` | 22 offline tests (wiring, clean prompts, malformed-JSON, controller rules a–e, escape hatch, ablation, e2e). |

### Modified (all additive / opt-in — default paths untouched)
| file | change |
|---|---|
| `src/pipeline/orchestrator.py` | Added optional ctor params `sequential_stages`, `sequential_controller` (default `None`). On the escalation path, when a controller is present and mode is `full_agentic`, run the staged pipeline instead of the parallel trio+consensus. Extracted the shared completion tail into `_finalize()`. |
| `src/agents/_sentiment_agent_variant.py` | Added `"sequential_sentiment_v1"` to `VALID_VARIANTS`. |
| `evaluate_pipeline.py` | Import the sequential components; `build_orchestrator` gains `sequential_config`; new variant branch builds the 3 stages + controller and injects them; added `--sentiment_agent_variant sequential_sentiment_v1` choice and `--seq_tau_intent/--seq_tau_revise/--seq_tau_low/--seq_no_primary_fallback` flags. |

## How to enable
```bash
# Mock (offline, no cost) — smoke only; mock cannot emit valid stage JSON
python evaluate_pipeline.py --dataset <data.jsonl> --mode full_pipeline \
  --pipeline_mode full_agentic --sentiment_agent_variant sequential_sentiment_v1 \
  --llm_client mock

# Real evaluation (spends money; run only after approval) — see command template below
```
Default runs (no `--sentiment_agent_variant`, or `default`) are **completely
unaffected** — the sequential objects are never built.

## Thresholds / config (exposed, not hardcoded-only)
| flag | constant | default | meaning |
|---|---|---|---|
| `--seq_tau_intent` | `TAU_INTENT` | 0.60 | high-confidence "no opinion" gate for Rule 1 |
| `--seq_tau_revise` | `TAU_REVISE` | 0.60 | confidence needed to trust a pragmatic revision |
| `--seq_tau_low` | `TAU_LOW` | 0.45 | below this polarity conf, weak cases fall back to primary |
| `--seq_no_primary_fallback` | `USE_PRIMARY_FALLBACK` | `True` (fallback on) | set flag → pure-sequential ablation (primary never used as fallback) |

Passed through `build_orchestrator(sequential_config={...})` → `SequentialController(...)`.

## JSON schemas (per stage, JSON-only)
```jsonc
// Stage 1 — Intent
{"opinion_expressed": true|false|"unclear", "target": "<str|null>",
 "speech_act": "evaluate|describe|ask|advise|quote|other",
 "use_vs_mention": "use|mention|platform_meta", "confidence": 0.0, "evidence": ["..."]}

// Stage 2 — Polarity  (conditioned on Stage 1)
{"label": "positive|negative|neutral", "confidence": 0.0, "mixed": false,
 "reasoning": "...", "evidence": ["..."]}

// Stage 3 — Pragmatic (conditioned on Stage 1+2)
{"keep_or_revise": "keep|revise", "final_label": "positive|negative|neutral",
 "confidence": 0.0, "reasoning": "...", "evidence": ["..."]}
```
Persisted trace (`state.extras["sequential_sentiment"]`): `intent`, `polarity`,
`pragmatic` (each incl. its `confidence` and raw text), `stage_events`
(retry/coerce/llm_error), `thresholds`, `decided_by`, `fallback_path`, `final_label`.
The controller also writes `final_output` **and** `consensus_output` so the existing
explainability agent and evaluator behave identically to the parallel path.

## Controller rules (deterministic, first match wins)
1. **No-opinion neutral** — `opinion_expressed==false` and `intent.conf ≥ TAU_INTENT`,
   **unless** pragmatics confidently (`≥ TAU_REVISE`) revises to a non-neutral label
   (the escape hatch) → `neutral`. `decided_by=intent_no_opinion`.
2. **Confident pragmatic revision** — `revise` and `prag.conf ≥ TAU_REVISE` →
   pragmatic label. `decided_by=pragmatic_revision`.
3. **Pragmatic keep** → Polarity label. `decided_by=polarity_kept`.
4. **Weak / conflicted** — a weak revision is discarded; if `USE_PRIMARY_FALLBACK` and
   `pol.conf < TAU_LOW` → primary label (`fallback_primary`); else Polarity label
   (`fallback_polarity`); last-resort neutral if neither available.

Primary participates **only** as router (unchanged) + safe fallback — never a voter.

## Error handling
- **One retry per stage** on malformed JSON (same temp-0 client), then a **safe
  coerced default** — never crashes.
- Safe defaults are designed to *degrade toward Polarity → primary → neutral*: a broken
  Intent stage becomes `unclear`/conf 0 (cannot force neutral — cascade guard); a broken
  Polarity/Pragmatic stage gets conf 0 (routes to the fallback branch / keeps Polarity).
- Invalid labels are rejected in-stage; the controller coerces any residual invalid final
  label to `neutral`. All coercions/retries are logged in `stage_events`.

## Tests passed
- **New:** `tests/test_sequential_sentiment_v1.py` — **22 passed**. Covers: default
  pipeline builds no sequential path; variant activates only when selected; the 3 stage
  prompts are JSON-only and name no dataset/benchmark; per-stage malformed-JSON
  retry→coerce; controller rules a) no-opinion→neutral, b) no-opinion + confident
  pragmatic implicit stance→pragmatic label (escape hatch), c) keep→polarity,
  d) revise→pragmatic, e) weak/conflicted→primary or polarity fallback; pure-sequential
  ablation; threshold config exposure; end-to-end escalated run produces a valid label +
  full persisted trace.
- **Full suite:** **967 passed** (was 945; +22), 0 failures.
- **CLI smoke** (`--llm_client mock`, forced escalation): runs end-to-end, all samples
  take the sequential path, valid labels emitted, no crash (mock triggers the
  coerce-to-default path by design — not a signal run).

## Evaluation command template (DO NOT run until approved)
First run on the **weak C3 generated primary** (the regime with headroom), per the design:
```bash
python evaluate_pipeline.py \
  --dataset <eesa_test.jsonl> \
  --mode both \
  --pipeline_mode full_agentic \
  --sentiment_agent_variant sequential_sentiment_v1 \
  --primary_model transformer \
  --transformer_checkpoint <C3_seed456_checkpoint> \
  --threshold <C3_escalation_threshold> \
  --llm_client openai --llm_model gpt-4o-mini \
  --seq_tau_intent 0.60 --seq_tau_revise 0.60 --seq_tau_low 0.45 \
  --output_dir experiments/outputs/multi_agent_bert/experiment_seqv1_c3 \
  --run_id seqv1_c3
# Baselines to beat: primary_only 0.6956/0.6830 ; full_agentic 0.7543/0.7387.
# Ablation: add --seq_no_primary_fallback for the pure-sequential variant.
# Strong-primary (Ahmed) run is OPTIONAL and only after C3 shows value.
```

## Status
Implementation + tests complete. **Awaiting approval before any paid (OpenAI) run.**
