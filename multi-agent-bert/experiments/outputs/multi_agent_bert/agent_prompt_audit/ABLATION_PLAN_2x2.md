# Ablation Plan — Fix #2 × Fix #3 (2×2)

**Plan only — do NOT run until approved.** Isolates the contribution of
primary-aware consensus (Fix #2, `w_primary`) and the primary-signal prompt block
(Fix #3, `agents_use_primary_signal`). Date: 2026-06-11.

## Fixed conditions (all four cells)
- model: **XLM-RoBERTa** (`experiments/checkpoints/eesa_xlm_roberta_base`)
- pipeline_mode: **full_agentic**, `--llm_client openai`, gpt-4o-mini
- threshold: **0.8**, dataset: **EESA test** (818), current code (Fix #1+#2+#3 in)

## Matrix
| Cell | `w_primary` | `agents_use_primary_signal` | Means |
|---|---|---|---|
| **A** | 0 | OFF | Original: blind agents, agents-only consensus (≈ prior clean 0.8472 baseline) |
| **B** | 1.0 | OFF | **Fix #2 only** (primary-aware consensus) — ≈ the 0.8447 run just measured |
| **C** | 0 | ON | **Fix #3 only** (primary-signal block; agents-only consensus) |
| **D** | 1.0 | ON | **Fix #2 + Fix #3** |

## Per-cell report
1. accuracy, 2. macro F1, 3. weighted F1, 4. per-class F1 (esp. negative/neutral),
5. escalation count/rate, 6. OpenAI calls + cost, 7. connection errors, 8. parse
errors, and **9. anchoring proxy** = among escalated samples, the rate at which
the **final label equals the primary_only prediction** (higher ⇒ more
anchoring/agreement with the primary). Compare A vs C (does the block alone pull
the agents toward the primary?) and B vs D (does the block add anything on top of
consensus anchoring, or just compound it?).

## Interpretation guide
- **C > A** → the primary-signal block helps on its own (agents adjudicate well).
- **C ≈ A with higher anchoring proxy** → the block mostly induces copying (low value).
- **D vs B** → whether the block adds value beyond Fix #2 (or double-anchors).
- Watch **negative F1** in every cell (the class the pipeline most affects).

## Cost / time
4 runs × ~109 escalated × 4 calls ≈ ~436 calls each ≈ **~$0.057/cell ≈ ~$0.23
total**, ~15 min/cell on a stable connection. Connectivity-probe each; if a cell
shows connection errors, mark it contaminated and re-run that cell only.

## Seam (IMPLEMENTED 2026-06-11 — two CLI flags, offline-tested)
`evaluate_pipeline.py` now exposes:
- `--consensus_primary_weight W` — sets the ConsensusAgent primary vote weight
  (Fix #2). **Unset → built-in default 1.0** (current behaviour). `0` = legacy
  agents-only. Wired via `build_orchestrator(consensus_primary_weight=...)` →
  `ConsensusAgent(weights={"primary": W})`. Consensus logic itself unchanged.
- `--agents_use_primary_signal` — presence forces the primary-signal block **ON**
  (Fix #3); omit → config default (off). Overrides `task_config` only.

Verified offline (897 tests pass; default weight still 1.0; mock end-to-end run
logs both overrides, 0 errors, no OpenAI). **Exact 2×2 commands** (each prefixed
with the env-load that the prior runs used; `--transformer_device cuda`):

```
# common args:
COMMON="--dataset data/Sentiment/processed/eesa_sentiment_test.jsonl \
  --config src/config/default.yaml --active_task sentiment_classification \
  --pipeline_mode full_agentic --mode full_pipeline --threshold 0.8 \
  --primary_model transformer --transformer_checkpoint experiments/checkpoints/eesa_xlm_roberta_base \
  --transformer_device cuda --llm_client openai --llm_model gpt-4o-mini"

# A: w_primary=0, signal OFF
python evaluate_pipeline.py $COMMON --consensus_primary_weight 0 \
  --output_dir .../ablation_2x2/A --run_id abl_A
# B: w_primary=1.0, signal OFF
python evaluate_pipeline.py $COMMON --consensus_primary_weight 1.0 \
  --output_dir .../ablation_2x2/B --run_id abl_B
# C: w_primary=0, signal ON
python evaluate_pipeline.py $COMMON --consensus_primary_weight 0 --agents_use_primary_signal \
  --output_dir .../ablation_2x2/C --run_id abl_C
# D: w_primary=1.0, signal ON
python evaluate_pipeline.py $COMMON --consensus_primary_weight 1.0 --agents_use_primary_signal \
  --output_dir .../ablation_2x2/D --run_id abl_D
```
(Each run loads `OPENAI_API_KEY`/`OPENAI_BASE_URL` from `Modified_Version/.env`,
as in prior paid runs.) **Awaiting approval before any of these are executed.**

## Guardrails
- Do not change router, prompts, or the default config (`w_primary=1.0`,
  signal off) as part of the ablation — the cells set their values per-run only.
- Report all four cells side-by-side plus the anchoring proxy; do not declare a
  winner from accuracy alone (the block's risk is anchoring, which the proxy
  catches).
