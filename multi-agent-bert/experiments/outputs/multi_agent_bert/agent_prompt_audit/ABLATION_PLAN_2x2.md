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

## Prerequisite to run (one small enabling change, NOT yet made)
`agents_use_primary_signal` is already config-wired (cells C/D just set it). But
**`w_primary` is not yet exposed to the CLI/config** (cells A/C need
`w_primary=0`). To run the matrix, add a minimal seam — the cleanest is a small
`scripts/run_primary_ablation.py` that builds the orchestrator programmatically
per cell:
```python
# pseudo: for each (w_primary, use_signal) in the 2x2:
task_config.agents_use_primary_signal = use_signal
orch = build_orchestrator(task_config, ..., primary_classifier=clf, llm_client=oai)
orch._consensus = ConsensusAgent(weights={"primary": w_primary})   # set the cell's weight
# run NEREvaluator/Evaluator over EESA test; save under
# experiments/outputs/multi_agent_bert/ablation_2x2/<cell>/
```
Alternatively wire `--consensus_primary_weight` + `--agents_use_primary_signal`
flags into `evaluate_pipeline.py` (also lets us sweep `w_primary` later). Either
is ~30 lines, offline-testable. **I will implement the chosen seam when you
approve running the ablation** — not before.

## Guardrails
- Do not change router, prompts, or the default config (`w_primary=1.0`,
  signal off) as part of the ablation — the cells set their values per-run only.
- Report all four cells side-by-side plus the anchoring proxy; do not declare a
  winner from accuracy alone (the block's risk is anchoring, which the proxy
  catches).
