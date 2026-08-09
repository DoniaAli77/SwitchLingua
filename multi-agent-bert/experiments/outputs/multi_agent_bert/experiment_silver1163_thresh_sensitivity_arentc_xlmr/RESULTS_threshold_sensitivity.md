# Threshold-Sensitivity Experiment — Full-ArEnTC XLM-R Full-Agentic on Silver-1163

Scope: **full-ArEnTC XLM-R only** (`experiments/checkpoints/topic_arentcv2_xlmr`). No mBERT, no
Topic-540, no route-all, no retraining, no prompt/weight tuning. Correctness = agreement with the
automatically-generated silver labels (`multi_llm_consensus_silver`), **not** confirmed human-gold
correctness. Results are reported as **threshold sensitivity only** — no threshold is called
"optimal," since all three are evaluated on the same silver corpus used to pick among them.

## Method (frozen, single agent run, no repeated LLM calls)

1. Reused the exact 1,163-row silver input/order/labels/preprocessing and the saved primary
   predictions from `experiment_silver_topic540/fullarentc_xlmr_1163_predictions.csv` (accuracy
   0.6346 / macro-F1 0.5636 / weighted-F1 0.6173 — unchanged, not recomputed).
2. Routing is monotonic in threshold (`confidence < threshold` routes), so the union of rows
   needing agent processing across {0.90, 0.95, 0.99} equals the set routed at 0.99: **221 of
   1,163 rows (19.00%)**.
3. Ran `evaluate_pipeline.py --pipeline_mode full_agentic --threshold 1.01` on **only those 221
   rows** — same topic agents/prompts/model (`gpt-4o-mini`)/consensus procedure/weights as the
   existing topic full-agentic experiments (no flags changed beyond checkpoint/dataset/output
   paths). `--threshold 1.01` simply forces 100% escalation of this pre-filtered subset; it does
   not change agent behavior.
4. Verified reproducibility: primary predictions/confidences recomputed live in this run matched
   the saved `fullarentc_xlmr_1163_predictions.csv` **exactly** for all 221 rows (0 label
   mismatches, max confidence diff 0.0) — confirms the cached primary confidences are a valid
   basis for reconstructing routing decisions at any threshold ≤ 0.99.
5. Reconstructed each threshold's result by re-partitioning the same 1,163 rows: routed rows use
   the cached agentic decision (a subset of the 221 already computed); non-routed rows keep the
   primary prediction unchanged. **One agent run, three thresholds, zero repeated LLM calls.**

## LLM usage — real, one-time (union-at-0.99 run)

| Metric | Value |
|---|---|
| Model | gpt-4o-mini |
| API calls | 884 (exactly 4/routed row: lexical + logic + contextual + explainability) |
| Prompt tokens | 691,203 |
| Completion tokens | 62,204 |
| Total tokens | 753,407 |
| Estimated cost | **$0.141** |

Per-threshold figures below are the **exact call count** (4 × routed rows, deterministic — no
branching in the pipeline) and a **proportional estimate** of tokens/cost (prorated from the
real run's average tokens/call, since the client only accumulates a running total, not
per-sample token counts).

## Headline: routing coverage and outcome by threshold

| Threshold | Routed n | Routed % | Primary errors routed | % of all primary errors routed | Primary-only acc/macroF1/weightedF1 | Final agentic acc/macroF1/weightedF1 | Δacc | Δmacro-F1 | Δweighted-F1 |
|---|---|---|---|---|---|---|---|---|---|
| 0.90 | 126 | 10.83% | 78/425 | 18.35% | 0.6346 / 0.5636 / 0.6173 | 0.6905 / 0.6363 / 0.6797 | +0.0559 | +0.0727 | +0.0624 |
| 0.95 | 150 | 12.90% | 92/425 | 21.65% | 0.6346 / 0.5636 / 0.6173 | 0.7016 / 0.6520 / 0.6923 | +0.0671 | +0.0884 | +0.0749 |
| 0.99 | 221 | 19.00% | 133/425 | 31.29% | 0.6346 / 0.5636 / 0.6173 | 0.7291 / 0.6849 / 0.7227 | +0.0946 | +0.1213 | +0.1054 |

Primary-only metrics are identical across rows (fixed baseline, not recomputed); only the routed
subset — and therefore the final blended result — changes with threshold.

## Correction/harm breakdown (routed rows only)

| Threshold | Wrong→Correct | Correct→Wrong | Wrong→Wrong | Correct→Correct (routed, unchanged) | Net gain (W→C − C→W) | Corrected:Harmed ratio |
|---|---|---|---|---|---|---|
| 0.90 | 68 | 3 | 10 | 45 | **+65** | 22.7 : 1 |
| 0.95 | 81 | 3 | 11 | 55 | **+78** | 27.0 : 1 |
| 0.99 | 115 | 5 | 18 | 83 | **+110** | 23.0 : 1 |

## Routed-subset accuracy: before vs. after agentic processing

| Threshold | Routed n | Accuracy before (primary) | Accuracy after (final) |
|---|---|---|---|
| 0.90 | 126 | 0.3810 | 0.8968 |
| 0.95 | 150 | 0.3867 | 0.9067 |
| 0.99 | 221 | 0.3982 | 0.8959 |

The routed subset is, by construction, the hardest slice for the primary model (accuracy ~38–40%
vs. 63.46% overall) — and the agents lift it to ~90% regardless of exactly where the threshold is
drawn.

## Estimated LLM usage per threshold (calls exact; tokens/cost prorated)

| Threshold | Calls | Tokens (est.) | Cost (est.) |
|---|---|---|---|
| 0.90 | 504 | ~429,544 | ~$0.0804 |
| 0.95 | 600 | ~511,362 | ~$0.0957 |
| 0.99 | 884 (real, exact) | 753,407 (real, exact) | $0.1410 (real, exact) |

## Per-class F1 — final agentic result, by threshold (primary-only F1 for reference)

| Label | Primary-only F1 | Final F1 @0.90 | Final F1 @0.95 | Final F1 @0.99 |
|---|---|---|---|---|
| business | 0.3871 | 0.5000 | 0.5254 | 0.5929 |
| education | 0.3604 | 0.4667 | 0.4754 | 0.5000 |
| health | 0.5714 | 0.6359 | 0.6540 | 0.6732 |
| shopping | 0.5274 | 0.5990 | 0.6082 | 0.6465 |
| medical | 0.5795 | 0.6328 | 0.6404 | 0.6591 |
| sports | 0.5904 | 0.6584 | 0.6667 | 0.7143 |
| tech | 0.7808 | 0.8116 | 0.8129 | 0.8298 |
| finance | 0.7686 | 0.7991 | 0.8043 | 0.8253 |
| social | 0.5070 | 0.6232 | 0.6809 | 0.7234 |

**Every class improves at every threshold — none deteriorates in aggregate F1.** Gains are
monotonic with threshold for all 9 classes.

## Confusion matrices — final agentic predictions (rows = true, cols = predicted)

### Threshold 0.90

| true\\pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|---|---|---|---|---|---|---|---|---|
| business | 58 | 1 | 2 | 12 | 3 | 8 | 41 | 27 | 2 |
| education | 2 | 28 | 3 | 0 | 38 | 2 | 10 | 2 | 0 |
| health | 0 | 0 | 69 | 3 | 3 | 8 | 7 | 0 | 0 |
| shopping | 0 | 0 | 18 | 59 | 2 | 6 | 3 | 2 | 1 |
| medical | 0 | 0 | 7 | 1 | 56 | 2 | 5 | 0 | 0 |
| sports | 0 | 0 | 0 | 2 | 0 | 53 | 1 | 5 | 2 |
| tech | 7 | 4 | 2 | 7 | 4 | 9 | 252 | 6 | 3 |
| finance | 8 | 0 | 4 | 18 | 0 | 5 | 8 | 185 | 4 |
| social | 3 | 2 | 22 | 4 | 0 | 5 | 0 | 4 | 43 |

### Threshold 0.95

| true\\pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|---|---|---|---|---|---|---|---|---|
| business | 62 | 1 | 2 | 11 | 3 | 7 | 40 | 27 | 1 |
| education | 2 | 29 | 2 | 0 | 38 | 2 | 10 | 2 | 0 |
| health | 0 | 0 | 69 | 3 | 3 | 8 | 7 | 0 | 0 |
| shopping | 0 | 0 | 18 | 59 | 2 | 6 | 3 | 2 | 1 |
| medical | 0 | 1 | 5 | 1 | 57 | 2 | 5 | 0 | 0 |
| sports | 0 | 0 | 0 | 2 | 0 | 53 | 1 | 5 | 2 |
| tech | 7 | 4 | 2 | 7 | 4 | 9 | 252 | 6 | 3 |
| finance | 8 | 0 | 3 | 18 | 0 | 5 | 8 | 187 | 3 |
| social | 3 | 2 | 20 | 2 | 0 | 4 | 0 | 4 | 48 |

### Threshold 0.99

| true\\pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|---|---|---|---|---|---|---|---|---|
| business | 75 | 1 | 2 | 12 | 2 | 5 | 36 | 20 | 1 |
| education | 2 | 31 | 1 | 0 | 38 | 2 | 9 | 2 | 0 |
| health | 0 | 0 | 69 | 3 | 3 | 8 | 7 | 0 | 0 |
| shopping | 0 | 0 | 15 | 64 | 2 | 4 | 3 | 2 | 1 |
| medical | 0 | 1 | 5 | 1 | 58 | 2 | 4 | 0 | 0 |
| sports | 0 | 0 | 0 | 1 | 0 | 55 | 1 | 5 | 1 |
| tech | 7 | 4 | 2 | 7 | 2 | 8 | 256 | 6 | 2 |
| finance | 12 | 0 | 2 | 17 | 0 | 3 | 7 | 189 | 2 |
| social | 3 | 2 | 19 | 2 | 0 | 4 | 0 | 2 | 51 |

## The 5 harmed rows (Correct→Wrong) at threshold 0.99 (the most permissive tested)

| segment_id | true | primary | final (harmed) | primary confidence |
|---|---|---|---|---|
| KBCMXaWRRE8_p0492 | finance | finance | business | 0.981 |
| CCIsVZbfO5A_p0126 | shopping | shopping | health | 0.567 |
| kVc5ZFmtvZU_p0027 | shopping | shopping | health | 0.821 |
| b6Fl15dIGgg_p0311 | finance | finance | tech | 0.871 |
| b6Fl15dIGgg_p0482 | tech | tech | business | 0.980 |

All 5 fall along the **same confusion pairs that already dominate the primary confusion
matrix** (business↔finance↔tech; shopping↔health) — the agents are not introducing a new
failure mode, they're occasionally losing a coin-flip on the same genuinely ambiguous pairs the
primary model also struggles with. This set is a strict subset at every lower threshold (3 of
these 5 rows are already routed at 0.90/0.95; the other 2 only enter the routed pool at 0.99).

## Per-row output (all 1,163 rows × 3 thresholds)

Saved per threshold: `threshold_090_rows.csv`, `threshold_095_rows.csv`, `threshold_099_rows.csv`
— columns: `segment_id, text, true_label, primary_pred, primary_conf, routed, final_pred,
final_conf, outcome` where `outcome ∈ {not_routed, corrected, harmed, still_wrong,
preserved_correct}`.

## Interpretation

**Does increasing the threshold allow the agents to correct more primary disagreements?**
Yes, monotonically. Wrong→Correct rises 68 → 81 → 115 as threshold rises 0.90 → 0.95 → 0.99, and
the share of *all* primary errors reached rises 18.35% → 21.65% → 31.29%. More routing coverage
surfaces strictly more correctable cases.

**Do the additional corrections outweigh damage to initially correct predictions?**
Yes, by a wide and stable margin. Net gain rises 65 → 78 → 110 while Correct→Wrong stays in the
3–5 range throughout — the corrected:harmed ratio holds around 23–27:1 at every threshold tested.
Widening the routing window does not measurably increase collateral damage to already-correct
predictions in this range.

**Is the weak overall result mainly caused by the confidence router, or do the agents also
struggle with real transcripts?**
The evidence points to the **router being the dominant bottleneck**, not agent competence. Even
at the most permissive threshold tested (0.99), only 31.29% of all primary errors are ever routed
— meaning **68.71% of primary errors are confidently wrong (≥0.99) and never reach the agents at
all**, regardless of which of these three thresholds is used. But on the cases the agents *do*
see, they perform strongly and consistently: routed-subset accuracy jumps from ~38–40% (primary)
to ~90% (final) at every threshold. The agents are not struggling with real transcripts when given
the chance — the router's blind spot for confidently-wrong primary predictions is what caps the
overall gain.

**Which topic classes benefit or deteriorate most?**
No class deteriorates in aggregate F1 at any threshold. `business` and `social` show the largest
gains (+0.21 and +0.22 F1 at threshold 0.99, from primary F1 of 0.387 and 0.507 respectively) —
both were the weakest primary-only classes and had the most headroom. `tech` and `finance` show
the smallest gains (+0.05 and +0.06 F1) — they were already the strongest primary-only classes
(F1 0.78 and 0.77), leaving less room for the agents to improve. The handful of harmed rows
cluster in the same business/finance/tech and shopping/health confusion pairs that dominate the
primary model's own confusion matrix — not a new failure mode introduced by the agentic layer.

## Scope confirmation

Only the full-ArEnTC XLM-R checkpoint was evaluated (per instruction). No mBERT, no Topic-540
checkpoints, no route-all baseline, no retraining, no prompt or consensus-weight tuning, no label
correction, no dataset modification. Stopped after this threshold-sensitivity analysis.

## Artifacts (all in this new folder — nothing in `experiment_silver_topic540/` was touched)

- `silver_routed_union099.jsonl` — the 221-row union subset (frozen input for the one agent run).
- `agentic_union099/` — raw `evaluate_pipeline.py` outputs (primary_only + full_pipeline
  predictions/metrics, `llm_usage.json`) for the cache-building run.
- `agentic_union099_run.log` — full run log (includes an earlier failed attempt with a missing
  `OPENAI_API_KEY`, 0 real calls, discarded; the successful rerun overwrote it).
- `threshold_090_rows.csv`, `threshold_095_rows.csv`, `threshold_099_rows.csv` — full per-row
  reconstruction for each threshold.
- `threshold_sensitivity_summary.json` — machine-readable summary (all metrics above, all
  thresholds).
- `scripts/analyze_threshold_sensitivity.py` — the reconstruction script (read-only over cached
  predictions; no new inference or agent calls).
