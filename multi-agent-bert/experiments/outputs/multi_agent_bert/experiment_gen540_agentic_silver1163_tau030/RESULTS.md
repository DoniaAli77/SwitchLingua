# Topic-540 XLM-R + full agentic pipeline at the primary-calibrated threshold τ = 0.30

**This is the main SwitchLingua experiment.** Both components originate from the proposed
framework: SwitchLingua generates the Topic-540 training corpus → XLM-R is trained on that
generated corpus → the agentic layer reviews its low-confidence Silver-1163 predictions.

Correctness throughout means **agreement with the automatically assigned silver topic label**
(`label_source = multi_llm_consensus_silver`), **not** human-gold correctness. Silver-1163 is an
automatically labeled real-transcription corpus, not a gold benchmark.

## Threshold provenance (stated explicitly)

τ=0.30 was selected **label-blind**, from the saved primary confidence column alone, to produce
selective coverage consistent with the operating range already established for the sentiment
primaries under the registry's locked rule *"router threshold calibrated per primary"*
(`EXPERIMENT_REGISTRY.md:1231` — XLM-R 0.9, Ahmed 0.7, C3 0.9, yielding 5–23% escalation).

It was **not** optimised using Silver correctness or agentic outcomes. **No performance-based
threshold sweep was performed.** No outputs from the aborted τ=0.90 or τ=0.60 executions are
mixed in (both are sealed in `ABORTED_gen540_route_all_th090_partial/` and
`ABORTED_gen540_tau060_partial/` and were never read by this experiment).

Coverage computed before any agent was invoked, reading only `confidence`:

| | Count | % |
|---|---|---|
| Routed (confidence < 0.30) | **169** | **14.53%** |
| Not routed (≥ 0.30) | 994 | 85.47% |

τ=0.90 was invalid for this primary (max confidence 0.7964 → 100% escalation), the same failure
mode the registry already documents for the Ahmed primary at τ=0.9.

## Configuration

| Item | Value |
|---|---|
| Primary checkpoint | `experiments/checkpoints/topic_gen540_xlmr` |
| Primary training data | `switchlingua_topic_train_540_60perlabel.jsonl` (540 rows, 60/label) |
| Primary base | `xlm-roberta-base` (fresh), 4 epochs, lr 2e-5, batch 16, fp16 |
| Evaluation corpus | `silver_full1163_ordered.jsonl` (1,163 rows, unsplit, untrained-on) |
| Pipeline mode | `full_agentic` |
| Router threshold | **0.30** (applied by the frozen Router, rows not pre-selected) |
| Agents | LLMLexical → LLMLogic → Contextual → Consensus → LLMExplainability |
| LLM | **gpt-4o-mini** (real API, `temperature=0`) |
| `agents_use_primary_signal` | `False` (frozen config value) |
| Consensus | primary-aware, w_primary = 1.0 (frozen) |
| Command | `evaluate_pipeline.py` `main()` called unchanged via `scripts/run_gen540_agentic_silver1163.py` |
| Run window (UTC) | 2026-08-09 12:19:23 → 12:34:56 |

## Headline comparison — identical 1,163 rows

| System | Training data | Correct | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|---|
| Primary only | SwitchLingua Topic-540 | 726/1163 | 0.6242 | 0.5599 | 0.6025 |
| **Full agentic (τ=0.30)** | SwitchLingua Topic-540 | **804/1163** | **0.6913** | **0.6536** | **0.6789** |
| **Δ** | | **+78** | **+0.0671** | **+0.0937** | **+0.0764** |

Arithmetic check: 726 + 86 (W→C) − 8 (C→W) = **804** ✓

## Routing and transitions

| Quantity | Value |
|---|---|
| Routed | 169 / 1163 (14.53%) |
| Not routed (kept primary, 0 LLM calls) | 994 (85.47%) |
| Primary errors total | 437 |
| Primary errors routed | 100 (**22.88%** of all primary errors) |
| primary-correct → final-correct (C→C) | 61 |
| **primary-wrong → final-correct (W→C)** | **86** |
| **primary-correct → final-wrong (C→W)** | **8** |
| primary-wrong → final-wrong (W→W) | 14 |
| **Net gain (W→C − C→W)** | **+78** |
| Corrected : harmed ratio | **10.8 : 1** |

Partition check: 61 + 86 + 8 + 14 = 169 = routed count ✓

### Routed-subset accuracy

| | Accuracy |
|---|---|
| Before (primary) | 0.4083 |
| After (agentic) | **0.8698** |

The routed slice is by construction the primary's weakest (40.8% vs 62.4% overall); the agents
lift it to 87.0%.

## Per-class precision / recall / F1

| Label | Support | P prim | R prim | F1 prim | P final | R final | F1 final | ΔF1 |
|---|---|---|---|---|---|---|---|---|
| business | 154 | 0.5775 | 0.2662 | 0.3644 | 0.6667 | 0.3506 | 0.4596 | **+0.0951** |
| education | 85 | 0.7308 | 0.2235 | 0.3423 | 0.8462 | 0.3882 | 0.5323 | **+0.1899** |
| health | 90 | 0.9459 | 0.3889 | 0.5512 | 0.8909 | 0.5444 | 0.6759 | **+0.1247** |
| shopping | 91 | 0.5294 | 0.6923 | 0.6000 | 0.5868 | 0.7802 | 0.6698 | +0.0698 |
| medical | 71 | 0.4841 | 0.8592 | 0.6193 | 0.5752 | 0.9155 | 0.7065 | +0.0872 |
| sports | 63 | 0.4667 | 0.7778 | 0.5833 | 0.5300 | 0.8413 | 0.6503 | +0.0670 |
| tech | 294 | 0.6505 | 0.8231 | 0.7267 | 0.7086 | 0.8435 | 0.7702 | +0.0435 |
| finance | 232 | 0.7458 | 0.7716 | 0.7585 | 0.7754 | 0.7888 | 0.7821 | +0.0236 |
| social | 83 | 0.5522 | 0.4458 | 0.4933 | 0.7059 | 0.5783 | 0.6358 | **+0.1424** |

**Every class improves. None deteriorates.** Largest gains: `education` (+0.19), `social`
(+0.14), `health` (+0.12) — the weakest-recall primary classes. Smallest: `finance` (+0.02) and
`tech` (+0.04), already the primary's strongest.

## Confusion matrices (rows = silver label, cols = predicted)

### Primary only

| true\\pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|---|---|---|---|---|---|---|---|---|
| business | 41 | 0 | 0 | 16 | 0 | 7 | 45 | 40 | 5 |
| education | 2 | 19 | 0 | 2 | 24 | 5 | 25 | 2 | 6 |
| health | 2 | 0 | 35 | 4 | 21 | 11 | 13 | 1 | 3 |
| shopping | 0 | 0 | 1 | 63 | 7 | 5 | 8 | 3 | 4 |
| medical | 0 | 0 | 0 | 2 | 61 | 1 | 6 | 0 | 1 |
| sports | 1 | 0 | 0 | 1 | 0 | 49 | 8 | 3 | 1 |
| tech | 11 | 5 | 0 | 6 | 2 | 13 | 242 | 8 | 7 |
| finance | 9 | 0 | 0 | 18 | 3 | 7 | 13 | 179 | 3 |
| social | 5 | 2 | 1 | 7 | 8 | 7 | 12 | 4 | 37 |

### Full agentic (τ=0.30)

| true\\pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|---|---|---|---|---|---|---|---|---|
| business | 54 | 0 | 0 | 12 | 0 | 7 | 39 | 37 | 5 |
| education | 2 | 33 | 0 | 1 | 20 | 4 | 18 | 2 | 5 |
| health | 1 | 0 | 49 | 3 | 16 | 8 | 11 | 1 | 1 |
| shopping | 0 | 0 | 3 | 71 | 1 | 5 | 5 | 3 | 3 |
| medical | 0 | 0 | 0 | 2 | 65 | 1 | 3 | 0 | 0 |
| sports | 0 | 0 | 0 | 1 | 0 | 53 | 6 | 2 | 1 |
| tech | 13 | 4 | 0 | 6 | 2 | 12 | 248 | 6 | 3 |
| finance | 8 | 0 | 0 | 19 | 3 | 6 | 11 | 183 | 2 |
| social | 3 | 2 | 3 | 6 | 6 | 4 | 9 | 2 | 48 |

## API usage — recorded, not estimated

| Metric | Value | Source |
|---|---|---|
| Calls | **676** | recorded (`llm_usage.json`), independently confirmed = 676 audit records |
| Prompt tokens | **529,166** | recorded from API `usage` |
| Completion tokens | **47,084** | recorded from API `usage` |
| Total tokens | **576,250** | recorded |
| Estimated cost | **$0.1076** | derived from recorded tokens × list price (only estimated field) |

676 = 169 routed × 4 agents. The 994 non-routed rows made **zero** LLM calls.

## Evidence audit — this run's execution is fully verified

Unlike the earlier full-ArEnTC run, the complete audit trail was preserved
(`audit_llm_calls.jsonl`, `audit_rows.jsonl`).

| Check | Result |
|---|---|
| Rows processed | 1163/1163, 0 errors |
| Routing decisions | 169 escalate + 994 accept = 1163 ✓ |
| Threshold in routing records | `{0.3}` only ✓ |
| Rows where routing ≠ (conf<0.30) | **0** ✓ |
| Non-routed rows whose final ≠ primary | **0** ✓ (primary retained exactly, as required) |
| This run's primary == previously saved primary predictions | **True** ✓ |
| Distinct sample_ids in call log | 169, exactly == routed set ✓ |
| Calls-per-row histogram | `{4: 169}` — every routed row got exactly 4 ✓ |
| Calls per agent | lexical 169, logic 169, contextual 169, explainability 169 ✓ |
| Model | `gpt-4o-mini` only ✓ |
| Calls with error / empty response | **0 / 0** ✓ |
| Distinct raw responses | 675/676 (canned/placeholder output would collapse) ✓ |
| Latency | min 0.82s, median 1.19s, max 11.07s (real network variance) ✓ |
| Leak markers in prompts (`true_label`, `gold`, `silver`, …) | **0 occurrences** ✓ |
| Lexical prompts byte-identical to gold-free reconstruction | **169/169** ✓ |
| `agents_use_primary_signal` | `False` ✓ |

**Silver label never entered any prompt** — proven two ways: no leak-marker substring appears in
any of the 676 prompts, and each lexical prompt is byte-identical to a prompt reconstructed from
`build_user_prompt(task_name, labels, label_descriptions, text, primary_signal=None)` with the
gold label absent by construction.

**Agents are not copying the gold label**: the lexical agent alone agrees with the silver label on
149/169 (88.2%) of routed rows — high (it is genuinely better than the 40.8% primary on this
slice) but not 100%, and the byte-identical prompt check rules out leakage as the cause.

## Interpretation — the central question

> **Does the full agentic pipeline improve the Topic-540 XLM-R primary when both are evaluated on
> Silver-1163?**

**Yes.** Reviewing only the 14.53% of rows the primary was least confident about, the agentic
layer raises accuracy 0.6242 → 0.6913 (**+0.0671**), macro-F1 0.5599 → 0.6536 (**+0.0937**), and
weighted-F1 0.6025 → 0.6789 (**+0.0764**). It corrects 86 primary errors while breaking 8
initially-correct predictions — a net gain of **+78 rows** at a corrected:harmed ratio of
**10.8:1** — and improves F1 for **all nine classes** with none deteriorating.

Two qualifications that matter for the thesis claim:

1. **The gain is bounded by the router, not by agent quality.** Only 100 of the primary's 437
   errors (22.88%) were routed at all; the other 337 were errors the primary made *confidently*
   (≥0.30) and the agents never saw. On the rows they did see, the agents lifted accuracy from
   0.4083 to 0.8698. The ceiling here is what the confidence gate surfaces, not what the agents
   can do with it.
2. **Correctness is agreement with automatic silver labels.** These numbers measure agreement
   with `multi_llm_consensus_silver`, not human-validated ground truth. Both the primary and the
   agentic layer are scored against the same automatic labels, so the *comparison* is fair, but
   the absolute values are not gold-benchmark accuracy.

The full-ArEnTC XLM-R results remain a **secondary large-synthetic-data comparison**, not the
main evidence for SwitchLingua.

## Artifacts

| File | Contents |
|---|---|
| `pipeline_out/…__primary_only_{predictions,metrics}.{csv,json}` | primary baseline, this run |
| `pipeline_out/…__full_pipeline_{predictions,metrics}.{csv,json}` | final agentic output |
| `pipeline_out/…__llm_usage.json` | recorded API calls + token counts |
| `audit_llm_calls.jsonl` | **676 records**: full prompt, full raw response, agent, model, timestamp, latency, error |
| `audit_rows.jsonl` | **1163 records**: primary label/conf/probabilities, routing decision, each agent's parsed label+confidence, consensus votes + rationale, final output, stage history |
| `tau030_rows.csv` | row-level: silver label, primary pred/conf, routed flag, final pred/conf, outcome |
| `tau030_summary.json` | machine-readable metrics (all of the above tables) |
| `run.log` | complete execution log |
| `scripts/run_gen540_agentic_silver1163.py` | runner (calls `evaluate_pipeline.main()` unchanged; observation-only instrumentation) |
| `scripts/analyze_gen540_tau030.py` | read-only analysis + evidence audit |

Scope: only the Topic-540 XLM-R primary at the single frozen τ=0.30. No mBERT, no retraining, no
Silver training, no threshold sweep, no prompt or weight tuning.
