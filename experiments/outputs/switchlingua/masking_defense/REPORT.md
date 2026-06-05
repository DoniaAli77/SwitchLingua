# SwitchLingua — Masking Defense: Detailed Experiment Report

**Date:** 2026-05-31
**Purpose:** Empirically defend the per-sentence scoring contribution of the modified
Arabic-English code-switching generation pipeline, against the original aggregate-scoring baseline.

---

## 1. Systems compared

| System | Codebase | Model | Scoring | Refiner | Tasks |
|--------|----------|-------|---------|---------|-------|
| **A** | Original_baseLine | gpt-4o | aggregate (1 score/scenario) | generic, whole-batch | topic |
| **B (contribution)** | Modified_Version | gpt-4o-mini | **per-sentence** (1 score/sentence) | targeted + task-aware + guardrail | topic, sentiment, NER |
| **C (control)** | Original_baseLine | gpt-4o-mini | aggregate | generic, whole-batch | topic |

This study isolates **architecture** by holding the model constant at **gpt-4o-mini** (B vs C).
System A (gpt-4o) was **not** run in this session — model effect (A vs C) is a separate question.

### Scoring formula (identical in both systems)
```
sentence_weighted_score = fluency*0.30 + naturalness*0.25 + cs_ratio*0.20 + socio_cultural*0.25
aggregate_score = mean(per-sentence weighted scores)   # verified: aggregate == mean of per-sentence
```
Each dimension is scored 0–10 by a gpt-4o-mini judge agent. **Task validation is a separate
pass/fail flag and is NOT part of the score.** Verified on all records that `aggregate == mean(sentence_scores)`.

---

## 2. Data generated (Step 1)

**Generator:** `step1_generate.py` → Modified_Version core, model = gpt-4o-mini.
**Config:** `Modified_Version/config/config2.yaml`, overridden to all 3 CS types.
**Scenario space:** 3 tasks (topic, sentiment, NER) × 3 CS types (Intrasentential, Intersentential,
Extra-sentential / Tag switching) = **54 scenario configs**.
**Fixed scenario attributes:** language pair Arabic (matrix) → English (embedded); cs_ratio target 70%;
First Person; Present tense; Expressive cs_function; single_turn; Male; age 18–25; College.
**Generation temperature = 0.7; judge/refiner temperature = 0.1.**

Two datasets were produced from the same scenario configs:

| Dataset | Refiner | Task validator | Scenarios | Sentences | Task split | CS-type split |
|---------|---------|----------------|-----------|-----------|------------|---------------|
| **RAW ("before")** `step1_raw_data/` | OFF | OFF | 54 | 245 | topic 12, sentiment 36, NER 6 | 18 / 18 / 18 |
| **FIXED ("after")** `step1_fixed_data/` | ON | OFF | 54 | 235 | topic 12, sentiment 36, NER 6 | 18 / 18 / 18 |

- RAW = grade each sentence once, **no fixing** → honest pre-refinement quality.
- FIXED = full pipeline with the refiner on (validator deliberately OFF so the *only* difference
  vs RAW is the refiner).
- Task split is uneven (sentiment-heavy) because config2.yaml enumerates more sentiment combinations.

**Infrastructure notes:** corporate TLS proxy (httpx `verify=False`, 60 s timeout); `.env` loaded with
`override=True`; OpenAI account hit its **daily 10,000-request limit (RPD)** twice during heavy runs.

---

## 3. Bug found and fixed (pipeline robustness)

**Symptom:** NER scenarios in FIXED mode (refiner ON) never terminated — infinite refinement loop.
**Root cause:** in `Modified_Version/core/node_engine.py` `RunRefinerAgent`, `refine_count` was
incremented only when a fix was *accepted*. For NER, improving a sentence breaks the required entities,
so the accept/reject guardrail correctly rejects the fix → `refine_count` stayed 0 → `meet_criteria`
re-routed to the refiner forever. Topic/sentiment were unaffected (their fixes get accepted).
**Fix (1 line, node_engine.py:1106):** count every refinement *attempt*, not just accepted ones; after
the budget (MAX_SENTENCE_REFINES=1) is spent the sentence is accepted as `budget_exhausted`.
**Verification:** 1 NER scenario terminated cleanly; topic unaffected; later all 6 NER completed.
**The guardrail behaviour was correct; only the loop counter was buggy.**

---

## 4. Step 2 — Masking measurement (the CATCH)

**Definition:** a *masking* scenario = aggregate score ≥ bar (average rule ACCEPTS the whole scenario)
**but** at least one sentence < bar (per-sentence rule CATCHES a weak sentence the average hides).
**Method:** `step2_count_masking.py` on RAW data. **Exclusion:** scenarios with <2 sentences are
excluded (a single-sentence scenario cannot "hide" anything). Exactly **1 scenario excluded** —
scenario index 49 (NER, 1 sentence) — leaving **53 usable** of 54.
Both decision rules applied to the **same** per-sentence scores → isolates only the rule (average vs each).

**Score distribution (RAW, 245 sentences):** mean **7.057**, **median 7.000**, IQR 6.8–7.3, range 5.9–8.15.
Scenario aggregates: mean 7.053, median 7.08, **max 7.71** (no scenario reaches 8).

### Masking rate by threshold (RAW, 53 scenarios)
| Bar | Masking scenarios | Masking rate | 95% CI (Wilson) | Both accept | Both refine |
|-----|-------------------|-------------|-----------------|-------------|-------------|
| 6.5 | 16 | 30.2% | 19.5–43.5% | 37 | 0 |
| **7.0** | **22** | **41.5%** | **29.3–54.9%** | 12 | 19 |
| 7.5 | 2 | 3.8% | — | 1 | 50 |
| 8.0 (pipeline default) | 0 | 0.0% | — | 0 | 53 |
| 8.5 | 0 | 0.0% | — | 0 | 53 |

**Key finding:** masking is threshold-dependent. At the pipeline default bar (8.0) there is **0% masking
on this data** — *because the maximum scenario aggregate is only 7.71*, the default bar sits **above the
model's entire aggregate range** and is effectively inoperative for gpt-4o-mini.

**Threshold justification (anti-bias):** the operating bar **7.0 = the median sentence quality**
(median 7.000, mean 7.057) — a principled, data-driven choice, **not** tuned to maximize masking. The
full curve is reported for transparency.

**Threshold sensitivity (honest limitation):** because the scores are tightly packed (IQR 6.8–7.3,
avg intra-scenario spread 0.794), the masking rate is sensitive to the bar — 41.5% at 7.0 but 3.8% at
7.5. This fragility is a direct consequence of gpt-4o-mini's uniform quality; a stronger generator
(wider quality spread) would make masking robust across a broader band of thresholds.

**Output:** `step2_counts/masking_by_threshold.csv`, `masking_cases.csv` (30 masked sentences at bar 7),
`spread_per_scenario.csv`.
**Example masking case:** scenario aggregate 7.225 (accepted), sentence scores [7.5, 7.0, **6.7**, 7.7];
masked sentence (6.7): *"لدي فكرة عن كيفية تحسين المبيعات. Maybe we should consider online marketing أكثر."*

**Decision:** report the full threshold curve (transparent); use operating bar 7.0 calibrated to the
model; confirm with humans (Step 3). gpt-4o (stronger generator, wider quality spread) was rejected on cost.

---

## 5. Step 3 — Human validation (the CONFIRM) — BUILT, NOT YET RUN

**Goal:** neutral bilingual annotators confirm the machine-flagged masked sentences are *genuinely*
weaker — removing the circularity of the AI grading its own output.
**Sheet:** `step3_human_check/human_check_sheet.csv` — **blind & shuffled (seed 42)**, **50 sentences**
(13 MASKED + 37 neighbours) drawn from 11 masking scenarios. Annotators rate each sentence:
`fluency_1to10`, `naturalness_1to10`, `cultural_1to10`, `is_real_codeswitch_yes_no`.
(CS-ratio is a measured proportion, not a Likert item, so it is the yes/no question.)
**Hidden answer key** (`answer_key.csv`) stores role (MASKED/neighbour) + the machine's per-dimension
scores → lets the analysis validate the AI judge dimension-by-dimension.
**Analyzer** (`step3_analyze_human_sheet.py`, auto-detects filled sheets): per-dimension MASKED vs
neighbour means + Mann-Whitney p; human-vs-machine Spearman per dimension; per-scenario composite sign
test; monolingual-leak %; inter-annotator Cohen's kappa + Spearman if ≥2 annotators.
**Status:** waiting on annotators. (Full pool available = 30 MASKED / 103 sentences if more power needed.)

---

## 6. Refiner effectiveness (the FIX)

### 6.1 Cross-run before/after (Step 4) — APPENDIX-ONLY SANITY CHECK (confounded, do not interpret)
`step4_before_after.py`: RAW (off) vs FIXED (on) datasets.
mean sentence 7.057→7.007; mean worst-per-scenario 6.674→6.602; % sentences below 7: 35.5→43.4;
% scenarios fully accepted (all ≥7): 24.1→18.5; **Mann-Whitney p = 0.25 (not significant)**.
**Invalid as a refiner test:** RAW and FIXED are *independent generations* (different sentences), so
generation randomness swamps the refiner's effect. Replaced by the within-sentence test below.

### 6.2 Clean within-sentence test (Test 6a) — POSITIVE
`refiner_clean_test.py`: for the **same** weak sentence, score the original (fresh), refine it, score the
result (fresh) — no cross-run confound. ~9 API calls/sentence.

| Sample | Mean before | Mean after | Mean Δ (95% CI) | Improved (95% CI) | Sign-test p |
|--------|------------|-----------|-----------------|-------------------|-------------|
| 5 (smoke) | 6.07 | 7.03 | +0.96 | 5/5 (100%) | 0.025 |
| 30 weakest | 6.41 | 7.17 | +0.77 | 29/30 (96.7%) | ≈0 |
| **ALL 87 weak (score<7)** | **6.64** | **7.24** | **+0.60 (0.51–0.68)** | **79/87 (90.8%, CI 82.9–95.3%)** | **≈0** |

**Conclusion:** once a weak sentence is handed to the refiner, it is genuinely improved ~91% of the time.
The improvement is both statistically significant (p≈0) and of practical magnitude (Δ≈+0.6, CI excludes 0).
**Output:** `step4_final_picture/refiner_within_sentence.csv`.

### 6.3 Test 6b — YOUR refiner vs ORIGINAL refiner — TIE (honest null)
`step6b_refiner_headtohead.py`: isolates the contribution's variable — **feedback granularity** —
holding prompt/model/sentence constant. Each weak sentence refined twice: with **per-sentence feedback**
(its own scores) vs **aggregate feedback** (scenario-average scores). 30 weak sentences, ~14 calls each.

| Method | Mean improvement (95% CI) | Wins |
|--------|---------------------------|------|
| YOURS (per-sentence feedback) | +0.747 (0.60–0.89) | 10 |
| ORIGINAL (aggregate feedback) | +0.808 (0.65–0.97) | 13 |
| ties | | 7 |
| **Paired diff (YOURS−ORIG) = −0.062, 95% CI −0.25 to +0.13 → straddles 0** | | |
| **Sign-test p = 0.53; YOURS win-share 10/23 = 43% (CI 26–63%)** | | |

**Conclusion:** per-sentence *feedback* does NOT produce better rewrites than aggregate feedback —
the CI of the paired difference straddles 0 and the win-share CI straddles 50%. The
refiners are statistically equivalent at the rewrite step. **Therefore the contribution is NOT "a better refiner."**
**Output:** `step4_final_picture/refiner_headtohead.csv`.
**NOT YET TESTED:** the *task-aware* refiner prompts (REFINER_TASK_*), which trigger only on task
failures — needs validator ON + task-failing sentences (a separate Test 6b-task).

---

## 6.5 Test 1 — Task-aware generation quality (automated; human confirmation pending)
`run_task_aware_eval.py` on the fresh pre-refinement sample (no regeneration). 40 sentences per
task (35 for NER). Task correctness = **blind gpt-4o-mini judge** not shown the target (sentiment =
re-classification; topic = relevance; NER = entity extraction + deterministic constraint check).
CS validity and CS-ratio are **deterministic** (objective). Fluency/naturalness are the pipeline's
own per-sentence judge scores. Outputs: `task_aware_eval/` (summary.csv/.json, details.jsonl, report.md).

| Task | n | Task-correct % (blind LLM) | CS-valid % (objective) | CS-ratio MAE vs 70% (objective) | Fluency | Naturalness |
|------|--:|--:|--:|--:|--:|--:|
| topic | 40 | 100.0 | 100.0 | 23.3 | 8.35 | 8.07 |
| sentiment | 40 | 70.0 | 87.5 | 22.1 | 8.05 | 8.05 |
| NER | 35 | **40.0** (English-only, final) | 97.1 | 13.8 | 8.40 | 8.54 |

**NER judge evolution (3 numbers — read carefully):**
- **45.7%** — original judge: hardcoded PER/ORG/LOC, parsed loose "TYPE: entity" text lines (fragile, under-counted).
- **62.9%** — constraint-aware but **lenient**: read types/range/must from constraints, strict JSON, 0 parse errors — but it followed the config's (contradictory) `allow_code_switched_entities: true` and **counted Arabic-script entities the pipeline does not accept**. Diagnostic only, NOT final.
- **40.0%** — **final, pipeline-consistent (Option A, English-only target entities):** required entities must be English/Latin-script (matching the generation + TaskValidator prompts); Arabic-script names are context and don't count (config now `target_entities_script: english`, `allow_code_switched_entities: false`). The judge enforces this deterministically (regex filter, field `arabic_script_ignored`).

**Why 40% < 62.9%:** the drop is mostly that the judge no longer extracts Arabic-script names at all, so failures surface as **missing_PER (19/21 failures)**. Honest reading: under the pipeline's real English-only policy, the model **rarely produces English-script PERSON entities** (e.g. "Elon Musk") — it writes person references in Arabic inside the Arabic-matrix sentence. NER is the weakest task at 40%, a genuine *generation* gap. (sentiment 72.5%→70.0% is ~1-sentence noise.)

**Reading:** strong surface quality (CS validity 87–100%, fluency/naturalness ~8) but weaker constraint
satisfaction — sentiment moderate (drag is the neutral class), **NER missing PERSON entities**, and the
realized CS ratio is **far from the 70% target** (off by ~14–23 points). **Fluency/naturalness stay ~8 even
where the task fails**, so quality scoring alone does not detect task-level failures — motivating the
task-aware validation and deterministic CS-ratio components. **Caveat:** task-correctness is a blind LLM
judge; needs human confirmation via `human_eval/consolidated_annotation_sheet.csv`. CS-validity and
CS-ratio are objective.

### 6.5.1 NER PER-prompt repair (diagnosed → fixed → promoted)
Diagnostic chain: Test 1 flagged NER weak → error analysis localized it to **missing_PER** → a
constraint-difficulty run (`ner_constraint_difficulty_*`) showed ORG is easy (90%) but **PERSON is the
bottleneck** (and difficulty is non-monotonic, so it's not just "too many constraints"). A controlled
before/after pilot (`ner_per_prompt_repair/`, 50/arm, Wilson CIs, core prompt untouched — variant in the
harness) then tested adding an explicit **English-script PERSON requirement + self-check** with
Arabic-friendly Latin names (Ahmed Ali, Sarah Hassan, …):

| arm | task-correct % (95% CI) | missing_PER | CS-ratio MAE | fluency / naturalness |
|---|---|---|---|---|
| current prompt | 22.5% (12.3–37.5) | 70% | 18.0 | 8.32 / 8.4 |
| **PER-focused prompt** | **56.8% (42.2–70.3)** | **25%** | 15.95 | 8.61 / 8.52 |

**CIs do not overlap → a real improvement (+34 pts), not run-to-run noise; no CS-ratio/naturalness
regression.** This repair was **promoted into the core `DATA_GENERATION_NER_PROMPT`** (generation prompt
only; TaskValidator, evaluation policy, and config unchanged). Post-promotion checks: mocked pipeline
tests pass, a real 12-sentence NER smoke parses and the TaskValidator still returns verdicts. Caveats:
still LLM-judged (human spot-check pending); absolute NER % is noisy run-to-run (~22–60% same config), so
the trustworthy result is the within-pilot **delta**, and the model still occasionally writes the person
name in Arabic script (which the English-only policy correctly rejects).

### 6.5.2 Final NER dev checkpoint (FROZEN)
**Old issue.** NER task-correctness was low under the English-only policy (Test 1: 40%). Error analysis +
a constraint-difficulty sweep localized it to **missing English-script entities — especially PER, then
EVENT and LOC** (the model writes these in Arabic, which the policy rejects); ORG was already easy.

**Fix.** A **generic, config-driven entity-guidance builder**: `node_engine.build_ner_entity_guidance()`
reads `must_include_types` + per-type metadata (`DEFAULT_ENTITY_GUIDANCE`, overridable by
`ner.entity_type_guidance` in config: description / script_rule / examples) and injects a dynamic block for
ONLY the required types into the NER prompt via `{ner_entity_guidance}`. `prompt.py` has **zero hardcoded
tag examples**; unknown tags get a safe fallback.

**Why better than hardcoded prompt examples.** Scalable (add a tag in config, never the prompt), task-aware
(only the required types are described), consistent with the English-only policy, and testable in isolation.

**Validation (generic vs baseline; missing-type drops are the robust signal):**
LOC 44→63%, PRODUCT 45→85%, **EVENT 39→84%** (missing EVENT 61%→5%), EVENT_LOC 25→83%.
**PER_EVENT remains hard** (11→22%): two Arabic-natural hard types compete for 2–3 entity slots (fixing
EVENT cost PER) — left unoptimized by design. A controlled same-session A/B confirmed **no PER regression**
from generalizing (generic 53% ≈ the PER pilot's 57%; an alarming 25% smoke was a low variance draw).

**Final verification.** Relevant tests pass: NER-guidance regression (5/5), task-generation mock, full
pipeline mock, per-instance scoring, refiner-guardrail (after updating 2 assertions that encoded the
*pre-loop-fix* behaviour — a rolled-back refine now spends the budget, count→1, to prevent the NER infinite
loop). `test_pipeline_output_reviewer` is a CLI utility (excluded); `*_real.py` need live API (skipped).
**Real core-pipeline smoke** (CodeSwitchingAgent graph, **real TaskValidator ON, refiner OFF**, English-only,
n=12): generated 12, **parse 12/12, CS-validity 12/12, fluency/naturalness 8.8/8.5, no errors**; task_correct
3/12 (25%, PER variance); **the real NER validator passed 0/12 — i.e. it is much stricter than the judge**
(an honest observation, consistent with NER difficulty). n was below the 20–30 target (the real pipeline
returned ~2 sentences/scenario).

**Caveats.** All NER scoring is LLM-judged (human confirmation pending); small n; pairwise variants
under-powered; same-config NER % is noisy run-to-run — trust deltas and missing-type drops, not absolutes.

**Status: NER FROZEN** — implemented, generic, tested. Known limitation: PER_EVENT / competing hard types
under 2–3 entity slots. No further NER prompt changes unless tests break.

## 6.6 Test 2 — TaskValidatorAgent necessity & effectiveness (real validator)
`run_task_validator_necessity.py` replays two acceptance policies over the Test 1 results — **no
regeneration**, but the **real TaskValidatorAgent is run** on the existing sentences (not an oracle).
Three separate signals: reference task-correctness = Test 1 blind judge; validator = real
TaskValidatorAgent verdict; quality = per-sentence weighted score ≥ 7.0. Policy A = quality_only;
Policy B = quality AND validator. Verdicts cached (`validator_verdicts.jsonl`); reference swappable for
human labels via `--labels`. Outputs: `task_validator/` (summary.csv/.json, report.md).

(Numbers below use the final English-only NER reference, so they are fully consistent with §6.5.)

| | A: quality-only | B: quality + validator |
|---|--:|--:|
| accepted | 86 | 62 |
| task-correct among accepted (precision) | 70.9% | **85.5%** |
| task-WRONG accepted | 25 | **9** |
| false-accept (of all wrong) | 78.1% | 28.1% |
| false-reject (of all correct) | — | 36.1% |

Validator as a standalone task detector (vs reference): precision **83.0%**, recall **88.0%**, agreement 78.3%.

**Per-task (the key finding):**
| Task | task-wrong accepted A→B | false-accept A→B |
|------|--:|--:|
| topic | 0 → 0 | — (validator only over-rejects, FN 17.5%) |
| sentiment | 5 → 5 | 45.5% → 45.5% (**no effect** — neutral errors slip past both) |
| NER | 17 → 4 | 89.5% → 21.1% (**validator earns its keep here**) |

**Reading:** adding the real (fallible) TaskValidator cuts task-wrong accepts 18→6 and lifts precision
79.1%→90.3%, but the benefit is **concentrated in NER**, **null for sentiment** (neutral class evades
both quality and validator), and it **over-rejects topic**. Honest claim: the validator is worth it
*specifically for entity-constrained tasks (NER)*, not as a blanket gate.

## 6.7 Test 4 — CS-ratio MEASUREMENT validation (PARTIAL; human counts pending)
`run_csratio_partial_validation.py` compares CS-ratio **measurement methods** on a **fixed 30-sentence
set** (20 real sentences sampled across topic/sentiment/NER/masked/control + 10 controlled edge cases).
**This is NOT a generation comparison — no pipeline was run.** Methods: (1) our deterministic counter
`compute_true_cs_stats`; (2) original-style **LLM-only** counting (gpt-4o-mini, temp 0.7, **repeated 3×**
to measure instability); (3) human manual counts (**columns blank → PENDING**). The script skips
human-reference metrics gracefully and reports instability + det-vs-LLM disagreement only.
Outputs: `csratio/` (validation_set.csv, validation_details.jsonl, llm_repeats.jsonl,
partial_summary.csv, partial_report.md).

| signal | value |
|---|--:|
| LLM-only repeats **disagree** across the 3 runs | **12/30 (40%)** |
| mean per-sentence LLM std (Arabic count / Arabic %) | 0.60 tokens / **2.32%** |
| deterministic counter variance | **0 (exact, free, reproducible)** |
| det vs LLM **is_code_switched** mismatch | 0/30 |
| mean det-vs-LLM Arabic-% abs diff | 5.04% |
| monolingual edge cases correct (det / LLM) | 2/2 / 2/2 |

**Reading:** the two methods **agree on the binary code-switch decision** (0/30 mismatch) and roughly on
the ratio (~5% mean gap), but the **LLM-only method is unstable** — 40% of sentences get a *different*
count on a re-run of the *same* sentence, whereas the deterministic counter is exact and free. Worst
wobble is on long mixed sentences (e.g. CS005 ratio std 2.6% with a 17-pt det-vs-LLM gap). **Which method
is more accurate is still PENDING** the manual human token counts; this partial run only establishes that
the baseline LLM measurement is non-reproducible while ours is deterministic.

## 7. Overall scorecard

| Claim | Evidence | Verdict |
|-------|----------|---------|
| Per-sentence scoring **catches** weak sentences aggregate scoring hides | 41.5% (54-scen Step 2); **35.6%** on the larger 101-scen calibration at bar 7 | ✅ supported |
| Refining a caught weak sentence **improves** it | +0.60, 79/87 (90.8%), p≈0 (Test 6a) | ✅ supported |
| Task-aware generation produces valid task data | topic 100%, sentiment 70%, **NER 40%** (English-only, final); CS-valid 87–100% | 🟡 mixed (topic strong; sentiment=neutral drag; NER weak — under-produces English-script PER); human-confirm pending |
| TaskValidator reduces task-wrong accepts | precision 70.9%→85.5%, task-wrong 25→9 (Test 2, real validator, English-only ref); benefit concentrated in NER | ✅ supported (NER); ❌ no effect on sentiment |
| Pipeline hits the requested CS ratio (70%) | CS-ratio MAE ≈ 14–23 pts off target (objective, Test 1) | ❌ off-target |
| Deterministic CS counter is more reliable than LLM-only counting | LLM-only disagrees with itself on 40% of sentences (Test 4, partial); deterministic = 0 variance; accuracy vs human PENDING | 🟡 reproducibility shown; accuracy pending |
| Quality scoring alone detects task failures | fluency/naturalness ~8 even where task fails (Test 1) | ❌ → motivates task-aware validation |
| Humans confirm masked sentences are genuinely weaker | Step 3 / consolidated sheet built, not run | 🟡 pending |
| Your refiner makes **better rewrites** than the original's | tie, p=0.53 (Test 6b) | ❌ not supported |
| At bar 8 (pipeline default) masking occurs | 0% on this model | ❌ (bar must be calibrated ~7) |

### Defensible thesis claim (precise, not overclaimed)
> The contribution is **per-sentence scoring as a detection/routing mechanism**, not a better refiner.
> Aggregate scoring lets a weak sentence escape unrefined in **41.5%** of scenarios (at a calibrated bar
> of 7.0, on gpt-4o-mini); per-sentence scoring **catches and routes** it to the refiner, which then
> improves it by **+0.60** (90.8% of cases). The two refiners are equivalent at the rewrite step
> (p=0.53), so the advantage is *what gets refined*, not *how well it is refined*.

---

## 8. Statistical methods
Implemented without scipy (numpy + math): Mann-Whitney U (normal approximation, tie-corrected),
Spearman rank correlation, sign test (binomial normal approximation), Cohen's kappa. Normal CDF via
`math.erf`.

## 9. Limitations / open items
1. **Human validation (Step 3) not yet run** — the CONFIRM step depends on it.
2. **Single model (gpt-4o-mini)** — uniformly mediocre quality (median 7.0, max aggregate 7.71), so the
   default bar 8 is above the whole range (0% masking there). Masking is reported at the median-quality
   bar 7.0 but is **threshold-sensitive** (41.5% at 7.0 → 3.8% at 7.5) because scores are tightly packed.
   A stronger generator (gpt-4o) would widen the spread and make masking robust across more thresholds.
3. **Cross-run before/after is confounded** — only the within-sentence test (6a) is valid for the refiner.
4. **Task-aware refiner untested** (Test 6b-task) — needs validator ON + task-failing sentences.
5. **Task imbalance** (sentiment 36 vs topic 12 vs NER 6) and **single refinement pass** (MAX=1).
6. **MASKED subset for humans = 13/50** (pilot); larger pool (30/103) available for more power.
7. **Test 4 CS-ratio accuracy PENDING human counts** — the partial run shows the LLM-only counter is
   non-reproducible (40% self-disagreement) vs our 0-variance deterministic counter, but *which is more
   accurate* needs the manual Arabic/English/other token counts (blank columns in `csratio_validation_set.csv`).

## 10. File index (experiments/outputs/switchlingua/masking_defense/)
- `PLAN.md` — step-by-step log (plain language) · `EVALUATION_PLAN.md` — full 6-test plan
- `step1_generate.py` · `step1_raw_data/Arabic.jsonl` (54) · `step1_fixed_data/Arabic.jsonl` (54)
- `step2_count_masking.py` · `step2_counts/{masking_by_threshold,masking_cases,spread_per_scenario}.csv`
- `step3_build_human_sheet.py` · `step3_analyze_human_sheet.py` · `step3_human_check/{human_check_sheet,answer_key}.csv`, `README.md`
- `step4_before_after.py` · `refiner_clean_test.py` · `step6b_refiner_headtohead.py`
- `step4_final_picture/{before_after_summary,worst_sentence_per_scenario,refiner_within_sentence,refiner_headtohead}.csv` + logs
- Pipeline fix: `Modified_Version/core/node_engine.py:1106`
