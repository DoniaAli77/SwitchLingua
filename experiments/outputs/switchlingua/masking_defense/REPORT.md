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
**Method:** `step2_count_masking.py` on RAW data; 53 usable scenarios (≥2 sentences).
Both decision rules applied to the **same** per-sentence scores → isolates only the rule (average vs each).

### Masking rate by threshold (RAW, 53 scenarios)
| Bar | Masking scenarios | Masking rate | Both accept | Both refine |
|-----|-------------------|-------------|-------------|-------------|
| 6.5 | 16 | 30.2% | 37 | 0 |
| **7.0** | **22** | **41.5%** | 12 | 19 |
| 7.5 | 2 | 3.8% | 1 | 50 |
| 8.0 (pipeline default) | 0 | 0.0% | 0 | 53 |
| 8.5 | 0 | 0.0% | 0 | 53 |

**Key finding:** masking is threshold-dependent. At the pipeline default bar (8.0) there is **0% masking
on this data** because gpt-4o-mini sentences are uniformly mediocre (~7), so every aggregate is < 8 and
the average rule refines everything anyway. Masking appears when the bar sits *inside* the score band:
at the calibrated operating bar **7.0, 41.5%** of scenarios hide a weak sentence the aggregate accepts.
**Average intra-scenario spread (max−min) = 0.794 (max 1.75).**

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

### 6.1 Cross-run before/after (Step 4) — CONFOUNDED null, superseded
`step4_before_after.py`: RAW (off) vs FIXED (on) datasets.
mean sentence 7.057→7.007; mean worst-per-scenario 6.674→6.602; % sentences below 7: 35.5→43.4;
% scenarios fully accepted (all ≥7): 24.1→18.5; **Mann-Whitney p = 0.25 (not significant)**.
**Invalid as a refiner test:** RAW and FIXED are *independent generations* (different sentences), so
generation randomness swamps the refiner's effect. Replaced by the within-sentence test below.

### 6.2 Clean within-sentence test (Test 6a) — POSITIVE
`refiner_clean_test.py`: for the **same** weak sentence, score the original (fresh), refine it, score the
result (fresh) — no cross-run confound. ~9 API calls/sentence.

| Sample | Mean before | Mean after | Mean Δ | Improved | Sign-test p |
|--------|------------|-----------|--------|----------|-------------|
| 5 (smoke) | 6.07 | 7.03 | +0.96 | 5/5 (100%) | 0.025 |
| 30 weakest | 6.41 | 7.17 | +0.77 | 29/30 (96.7%) | ≈0 |
| **ALL 87 weak (score<7)** | **6.64** | **7.24** | **+0.60** | **79/87 (90.8%)** | **≈0** |

**Conclusion:** once a weak sentence is handed to the refiner, it is genuinely improved ~91% of the time.
**Output:** `step4_final_picture/refiner_within_sentence.csv`.

### 6.3 Test 6b — YOUR refiner vs ORIGINAL refiner — TIE (honest null)
`step6b_refiner_headtohead.py`: isolates the contribution's variable — **feedback granularity** —
holding prompt/model/sentence constant. Each weak sentence refined twice: with **per-sentence feedback**
(its own scores) vs **aggregate feedback** (scenario-average scores). 30 weak sentences, ~14 calls each.

| Method | Mean improvement | Wins |
|--------|------------------|------|
| YOURS (per-sentence feedback) | +0.747 | 10 |
| ORIGINAL (aggregate feedback) | +0.808 | 13 |
| ties | | 7 |
| **Sign-test p = 0.53 → no difference** | | |

**Conclusion:** per-sentence *feedback* does NOT produce better rewrites than aggregate feedback. The
refiners are equivalent at the rewrite step. **Therefore the contribution is NOT "a better refiner."**
**Output:** `step4_final_picture/refiner_headtohead.csv`.
**NOT YET TESTED:** the *task-aware* refiner prompts (REFINER_TASK_*), which trigger only on task
failures — needs validator ON + task-failing sentences (a separate Test 6b-task).

---

## 7. Overall scorecard

| Claim | Evidence | Verdict |
|-------|----------|---------|
| Per-sentence scoring **catches** weak sentences aggregate scoring hides | 41.5% of scenarios at bar 7 (Step 2) | ✅ supported |
| Refining a caught weak sentence **improves** it | +0.60, 79/87 (90.8%), p≈0 (Test 6a) | ✅ supported |
| Humans confirm masked sentences are genuinely weaker | Step 3 built, not run | 🟡 pending |
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
2. **Single model (gpt-4o-mini)** — uniformly mediocre quality (~7); masking at the default bar 8 is 0%.
   A stronger generator (gpt-4o) would widen the quality spread and likely show masking at higher bars.
3. **Cross-run before/after is confounded** — only the within-sentence test (6a) is valid for the refiner.
4. **Task-aware refiner untested** (Test 6b-task) — needs validator ON + task-failing sentences.
5. **Task imbalance** (sentiment 36 vs topic 12 vs NER 6) and **single refinement pass** (MAX=1).
6. **MASKED subset for humans = 13/50** (pilot); larger pool (30/103) available for more power.

## 10. File index (experiments/outputs/switchlingua/masking_defense/)
- `PLAN.md` — step-by-step log (plain language) · `EVALUATION_PLAN.md` — full 6-test plan
- `step1_generate.py` · `step1_raw_data/Arabic.jsonl` (54) · `step1_fixed_data/Arabic.jsonl` (54)
- `step2_count_masking.py` · `step2_counts/{masking_by_threshold,masking_cases,spread_per_scenario}.csv`
- `step3_build_human_sheet.py` · `step3_analyze_human_sheet.py` · `step3_human_check/{human_check_sheet,answer_key}.csv`, `README.md`
- `step4_before_after.py` · `refiner_clean_test.py` · `step6b_refiner_headtohead.py`
- `step4_final_picture/{before_after_summary,worst_sentence_per_scenario,refiner_within_sentence,refiner_headtohead}.csv` + logs
- Pipeline fix: `Modified_Version/core/node_engine.py:1106`
