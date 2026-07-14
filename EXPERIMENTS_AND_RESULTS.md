# SwitchLingua — Experiments and Results: Authoritative Scientific History

This document reconstructs the complete scientific history of the SwitchLingua research for
the thesis Experiments & Results chapter. It is **exhaustive and chronological**, not a
summary. It spans **two tracks**:

- **Track 1 — SwitchLingua data generation** (the *Modified_Version* pipeline: per-sentence
  scoring, TaskValidator, task-aware refiner, deterministic CS-ratio; the generation
  contribution) and the **generation experiments** that produced the training data.
- **Track 2 — Multi-Agent downstream evaluation** (a confidence-aware primary + specialist-agent
  pipeline used as an *extrinsic* quality test of the generated data, and as a standalone
  classification system). Track 2 is documented in full in the companion file
  `multi-agent-bert/experiments/outputs/multi_agent_bert/EXPERIMENT_REGISTRY.md` and is
  re-cast here into the same 11-field structure in **Part 2**.

Every experiment is reported with: (1) Experiment Name, (2) Research Question, (3) Motivation,
(4) Hypothesis, (5) Baseline, (6) Methodology, (7) Experimental Setup, (8) Results (every
reported metric, including negative results), (9) Analysis, (10) Conclusions, (11) Impact, and
(12) Thesis Relevance.

### A necessary clarification about metrics (read before Results)
No experiment in this project reports **BLEU, ROUGE, or BERTScore**. These are reference-based
metrics for tasks with a gold target text; open-ended Arabic–English code-switched *generation*
has no reference string, so they are inapplicable and were not used. The generation track is
instead evaluated with: **quality-judge scores** (fluency, naturalness, social-cultural, 0–10),
**deterministic CS-validity** (is the sentence genuinely code-switched?), **deterministic
CS-ratio** (Arabic:English token proportion vs a target), **task-correctness** (a blind LLM
judge re-deriving the label), **TaskValidator precision/recall**, a **masking rate**
(per-sentence vs aggregate scoring), and a **human evaluation** (built, not yet run). The
downstream track (Part 2) is evaluated with **accuracy, macro-F1, weighted-F1**, escalation
transitions (W→C, C→W, net), and McNemar significance. Where a metric family (e.g. human eval)
was *built but not executed*, that is stated explicitly rather than filled in.

All generation experiments use **GPT-4o-mini** unless noted (the NeurIPS-2025 baseline used
GPT-4o). Systems: **B = Modified_Version** (the contribution); **C = Original_baseLine at
gpt-4o-mini** (architecture control); **A = Original_baseLine at gpt-4o** (the published
baseline, model-different).

---

# PART 1 — TRACK 1: THE SWITCHLINGUA GENERATION FRAMEWORK

---

## G0 — The Modified_Version Generation Pipeline (the contribution)
**1. Experiment Name:** Modified_Version task-aware, per-sentence code-switching generation
pipeline (System B).
**2. Research Question:** Can the SwitchLingua generation pipeline be redesigned so that (a) it
supports multiple NLP tasks (topic, sentiment, NER) in one run, (b) it scores and refines
**each sentence individually** rather than one aggregate score per scenario, and (c) it
validates that each generated sentence actually satisfies its task constraint?
**3. Motivation:** The NeurIPS-2025 baseline (Original_baseLine) generates for a single implicit
task, produces **one aggregate quality score per scenario** (5 sentences), refines the whole
batch generically, re-uses **stale pre-refinement scores**, and computes CS-ratio once for the
batch. A single aggregate score can **hide a weak sentence** inside an otherwise-good scenario;
there is no task-level validity check; CS-ratio is LLM-estimated and non-reproducible.
**4. Hypothesis:** Per-sentence scoring will detect weak sentences that aggregate scoring hides;
a TaskValidator and deterministic CS-ratio will catch task/CS failures that quality scores miss.
**5. Baseline:** Original_baseLine (aggregate scoring, generic refiner, single-task, GPT-4o).
**6. Methodology — exactly what changed:**
- **Multi-task, nested config** (`config2.yaml`): `task ∈ {topic, sentiment, ner}` with a
  `shared` block + per-task sections (sentiment: labels/intensity/ambiguity; NER:
  entity_types/min/max/must_include/allow_cs_entities; topic: sub-domains). Scenarios built by
  `itertools.product` (default 18).
- **New `TaskValidatorAgent`** (topic/sentiment/NER prompts) producing `per_instance_results`
  (passed / confidence / predicted_label / notes) — one verdict per sentence.
- **Per-sentence scoring:** CS-ratio computed per sentence
  (`cs_ratio_results_per_instances[i]`); `SummarizeResult` builds `sentence_records` and a
  per-sentence `weighted_score`; `failing_sentence_indices` = sentences < 8.0.
- **Selective, task-aware refiner:** rewrites **only** failing sentences with task-specific
  prompts (`REFINER_TASK_{TOPIC,SENTIMENT,NER}`) that must preserve the task label/entities;
  a **guardrail** re-validates + re-scores and **rolls back on regression**.
- **Re-evaluation after refine:** the four quality agents **re-run** for fresh scores (baseline
  reuses stale scores).
- **Deterministic CS-ratio:** `utils.compute_true_cs_stats` counts Arabic vs Latin tokens
  (0 variance) as a ground-truth complement to the LLM CS estimate.
- **Generic, config-driven NER guidance:** `build_ner_entity_guidance()` +
  `DEFAULT_ENTITY_GUIDANCE` + `{ner_entity_guidance}` placeholder (English-only entity policy),
  no hardcoded tag examples.
- **Model change:** GPT-4o → **GPT-4o-mini** (cost).
- Prompt count 6 → **14+** (per task × agent).
**7. Experimental Setup:** LangGraph `StateGraph`; agents = DataGeneration → TaskValidator →
{Fluency, Naturalness, CSRatio, SocialCultural} (parallel) → SummarizeResult → Refiner?(loop
MAX=1) → Acceptance. Scoring formula (identical in B and C):
`weighted = 0.30·fluency + 0.25·naturalness + 0.20·cs_ratio + 0.25·socio_cultural`; task
validity is a **separate pass/fail flag, not in the score**. Generation temp 0.7; judge/refiner
temp 0.1. Corporate TLS proxy (`verify=False`); daily 10,000-request (RPD) cap.
**8. Results:** Architecture delivered and frozen (2026-06-05). Full run (2026-04-01, 18
scenarios): avg overall **7.27/10** (target > 8, below), fluency ~8.2, naturalness ~8.0,
CS-ratio accuracy ~50% (target 70%, below), TaskValidator pass rate ~40% (target > 80%, below).
Automated non-API tests pass (refiner guardrail 4/4, NER guidance 5/5, per-instance scoring 2/2,
per-sentence-vs-scenario policy test PASS).
**9. Analysis:** The redesign is architecturally complete, but on GPT-4o-mini the *content*
quality is uniformly mediocre (overall ~7.3, never reaching the 8.0 bar), CS-ratio is off
target, and task-validity pass rate is low — exactly the failures the new per-sentence /
validator machinery is designed to *detect*. The value of the contribution therefore had to be
demonstrated as **detection/routing**, not as raw score lift (see G2–G10).
**10. Conclusions:** The task-aware, per-sentence architecture is realized; whether it
*empirically* beats aggregate scoring is deferred to the masking-defense experiments.
**11. Impact:** Became System B — the object of every subsequent generation experiment and the
generator of all downstream training data (Track 2). Froze the NER path (G8).
**12. Thesis Relevance:** The core *methods* chapter artifact — the generation pipeline whose
per-sentence-scoring contribution the Experiments chapter must defend.

---

## G1 — NER Infinite-Loop Guardrail Fix
**1. Experiment Name:** Refiner loop-counter fix for NER scenarios.
**2. Research Question:** Why do NER scenarios never terminate when the refiner is on?
**3. Motivation:** In FIXED-mode runs, NER scenarios looped forever, blocking any NER data.
**4. Hypothesis:** The refine budget is not being consumed on rejected fixes.
**5. Baseline:** Pre-fix `node_engine.RunRefinerAgent`.
**6. Methodology:** Root-caused: `refine_count` incremented **only when a fix was accepted**.
For NER, improving fluency tends to break the required entities, so the guardrail **correctly
rejects** the fix → `refine_count` stayed 0 → `meet_criteria` re-routed to the refiner forever.
Topic/sentiment fixes get accepted, so they were unaffected. **Fix (1 line,
`node_engine.py:1106`):** count **every** refinement *attempt*; after the budget
(`MAX_SENTENCE_REFINES=1`) is spent, accept the sentence as `budget_exhausted`.
**7. Experimental Setup:** Real pipeline, NER scenarios.
**8. Results:** 1 NER scenario terminated cleanly; topic unaffected; later all 6 NER scenarios
completed. The guardrail behaviour was correct; only the counter was buggy.
**9. Analysis:** A genuine interaction between a correct guardrail (reject entity-breaking fixes)
and an incorrect loop counter. Distinct from the *downstream* System-B NER loop bug noted in the
multi-agent track — same conceptual class (guardrail rejects fixes, counter only advanced on
accepted fixes), different codebase.
**10. Conclusions:** Hypothesis confirmed; robustness restored.
**11. Impact:** Unblocked all NER generation and the NER quality experiments (G6–G8).
**12. Thesis Relevance:** A pipeline-robustness footnote / limitation-and-fix in Methods.

---

## G2 — Masking Measurement: does per-sentence scoring catch what aggregate hides?
**1. Experiment Name:** Masking-defense Step 2 (the CATCH).
**2. Research Question:** In what fraction of scenarios does the **aggregate** rule accept a
scenario (mean ≥ bar) while **at least one sentence** is below the bar (per-sentence rule catches
it)?
**3. Motivation:** This is the central empirical defense of the per-sentence contribution.
**4. Hypothesis:** A non-trivial fraction of scenarios "mask" a weak sentence under aggregate
scoring.
**5. Baseline:** Aggregate scoring (mean of per-sentence scores) — verified `aggregate == mean`.
**6. Methodology:** Generate a RAW dataset (refiner OFF, validator OFF) so scoring is honest
pre-refinement quality; apply **both** decision rules to the **same** per-sentence scores (isolates
only the rule). Masking scenario = aggregate ≥ bar AND min sentence < bar. Exclude scenarios with
<2 sentences (1 excluded: NER scenario idx 49). Report the full threshold curve; operating bar =
**7.0 = median sentence quality** (principled, not tuned to maximize masking).
**7. Experimental Setup:** System B, GPT-4o-mini; 54 scenario configs (3 tasks × 3 CS types);
RAW = 245 sentences across 53 usable scenarios. Score distribution: mean **7.057**, median
**7.000**, IQR 6.8–7.3, range 5.9–8.15; scenario aggregates mean 7.053, **max 7.71**.
**8. Results (masking rate by bar, RAW, 53 scenarios):**
| Bar | Masking scenarios | Rate | 95% CI (Wilson) |
|---|---|---|---|
| 6.5 | 16 | 30.2% | 19.5–43.5% |
| **7.0** | **22** | **41.5%** | **29.3–54.9%** |
| 7.5 | 2 | 3.8% | — |
| 8.0 (pipeline default) | 0 | 0.0% | — |
| 8.5 | 0 | 0.0% | — |

On a larger **101-scenario** calibration set, masking at bar 7.0 = **35.6%**. Example masked
sentence (scenario aggregate 7.225 accepted; sentence scores [7.5, 7.0, **6.7**, 7.7]).
**9. Analysis:** Per-sentence scoring catches a weak sentence in **~36–41%** of scenarios that
aggregate scoring accepts — a substantial, real effect. **But it is threshold-sensitive**: at
bar 8.0 (the pipeline default) masking is **0%**, because GPT-4o-mini's entire aggregate range
tops out at 7.71 — the default bar sits above the whole distribution and is inoperative. The
fragility (41.5% at 7.0 → 3.8% at 7.5) is a direct consequence of the model's uniformly mediocre,
tightly-packed quality (IQR 0.5, intra-scenario spread 0.79). A stronger generator (GPT-4o,
wider spread) would make masking robust across a broader band.
**10. Conclusions:** Hypothesis confirmed at a **calibrated** bar (7.0 = median), with an honest
threshold-sensitivity caveat. The defensible claim is *detection*, at a model-calibrated bar.
**11. Impact:** The headline evidence for the contribution; motivated threshold calibration (G3)
and the human confirmation step (G11). Reframed the thesis claim to "per-sentence scoring as a
detection/routing mechanism."
**12. Thesis Relevance:** The primary result defending the generation contribution.

---

## G3 — Threshold Calibration (larger scenario set)
**1. Experiment Name:** Per-sentence threshold calibration (101 scenarios).
**2. Research Question:** Does the masking effect hold on a larger, calibrated scenario set, and
what operating bar is principled?
**3. Motivation:** The 54-scenario masking rate is threshold-sensitive; a larger set and an
explicit calibration guard against cherry-picking the bar.
**4. Hypothesis:** Masking persists at the median-calibrated bar on more data.
**5. Baseline:** G2's 54-scenario result (41.5% at 7.0).
**6. Methodology:** Recompute masking on a 101-scenario set; calibrate the bar to the median
sentence quality; report the full curve (`per_sentence/threshold_calibration_report.md`).
**7. Experimental Setup:** System B, GPT-4o-mini; per-sentence weighted scores.
**8. Results:** Masking **35.6%** at bar 7.0 on 101 scenarios (vs 41.5% on 54). Bar 8.0 remains
~0% (model ceiling ~7.7).
**9. Analysis:** The effect is **stable in sign and order of magnitude** (~36–41%) across sample
sizes at the calibrated bar, strengthening G2. The absolute rate depends on the bar because the
score distribution is narrow.
**10. Conclusions:** Masking is a **robust, model-calibrated** phenomenon (~⅓ of scenarios),
confirmed on the larger set; hypothesis confirmed.
**11. Impact:** Firmed up the headline masking number reported alongside G2.
**12. Thesis Relevance:** The robustness check that lets the masking claim be stated with a CI.

---

## G4 — Refiner Effectiveness (within-sentence, the FIX)
**1. Experiment Name:** Masking-defense Test 6a — clean within-sentence refiner test.
**2. Research Question:** Once a weak sentence is routed to the refiner, is it actually improved?
**3. Motivation:** Detection (G2) is only valuable if the routed sentence is then fixed. A naïve
cross-run before/after (Step 4) is confounded (RAW and FIXED are independent generations).
**4. Hypothesis:** Refining a caught weak sentence improves its score.
**5. Baseline:** The same sentence's own pre-refine score (within-sentence, no cross-run confound).
**6. Methodology:** For the **same** weak sentence: score original (fresh) → refine → score result
(fresh). ~9 API calls/sentence. The confounded cross-run Step 4 (mean 7.057→7.007, Mann-Whitney
**p=0.25, not significant**) is retained only as an appendix sanity check and explicitly **not
interpreted**.
**7. Experimental Setup:** System B refiner, GPT-4o-mini; weak = score < 7.
**8. Results:**
| Sample | Before | After | Δ (95% CI) | Improved (95% CI) | Sign-test p |
|---|---|---|---|---|---|
| 5 (smoke) | 6.07 | 7.03 | +0.96 | 5/5 (100%) | 0.025 |
| 30 weakest | 6.41 | 7.17 | +0.77 | 29/30 (96.7%) | ≈0 |
| **87 weak (<7)** | **6.64** | **7.24** | **+0.60 (0.51–0.68)** | **79/87 (90.8%, 82.9–95.3%)** | **≈0** |
**9. Analysis:** A routed weak sentence is genuinely improved ~91% of the time; the effect is
statistically significant (p≈0) and of practical magnitude (Δ≈+0.6, CI excludes 0). The
confounded cross-run test's null (p=0.25) is an artifact of comparing independent generations —
correctly discarded in favour of the within-sentence design.
**10. Conclusions:** Hypothesis confirmed — the refiner reliably fixes what per-sentence scoring
catches.
**11. Impact:** Completes the "catch → fix" chain (G2 catches, G4 fixes); set up the question of
*whether the per-sentence feedback itself* makes the fix better (G5).
**12. Thesis Relevance:** The second half of the contribution's evidence (detection + repair).

---

## G5 — Per-Sentence vs Aggregate Refiner Feedback (head-to-head, the NULL)
**1. Experiment Name:** Masking-defense Test 6b — refiner feedback-granularity head-to-head.
**2. Research Question:** Does giving the refiner **per-sentence** feedback (its own scores)
produce better rewrites than **aggregate** feedback (scenario-average scores)?
**3. Motivation:** To isolate whether the contribution is "a better refiner" or "better routing."
**4. Hypothesis (to be tested, not assumed):** Per-sentence feedback yields better rewrites.
**5. Baseline:** Aggregate-feedback refinement of the same sentence (prompt/model/sentence held
constant; only feedback granularity varies).
**6. Methodology:** Each weak sentence refined twice — with per-sentence vs aggregate feedback;
30 weak sentences, ~14 calls each; paired comparison.
**7. Experimental Setup:** System B refiner, GPT-4o-mini.
**8. Results:**
| Method | Mean improvement (95% CI) | Wins |
|---|---|---|
| Per-sentence feedback (ours) | +0.747 (0.60–0.89) | 10 |
| Aggregate feedback (original) | +0.808 (0.65–0.97) | 13 |
| ties | — | 7 |

Paired diff (ours − original) = **−0.062, 95% CI −0.25 to +0.13** (straddles 0). Sign-test
**p=0.53**; our win-share 10/23 = 43% (CI 26–63%, straddles 50%).
**9. Analysis:** **Negative/null result.** Per-sentence *feedback* does not produce better
rewrites than aggregate feedback — the two refiners are statistically equivalent at the rewrite
step. Therefore the contribution is emphatically **not "a better refiner."** Its advantage is
**what gets refined** (routing), not **how well** it is refined.
**10. Conclusions:** Hypothesis **rejected**; this sharpened (not weakened) the thesis claim.
**11. Impact:** Reframed the entire contribution to *detection/routing*; the defensible thesis
claim is written around this null (see Overall Scorecard, G-summary). The **task-aware** refiner
prompts (REFINER_TASK_*) remain untested (a separate Test 6b-task, deferred).
**12. Thesis Relevance:** The crucial honesty result that defines what the contribution *is* and
*is not*.

---

## G6 — Task-Aware Generation Quality (Test 1)
**1. Experiment Name:** Task-aware generation quality evaluation.
**2. Research Question:** Do the task-aware prompts produce sentences that (a) actually satisfy
the intended task, (b) are genuinely code-switched, (c) hit the target CS ratio, and (d) read
fluently/naturally?
**3. Motivation:** Task-aware generation is only useful if the generated label is *correct*;
quality scores alone may not detect task-level failures.
**4. Hypothesis:** Surface quality will be high; task-correctness and CS-ratio adherence may lag.
**5. Baseline:** None (characterization of System B output).
**6. Methodology:** On the fresh pre-refinement sample (refiner OFF), 40 sentences/task (35 NER).
Task-correctness by a **blind GPT-4o-mini judge** not shown the target (sentiment =
re-classification; topic = relevance; NER = entity extraction + deterministic constraint check).
CS-validity and CS-ratio are **deterministic/objective**. Fluency/naturalness = pipeline judge
scores.
**7. Experimental Setup:** System B, GPT-4o-mini; per_sentence/validation_raw.
**8. Results:**
| Task | n | Task-correct % (blind LLM) | CS-valid % | CS-ratio MAE vs 70 | Fluency | Naturalness |
|---|---|---|---|---|---|---|
| topic | 40 | **100.0** | 100.0 | 23.3 | 8.35 | 8.07 |
| sentiment | 40 | **70.0–72.5** | 87.5 | 22.1 | 8.05 | 8.05 |
| NER | 35 | **40.0** (English-only, final) | 97.1 | 13.8 | 8.40 | 8.54 |

NER judge evolution (documented explicitly): **45.7%** (original loose judge) → **62.9%**
(constraint-aware but lenient, counted Arabic-script entities) → **40.0%** (final,
pipeline-consistent English-only policy; failures surface as missing_PER, 19/21). Sentiment drag
is the **neutral** class (factual/descriptive sentences read as mildly polar or vice-versa).
**9. Analysis:** Strong surface quality (CS-valid 87–100%, fluency/naturalness ~8) but weaker
**constraint satisfaction** — topic near-perfect; sentiment moderate (neutral drag); **NER
weakest (40%)** because under the English-only policy the model rarely emits English-script
PERSON entities (it writes names in Arabic script). **Crucially, fluency/naturalness stay ~8
even where the task fails** — so quality scoring alone cannot detect task-level failures. Realized
CS-ratio is 14–23 points off the 70% target.
**10. Conclusions:** Mixed: task-aware generation works for topic, is moderate for sentiment,
and under-produces English-script entities for NER; quality scores do **not** substitute for a
task-validity check.
**11. Impact:** Directly motivated (a) the TaskValidator necessity study (G9) — since quality
misses task failures; (b) the NER PER-prompt repair (G7) and generic entity-guidance (G8); and
(c) informed the sentiment data-generation filtering (G12–G16).
**12. Thesis Relevance:** The characterization that justifies both the TaskValidator and the
deterministic CS metrics as necessary components.

---

## G7 — NER PERSON-Entity Prompt Repair
**1. Experiment Name:** NER PER-prompt repair pilot.
**2. Research Question:** Can an explicit English-script PERSON requirement (+ self-check) raise
NER task-correctness without regressing CS-ratio/naturalness?
**3. Motivation:** Test 1 (G6) localized NER failure to **missing_PER**; a constraint-difficulty
run showed ORG is easy (~90%) but **PERSON is the bottleneck** (difficulty non-monotonic).
**4. Hypothesis:** A PER-focused prompt with Arabic-friendly Latin names (Ahmed Ali, Sarah
Hassan…) raises task-correctness and cuts missing_PER.
**5. Baseline:** Current NER prompt (same-session control arm).
**6. Methodology:** Controlled before/after pilot, 50/arm, Wilson CIs; core prompt untouched
(variant in the harness); adds an explicit English-script PERSON requirement + self-check.
**7. Experimental Setup:** System B NER generation, GPT-4o-mini.
**8. Results:**
| Arm | Task-correct % (95% CI) | missing_PER | CS-ratio MAE | fluency / naturalness |
|---|---|---|---|---|
| current prompt | 22.5% (12.3–37.5) | 70% | 18.0 | 8.32 / 8.4 |
| **PER-focused** | **56.8% (42.2–70.3)** | **25%** | 15.95 | 8.61 / 8.52 |
**9. Analysis:** Non-overlapping CIs → a **real +34-point improvement**, not run-to-run noise; no
CS-ratio or naturalness regression. Absolute NER % is noisy run-to-run (~22–60% same config), so
the trustworthy signal is the **within-pilot delta**. The model still occasionally writes the
name in Arabic script (correctly rejected by the English-only policy).
**10. Conclusions:** Hypothesis confirmed; PER guidance is the effective lever for NER.
**11. Impact:** **Promoted into the core `DATA_GENERATION_NER_PROMPT`** (generation prompt only;
validator/policy/config unchanged); generalized in G8.
**12. Thesis Relevance:** A diagnosed→fixed→promoted improvement demonstrating the value of
constraint-explicit prompting.

---

## G8 — Generic Config-Driven NER Entity Guidance (FROZEN)
**1. Experiment Name:** Generic NER entity-guidance builder.
**2. Research Question:** Can the PER fix be generalized to *any* entity type via config, rather
than hardcoding examples per tag in the prompt?
**3. Motivation:** Beyond PER, the model under-produces English-script EVENT, LOC, PRODUCT
entities; hardcoding per-tag examples does not scale.
**4. Hypothesis:** A config-driven guidance block for only the required types lifts missing-type
production across tags without a PER regression.
**5. Baseline:** Hardcoded-example prompt (per type).
**6. Methodology:** `build_ner_entity_guidance()` reads `must_include_types` + per-type metadata
(`DEFAULT_ENTITY_GUIDANCE`; overridable via `ner.entity_type_guidance`: description/script_rule/
examples) and injects a dynamic block for **only the required types** into `{ner_entity_guidance}`;
`prompt.py` has **zero hardcoded tag examples** (unknown tags get a safe fallback). Validation via
missing-type-drop (the robust signal); same-session A/B for PER regression.
**7. Experimental Setup:** System B NER, GPT-4o-mini.
**8. Results (generic vs baseline, task-correct %):** LOC 44→63%, PRODUCT 45→85%, **EVENT 39→84%**
(missing EVENT 61%→5%), EVENT_LOC 25→83%. **PER_EVENT remains hard (11→22%)** — two Arabic-natural
hard types competing for 2–3 entity slots (fixing EVENT costs PER). Same-session A/B: **no PER
regression** (generic 53% ≈ PER-pilot 57%). Real core-pipeline smoke (real TaskValidator ON,
refiner OFF, English-only, n=12): parse 12/12, CS-valid 12/12, fluency/naturalness 8.8/8.5;
task-correct 3/12 (25%, PER variance); the real NER validator passed **0/12** — much stricter than
the judge. Regression tests: NER-guidance 5/5, task-gen mock, full-pipeline mock, per-instance
scoring, refiner-guardrail (2 assertions updated for the post-loop-fix behaviour).
**9. Analysis:** Generic guidance lifts every required-but-missing type substantially and scales
by config, not prompt edits. **PER_EVENT** is the residual hard case (competing hard types under a
tight entity budget), left unoptimized by design. The real TaskValidator being far stricter than
the judge (0/12) is an honest signal that NER remains the hardest task.
**10. Conclusions:** Hypothesis confirmed for isolated types; PER_EVENT is a known limitation.
**11. Impact:** **NER FROZEN** (commit e152b4d): implemented, generic, tested; no further NER
prompt changes unless tests break.
**12. Thesis Relevance:** The final NER methodology + its explicit limitation (competing hard
entity types).

---

## G9 — TaskValidator Necessity & Effectiveness (Test 2)
**1. Experiment Name:** TaskValidatorAgent necessity study.
**2. Research Question:** Does adding the (real, fallible) TaskValidator as an acceptance gate
reduce **task-wrong accepts** over a quality-only gate — and for which tasks?
**3. Motivation:** Test 1 (G6) showed quality scores stay ~8 even when the task is wrong, so a
quality-only gate lets task-wrong sentences through.
**4. Hypothesis:** quality + validator accepts fewer task-wrong sentences than quality-only.
**5. Baseline:** Policy A = quality-only (weighted score ≥ 7.0).
**6. Methodology:** Replay two acceptance policies over the Test-1 sentences (**no regeneration**),
running the **real TaskValidatorAgent** (not an oracle). Reference task-correctness = the Test-1
blind judge (final English-only NER ref). Policy B = quality AND validator. Verdicts cached.
**7. Experimental Setup:** System B validator, GPT-4o-mini.
**8. Results:**
| | A: quality-only | B: quality + validator |
|---|---|---|
| accepted | 86 | 62 |
| precision (task-correct among accepted) | 70.9% | **85.5%** |
| task-wrong accepted | 25 | **9** |
| false-accept (of all wrong) | 78.1% | 28.1% |
| false-reject (of all correct) | — | 36.1% |

Validator as a standalone detector vs reference: precision **83.0%**, recall **88.0%**, agreement
78.3%. **Per-task:** topic 0→0 task-wrong (validator only over-rejects, FN 17.5%); **sentiment
5→5 (no effect — neutral errors evade both)**; **NER 17→4 (false-accept 89.5%→21.1% — validator
earns its keep)**.
**9. Analysis:** The validator lifts precision 70.9%→85.5% and cuts task-wrong accepts 25→9, but
the benefit is **concentrated in NER** (entity-constrained), **null for sentiment** (the neutral
class evades both quality and validator), and it **over-rejects topic**. Honest claim: worth it
*specifically for entity-constrained tasks*, not as a blanket gate.
**10. Conclusions:** Hypothesis confirmed **for NER**, rejected for sentiment/topic.
**11. Impact:** Justified keeping the TaskValidator (as a task-specific, not universal, gate) and
directly shaped the sentiment data-generation filter (which relies on the validator + a
deterministic CS-validity filter + a human spot-check for neutral, G12).
**12. Thesis Relevance:** The evidence that scopes the TaskValidator's contribution honestly.

---

## G10 — CS-Ratio Measurement Validation (Test 4, PARTIAL)
**1. Experiment Name:** CS-ratio measurement-method validation.
**2. Research Question:** Is the deterministic CS-ratio counter more reliable than the baseline
LLM-only counting?
**3. Motivation:** The baseline estimates CS-ratio with an LLM; the contribution adds a
deterministic token counter. Reliability must be shown.
**4. Hypothesis:** Deterministic counting is reproducible; LLM-only counting is not.
**5. Baseline:** Original-style LLM-only counting (GPT-4o-mini, temp 0.7).
**6. Methodology:** Fixed 30-sentence set (20 real across tasks/masked/control + 10 controlled
edge cases). Methods: (1) deterministic `compute_true_cs_stats`; (2) LLM-only, **repeated 3×** to
measure instability; (3) human counts (**blank → PENDING**). **Measurement-only — no generation
run.**
**7. Experimental Setup:** GPT-4o-mini for the LLM counter.
**8. Results:**
| Signal | Value |
|---|---|
| LLM-only repeats disagree across 3 runs | **12/30 (40%)** |
| mean per-sentence LLM std (Arabic count / %) | 0.60 tokens / **2.32%** |
| deterministic counter variance | **0 (exact, free, reproducible)** |
| det vs LLM binary is_code_switched mismatch | 0/30 |
| mean det-vs-LLM Arabic-% abs diff | 5.04% |
| monolingual edge cases correct (det / LLM) | 2/2 / 2/2 |
**9. Analysis:** The two methods agree on the **binary** CS decision (0/30 mismatch) and roughly
on the ratio (~5% mean gap), but the **LLM-only counter is non-reproducible** — 40% of sentences
get a different count on a re-run of the *same* sentence — whereas the deterministic counter is
exact and free. Which is more *accurate* is **PENDING** human token counts.
**10. Conclusions:** Reproducibility demonstrated; accuracy pending. Hypothesis confirmed for
reproducibility.
**11. Impact:** Justified using the deterministic counter as the ground-truth CS metric in all
generation experiments and the downstream data-generation filters.
**12. Thesis Relevance:** The reliability argument for the deterministic CS-ratio component.

---

## G11 — Human Evaluation (BUILT, NOT RUN)
**1. Experiment Name:** Blind human validation of masked sentences + AI-judge agreement.
**2. Research Question:** Do neutral bilingual annotators confirm that machine-flagged masked
sentences are genuinely weaker, and does the AI judge agree with humans dimension-by-dimension?
**3. Motivation:** Removes the circularity of the AI grading its own output; the CONFIRM step for
G2.
**4. Hypothesis:** Masked sentences rate lower than neighbours; AI judge correlates with humans.
**5. Baseline:** Machine per-dimension scores (hidden key).
**6. Methodology:** Blind, shuffled (seed 42) sheets. Two instruments: (a) masking sheet — 50
sentences (13 MASKED + 37 neighbours) from 11 scenarios, rated fluency/naturalness/cultural
1–10 + is_real_codeswitch yes/no; larger pool available (30 MASKED / 103). (b) Consolidated
blind sheet — **86 rows** unlocking task-correctness by task, CS-validity, acceptability,
AI-judge-vs-human and Validator-vs-human agreement, masked-vs-control quality,
neutral-sentiment dispute resolution, NER English-script compliance. Analyzers built and verified
on dummy completed input (9 analyses; Mann-Whitney, Spearman, sign test, Cohen's kappa;
scipy-free). Plus CS-ratio human token counts (30 rows) → Arabic/English/other MAE, ratio MAE,
detection accuracy, boundary error for both counters.
**7. Experimental Setup:** GPT-4o-mini machine scores as the comparison key.
**8. Results:** **PENDING — annotators have not completed the sheets.** No human metrics exist yet;
the analysis pipeline is verified end-to-end on dummy data only.
**9. Analysis:** This is the one piece of the masking-defense argument that remains machine-only;
its absence is the principal open limitation of Track 1.
**10. Conclusions:** Not yet answerable; infrastructure complete.
**11. Impact:** Gates the strongest form of the masking claim (human-confirmed).
**12. Thesis Relevance:** Explicitly reported as **future work / open limitation** in the thesis —
never as a completed result.

---

## G12 — SwitchLingua Sentiment Data Generation (Experiment C pilot v1)
**1. Experiment Name:** Experiment C — SwitchLingua sentiment generation pilot (114 balanced).
**2. Research Question:** Can the Modified pipeline generate a **trainable, balanced,
CS-valid** Arabic–English sentiment dataset for the downstream track?
**3. Motivation:** Track 2 needs SwitchLingua-generated sentiment data to test extrinsic quality.
**4. Hypothesis:** After task-validity + CS-validity + quality filtering, a balanced set can be
extracted.
**5. Baseline:** None (first sentiment generation run).
**6. Methodology:** Config `config_sentiment_expC.yaml` (324 sentiment scenarios by design);
quality threshold (weighted) 7.0; filter funnel = TaskValidator pass → deterministic CS-valid →
quality ≥ 7.0 → de-dup; then down-sample to the smallest label. NER frozen.
**7. Experimental Setup:** System B, GPT-4o-mini; ~50–70 API requests/scenario.
**8. Results:** **PARTIAL RUN** — daily 10,000 RPD cap hit at ~scenario 130; 324 attempted, ~130
succeeded, **629 raw instances**. Filter funnel: 629 → TaskValidator-passed 612 → +CS-valid 172 →
+quality≥7.0 141 → +dedup **141 kept**. Kept per label (pre-balance): positive 54, negative 49,
neutral 38 → **balanced 38/label = 114 total**. Example kept sentences span ar% 35–71, q 7.0–8.0.
**9. Analysis:** The modified pipeline is **request-heavy** (per-sentence scoring + validator +
refine loop ⇒ ~50–70 requests/scenario), so a 324-scenario run needs >20k requests — more than one
day's cap; only ~40% of scenarios yielded data. The dominant filter loss is **CS-validity** (612
validator-passed → only 172 CS-valid) — i.e. most generated sentences are not genuinely
code-switched (diagnosed in G13). Neutral is the scarcest kept label (38), capping the balanced
size. Residual risk of mildly-polar "neutral" remains (prompt frozen; neutral quality enforced
post-hoc by the validator).
**10. Conclusions:** A balanced 114-sentence pilot was produced; the binding constraints are the
RPD cap and CS-validity yield.
**11. Impact:** First standalone SwitchLingua sentiment set; exposed the CS-validity yield problem
(→ G13 diagnosis, G14 fix) and the scaling constraint (→ G15). Fed the earliest downstream
transfer tests.
**12. Thesis Relevance:** The bridge from Track 1 (generation) to Track 2 (extrinsic evaluation).

---

## G13 — CS-Validity Failure Diagnosis
**1. Experiment Name:** CS-validity failure diagnosis (690 instances).
**2. Research Question:** Why do only ~30% of generated sentiment sentences pass the CS-validity
filter?
**3. Motivation:** CS-validity is the dominant yield bottleneck in G12.
**4. Hypothesis:** Failures are dominated by fully-Arabic (monolingual) output driven by the 70%
Arabic target.
**5. Baseline:** N/A (read-only diagnosis; no prompt/config change).
**6. Methodology:** Analyze 690 non-empty instances (pilot_v1 + daily runs); characterize
failures by script, label, topic, cs_type, cs_function, intensity.
**7. Experimental Setup:** Deterministic token analysis.
**8. Results:** CS-valid **209/690 (30%)**; CS-fail **481 (70%)** = fully-Arabic **479**,
fully-English **2**, no-letters 0. Mean Arabic share **86.5%** over all non-empty (above the 70%
target), **56.3%** over valid CS (12/209 fragile = exactly 1 English token). Fail rate by label:
**positive 77% (worst), negative 69%, neutral 62% (best)** — "neutral goes monolingual" myth
busted. Fail by cs_type: **Intersentential 74% > Intrasentential 65%**. Fail by intensity: low
74% > medium 71% > high 64%. Topic fail 56–80% (sports best, health worst).
**9. Analysis:** The single failure mode — **70% of sentences are 100% Arabic with zero English
tokens** — is the whole yield problem. When the model *does* code-switch, output is healthy (56%
Arabic, only 6% fragile), so the generator *can* code-switch but frequently emits pure Arabic.
The 70% Arabic target biases short single sentences to all-Arabic; **Intersentential** switching
makes a single-sentence instance monolingual by design (mismatched with a *per-sentence*
CS-validity filter).
**10. Conclusions:** Hypothesis confirmed; root cause = too-Arabic target + Intersentential
cs_type, not the filter (the filter correctly rejects genuine monolingual text).
**11. Impact:** Prescribed the **config-only** fix tested in G14 (lower cs_ratio + drop
Intersentential); explicitly rejected loosening the CS-validity definition (would pollute the
dataset) and prompt changes (frozen).
**12. Thesis Relevance:** The diagnostic that explains the generation dataset's yield economics.

---

## G14 — CS-Validity Fix Pilot (v2, config-only)
**1. Experiment Name:** Pilot v2 — CS-validity config fix.
**2. Research Question:** Does lowering the CS-ratio target and restricting to Intrasentential
switching raise CS-validity yield?
**3. Motivation:** G13 prescribed exactly this config-only lever.
**4. Hypothesis:** `cs_ratio ∈ {50,60,70}` + `cs_type = Intrasentential` raises CS-valid yield.
**5. Baseline:** Pilot v1 (30% CS-valid).
**6. Methodology:** Config-only change (`cs_ratio: [50%,60%,70%]`, `cs_type: [Intrasentential]`);
no prompt/NER/pipeline change; isolated run, **not merged** into the training set.
**7. Experimental Setup:** System B, GPT-4o-mini; quality threshold 7.0.
**8. Results:** 40 scenarios requested, **6 completed** (34 failed to 429/other — RPD).
CS-valid rate **30% → 43% (12/28)**. By target: **60% → 70% valid (7/10)** vs **70% → 28% valid
(5/18)** — the lower target is much better. Fully-Arabic share of failures 99.6% → 100% (16/16).
Filter funnel 28 raw → 10 kept (loss: not_cs_valid 16, low_quality 2). Kept by label: positive 0,
negative 7, neutral 3.
**9. Analysis:** The config-only fix **improves CS-validity (+13 points)** and confirms G13's
mechanism — a 60% Arabic target more than doubles yield vs 70%. But the run again hit the RPD cap
(only 6/40 completed), and label balance skews negative, so the isolated pilot is small.
**10. Conclusions:** Hypothesis confirmed; lower CS-ratio + Intrasentential is the right recipe.
**11. Impact:** Recipe adopted for the scaled GEN builds (G15); documented that a
generation-instruction requiring ≥1 English insertion (a prompt change) is the higher-leverage
but currently-frozen next step.
**12. Thesis Relevance:** The config-tuning result that operationalizes the diagnosis into a
production recipe.

---

## G15 — Scaled Generated Sentiment Datasets (GEN-240 / 480 / 960)
**1. Experiment Name:** Scaled standalone SwitchLingua sentiment datasets.
**2. Research Question:** Can the generation recipe be scaled to balanced datasets of 240, 480,
and 960 sentences for the downstream scaling study?
**3. Motivation:** Track 2's standalone-transfer experiments (C1/C2/C3) require size-graded,
balanced, CS-valid datasets.
**4. Hypothesis:** The recipe scales to larger balanced sets across multiple daily runs.
**5. Baseline:** Pilot v1 (114 balanced).
**6. Methodology:** Repeated generation across days (working around the RPD cap), merged and
cross-deduplicated; balanced to equal labels; quality-filtered (weighted ≥ 7.0). Dataset cards
under `data/Sentiment/generated/merged/`.
**7. Experimental Setup:** System B, GPT-4o-mini.
**8. Results — GEN-960 profile:** n=960, **balanced 320/320/320** (neg/neu/pos), Arabic %
**52.7**, English % **47.3**, **CMI mean 40.9** (median 42.1; hist 0–20:17, 20–40:297, 40–60:646),
length mean 14.1 (median 14, p90 18), quality 7.0–9.0. GEN-240 and GEN-480 are size-graded balanced
subsets built by the same recipe.
**9. Analysis:** The scaled sets are **more balanced Arabic–English (CMI ~41)** than the raw
70%-target output — a direct effect of the G14 recipe (lower CS-ratio + Intrasentential). This
balanced, cleaner-register profile later becomes central to the downstream augmentation finding
(the generated data is off-domain vs EESA's Arabic-dominant, dialectal, noisy social-media text —
see Part 2).
**10. Conclusions:** Hypothesis confirmed; three size-graded balanced CS-valid datasets produced.
**11. Impact:** Directly enabled downstream C1-240, C2-480, C3-960 (standalone transfer +
scaling), the augmentation E-series, and the weak-primary agentic experiments.
**12. Thesis Relevance:** The final standalone generated datasets — the primary *product* of the
generation pipeline that the thesis evaluates extrinsically.

---

## G16 — Generation-Config Sensitivity Variants (V1_lowerCS, V2a_register)
**1. Experiment Name:** GEN sensitivity pilots (config variants).
**2. Research Question:** How sensitive are the generated data's properties (Arabic ratio, CMI,
CS-valid yield, length) to generation-config choices (lower CS-ratio; register)?
**3. Motivation:** Characterize the controllability of the generator (not EESA-tailored) and
whether variants could better match a target domain.
**4. Hypothesis:** Lowering CS-ratio raises Arabic share and lowers CMI; register changes shift
CMI/length.
**5. Baseline:** GEN-960 (AR 52.7, CMI 40.9, len 14.1).
**6. Methodology:** Isolated pilots, cross-deduped vs GEN-960 and each other; balanced subsets.
**7. Experimental Setup:** System B, GPT-4o-mini.
**8. Results:**
| Variant | scenarios (done/fail) | raw | kept / balanced | CS-valid yield | AR:EN | CMI mean | len mean | vs GEN-960 |
|---|---|---|---|---|---|---|---|---|
| **V1_lowerCS** | 90/0 | 429 | 88 / **56** (neg16/pos20/neu20) | 29.4% | **62.3:37.7** | **32.9** | 14.4 | AR +9.6pp, CMI −8.0, len +0.3 |
| **V2a_register** | 37/33 | 184 | 76 / **49** (neu20/neg9/pos20) | 52.2% | 51.1:48.9 | 43.1 | 12.5 | AR −1.6pp, CMI +2.2, len −1.6 |
**9. Analysis:** The generator is **controllable**: V1_lowerCS shifts toward more Arabic and lower
CMI (closer to EESA's Arabic-dominance) but at a lower CS-valid yield (29.4%); V2a_register raises
CS-valid yield (52.2%) and CMI while shortening sentences. These confirm CS-ratio and register are
real, independent knobs on the output distribution.
**10. Conclusions:** Hypothesis confirmed; the config sensitivity is characterized.
**11. Impact:** V1_lowerCS was carried forward into a downstream variant dataset (a 480-sample
V1_lowerCS build used in the augmentation/config-sensitivity line, per the downstream track);
V2a_register was shelved. Establishes that domain-matching a target corpus is possible via config
but trades off CS-validity yield.
**12. Thesis Relevance:** The controllability analysis — evidence that generation properties are
config-steerable, relevant to the augmentation-domain-mismatch discussion in Part 2.

---

## Track 1 — Overall Scorecard (from the masking-defense report)
| Claim | Evidence | Verdict |
|---|---|---|
| Per-sentence scoring **catches** weak sentences aggregate hides | 41.5% (54-scen), **35.6%** (101-scen) at bar 7.0 | ✅ supported (calibrated bar) |
| Refining a caught weak sentence **improves** it | +0.60, 79/87 (90.8%), p≈0 | ✅ supported |
| Per-sentence *feedback* makes **better rewrites** | tie, paired diff −0.062, p=0.53 | ❌ not supported (null) |
| Task-aware generation produces valid task data | topic 100%, sentiment 70%, NER 40%; CS-valid 87–100% | 🟡 mixed |
| TaskValidator reduces task-wrong accepts | precision 70.9%→85.5%, 25→9; NER-concentrated | ✅ NER; ❌ sentiment |
| Pipeline hits the 70% CS ratio | MAE ~14–23 pts off | ❌ off-target |
| Deterministic CS counter > LLM-only counting | LLM-only 40% self-disagreement; det 0 variance | 🟡 reproducibility yes; accuracy pending |
| Quality scoring alone detects task failures | fluency/naturalness ~8 even when task fails | ❌ → motivates validator |
| Humans confirm masked sentences are weaker | built, not run | 🟡 pending |
| At bar 8 (default) masking occurs | 0% on gpt-4o-mini | ❌ (bar must be ~7) |

**Defensible thesis claim (Track 1):** *The contribution is per-sentence scoring as a
detection/routing mechanism, not a better refiner. Aggregate scoring lets a weak sentence escape
unrefined in ~36–41% of scenarios (calibrated bar 7.0, gpt-4o-mini); per-sentence scoring catches
and routes it to the refiner, which improves it by +0.60 (90.8% of cases). The two refiners are
equivalent at the rewrite step (p=0.53) — the advantage is what gets refined, not how well.*

---

*Part 1 (Track 1: generation framework) ends here. No metrics above are invented; every value is
transcribed from the source reports under `experiments/outputs/switchlingua/` and
`multi-agent-bert/data/Sentiment/generated/`.*

---

# PART 2 — TRACK 2: MULTI-AGENT DOWNSTREAM EVALUATION (index)

The 32 downstream experiments are documented **in full, in an even richer per-experiment
structure**, in the companion registry
`multi-agent-bert/experiments/outputs/multi_agent_bert/EXPERIMENT_REGISTRY.md` (Parts I–IV:
per-experiment entries, architecture/prompt/aggregation/gate/model/training/negative-results
tables, and the verbatim agent prompts). To avoid duplicating that authoritative document, Part 2
is indexed here — every experiment is listed (none skipped) with its headline result and status;
the registry holds the Research Question / Motivation / Hypothesis / Methodology / Setup / Results
/ Analysis / Conclusion / Impact for each. All downstream numbers are EESA sentiment test (818)
unless noted; primary = XLM-R; agents = GPT-4o-mini unless noted.

| # | Experiment | Headline result | Status |
|---|---|---|---|
| D1 | Experiment A — EESA mBERT reference | primary_only 0.7971 / 0.7833 | Historical |
| D2 | Experiment A — EESA XLM-R reference | primary_only **0.8240 / 0.8088** | Reference |
| D3 | Real-LLM full_agentic pilot (@0.6) | XLM-R 0.8240→**0.8399** (net +13 esc) | Accepted |
| D4 | Real-LLM threshold sweep (0.6–0.9) | acc peaks **0.8460**@0.8; more escalation ≠ harm | Accepted |
| D5 | Prompt audit + 4 correctness fixes | Fix#2 +0.064 (paper_style); Fix#3 anchoring, off | Accepted |
| D6 | Consensus 2×2 ablation (th 0.8 & 0.9) | **best 0.8509 / 0.8401** (cell B @0.9) | Accepted/locked |
| D7 | Experiment C1 — generated-240 | primary_only 0.5905 / 0.5619 (transfer pilot) | Superseded |
| D8 | C2/C3 + 3-seed stability | 240→480→960 = 0.59→**0.65**→**0.67**; "480>960" retracted | Accepted |
| D9 | C3-960 full_agentic (seed 456) | 0.6956→**0.7543** (**Δ +0.059**, W→C 71/C→W 23) | Current best (weak, 4o) |
| D10 | E0 — EESA-only Adafactor | **0.8533 / 0.8409** (best trained primary) | Current best (trained) |
| D11 | E3 / LR / ratio / diagnosis (augmentation) | E0→E3 **−0.012**; domain mismatch (~10% vocab) | Rejected (augmentation) |
| D12 | T1 / T2 — topic (ARENTC, 9-class) | primary **0.9946 / 0.9947**; agents net +3 / −6 (noise) | Accepted (primary_only) |
| D13 | Ahmed external baseline | **0.9254 / 0.9207** (strongest EESA model) | Reference ceiling |
| D14 | Ahmed frozen-primary full_agentic (default) | 0.9254→0.9205 (**net −4**) | Historical |
| D15 | Agent-behaviour comparison | the **primary-strength curve** (agent ceiling ~0.75) | Foundational |
| D16 | semantic_v1 (Design A) | 0.9230 (net −2); agreement 92%→84.5% | Superseded |
| D17 | Polarity decomposition ablation A/B/C/D | **C (Lex+Pol+Ctx) 0.9267, net +1** (first > primary) | C lead; A/D retired |
| D18 | Design E — Intent as 4th voter | 0.9267, net +1 (ties C, +30% cost) | Opt-in |
| D19 | Design F — remove Lexical | 0.9218, **net −3** (over-neutralization) | Rejected |
| D20 | Design G — non-voting IntentGate | **0.9279 / 0.9242, net +2** (best @4o-mini) | Lead @4o |
| D21 | Design G2 — selective gate | ties G @4o (0.9279); lost 3/4 platform blocks | Retired@4o (revived@4.1) |
| D22 | Design v3 — pragmatic Contextual | ties G (0.9279); Contextual +0.024, system flat | Opt-in |
| D23 | Consensus investigation (loss/rescoring/weights) | all simple re-fusion ties or loses to G (−1 to −4) | Rejected (simple fusion) |
| D24 | Sequential v1 (Ahmed) | 0.9242, **net −1** (Stage-3 inert) | Rejected (strong primary) |
| D25 | Sequential v2 (Ahmed) | 0.9120, **net −11** (over-calls) | Rejected (strong primary) |
| D26 | Design G on C3 (weak primary) | 0.6956→**0.7604** (**net +53, p≪0.001**) | Current best (weak, 4o) |
| D27 | Sequential v2 on C3 | **+47** (< G's +53; 2.3× breakage) | G wins |
| D28 | Stronger-model diagnostic (4.1-mini) | fixes **4/18** residual failures (0 from noise) | Accepted diagnostic |
| D29 | G @ gpt-4.1-mini (Ahmed) | **0.9291, net +3** (best-yet, non-significant) | Best-yet (n.s.) |
| D30 | Why 4.1-mini broke cases | gate × model-strength interaction (over-veto) | Diagnosis |
| D31 | Wash diagnosis → semantic_v2_disambig | **net −1 to −4** (can't prompt past ceiling) | Rejected |
| D32 | GPT-4.1-mini gate ablation (G/G2/C) | **G2@4.1-mini = 0.9303 / 0.9262** (first > 0.930) | **Current best (strong)** |

*Full 11-field detail for D1–D32: `EXPERIMENT_REGISTRY.md`.*

---

# RESEARCH EVOLUTION — from the first SwitchLingua idea to the final pipeline

**1. The starting point (the NeurIPS-2025 SwitchLingua baseline).** The project began with a
published Arabic–English code-switch *generation* pipeline (System A / Original_baseLine, GPT-4o):
a LangGraph of a generation agent, four quality-judge agents (fluency, naturalness, CS-ratio,
social-cultural), and a generic refiner, producing **one aggregate score per scenario** for a
single implicit task. The scientific question that seeded this thesis was: *does aggregate scoring
hide weak sentences, and can task-awareness make the generated data usable for downstream NLP?*

**2. Redesign into a task-aware, per-sentence pipeline (System B).** The Modified_Version pipeline
(G0) reworked the baseline into a **multi-task** (topic/sentiment/NER), **per-sentence-scored**
system with a new **TaskValidatorAgent**, a **selective task-aware refiner** with a rollback
guardrail, **deterministic CS-ratio counting**, and generic config-driven NER guidance — moved to
**GPT-4o-mini** for cost. Early full runs exposed the honest reality of that model: overall quality
~7.3 (never reaching the 8.0 bar), CS-ratio ~50% (target 70%), TaskValidator pass ~40% — i.e. the
new machinery kept *detecting* failures the baseline would have hidden.

**3. The masking-defense program (proving the contribution).** Rather than claim a quality lift
(which GPT-4o-mini could not deliver), the contribution was defended as **detection/routing**. The
central experiment (G2/G3) showed per-sentence scoring **catches a weak sentence in ~36–41% of
scenarios** that aggregate scoring accepts, at a *calibrated* bar (7.0 = median quality) — with the
honest caveat that at the default bar 8.0 masking is 0% because the model's whole range tops out at
7.71. The refiner then genuinely **improves a routed weak sentence (+0.60, 90.8%, p≈0)** (G4). The
pivotal moment was G5: a head-to-head showed **per-sentence feedback does *not* produce better
rewrites than aggregate feedback (tie, p=0.53)** — a null result that *reframed the entire thesis
claim* from "a better refiner" to "better *routing* — what gets refined, not how well." Supporting
studies characterized task-aware quality (G6: topic strong, sentiment moderate, **NER weakest**),
repaired NER via PER-explicit (G7, +34 pts) then generalized it (G8, FROZEN), scoped the
TaskValidator to **entity-constrained tasks only** (G9: helps NER, null for sentiment), and showed
the deterministic CS counter is **reproducible where LLM counting is not** (G10). The one missing
piece — **human confirmation (G11)** — was built but never run, and is carried as the principal open
limitation. The pipeline was then **frozen (2026-06-05)**.

**4. Turning the generator into a downstream dataset.** To test the generated data's *extrinsic*
value, the frozen pipeline generated sentiment data (G12: a 114-sentence balanced pilot). This
immediately surfaced two constraints: the pipeline is **request-heavy** (~50–70 API calls/scenario,
so a 324-scenario run exceeds the 10k daily cap) and **CS-validity yield is only ~30%**. A
read-only diagnosis (G13) found the cause — **70% of generated sentences were fully Arabic** because
the 70% Arabic target over-biases short sentences to monolingual, worsened by Intersentential
switching. A **config-only fix** (G14: lower CS-ratio + Intrasentential-only) lifted CS-validity
30%→43%, which was scaled into the balanced **GEN-240/480/960** datasets (G15, CMI ~41,
balanced Arabic–English) and characterized for config-sensitivity (G16: CS-ratio and register are
real, independent knobs).

**5. Extrinsic evaluation via a multi-agent classifier (Track 2).** A separate confidence-aware
multi-agent classification pipeline (a fine-tuned primary + escalation to LLM specialist agents +
consensus) was built to *evaluate* the generated data and, in its own right, to study when agents
help. The reference primaries were established (D1/D2: mBERT 0.7971, XLM-R **0.8240**); real LLM
agents were shown to help where mock agents hurt (D3/D4, up to 0.8460); four correctness fixes were
audited and a controlled 2×2 locked the best XLM-R setting (D5/D6, **0.8509**). The generated data
then proved itself as **standalone** training material — transferring to real EESA and **scaling
240→480→960** (D7/D8), with the multi-agent layer delivering its **largest rescue on the weak
generated primary (+0.059**, D9). Crucially, the same generated data **failed as naive
augmentation** of real EESA (D11, −0.012) because of a **domain/register mismatch** (the generated
data is more balanced, cleaner, and shares only ~10% of EESA's vocabulary) — a finding that ties
directly back to the generation-config profile (G15/G16). Topic (D12) extended the framework to a
second task and anchored the near-perfect end of the curve.

**6. The agentic-design arc and the ceiling.** Introducing Ahmed's external model as a very strong
**frozen primary** (D13, 0.9254) reframed everything: the agentic layer went slightly negative
(D14, −4). Consolidating C3 / EESA / Ahmed produced the project's unifying law (D15): **Δ ≈
(agent-ceiling − primary-on-escalated) × escalation-rate**, agent ceiling ≈ 0.75. The rest of the
arc was a disciplined attempt to beat that ceiling on the strong primary — prompt refinement (D16),
a **Polarity** decomposition that first exceeded the primary (D17, C = 0.9267), ablations pinning
the necessary agent set (D18/D19), and the breakthrough **non-voting IntentGate** (D20, G =
0.9279) that fixed the meta-comment cluster a vote never could. A full consensus investigation
(D23) and both sequential architectures (D24/D25) confirmed, from five independent directions, that
**you cannot prompt or re-architect past the ceiling on a strong primary**. The **weak-primary C3
check** made the thesis undeniable (D26: **+53, p≪0.001**; D27: sequential mirrors it, +47 but
loses to G's damage control). The last lever was **model quality**: GPT-4.1-mini recovered the
compliance/obscured-cue slice (D28/D29, 0.9291), a diagnosis showed the gate over-vetoes under a
stronger model (D30), a prompt-disambiguation attempt failed (D31), and the **gate ablation revived
the selective gate** — **G2 @ gpt-4.1-mini = 0.9303 / 0.9262**, the best configuration on record and
the first to cross 0.930 (D32).

**7. Why the final approach was selected.** For **generation**, per-sentence scoring + a task-aware
selective refiner + deterministic CS metrics were retained because they *detect and route* failures
aggregate scoring hides (G2–G4), even though they are not a better refiner (G5) and the TaskValidator
pays off only for entity-constrained tasks (G9). For the **generated datasets**, the low-CS-ratio +
Intrasentential recipe was selected because it is the only config-lever that raised CS-validity
without loosening the (correct) filter (G13/G14), and the data's value was confirmed to be
**standalone, not augmentation** (D9 vs D11). For **downstream inference**, **Design G2 (Lexical +
Polarity + Contextual + selective IntentGate) at gpt-4.1-mini** was selected for the strong-primary
regime (best point estimate, first > 0.930), and **Design G at gpt-4o-mini** for the weak-primary
regime (+0.065, highly significant) — with per-primary threshold calibration and primary-aware
consensus (Fix #2 on, Fix #3 off) as the fixed defaults.

---

# KEY SCIENTIFIC FINDINGS

### Generation contribution (Track 1)
1. **Per-sentence scoring catches weak sentences aggregate scoring hides** — ~36–41% of scenarios at
   a calibrated bar (7.0 = median), 95% CI 29–55%. The contribution is **detection/routing**.
2. **The routed weak sentence is genuinely improved** by the refiner: **+0.60, 90.8%, p≈0**.
3. **(Negative) Per-sentence *feedback* is not a better refiner** than aggregate feedback — paired
   diff −0.062, **p=0.53**. The advantage is *what* gets refined, not *how well*.
4. **Task-aware generation is task-dependent:** topic 100%, sentiment ~70% (neutral drag), **NER 40%**
   (English-only PERSON under-production). Surface quality (fluency/naturalness ~8, CS-valid 87–100%)
   stays high **even when the task fails** → quality scores cannot detect task-level failures.
5. **The TaskValidator earns its keep only for entity-constrained tasks (NER)** — precision
   70.9%→85.5%, task-wrong 25→9, but **null for sentiment** (neutral evades it) and it over-rejects
   topic. Not a blanket gate.
6. **Deterministic CS counting is reproducible; LLM-only counting is not** (40% self-disagreement vs
   0 variance); accuracy vs humans is still pending.
7. **NER improvements are diagnosable and promotable:** PER-explicit prompting +34 pts; a generic
   config-driven entity-guidance builder lifts every required-but-missing type (EVENT 39→84% etc.),
   with **PER_EVENT** (competing hard types under a tight entity budget) the residual limitation.

### Generated data properties (Track 1 → Track 2)
8. **CS-validity yield is governed by the CS-ratio target:** a 70% Arabic target makes **~70% of
   sentences fully Arabic**; lowering to 50–60% + Intrasentential-only roughly doubles yield.
9. **The generator is config-steerable** (CS-ratio and register are independent knobs on Arabic%,
   CMI, and length) — GEN-960 sits at balanced AR/EN (52.7/47.3), CMI ~41.
10. **"Neutral goes monolingual" is a myth** — neutral had the *best* CS-validity (62% fail) and
    positive the worst (77%).

### Downstream / agentic (Track 2)
11. **Agent value follows the primary-strength curve:** Δ ≈ (ceiling − primary-on-escalated) × rate,
    ceiling ≈ 0.75. Agents help a weak primary (**C3 +0.059/+53**), are neutral near parity, and
    **slightly hurt a very strong primary** (Ahmed −0.005) and a near-perfect one (topic ~0).
12. **Generated SwitchLingua data is genuinely useful as *standalone* training data** (scales
    240→480→960; largest agentic rescue), but **useless-to-harmful as naive augmentation** of real
    EESA (−0.012), due to a **domain/register mismatch** (~10% vocab overlap) — not a generation
    defect.
13. **A non-voting veto beats a vote** for the same pragmatic signal (12/12 suppressed as a vote, 0
    missed as a veto) — the IntentGate is the only aggregation change that ever helped.
14. **Gate aggressiveness must scale inversely with model strength** — the full gate over-vetoes
    under gpt-4.1-mini; the selective gate (G2) is the right amount, giving the best config 0.9303.
15. **Best configurations:** strong primary = **G2 @ gpt-4.1-mini (0.9303/0.9262)**; weak primary =
    **G @ gpt-4o-mini (0.7604, +0.065, p≪0.001)**; best trained primary = **E0 XLM-R Adafactor
    (0.8533)**; best EESA model overall = **Ahmed external (0.9254)**.

### Negative findings (kept, not hidden)
16. Per-sentence refiner feedback ≈ aggregate feedback (p=0.53).
17. Generated data as augmentation: neutral-to-harmful (−0.012; −0.034 when it dominates the mix).
18. Every strong-primary prompt/topology intervention failed: semantic_v1 (−2), v3 (0), sequential
    v1 (−1) / v2 (−11), disambig (−1 to −4) — **you cannot prompt past the ceiling**.
19. All simple consensus re-fusion rules tie or lose to G (−1 to −4).
20. Topic/near-perfect-primary agents are net noise (+3 / −6).
21. Mock (non-LLM) agents hurt the primary; the primary-signal prompt block only induces anchoring.

### Lessons learned
22. **Calibrate to the model, report the curve:** the masking bar and the router threshold both must
    be model-specific; a fixed global value (bar 8, threshold 0.9) transfers poorly.
23. **Isolate confounds:** cross-run before/after is invalid for a refiner (p=0.25 artifact); only
    within-sentence tests measure it (p≈0). Single-seed dataset comparisons are unreliable (the
    "480>960" artifact); use ≥3 seeds.
24. **Serialize what you'll need to analyze:** agent/stage confidences were never stored, which
    blocked all confidence-calibrated and learned-consensus analysis offline.
25. **The information floor is real:** ~⅔ of hard escalated errors are Bayes-irreducible from the
    text alone; no aggregation recovers them — only a better primary or a stronger/knowledge-augmented
    model can.

### Research decisions
26. Freeze the generation pipeline (2026-06-05) and defend it as **routing, not refinement**.
27. Fix generation CS-validity by **config only** (never by loosening the correct CS-validity filter,
    never by unfreezing prompts except the one promoted NER PER line).
28. Use generated data as **standalone** training data, not as an EESA augmenter.
29. Lock downstream consensus defaults: **primary-aware Fix #2 on (w_primary=1.0), Fix #3 off**;
    calibrate the router threshold **per primary**.
30. Adopt **Design G/G2 (Lexical + Polarity + Contextual + IntentGate)** as the sentiment
    architecture; use the **selective gate under the stronger model**; keep the **default trio
    (Lexical/Logical/Contextual)** for topic.
31. Report **human evaluation as pending future work**, never as a completed result.

---

*End of report. Track 1 (G0–G16) is documented in full above; Track 2 (D1–D32) is indexed here and
documented in full in `EXPERIMENT_REGISTRY.md`. Every metric is transcribed from the project's own
reports; none are invented, and negative/null results are retained throughout.*
