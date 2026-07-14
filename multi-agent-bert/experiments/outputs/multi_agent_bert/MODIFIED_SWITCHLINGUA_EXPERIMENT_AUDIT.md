# Modified Task-Aware SwitchLingua — Complete Experimental Audit

**Purpose.** Exhaustive reconstruction of every experiment, ablation, diagnosis, and evaluation
belonging to the **Modified Task-Aware SwitchLingua** contribution (the *data-generation* track and
its downstream utilization), for writing Chapter 5.2 of the thesis. Multi-agent classifier
experiments are excluded **except** where they were used to evaluate generated data.

**Method.** Reconstructed by reading the repository's experiment reports, dataset cards, generation
reports, diagnoses, and status files (not filenames alone). Every number below is quoted from a
source document; sources are named per experiment. Where a later run corrected an earlier reading,
the corrected version is authoritative and the change is noted.

**What the framework is.** "Modified Task-Aware SwitchLingua" = **System B** (`Modified_Version`),
the modified fork of the LinguaMaster/SwitchLingua generation pipeline. Its five contributions over
the **System C / Original baseline** (`Original_baseLine`, aggregate scoring, topic-only):
1. **Per-sentence scoring** (score each sentence, not the scenario average) — the detection/routing core.
2. **TaskValidatorAgent** (topic / sentiment / NER pass-fail, separate from the quality score).
3. **Targeted + task-aware refiner** with a **guardrail** (re-validate + re-score, rollback on regression).
4. **Deterministic CS-ratio counter** (`compute_true_cs_stats`; Arabic vs Latin token counts, 0 variance).
5. **Generic config-driven NER entity guidance** (`build_ner_entity_guidance()`, English-only entity policy).
All generation uses **gpt-4o-mini** (temp 0.7 generate, 0.1 judge/refine). Framework **code FROZEN
2026-06-05**.

---

## ⚠️ Critical correction to earlier project notes
Earlier consolidations stated "no two-stage / pretraining experiment exists." **That is now false.**
On **2026-07-05** a two-stage experiment (`EXPERIMENT_TWOSTAGE_GEN_AUGMENTATION.md`) was run and is
**the one augmentation setup that works** (+1.2 to +2.2 pts). The generated data's value is therefore
threefold, not twofold. This audit supersedes the earlier claim. It also folds in the later
`V1_lowerCS` sensitivity variant and the corrected primary-strength curve (mid-primary E0 gain is
now **significant**, p≈0.002).

---

# PART 1 — CLUSTERS AND EXPERIMENTS

Six clusters emerged from the repository (not forced):
- **Cluster A — Generation-framework validation** (the frozen SwitchLingua thesis: masking, refiner,
  task-aware quality, validator, CS-ratio, NER).
- **Cluster B — CS-validity diagnosis and the config-only fix.**
- **Cluster C — Corpus construction & scaling** (pilots → 240 → 480 → 960).
- **Cluster D — Generator-config sensitivity** (V1_lowerCS, V2a_register).
- **Cluster E — Generated-corpus utilization** (standalone transfer, mixing, two-stage, downstream
  weak-primary).
- **Cluster F — Human validation** (built, pending).

═══════════════════════════════════════════════════════════════════════

## CLUSTER A — Generation-Framework Validation
*(System B vs System C; model held constant at gpt-4o-mini to isolate architecture; frozen 2026-06-05.
Main source: `experiments/outputs/switchlingua/masking_defense/REPORT.md`, 2026-05-31.)*

### A1. Masking — per-sentence scoring catches what aggregate scoring hides
1. **Name.** Masking measurement (the "CATCH").
2. **Motivation.** Core thesis claim: aggregate (scenario-mean) scoring accepts a whole scenario on
   its average and thereby *hides* an individual weak sentence; per-sentence scoring catches it and
   routes it to the refiner. Hypothesis: a non-trivial fraction of aggregate-accepted scenarios
   contain a sub-threshold sentence.
3. **Design.** Both decision rules applied to the *same* per-sentence scores (isolates rule, not
   generation). A scenario is *masked* iff aggregate ≥ bar **but** ≥1 sentence < bar. Scenarios with
   <2 sentences excluded. Reported as a full threshold sweep (anti-cherry-pick); operating bar =
   **7.0** = the model's **median** sentence quality. Two runs: 54-scenario (`step2`) and a larger
   101-scenario calibration (`per_sentence/threshold_calibration_report.md`).
4. **Datasets.** Pre-refinement (refiner OFF) generations. 54-scen run: 245 sentences, 53 usable.
   101-scen run: 458 sentences, 101 usable of 104.
5. **Models.** Generator gpt-4o-mini; quality judged by gpt-4o-mini judge. CS detection deterministic.
6. **Metrics.** Masking rate (+ Wilson CI), both-accept / both-refine counts, weak-sentence leakage,
   monolingual leakage, valid-CS%; score distribution.
7. **Results.**
   - Score distribution (RAW, 245 sent): **mean 7.057, median 7.000**, IQR 6.8–7.3, range 5.9–8.15;
     max scenario aggregate **7.71** (no scenario reaches 8).
   - **54-scenario sweep (53 usable):** bar 6.5 → 30.2% (16 scen); **bar 7.0 → 41.5% (22 scen, CI
     29.3–54.9%)**; bar 7.5 → 3.8% (2); **bar 8.0 (pipeline default) → 0.0%**; bar 8.5 → 0.0%.
   - **101-scenario sweep (458 sent):** bar 6.5 → 23.8% (CI 16.5–32.9); **bar 7.0 → 35.6% (CI
     27.0–45.4)**, weak-sent leak 16.4%, monolingual leak 2.1%, valid-CS 97.9%; bar 7.5 → 7.9%;
     **bar 8.0 → 0.0%**.
   - Example masked scenario: aggregate 7.225 (accepted), sentence scores [7.5, 7.0, **6.7**, 7.7].
8. **Interpretation.** Claim **supported** at the calibrated bar (35.6–41.5%). But masking is
   **threshold-sensitive** (41.5% at 7.0 → 3.8% at 7.5) because gpt-4o-mini's scores are tightly
   packed; at the pipeline default bar 8.0 masking is **0%** — the default bar sits above the model's
   entire aggregate range (max 7.71) and is inoperative for this model. Honest limitation, stated.
9. **Thesis relevance.** **§5.2 (framework evaluation) — the central contribution result.** The
   headline number is **35.6% at bar 7.0 (101-scenario, larger sample)**; report the 41.5% (54-scen)
   as the first run and the full sweep for transparency.

### A2. Refiner effectiveness — within-sentence (the "FIX")
1. **Name.** Clean within-sentence refiner test (Test 6a).
2. **Motivation.** After a weak sentence is *caught*, does the refiner actually *improve* it?
3. **Design.** Same sentence scored fresh → refined → scored fresh (no cross-run confound). ~9 API
   calls/sentence. Samples: 5 smoke, 30 weakest, all 87 weak (<7).
4. **Datasets.** The weak sentences (score <7) from the framework's own generations.
5. **Models.** gpt-4o-mini generate/refine/judge.
6. **Metrics.** Mean before/after, mean Δ (95% CI), % improved (CI), sign-test p.
7. **Results.**
   - 5 smoke: 6.07 → 7.03, **Δ +0.96**, 5/5 (100%), p 0.025.
   - 30 weakest: 6.41 → 7.17, **Δ +0.77**, 29/30 (96.7%), p≈0.
   - **All 87 weak: 6.64 → 7.24, Δ +0.60 (CI 0.51–0.68), improved 79/87 (90.8%, CI 82.9–95.3%),
     p≈0.**
8. **Interpretation.** **Supported.** A caught weak sentence is genuinely improved ~91% of the time,
   Δ≈+0.6 (CI excludes 0).
9. **Thesis relevance.** **§5.2 — the "does refinement help" result** (pairs with A1's "detection").

### A3. Refiner head-to-head — per-sentence vs aggregate feedback (the honest NULL)
1. **Name.** Refiner feedback-granularity head-to-head (Test 6b).
2. **Motivation.** Is the contribution a *better refiner*, or only better *routing*? Isolate the
   feedback variable (per-sentence scores vs scenario-average scores), holding prompt/model/sentence
   constant.
3. **Design.** Each of 30 weak sentences refined twice — with per-sentence feedback (YOURS) vs
   aggregate feedback (ORIGINAL). ~14 calls each. Paired.
4. **Datasets.** 30 weak sentences. 5. **Models.** gpt-4o-mini.
6. **Metrics.** Mean improvement per method (CI), wins/ties, paired diff (CI), sign-test p, win-share.
7. **Results.** YOURS +0.747 (CI 0.60–0.89), 10 wins; ORIGINAL +0.808 (CI 0.65–0.97), 13 wins; 7
   ties. **Paired diff (YOURS−ORIG) = −0.062 (CI −0.25 to +0.13, straddles 0); sign-test p = 0.53;
   YOURS win-share 10/23 = 43% (CI 26–63%).**
8. **Interpretation.** **NULL — refiners statistically equivalent at the rewrite step.** This is a
   *deliberate, honesty-defining* result: the contribution is **detection/routing (what gets
   refined), NOT a better refiner (how well it is refined).**
9. **Thesis relevance.** **§5.2 (must be included) — it precisely scopes the claim and is the single
   most defensible framing in the whole framework.** *Do not omit; it is what makes the contribution
   claim honest.*

### A4. Cross-run before/after — CONFOUNDED (appendix only)
1. **Name.** RAW-vs-FIXED cross-run before/after (Step 4). 2. **Motivation.** Naive "did the refiner
   improve the dataset" check. 3. **Design.** Two *independent* generations (refiner OFF vs ON).
4–5. 54 scenarios each, gpt-4o-mini. 6–7. **Results.** mean sentence 7.057→7.007; worst-per-scenario
   6.674→6.602; % below 7: 35.5→43.4; % scenarios fully accepted 24.1→18.5; **Mann-Whitney p = 0.25
   (n.s.).** 8. **Interpretation.** **Invalid as a refiner test** — RAW and FIXED are different
   sentences, so generation randomness swamps the effect. Superseded by A2 (within-sentence).
9. **Thesis relevance.** **Appendix / methodology footnote only** (document why it was replaced).

### A5. Task-aware generation quality (Test 1)
1. **Name.** Task-aware generation quality. 2. **Motivation.** Does the pipeline produce *valid*
   task-specific data (topic / sentiment / NER), and does quality scoring alone detect task failure?
3. **Design.** Blind gpt-4o-mini judge (not shown the target): sentiment = re-classification, topic =
   relevance, NER = constraint-aware entity extraction + deterministic check. CS-validity and
   CS-ratio deterministic. 40 sentences/task (35 NER). Pre-refinement sample. 4. **Datasets.** Fresh
   validation set. 5. **Models.** gpt-4o-mini generator + blind judge.
6. **Metrics.** Task-correct %, CS-valid %, CS-ratio MAE vs 70%, fluency, naturalness.
7. **Results.**
   | Task | n | Task-correct | CS-valid | CS-ratio MAE | Fluency | Naturalness |
   |---|--:|--:|--:|--:|--:|--:|
   | topic | 40 | **100.0%** | 100.0% | 23.3 | 8.35 | 8.07 |
   | sentiment | 40 | **72.5%** (→70.0 final) | 87.5% | 22.1 | 8.05 | 8.05 |
   | NER | 35 | **40.0%** (English-only, final) | 97.1% | 13.8 | 8.40 | 8.54 |
   - **NER judge evolution (report carefully):** 45.7% (original fragile parser) → 62.9%
     (constraint-aware but *lenient*, counted Arabic-script entities) → **40.0%** (final,
     pipeline-consistent English-only policy; failures surface as **missing_PER 19/21**).
8. **Interpretation.** Strong surface quality (CS-valid 87–100%, fluency/naturalness ~8) but weaker
   constraint satisfaction: topic strong, **sentiment moderate (neutral-class drag)**, **NER weak
   (under-produces English-script PERSON entities)**, and realized **CS-ratio is 14–23 pts off the
   70% target**. Crucially **fluency/naturalness stay ~8 even where the task fails** → quality scoring
   alone does NOT detect task failure → **motivates the TaskValidator and deterministic CS-counter**.
9. **Thesis relevance.** **§5.2 (generated-data quality) — establishes both the quality profile and
   the motivation for A6/A7.**

### A6. TaskValidator necessity & effectiveness (Test 2)
1. **Name.** TaskValidatorAgent necessity. 2. **Motivation.** Does a task-validation pass reduce
   task-wrong accepts beyond quality filtering? 3. **Design.** Replay two acceptance policies over the
   Test 1 results (no regeneration) with the **real** (fallible) TaskValidator run on the sentences:
   Policy A = quality-only (≥7.0); Policy B = quality AND validator. Reference = Test 1 blind judge.
4. **Datasets.** Test 1 set (English-only NER reference). 5. **Models.** Real TaskValidatorAgent
   (gpt-4o-mini). 6. **Metrics.** Accepted, precision, task-wrong accepted, false-accept, false-reject.
7. **Results.**
   | | A: quality-only | B: quality + validator |
   |---|--:|--:|
   | accepted | 86 | 62 |
   | precision (task-correct among accepted) | 70.9% | **85.5%** |
   | task-WRONG accepted | 25 | **9** |
   | false-accept (of all wrong) | 78.1% | 28.1% |
   | false-reject (of all correct) | — | 36.1% |
   - Validator standalone (vs reference): **precision 83.0%, recall 88.0%, agreement 78.3%.**
   - **Per-task:** topic 0→0 (validator only over-rejects, FN 17.5%); **sentiment 5→5 (NO effect** —
     neutral errors evade both, false-accept 45.5%→45.5%); **NER 17→4 (false-accept 89.5%→21.1% —
     the validator earns its keep here).**
8. **Interpretation.** **Supported for NER, null for sentiment, over-rejects topic.** Honest claim:
   the validator is worth it *specifically for entity-constrained tasks*, not as a blanket gate.
9. **Thesis relevance.** **§5.2 (framework component evaluation).**

### A7. CS-ratio measurement validation (Test 4, PARTIAL)
1. **Name.** Deterministic vs LLM-only CS-ratio measurement. 2. **Motivation.** Is the deterministic
   CS counter more reliable than the baseline LLM-only counting? 3. **Design.** Fixed 30-sentence set
   (20 real + 10 controlled edge cases). Three methods: deterministic `compute_true_cs_stats`;
   LLM-only counting (gpt-4o-mini, temp 0.7, **repeated 3× to measure instability**); human (blank →
   PENDING). **No pipeline run — measurement only.** 4. **Datasets.** 30-sentence CS-ratio set.
5. **Models.** gpt-4o-mini LLM counter. 6. **Metrics.** LLM self-disagreement, per-sentence LLM std,
   deterministic variance, det-vs-LLM binary CS mismatch, mean ratio gap.
7. **Results.** **LLM-only disagrees with itself on 12/30 (40%)** across 3 runs of the *same*
   sentence; mean per-sentence LLM std 0.60 tokens / **2.32%**; **deterministic variance = 0**
   (exact, free); det-vs-LLM `is_code_switched` mismatch **0/30**; mean det-vs-LLM Arabic-% abs diff
   **5.04%**; monolingual edge cases 2/2 correct both.
8. **Interpretation.** **Reproducibility shown** (LLM non-reproducible, deterministic exact); the two
   agree on the binary CS decision and roughly on ratio. **Accuracy vs human is PENDING** the manual
   token counts.
9. **Thesis relevance.** **§5.2 (framework component) — reproducibility result; flag accuracy as
   pending/future work.**

### A8. NER PERSON-prompt repair (diagnostic → controlled A/B → promotion)
1. **Name.** NER missing-PER repair. 2. **Motivation.** Test 1 flagged NER weakest (40%); error
   analysis localized it to **missing_PER**; a constraint-difficulty run showed **ORG easy (90%) but
   PERSON is the bottleneck** (and difficulty non-monotonic). 3. **Design.** Controlled before/after
   pilot (50/arm, Wilson CIs, core prompt untouched — variant in harness): add explicit
   English-script PERSON requirement + self-check with Arabic-friendly Latin names.
4. **Datasets.** NER pilot (`ner_per_prompt_repair/`). 5. **Models.** gpt-4o-mini.
6. **Metrics.** Task-correct % (CI), missing_PER, CS-ratio MAE, fluency/naturalness.
7. **Results.** current prompt **22.5% (CI 12.3–37.5), missing_PER 70%**; **PER-focused prompt 56.8%
   (CI 42.2–70.3), missing_PER 25%** → **+34 pts, CIs do not overlap, no CS-ratio/naturalness
   regression.** Promoted into core `DATA_GENERATION_NER_PROMPT`.
8. **Interpretation.** **Real improvement** (delta, not noise). Absolute NER % noisy run-to-run
   (~22–60% same config), so trust the within-pilot delta.
9. **Thesis relevance.** **§5.2 (NER generation) or Methodology** (prompt engineering that was
   validated then frozen).

### A9. NER generic config-driven entity guidance
1. **Name.** Generic NER entity-guidance builder. 2. **Motivation.** Generalize the PER fix to all
   required entity types without hardcoding prompt examples. 3. **Design.** `build_ner_entity_guidance()`
   reads `must_include_types` + per-type metadata (`DEFAULT_ENTITY_GUIDANCE`, config-overridable),
   injects a dynamic block for *only* the required types via `{ner_entity_guidance}`; zero hardcoded
   tag examples. 4–5. NER A/B pilots, gpt-4o-mini. 6. **Metrics.** Per-type task-correct %,
   missing-type drop. 7. **Results.** **LOC 44→63%, PRODUCT 45→85%, EVENT 39→84% (missing EVENT
   61%→5%), EVENT_LOC 25→83%; PER_EVENT remains hard 11→22%** (two Arabic-natural types compete for
   2–3 slots); controlled A/B confirmed **no PER regression** from generalizing. **NER FROZEN**
   (commit e152b4d). 8. **Interpretation.** Scalable, task-aware, English-only-consistent; the one
   remaining weak spot (PER_EVENT) left unoptimized by design. 9. **Thesis relevance.** **§5.2 (NER)
   or Methodology.**

### A10. Pipeline robustness — NER infinite-loop fix
1. **Name.** NER refiner infinite-loop bug fix. 2. **Motivation.** NER scenarios never terminated in
   refiner-ON mode. 3. **Root cause.** `refine_count` incremented only on *accepted* fixes; NER fixes
   correctly rejected by the guardrail (a "better" sentence breaks required entities) → counter stayed
   0 → infinite re-route. 4. **Fix.** Count every refinement *attempt* (`node_engine.py:1106`); after
   budget (MAX_SENTENCE_REFINES=1) the sentence is accepted as `budget_exhausted`. 5. **Verification.**
   1 NER scenario terminated cleanly; all 6 NER completed; guardrail behaviour was correct, only the
   counter was buggy. 6. **Thesis relevance.** **Methodology / Appendix** (implementation robustness).

═══════════════════════════════════════════════════════════════════════

## CLUSTER B — CS-Validity Diagnosis and the Config-Only Fix
*(Source: `generated_sentiment_data/CS_VALIDITY_DIAGNOSIS.md`, `pilot_v2_csfix/PILOT_V2_REPORT.md`.)*

### B1. CS-validity failure diagnosis
1. **Name.** CS-validity failure diagnosis (Exp C sentiment generation). 2. **Motivation.** The first
   240-sample generation transferred poorly (0.59); diagnose whether the generated data is even
   code-switched. 3. **Design.** Read-only analysis of 690 non-empty instances (pilot_v1 + daily runs);
   deterministic CS check; break failures down by label / topic / cs_type / intensity. 4. **Datasets.**
   690 generated sentiment instances. 5. **Models.** gpt-4o-mini generations; deterministic analyzer.
6. **Metrics.** CS-valid %, failure-mode breakdown, mean Arabic share, per-factor fail rates.
7. **Results.** **CS-valid 209/690 (30%); CS-FAIL 481 (70%), of which fully-Arabic 479, fully-English
   2.** Mean Arabic share **86.5% over all**, **56.3% over valid CS** (12/209 valid are fragile = 1
   English token). **By label:** positive 77% fail, negative 69%, **neutral 62% (best)**. **By
   cs_type:** Intersentential 74% fail vs Intrasentential 65%. **By intensity:** low 74, med 71, high
   64. **By topic:** health 80% … sports 56%. 8. **Interpretation.** Root cause = **the 70% Arabic
   target overshoots to 86.5% mean Arabic → short single sentences go fully monolingual**;
   Intersentential switching makes a single-sentence instance monolingual by design (mismatched with a
   per-sentence CS filter). **Label myth busted: neutral is NOT the culprit (best), positive is worst.**
   Recommended **config-only fix** (no prompt/filter change): add cs_ratio 50/60%, drop Intersentential.
   Filter deliberately **not loosened** (it correctly rejects monolingual). 9. **Thesis relevance.**
   **§5.2.1 (generated dataset analysis) — the key generation-quality diagnostic and the justification
   for the config evolution.**

### B2. Pilot v2 — CS-validity fix validation
1. **Name.** Pilot v2 (CS-fix). 2. **Motivation.** Validate the config-only fix. 3. **Design.**
   `cs_ratio: [50,60,70]`, `cs_type: [Intrasentential]` — no prompt/NER/pipeline change. Isolated,
   **not merged**. 4. **Datasets.** 40 scenarios requested, **6 completed** (34 failed on 429 quota).
5. **Models.** gpt-4o-mini. 6. **Metrics.** CS-valid rate; rate by cs_ratio. 7. **Results.**
   **CS-valid 30% → 43% (12/28)**; by cs_ratio target **60% → 7/10 (70%)**, **70% → 5/18 (28%)**;
   fully-Arabic share of failures 100%. Kept 10 (validator+CS-valid+quality+dedup). 8.
   **Interpretation.** Config-only fix **improved yield** (30→43%). Tiny pilot (6 scenarios); in this
   pilot 60% beat 70%. 9. **Thesis relevance.** **§5.2.1 (dataset construction) — validates the fix.**

### B3. Scale-up empirical CS-validity (the correction)
1. **Name.** CS-validity at scale (50% vs 60%). 2. **Motivation.** Does the tiny-pilot "60% best"
   hold at accumulation scale? 3. **Design.** Observed CS-valid yield during the v3 accumulation.
4–7. **Results.** At scale **50% ≈ 49% CS-valid, 60% ≈ 40%** — the **opposite** of the v2 pilot
   (where 60% won). 8. **Interpretation.** Small-pilot ranking was unreliable; 50% became the primary
   target. 9. **Thesis relevance.** **§5.2.1 — a methodology caveat (small-pilot fragility).**

═══════════════════════════════════════════════════════════════════════

## CLUSTER C — Corpus Construction & Scaling
*(Sources: `pilot_v1/GENERATION_REPORT.md`, `merged/DATASET_CARD_EXP_C{,_480,_960}.md`, `MERGE_REPORT.md`.)*

### C1. Pilot generation (pilot_v1)
1. **Name.** Exp C sentiment generation pilot. 2. **Motivation.** First generation of trainable
   sentiment data with System B. 3. **Design.** `config_sentiment_expC.yaml` (324 sentiment scenarios,
   cs_ratio 70%, Intra+Inter). 4. **Datasets/funnel.** 324 scenarios, **~130 succeeded** (quota),
   **629 raw → 612 TaskValidator → 172 CS-valid → 141 quality≥7 → 141 dedup → balanced 114 (38/label)**.
   Pipeline is **request-heavy** (~50–70 API calls/scenario ⇒ full run needs ~20k > daily 10k cap).
5. **Models.** gpt-4o-mini. 6–7. quality 7.0–8.4. 8. **Interpretation.** Yield is the bottleneck (the
   CS-validity failure, Cluster B). 9. **Thesis relevance.** **§5.2.1 (construction) — first artifact.**

### C2. Balanced datasets 240 → 480 → 960 (accumulation)
1. **Name.** C-series dataset construction (C1=240, C2=480, C3=960). 2. **Motivation.** Scale the
   generated corpus; test whether more data → better transfer. 3. **Design.** Resume-safe,
   append-only accumulation (`scenario_id` manifest); balanced by down-sampling to smallest label;
   5-filter funnel (non-empty → TaskValidator → deterministic CS-valid → quality≥7.0 → dedup).
   **Config evolution (config-only):** v3 (cs_ratio 50/60, Intra, age 18–25, Present) → v4 (add age
   26–40; scenario space 324→648) → v5 (add Past tense; negative top-up). 4. **Datasets.**
   | set | per label | total | pre-balance pool | cs_ratio mix (50/60/70) | quality | source configs |
   |---|--:|--:|---|---|---|---|
   | **C1 (240)** | 80 | 240 | pos 113 / neg 96 / neu 80 | 69 / 72 / 99 | 7.0–8.4 | pilot_v1(94)+run0613(141)+run0606(5) |
   | **C2 (480)** | 160 | 480 | pos 199 / neg 162 / neu 165 | 206 / 170 / 104 | 7.0–8.95 | v3 windows + pilot_v1 legacy |
   | **C3 (960)** | 320 | 960 | — | 477 / 381 / 102 | 7.0–9.0 | v3+v4+v5; age 18-25:556/26-40:404 |
   Every example CS-valid, validator-passed, quality≥7, 0 dups (verified per card).
5. **Models.** gpt-4o-mini generator; deterministic filters. 6–7. See table. 8. **Interpretation.**
   Heterogeneous configs (70% Arabic is a shrinking legacy minority: 99/240 → 104/480 → 102/960);
   acceptable because the sentiment label is cs_ratio-independent. **Negative had the lowest yield**
   (drove the v5 top-up); **neutral is the binding/hardest class**. 9. **Thesis relevance.** **§5.2.1
   (generated dataset analysis) — the dataset the thesis contributes; report the funnel + statistics.**

### C3. Generation statistics of the final corpus (GEN-960 profile)
1. **Name.** GEN-960 distributional profile. 2. **Motivation.** Characterize the delivered corpus.
3–6. **Metrics.** AR/EN share, CMI, length, quality, balance. 7. **Results (GEN-960).** n=960;
   **AR 52.7% / EN 47.3%; CMI mean 40.9 / median 42.1** (hist: 0–20:17, 20–40:297, 40–60:646);
   **length mean 14.1 / median 14 / p90 18; quality 7.0–9.0; balanced 320/320/320.** (Contrast EESA:
   AR ~73%, CMI ~24, per the augmentation diagnosis.) 8. **Interpretation.** Generated corpus is
   **more balanced AR–EN and higher-CMI than EESA** — the domain gap that governs augmentation
   (Cluster E). 9. **Thesis relevance.** **§5.2.1 (dataset analysis) — the intrinsic characterization.**

═══════════════════════════════════════════════════════════════════════

## CLUSTER D — Generator-Config Sensitivity
*(Source: `gen_sensitivity/PILOT_REPORT.md`, `EXPERIMENT_V1_LOWERCS_SENSITIVITY.md`.)*

### D1. V1_lowerCS variant (Arabic↑ / CMI↓)
1. **Name.** V1_lowerCS generation variant. 2. **Motivation.** Does making the generated data
   *closer to EESA* (more Arabic-dominant, lower CMI) improve transfer/augmentation? 3. **Design.**
   cs_ratio 70/80 → Arabic-dominant. 4. **Datasets.** Pilot: 90 scenarios, 429 raw, 88 valid, balanced
   56; scaled to **V1-480** for downstream. 5. **Models.** gpt-4o-mini. 6. **Metrics.** CS-valid
   yield, AR:EN, CMI, length. 7. **Results (pilot).** **CS-valid 29.4%; AR:EN 62.3:37.7 (GEN-960
   52.7:47.3); CMI mean 32.9 (GEN-960 40.9); length 14.4/13/18.** vs GEN-960: AR +9.6pp, CMI −8.0.
   *(Downstream results in Cluster E: E-series.)* 8. **Interpretation.** Achieves the intended shift
   toward EESA (AR↑, CMI↓) at the same length; yield slightly lower. 9. **Thesis relevance.** **§5.2.1
   (dataset variants) + §5.2.3 (its downstream sensitivity, Cluster E).**

### D2. V2a_register variant (register-focused) — PILOTED, SHELVED
1. **Name.** V2a_register generation variant. 2. **Motivation.** Improve *register/authenticity*
   (a different lever than Arabic%/CMI) and CS-valid yield. 3. **Design.** cs_ratio 50/60,
   register-oriented. 4. **Datasets.** Pilot: 70 scenarios, **37 completed** (33 failed), 184 raw, 76
   valid, balanced 49. 5. **Models.** gpt-4o-mini. 6–7. **Results.** **CS-valid 52.2% (best yield of
   any variant); AR:EN 51.1:48.9; CMI mean 43.1; length 12.5/12/16.** 8. **Interpretation.** Highest
   CS-valid yield, but **SHELVED — never scaled or tested downstream** (per project memory). 9.
   **Thesis relevance.** **§5.2.1 (mention as an explored-but-not-pursued variant) or "abandoned
   experiments" — do NOT present as a result; it has no downstream number.**

═══════════════════════════════════════════════════════════════════════

## CLUSTER E — Generated-Corpus Utilization (Training)
*(Sources: C-series reports + seed-stability; `EXPERIMENT_E_AUGMENTATION_*`;
`EXPERIMENT_TWOSTAGE_GEN_AUGMENTATION.md`; `EXPERIMENT_V1_LOWERCS_SENSITIVITY.md`;
`EXPERIMENT_G_C3_RESULTS.md` / `EXPERIMENT_CONSOLIDATED_FINDINGS.md`. All XLM-R, EESA test 818.)*

### E1. Standalone transfer — generated-only training (C1/C2/C3 + seed stability)
1. **Name.** Generated-only transfer + size scaling. 2. **Motivation.** Does generated data alone
   train a classifier that transfers to real EESA, and does more data help? 3. **Design.** Fine-tune
   XLM-R on generated data only; dev/test = real EESA. Adafactor, eff-batch 16, fp16, 4 epochs,
   max_len 256. 3-seed stability for C2/C3. 4. **Datasets.** C1 240, C2 480, C3 960; EESA test 818.
5. **Models.** xlm-roberta-base. 6. **Metrics.** Accuracy, macro F1 (mean±std). 7. **Results.**
   | set | size | accuracy | macro F1 | note |
   |---|--:|---|---|---|
   | C1 | 240 | 0.5905 | 0.5619 | single seed (+15pp over 0.444 majority) |
   | C2 | 480 | **0.6500 ± 0.016** | 0.6345 ± 0.017 | 3-seed |
   | C3 | 960 | **0.6695 ± 0.024** | 0.6592 ± 0.021 | 3-seed; more data → better |
   | C3 seed-456 | 960 | 0.6956 | 0.6830 | best-dev checkpoint |
   (Reference: EESA-trained XLM-R ≈ 0.824–0.853.) 8. **Interpretation.** **Real transfer signal, scales
   240→480→960**; the single-run "480>960" was a seed artifact (retracted by the 3-seed check).
   Confusion sensible (not collapsed). 9. **Thesis relevance.** **§5.2.2 (standalone training
   performance) — the core utilization result.**

### E2. C-V1 standalone transfer (the sensitivity answer)
1. **Name.** V1_lowerCS standalone transfer. 2. **Motivation.** Does EESA-proximity (Arabic↑/CMI↓)
   improve standalone transfer? 3. **Design.** V1-480, 3 seeds, exact C2 recipe. 4–6. XLM-R, EESA
   test. 7. **Results.** **C-V1 (V1-480) = 0.6304 ± 0.017** vs **C2 (GEN-480) 0.6500 ± 0.016 (−0.020,
   ~1.2σ)** and C3-960 (−0.039). V1 models **over-predict neutral, under-predict negative**. 8.
   **Interpretation.** **Proximity HURT standalone transfer** — the lower code-switching intensity
   (CMI 33 vs 41) removes the very signal transfer needs. 9. **Thesis relevance.** **§5.2.2 /
   §5.2.4 (discussion) — the sensitivity finding.**

### E3. Mixing augmentation — real EESA + generated together
1. **Name.** Mixing augmentation (E-series). 2. **Motivation.** Does adding generated data to real
   EESA training help? 3. **Design.** Fresh XLM-R, Adafactor, matched compute; E0 = EESA-only control;
   E3 = EESA+GEN-960; low-resource LR = 10/25/50% EESA ± GEN-960; ratio sweep (gen 0.25/0.5/1.0).
4. **Datasets.** EESA train + GEN-960; EESA test. 5. **Models.** XLM-R. 6. **Metrics.** Accuracy.
7. **Results.**
   | EESA% | real-only | + GEN-960 (mix) | Δ |
   |---|---|---|---|
   | 10% | 0.7751 | 0.7408 | **−0.034** |
   | 25% | 0.7873 | 0.7689 | **−0.018** |
   | 50% | 0.8166 | 0.8142 | −0.002 |
   | 100% | **0.8533 (E0)** | 0.8411 (E3) | **−0.012** |
   Ratio sweep: all within ±0.026 single-seed noise; best +0.006 (noise). 8. **Interpretation.**
   **Mixing is neutral-to-harmful at every ratio** — harm monotone in generated fraction; the
   optimizer, not the data, explained E3's apparent gain over the old AdamW baseline. 9. **Thesis
   relevance.** **§5.2.3 (generated data augmentation) — the negative result.**

### E4. E-V1 augmentation (proximity as a less-harmful augmenter)
1. **Name.** V1_lowerCS augmentation. 2–3. EESA{10,25,50}% + V1-gen (capped ≤50%), seed 42.
4–6. XLM-R, EESA test. 7. **Results.**
   | EESA% | real-only | +V1 | +GEN-960 | ΔV1−real | ΔV1−960 |
   |---|---|---|---|---|---|
   | 10% | 0.7751 | 0.7531 | 0.7408 | **−0.022** | **+0.012** |
   | 25% | 0.7873 | 0.7885 | 0.7689 | +0.001 | +0.020 |
   | 50% | 0.8166 | 0.8178 | 0.8142 | +0.001 | +0.004 |
8. **Interpretation.** **vs real-only: neutral-to-harmful** (hurts at 10%, neutral at 25/50%);
   **vs GEN-960: consistently less harmful** (+0.01–0.02). Proximity helps *relatively* (a
   less-distorting perturbation) but never beats real-only. **The divergence** (proximity HURTS
   standalone but HELPS-vs-960 augmentation) reconciles the sensitivity story cleanly. Single-seed.
9. **Thesis relevance.** **§5.2.3 + §5.2.4 (discussion).**

### E5. Two-stage augmentation — generated PRETRAIN → real FINE-TUNE (THE WIN)
1. **Name.** Two-stage (DAPT) augmentation. 2. **Motivation.** Mixing lets gen *distort* the decision
   boundary; use gen only to *warm up the representation* and let real data have the last word.
3. **Design.** Pretrain XLM-R on GEN-960 (the `expC3` checkpoint) → fine-tune on real EESA%. Adafactor,
   max_steps 400, load_best, seed 42, primary_only. 4. **Datasets.** GEN-960 (pretrain) + EESA
   {10,25,50,100}% (fine-tune); EESA test. 5. **Models.** XLM-R. 6. **Metrics.** Accuracy + macro F1.
7. **Results.**
   | EESA% real | real-only | mixing (+960) | **two-stage** | 2-stage vs real-only | vs mixing |
   |---|---|---|---|---|---|
   | 10% | 0.7751 | 0.7408 | 0.7604 | **−0.0147 (HURT)** | +0.0196 |
   | **25%** | 0.7873 | 0.7689 | **0.8093** | **+0.0220 ✅** | +0.0403 |
   | **50%** | 0.8166 | 0.8142 | **0.8313** | **+0.0147 ✅** | +0.0171 |
   | **100%** | 0.8533 (E0) | 0.8411 (E3) | **0.8655** | **+0.0122 ✅** | +0.0244 |
   Macro F1 confirms: 10% −0.012, **25% +0.021, 50% +0.020, 100% +0.017**. 8. **Interpretation.**
   **First augmentation win in the whole campaign.** Same gen data that *hurt* when mixed *helps* when
   used as pretraining — the *method* matters, not the data. **Persists at full data** (+0.012/+0.017),
   making **two-stage-full 0.8655 a new best full-EESA primary (> E0 0.8533)**. Only 10% hurts (too
   little real data to steer the gen-heavy init). **Single seed — needs 3-seed replication before a
   strong claim.** 9. **Thesis relevance.** **§5.2.3 (generated data augmentation) — the positive
   result and the headline of the data contribution; the recipe: pretrain-on-gen → finetune-on-real.**

### E6. Augmentation failure diagnosis (domain mismatch)
1. **Name.** Why mixing fails. 2. **Motivation.** Explain E3/LR/ratio negatives. 3. **Design.**
   Distributional profiling of EESA vs GEN-960. 4–7. **Results.** EESA: AR ~73%, CMI ~24, dialectal
   noisy social-media, ~49% train vocab coverage of EESA test. GEN-960: AR ~53%, CMI ~41,
   cleaner/MSA, **~10% (0.098) vocab overlap with EESA test.** 8. **Interpretation.** Mixing adds
   **off-domain signal**; two-stage retains only representation-level signal because the real
   fine-tune overwrites the off-distribution parts. 9. **Thesis relevance.** **§5.2.4 (discussion) —
   the mechanism uniting E3/E4/E5.**

### E7. Generated data as the weak primary the agents rescue (downstream payoff)
1. **Name.** C3 generated primary + multi-agent layer. 2. **Motivation.** Generated data's *second*
   value: it builds the weak primary where the multi-agent layer delivers its largest gain.
3. **Design.** XLM-R trained on C3-960 (weak) as the primary; best agent config (G/G2 @ gpt-4.1-mini).
4. **Datasets.** C3-960 primary; EESA test. 5. **Models.** XLM-R primary + GPT-4.1-mini agents.
6. **Metrics.** Accuracy, net escalated, significance. 7. **Results.** primary 0.6956 → **0.7665/0.7677
   with agents (+0.071, +58/+59 net, p ≪ 0.001)** — the largest agentic gain of any regime. 8.
   **Interpretation.** Generated data, *as a substitute for scarce real data*, creates the regime
   where the agentic contribution matters most. 9. **Thesis relevance.** **Bridges §5.2 (data) and
   §5.3 (multi-agent) — cite in both; it is the one place the two contributions connect.**

═══════════════════════════════════════════════════════════════════════

## CLUSTER F — Human Validation (BUILT, NOT RUN)
1. **Name.** Human confirmation of masking + CS-ratio accuracy. 2. **Motivation.** Remove circularity
   (AI grading its own output). 3. **Design.** (a) BLIND shuffled sheet (seed 42): masking sheet 50
   sentences (13 MASKED + 37 neighbours from 11 scenarios; larger pool 30/103 available); consolidated
   86-row BLIND sheet + hidden key; analyzer computes per-dimension MASKED-vs-neighbour means +
   Mann-Whitney, human-vs-machine Spearman, per-scenario sign test, monolingual leak, Cohen's κ. (b)
   CS-ratio: 30-sentence manual Arabic/English/other token counts → MAE/detection/boundary for both
   methods. 4–7. **Status: waiting on annotators — NOT run.** Analyzers verified on dummy input.
8. **Interpretation.** The "CONFIRM" step of the framework; without it, task-correctness, CS-ratio
   accuracy, and masking are LLM-judged only. 9. **Thesis relevance.** **§5.2.4 / Threats to validity /
   Future work — must be stated as a limitation (the framework's human validation is pending).**

═══════════════════════════════════════════════════════════════════════

# PART 2 — TIMELINE (chronological, with causal links)

```
2026-05-31  A: MASKING DEFENSE (framework validation, System B vs C)
            • per-sentence catches 41.5%/35.6% what aggregate hides ── the contribution
            • refiner improves caught sentence +0.60 (within-sentence)
            • refiner head-to-head = TIE (p=0.53) ── scopes claim to "detection, not better refiner"
            • Test1 quality: topic100 / sent72.5 / NER40 ── NER weak, sentiment neutral-drag
            • Test2 validator: helps NER (17→4), null sentiment ── validator is entity-task-specific
            • Test4 CS-ratio: LLM 40% self-disagree vs deterministic 0 ── reproducibility
            • NER PER repair +34, generic guidance (EVENT 39→84…) ── promoted, then FROZEN
                    │  (framework proven on a small mixed sample; sentiment yield unknown at scale)
                    ▼
2026-06-05  FRAMEWORK FREEZE (no more prompt/core/NER changes)
                    │  (turn the frozen pipeline into a sentiment DATA GENERATOR for Exp C)
                    ▼
2026-06-06→13  C1: first sentiment generation (pilot_v1) → 240 balanced (80/label)
                    │  train XLM-R → 0.59 on EESA ("looks bad")
                    ▼
2026-06-13  B1: CS-VALIDITY DIAGNOSIS → only 30% CS-valid; 70% fully-Arabic (70% target overshoots)
                    │  (root cause = config, not filter/prompt)
                    ▼
            B2: pilot v2 CS-fix (cs_ratio 50/60, Intra-only) → 30%→43% CS-valid  ✅
                    │  (adopt as config v3; accumulate)
                    ▼
2026-06-14→21  C2/C3: scale 240 → 480 → 960 (config v3→v4→v5, resume-safe accumulation)
                    │  B3: at scale 50% (49%) > 60% (40%) — corrects the tiny-pilot ranking
                    ▼
2026-06-21→ E1: standalone transfer scales 240(0.59)→480(0.65)→960(0.67, 3-seed)  ✅
                    │  (does gen help a REAL classifier? test augmentation)
                    ▼
2026-06-25  E3/E6: mixing augmentation FAILS at every ratio (−0.034…−0.012) → domain-mismatch diagnosis
                    │  (can a more-EESA-like generator help?)
                    ▼
2026-07-02  D1/E2/E4: V1_lowerCS variant → standalone WORSE (0.630<0.650); augmentation less-harmful
                    │  than 960 but still ≤ real-only.  (config tuning of the generator did NOT help)
                    ▼
2026-07-05  E5: TWO-STAGE (gen pretrain → real finetune) → +2.2/+1.5/+1.2 at 25/50/100%  ✅ THE WIN
            E7: C3 weak primary + agents → +0.071 (p≪0.001) ── generated data's second value
```

**Why each step led to the next:** framework validation (A) proved the *mechanism* on a mixed sample
but exposed that **sentiment yield was untested at scale** → generating at scale (C1) surfaced the
**CS-validity yield problem** (B1) → the **config-only fix** (B2/B3) → **scaling** (C2/C3) → **standalone
transfer works and scales** (E1) → the obvious question "does it *augment* a real model?" → **mixing
fails** (E3) → **why?** domain mismatch (E6) → "make the generator more EESA-like?" → **V1 doesn't help
standalone, only relatively** (D1/E2/E4) → "so is augmentation hopeless?" → **change the *method*, not
the data: two-stage works** (E5). In parallel, the generated corpus fed the **weak-primary regime** where
the multi-agent contribution pays off (E7).

═══════════════════════════════════════════════════════════════════════

# PART 3 — MISSING / ABANDONED / PARTIAL / SUPERSEDED

| Item | Status | Why |
|---|---|---|
| **Human annotation (masking + CS-ratio accuracy)** | **BUILT, NOT RUN** | Waiting on annotators; the framework's "CONFIRM" step. Task-correctness, masking, and CS-ratio accuracy remain LLM-judged only. **Biggest gap.** |
| **V2a_register variant** | **PILOTED, SHELVED** | Best CS-valid yield (52.2%) but never scaled or tested downstream — no transfer/augmentation number exists. Present as "explored, not pursued," not as a result. |
| **Cross-run before/after (Step 4)** | **SUPERSEDED (confounded)** | Independent generations confound the refiner test; replaced by the within-sentence test (A2). Appendix-only. |
| **NER judge 45.7% / 62.9%** | **SUPERSEDED** | Fragile parser (45.7) and lenient Arabic-counting judge (62.9); final English-only figure is 40.0%. Report 40.0% only; the others are methodology footnotes. |
| **Task-aware refiner (REFINER_TASK_*)** | **NOT ISOLATED** | The one refiner path never tested alone (needs validator ON + task-failing sentences; "Test 6b-task"). |
| **Full System A (gpt-4o) vs C model comparison** | **NOT RUN (by design)** | Masking study held the model constant to isolate architecture; the model effect (A vs C) is a separate, deliberately-unrun question. |
| **PER_EVENT NER (competing hard types)** | **KNOWN LIMITATION** | 11→22% only; fixing EVENT cost PER under 2–3 slots. Left unoptimized. |
| **Single-config clean corpus** | **NOT PRODUCED** | 240/480/960 are heterogeneous (legacy 70% rows). Acceptable for sentiment; note it. |
| **3-seed two-stage; 3-seed E-V1** | **PARTIAL (single-seed)** | E5 and E4 are single-seed; the two-stage win needs seed replication before a strong claim. |
| **"480 > 960" standalone reading** | **RETRACTED** | Single-seed artifact; 3-seed check shows 960 ≥ 480. |
| **CS-ratio hits 70% target** | **FAILED** | Realized ratio 14–23 pts off target (objective). Motivates the deterministic counter but is itself a negative. |
| **Two-stage + agent layer stacked** | **NOT RUN** | Recommended follow-up (does the +2.2 two-stage gain stack on the agent gain?). |

═══════════════════════════════════════════════════════════════════════

# PART 4 — RECOMMENDED STRUCTURE FOR SECTION 5.2
*(Derived only from experiments that were actually performed.)*

**5.2 Evaluation of the Modified Task-Aware SwitchLingua Framework**

**5.2.1 Generation-Framework Evaluation** *(Cluster A + F — the contribution itself)*
- Per-sentence vs aggregate scoring: the masking result (35.6% at bar 7.0, full sweep) — A1
- Refiner effectiveness (within-sentence +0.60) and the honest scope (head-to-head tie) — A2, A3
- Task-aware generation quality (topic/sentiment/NER; quality ≠ task-correctness) — A5
- TaskValidator (NER-concentrated benefit) and deterministic CS-ratio (reproducibility) — A6, A7
- NER generation improvements (PER repair, generic guidance) — A8, A9
- *(State the human-validation gap up front — Cluster F.)*

**5.2.2 Generated Dataset Analysis** *(Clusters B + C + D — the delivered corpus)*
- CS-validity diagnosis and the config-only fix (30%→43%; root cause) — B1, B2, B3
- Corpus construction & scaling (240/480/960; funnel; config v3→v5) — C1, C2
- Distributional profile (AR/EN, CMI, length, quality) and the EESA gap — C3
- Generator-config sensitivity (V1_lowerCS; V2a_register mentioned as explored) — D1, D2

**5.2.3 Generated Data Utilization** *(Cluster E — does it help a classifier?)*
- Standalone training performance (scales 240→480→960; C-V1 sensitivity) — E1, E2
- Augmentation: mixing fails at every ratio; **two-stage pretraining is the win** (+1.2 to +2.2) — E3, E4, E5
- *(The downstream weak-primary payoff, E7, is cited here and picked up in §5.3.)*

**5.2.4 Discussion**
- Why mixing fails but two-stage works (domain mismatch; real fine-tune has the last word) — E6
- The proximity divergence (V1: hurts standalone, less-harmful augmenter) — E2/E4
- Limitations & threats to validity: **human validation pending**; single-model generator; single-seed
  two-stage; heterogeneous configs; CS-ratio off-target; V2a shelved.

**Do NOT create** sections for: "two-stage vs intermediate vs transfer" as separate experiments (only
two-stage exists); a NER *evaluation* results section beyond generation quality (NER was generated and
prompt-repaired, never used to train a downstream NER classifier); human-eval *results* (built, not run).

═══════════════════════════════════════════════════════════════════════

# PART 5 — ONE-LINE FINDINGS (for quick citation)

- **Contribution = detection/routing, not a better refiner** (masking 35.6% caught; head-to-head tie p=0.53).
- **Refiner improves caught sentences** +0.60 (90.8%, p≈0).
- **Quality scoring ≠ task-correctness** (fluency ~8 even when the task fails) → motivates the validator.
- **TaskValidator earns its keep on NER** (task-wrong 17→4), **null on sentiment** (neutral evades both).
- **Deterministic CS counter is reproducible** (0 variance) vs LLM-only (40% self-disagreement); accuracy pending.
- **Generation yield problem = 70% Arabic overshoots to 86.5%** → 70% fully-Arabic; config-only fix 30%→43%.
- **Generated-only transfer is real and scales** 240(0.59) → 480(0.65) → 960(0.67, 3-seed).
- **Making the generator more EESA-like (V1) did NOT help** (standalone worse; augmentation only less-harmful).
- **Mixing augmentation fails at every ratio** (−0.034 … −0.012); **two-stage pretraining is the one win**
  (+2.2/+1.5/+1.2 at 25/50/100% real; new best full-EESA primary 0.8655).
- **Generated data's second value: it builds the weak primary the agents rescue** (+0.071, p≪0.001).
- **Human validation of the framework is built but NOT run** — the standing limitation.

*Sources: `masking_defense/REPORT.md`, `per_sentence/threshold_calibration_report.md`,
`task_aware_eval/*`, `task_validator/*`, `csratio/*`, `CS_VALIDITY_DIAGNOSIS.md`,
`pilot_v1/GENERATION_REPORT.md`, `pilot_v2_csfix/PILOT_V2_REPORT.md`, `gen_sensitivity/PILOT_REPORT.md`,
`merged/DATASET_CARD_EXP_C{,_480,_960}.md`, `EXPERIMENT_V1_LOWERCS_SENSITIVITY.md`,
`EXPERIMENT_E_AUGMENTATION_*`, `EXPERIMENT_TWOSTAGE_GEN_AUGMENTATION.md`,
`EXPERIMENT_CONSOLIDATED_FINDINGS.md`, `EXPERIMENT_MASTER_SUMMARY.md`, `FINAL_STATUS.md`,
`PROJECT_STATUS_A_TO_Z.md`.*
