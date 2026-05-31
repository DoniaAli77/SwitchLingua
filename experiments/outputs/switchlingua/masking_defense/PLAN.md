# Masking Defense — My Home Base

> **Come back here anytime.** This file holds the plan, what's done, and where everything is saved.

---

## The goal (in one sentence)

**Prove that looking at each sentence is better than looking at the average — because the average can hide one bad sentence.**

That hidden bad sentence is called **"masking."**

---

## Why this matters

- **Old system (System C):** looks at the average. A bad sentence can hide behind good ones and escape.
- **My system (System B):** looks at each sentence. It catches the bad one.

---

## The 4 steps

- [ ] **Step 1 — Make more examples (TWO runs: "before" and "after")**
  Run the pipeline to get ~60–100 sentences (we only have 8 now).
  We do it **twice**, because the masking story needs two photos:
    - **Run 1 — fixing OFF** ("before" photo): grade each sentence once, stop.
      Shows the mess (the bad sentences). → used by Step 2.
      Saved to: `masking_defense/step1_raw_data/` (refiner OFF)
    - **Run 2 — fixing ON** ("after" photo): let the system catch + fix bad ones.
      Shows the fix works. → used by Step 4.
      Saved to: `masking_defense/step1_fixed_data/` (refiner ON)
  *Why two runs:* the pipeline only saves the AFTER grades. The only way to see
  the raw "before" mess is a separate run with fixing off.

- [ ] **Step 2 — Count the hidden bad sentences**
  In the new data, count how many bad sentences the average hid.
  *Why:* Real numbers, not the fake hand-typed ones we have now.
  Saved to: `masking_defense/step2_counts/`

- [ ] **Step 3 — Ask real people**
  Show the bad sentences to Arabic-English speakers. Ask: "Is this bad?"
  *Why:* Proves it's not just the computer's opinion.
  Saved to: `masking_defense/step3_human_check/`

- [ ] **Step 4 — Show my system is better (before vs after)**
  Make one simple picture comparing the worst sentence per scenario:
  "before" (fixing off) vs "after" (fixing on). The bad sentences the average
  would let through get caught and fixed → the floor rises.
  *Why:* This is the picture that wins the defense.
  Saved to: `masking_defense/step4_final_picture/`

---

## Rule: do them in order

Steps 2, 3, 4 all need the data from **Step 1**. So Step 1 is first.

---

## Progress log

| Date | What I did | Notes |
|------|-----------|-------|
| 2026-05-30 | Made this plan | Starting fresh. 8 examples so far. |
| 2026-05-30 | Checked the code works | ✅ Both systems load, all 3 tasks build. Safe test saved as `_healthcheck.py`. Note: config makes only 1 CS type by default — use the `--cs_types` setting in Step 1 to get all 3. |
| 2026-05-30 | Confirmed clean test | ✅ Aggregate score = average of per-sentence scores (all 8 match). So we test ONLY average-vs-each. Proof saved as `_check_same_scale.py`. |
| 2026-05-30 | Step 1 Run 1 (before, refiner OFF) | ✅ 54 scenarios, 245 sentences, CS types 18/18/18, fixing=0. Saved to `step1_raw_data/`. |
| 2026-05-30 | Step 1 Run 2 (after, refiner ON, validator OFF) | ⚠️ 48/54 done (217 sentences, all fixed). Last 6 (NER) FAILED — OpenAI quota ran out (429 insufficient_quota). Saved to `step1_fixed_data/`. Need to add OpenAI credit, then re-run the 6 NER in fixed mode to complete the pair. |
| 2026-05-30 | Step 2 — counted REAL masking | ⚠️ KEY FINDING: masking depends on the passing bar (threshold). On this gpt-4o-mini data sentences score ~7, so at the default bar **8.0 there is NO masking (0%)** — everything fails the average too. Masking IS real lower down: **bar 7.0 → 41.5% (22/53), bar 6.5 → 30%**. Avg intra-scenario spread 0.79. Outputs in `step2_counts/`. |
| 2026-05-30 | DECISION: Option A | Chose Option A (report the threshold curve, operating bar ~7) — gpt-4o too expensive. Honesty safeguards: show the FULL curve (incl 0% at 8), justify bar from the data, confirm with humans in Step 3. Exported **30 masked sentences at bar 7** → `step2_counts/masking_cases.csv` (material for Step 3). |
| 2026-05-30 | Step 2 caveat noted | Masked sentences score ~6.7–6.8 vs neighbours ~7.0–7.7 → SMALL gaps (model is uniform). Step 3 human check must confirm these flagged sentences are genuinely worse (answers "is a 0.5 gap real or noise?"). |
| 2026-05-30 | NER-fixed BUG found | The 6 NER scenarios in FIXED mode infinite-loop (not network/quota). Cause: refiner tries to improve an NER sentence → breaks entity rules → guardrail correctly rejects the fix → but the loop counter only advances on ACCEPTED fixes, so it retries forever. Topic/sentiment finish fine (their fixes get accepted). Guardrail behaviour is CORRECT; the loop counter is the bug. |
| 2026-05-30 | NER-fixed BUG FIXED ✅ | One-line fix in `Modified_Version/core/node_engine.py` RunRefinerAgent: count every refine ATTEMPT (not just accepted) so budget is spent and sentence is accepted as 'budget_exhausted'. VERIFIED: 1 NER scenario now finishes (was infinite). Did NOT break topic (topic generated+graded fine in test). NOTE: account hit its DAILY 10,000-request limit (RPD) — drained by the earlier loop — so no more generation today; resets in ~24h. |
| 2026-05-30 | NER top-up partial | Re-ran 6 NER fixed AFTER the bug fix: **2 completed cleanly** (fix confirmed in a real run), 4 hit the daily RPD cap again (10,000/10,000 used). Merged the 2 into `step1_fixed_data/` → now **50 scenarios** (topic 12, sentiment 36, ner 2). **4 NER still pending — finish tomorrow when daily limit resets.** Not blocking: masking uses raw (complete 54); Step 4 uses the 50 pairs. |
| 2026-05-30 | Step 3 sheet BUILT | Blind human-check sheet built in `step3_human_check/`: **103 sentences** (30 MASKED + 73 neighbours) from 22 masking scenarios, shuffled (seed 42). Annotators rate overall_quality_1to5 + is_real_codeswitch_yes_no. `human_check_sheet.csv` (give to annotators), `answer_key.csv` (KEEP HIDDEN), README with instructions. |
| 2026-05-30 | Step 3 analysis READY | `step3_analyze_human_sheet.py` built + tested (on throwaway dummy, since deleted). Stats by hand (no scipy): Mann-Whitney p (MASKED vs neighbour quality), Spearman (machine vs human), per-scenario sign test, monolingual-leak %, and Cohen's kappa + Spearman agreement if ≥2 annotators. Auto-detects filled sheets in `step3_human_check/`. WAITING ON: annotators to fill `human_check_sheet.csv`, save as e.g. `annotator1.csv` in that folder, then run the script. |
| 2026-05-31 | NER-fixed COMPLETE ✅ | Daily limit reset. Regenerated all 6 NER fixed fresh (6 ok, 0 failed) and swapped into `step1_fixed_data/` → now **full 54 scenarios** (topic 12, sentiment 36, ner 6), matching the 54 raw. Step 1 fully complete (raw 54 + fixed 54). |
| 2026-05-31 | Step 3 sheet REDESIGNED | User chose per-dimension Likert. Sheet now rates **fluency / naturalness / cultural each 1-10** + is_real_codeswitch yes/no, on a balanced **50-sentence subset** (11 whole scenarios kept together; 13 MASKED + 37 neighbours). Answer key now also stores the MACHINE's per-dimension scores → analysis validates the AI judge dimension-by-dimension. Analyzer `step3_analyze_human_sheet.py` updated to match. NOTE: 13 MASKED is the pilot subset; for more statistical power, raise SUBSET_TARGET in `step3_build_human_sheet.py` (full set = 30 MASKED / 103 sentences). |
| 2026-05-31 | Step 4 — NEGATIVE result ⚠️ | Before(off) vs After(on): mean sentence 7.057→7.007, mean worst 6.674→6.602, % below bar7 35.5→43.4, fully-accepted 24.1→18.5%. **Mann-Whitney p=0.25 (NOT significant)** — refiner did NOT improve quality (point estimates slightly worse). CAVEAT: raw & fixed are INDEPENDENT generations, so this cross-run test is confounded by generation randomness — NOT a clean refiner test. Chose option (2): clean within-sentence test. |
| 2026-05-31 | CLEAN refiner test — POSITIVE ✅ | `refiner_clean_test.py`: same weak sentence, fresh before vs (refine→fresh after), no cross-run confound. **FULL 30-sentence result: mean before 6.41 → after 7.17, mean delta +0.77, median +0.83, improved 29/30 (96.7%), sign-test p≈0.** The refiner GENUINELY improves weak sentences. The Step 4 cross-run null (p=0.25) was purely the generation-randomness confound. This is the correct Test 6a result → `step4_final_picture/refiner_within_sentence.csv` + `refiner_clean_test.log`. |
| 2026-05-31 | Clean refiner test — FULL (all 87 weak) ✅ | Expanded to ALL weak sentences (score<7): **mean before 6.64 → after 7.24, mean delta +0.60, median +0.65, improved 79/87 (90.8%), sign-test p≈0.** Robust: the refiner lifts ~91% of weak sentences. "My refiner works" is firmly established. |
| 2026-05-31 | Test 6b — TIE (honest null) | YOUR per-sentence feedback vs ORIGINAL aggregate feedback, same sentence/prompt/model, 30 weak sentences: YOURS +0.747, ORIG +0.808, wins YOURS 10 / ORIG 13 / ties 7, sign-p=0.53. **No advantage — once a sentence is handed to the refiner, feedback granularity doesn't change rewrite quality.** REFRAME: the contribution is NOT "a better refiner" — it's the per-sentence SCORING that ROUTES the masked sentence to refinement at all (the aggregate lets it escape). Defend detection+routing (Step 2: 41.5% + clean test +0.60), NOT refiner rewrite quality. `step4_final_picture/refiner_headtohead.csv`. NOTE: task-aware refiner prompts (task_fail path) still UNtested — would need validator ON + task-failing sentences (separate Test 6b-task). |

*(We add a new row each time we do something.)*

---

## What exactly are we testing? (read this if confused)

We test **ONE thing only: average vs each-sentence.**

Same grades, two ways to look:
- Old way → look at the **average** (one number).
- My way → look at **each sentence**.

We do NOT change the sentences, the scoring, or anything else.
Proof: the "average score" in the data is literally the average of the
per-sentence scores (checked all 8 examples — they match exactly).
Re-check anytime with: `_check_same_scale.py`

⚠️ **Trap 1:** Do NOT compare System B's text vs System C's text for masking —
that changes too many things at once. For masking, always use the SAME grades,
looked at two ways.

⚠️ **Trap 2 (about the new parts):**
- The **task-validation agent** does NOT affect the test. The grade =
  fluency + naturalness + cs-ratio + culture only. Task validation is a
  separate yes/no stamp, not part of the grade. ✅ Safe.
- The **smart refiner** DID touch our old 8 examples — it already fixed the
  bad sentences before saving (we saw `[1,1,1,1]` = "fixed once"). That HIDES
  masking. So in **Step 1 we turn the fixing OFF**: just make sentences and
  grade once, no fixing. Then we can see the bad sentences in their raw state.
  Re-check anytime with: `_check_contamination.py`

---

## Important notes (don't forget)

- The old "bad sentence" example uses **fake numbers** typed by hand. We are replacing it with **real numbers** in Step 2.
- Step 3 needs **real people** — only I can find the Arabic-English speakers. Claude builds the sheets; I get the people.
- The winning result is **one picture** (Step 4): my system has fewer bad sentences in the final data.
