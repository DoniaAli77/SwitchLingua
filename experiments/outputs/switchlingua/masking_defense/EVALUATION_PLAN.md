# Evaluation Plan — Defending the Whole Pipeline

> **Simple idea:** You added 4 new things. Each needs 1 test to prove it was worth it.
> The masking work (Test 3) is part of this bigger plan.

---

## The secret recipe (every test is the same shape)

> **1.** One set of sentences  →  **2.** do one thing **two ways**  →  **3.** let **humans** decide who's right.

All tests below are just this recipe applied to a different feature.

---

## Your 4 new features → 4 tests

### Test 1 — Can it do the 3 jobs well? (topic, sentiment, NER)
- **Do:** make sentences for all 3 jobs; ask people if they're good
  (right label, real code-switching, good language, acceptable overall).
- **Proves:** the pipeline is task-aware, not just topic/style.
- **Sample:** ~100 per job for auto-judge; humans check a subset (~30–50 per job).

### Test 2 — Is the task-checker (TaskValidator) worth it?
- **Do:** same sentences, accept them two ways:
  - WITHOUT checker (accept on quality grades only)
  - WITH checker (accept only if quality good AND task passes)
- **Compare:** which way lets fewer WRONG-task sentences through.
- **Proves:** a sentence can be fluent + code-switched but still wrong
  (wrong sentiment / bad NER tags). The checker catches that.
- **Reference:** human task-correctness labels.

### Test 3 — Is grading each sentence better than the average? ← MASKING (our current work)
- **Do:** same per-sentence grades, two decision rules:
  - Scenario-level: average the grades, accept/reject all together
  - Per-sentence: accept/reject each sentence on its own
- **Keep CS ratio method fixed (deterministic) in both** so only granularity differs.
- **Headline metric:** monolingual leakage — fully Arabic/English sentences
  accepted into a code-switching dataset (an undeniable failure).
- **Also:** valid-CS rate among accepted, human agreement, false-positive rate, # of decisions (cost).
- **Real data is the main evidence.** Planted edge cases only as a labeled stress test.
- **Proves:** per-sentence scoring catches bad sentences the average hides.

### Test 4 — Is the calculator better than the AI at counting code-switching?
- **Do:** same sentences, count CS three ways: AI, calculator (deterministic), human (truth).
- **Compare:** token-count error, ratio error (MAE), wrong accept/reject near the target,
  variance over 3 AI runs (stability), monolingual detection accuracy.
- **Proves:** deterministic counting is more accurate and more reliable than LLM-only.
- **Separate from Test 3.**

---

### Test 6 — Refinement effectiveness  [CORE]
Your refiner differs from the original in TWO ways → two sub-tests:
  (1) TARGETED: fixes only the failing sentence (per-sentence), and
  (2) TASK-AWARE: when a sentence fails the JOB, a task-specific fixer that
      knows the task rewrites it (vs the original's generic "rewrite this").
      Code: node_engine.py RunRefinerAgent, lines ~1031-1046.

#### Test 6a — Quality refinement helps (the payoff half of masking)
- **Why:** Test 3 proves we CATCH the bad sentence; 6a proves FIXING it helps.
- **Do:** same scenarios before vs after the refiner:
  acceptance rate, average score, worst-sentence score, improvement on the
  previously-failing sentences, guardrail rollback rate, 3 before→after examples.
- **Stats:** paired Wilcoxon on refined sentences' scores.
- **Validator:** OFF.  **Data already exists:** `step1_raw_data/` vs `step1_fixed_data/`.

#### Test 6b — Task-aware refinement beats generic refinement
- **Why:** defends the task-aware fixer specifically (a separate contribution).
- **Do:** take the SAME task-failing sentences → fix them two ways:
  generic fixer (original REFINER_PROMPT) vs your task-aware fixer
  (REFINER_TASK_* prompts) → measure which actually fixes the JOB.
- **Metric:** task-correctness after refinement (human / validator judged);
  also CS validity preserved, # rollbacks.
- **Validator:** ON (needed to flag task failures).  **Data:** NEEDS a small new run.
- **Note:** our current runs had the validator OFF, so the task-aware path was
  never exercised — that was correct for keeping the masking test clean.

---

## 1 optional extra (only if time)

### Test 5 — Compact ablation (turn features on one at a time)
Raw prompt → +TaskValidator → +per-sentence → +deterministic CS → +full refinement.
Watch task-correctness, CS validity, acceptance rate, average score, refine count climb.
Shows each piece adds value, and ties all contributions together.

---

## 3 must-add rules (keep it research-level)

1. **Real sentences first**, made-up edge cases only as a clearly labeled stress test.
2. **A small stats check:** McNemar's test for Tests 2 & 3 (same items, two accept/reject
   decisions → paired yes/no), plus Cohen's kappa for human agreement.
3. **Don't exhaust human helpers:** auto-judge everything, humans check a representative subset.
   (NER human checking is the hardest — budget more there.)

---

## Priority order
1. Task-aware generation quality (Test 1)
2. TaskValidator necessity (Test 2)
3. Per-sentence vs scenario — CATCH (Test 3) ← in progress now
4. Quality refinement — FIX (Test 6a)  [pairs with Test 3 = full masking story; data ready]
5. Task-aware refinement vs generic (Test 6b)  [defends the task-aware fixer; needs new run]
6. Deterministic CS ratio vs LLM (Test 4)
7. Ablation (Test 5, if time)

---

## How our current work fits
Step 1 (raw vs fixed runs in `step1_raw_data/` and `step1_fixed_data/`) feeds **Test 3**
(per-sentence vs scenario) and **Test 6** (before vs after refinement).
See `PLAN.md` for the step-by-step masking work.
