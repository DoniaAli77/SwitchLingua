# SwitchLingua — FINAL STATUS (Freeze)

**Frozen on:** 2026-06-05. Codewise freeze check passed. No further code changes unless a bug appears.
**Main detailed report:** [`masking_defense/REPORT.md`](masking_defense/REPORT.md) (sections referenced below).
**Systems:** B = `Modified_Version` (contribution: per-sentence scoring, TaskValidator, targeted + task-aware
refiner, deterministic CS-ratio, generic config-driven NER guidance). C = `Original_baseLine` (control).
All experiments use gpt-4o-mini.

---

## A. CODE COMPLETE (frozen)

### Pipeline (Modified_Version/core)
- Per-sentence scoring + per-instance refine routing.
- `TaskValidatorAgent` (topic / sentiment / NER); refiner guardrail (re-validate + re-score, rollback on regression).
- Generic config-driven NER guidance: `DEFAULT_ENTITY_GUIDANCE` + `build_ner_entity_guidance()` + `{ner_entity_guidance}` placeholder (English-only entity policy). **NER FROZEN** (commit e152b4d).
- NER infinite-loop fix: refine counter advances on every attempt (`node_engine.py:~1106`).
- Deterministic CS-ratio counter: `utils.compute_true_cs_stats` (Arabic vs Latin token counts; 0 variance).

### Automated tests — ALL PASS, no real API (mocked LLM / pure functions)
| Test file | Covers | Result |
|---|---|---|
| `Modified_Version/core/test files/test_refiner_guardrail.py` | accept / rollback / task-fix / quality-regression + loop-fix budget | 4/4 PASS |
| `Modified_Version/core/test files/test_ner_guidance.py` | generic NER guidance builder + prompt placeholder | 5/5 PASS |
| `Modified_Version/core/test files/test_per_instance_scoring.py` | per-instance vs aggregate weighting | 2/2 PASS (import-path bug fixed during freeze) |
| `experiments/switchlingua/test_per_sentence_vs_scenario.py` | policy disagreement / variance / masking sim / refiner targeting | PASS |

### Analysis / runner scripts (run, reusable)
- `analyze_consolidated_human_eval.py` — 9 analyses; full-sheet **and** BLIND `--annotations/--key` merge. **Verified on dummy completed input.**
- `build_consolidated_annotation_sheet.py` / `build_blind_sheet.py` — consolidated sheet → BLIND + KEY.
- `build_csratio_set.py` — fixed 30-sentence CS-ratio set (20 real + 10 controlled edge cases).
- `run_csratio_partial_validation.py` — deterministic vs LLM-only counting; `--reuse-llm` (offline), `--set/--outdir`. **Verified blank (PENDING) and filled (human metrics) modes, no API.**
- Earlier frozen runners: `run_threshold_sweep.py`, `run_task_aware_eval.py`, `run_task_validator_necessity.py`, `run_ner_*` (guidance/AB/coverage), `refiner_clean_test.py` / `step6b_refiner_headtohead.py` (in masking_defense/).

---

## B. AUTOMATED RESULTS COMPLETE

| Result | Finding | Where |
|---|---|---|
| **Masking** (per-sentence catches what aggregate hides) | 41.5% (54-scen) / **35.6%** (101-scen) at calibrated bar 7.0; 0% at bar 8 (scores tightly packed) | REPORT §4 · `masking_defense/step2_counts/` |
| **Refiner improves a caught weak sentence** | +0.60, 79/87 (90.8%), p≈0 (within-sentence) | REPORT §6.2 |
| **Your refiner vs original refiner** | TIE, p=0.53 (advantage is *what* gets refined, not *how well*) | REPORT §6.3 |
| **Task-aware generation quality** | topic 100%, sentiment ~70% (neutral drag), NER 40% (English-only) | REPORT §6.5 · `task_aware_eval/` |
| **TaskValidator necessity** | precision 70.9%→85.5%, task-wrong accepts 25→9; benefit concentrated in NER, null for sentiment | REPORT §6.6 · `task_validator/` |
| **Generic NER guidance** | EVENT 39→84%, LOC 44→63%, PRODUCT 45→85%; no PER regression; FROZEN | REPORT §6.5.1–6.5.2 |
| **CS-ratio measurement (Test 4, PARTIAL)** | LLM-only counter disagrees with itself on **12/30 (40%)** sentences; deterministic = **0 variance**; both agree on binary CS (0/30); mean ratio gap 5.0% | REPORT §6.7 · `csratio/csratio_partial_report.md` |

---

## C. HUMAN / MANUAL PENDING

1. **BLIND human annotation** (86 rows) — give annotators `human_eval/consolidated_human_annotation_sheet_BLIND.csv`
   (keep `consolidated_human_annotation_key.csv` hidden); save completed as `*_completed.csv`, then:
   ```
   python experiments/switchlingua/analyze_consolidated_human_eval.py \
     --annotations <completed_BLIND.csv> --key human_eval/consolidated_human_annotation_key.csv
   ```
   Unlocks: task-correctness by task, CS-validity, acceptability, AI-judge-vs-human & Validator-vs-human
   agreement, masked-vs-control quality, neutral-sentiment dispute resolution, NER English-script compliance.

2. **CS-ratio human token counts** (30 rows) — fill `csratio/csratio_validation_set.csv`
   (`human_arabic_token_count` / `human_english_token_count` / `human_other_token_count`), then:
   ```
   python experiments/switchlingua/run_csratio_partial_validation.py --reuse-llm
   ```
   (`--reuse-llm` reuses the cached LLM counts — no new API calls.) Unlocks: Arabic/English/other count MAE,
   ratio MAE, monolingual + code-switch detection accuracy, boundary error — for **both** deterministic and LLM-only.

---

## D. OPTIONAL FUTURE WORK (not required; out of current scope)

- **Stronger generator (gpt-4o)** to widen the score spread → masking robust across more thresholds (current model is uniformly mediocre, so masking is threshold-sensitive).
- **Task-aware refiner test** (validator ON + task-failing sentences) — the one refiner path not yet isolated.
- **Larger / balanced task sample** (current: sentiment-heavy; single refinement pass MAX=1).
- **Full Original-vs-Modified generation comparison** (`run_csratio_validation.py` A/B/C post-hoc, `score_system_comparison.py`) — deliberately NOT run here; Test 4 is measurement-only.
- **Hard NER case** `PER_EVENT` (competing entity types) — remaining weak spot after guidance.

---

## Freeze-check log (this pass)
1. ✅ All non-API tests pass (4/4 files) — fixed one stale import path in `test_per_instance_scoring.py`.
2. ✅ Human-eval analyzer runs on dummy completed BLIND + key (all 9 analyses produced).
3. ✅ CS-ratio partial script reruns blank (PENDING) **and** filled-dummy (human MAE/detection/boundary), offline via `--reuse-llm`.
4. ✅ This status file written.
