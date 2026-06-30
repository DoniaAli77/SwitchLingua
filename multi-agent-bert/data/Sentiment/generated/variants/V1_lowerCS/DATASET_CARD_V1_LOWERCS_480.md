# Dataset Card — V1 Lower-CS / Arabic-dominant (480)

Generation-config **sensitivity** variant (NOT EESA-tailored). Built by the frozen Modified_Version
SwitchLingua pipeline (`gpt-4o-mini`) from `config_sentiment_genV1_lowerCS.yaml`. Isolated accumulation
(own dir + manifest); **not** merged from pilot_v1 / GEN daily_runs / GEN-960 / V2a. No core/prompt change.

## Files
- `variants/V1_lowerCS/switchlingua_sentiment_v1_lowerCS_480.csv`  (UTF-8-SIG, 480 + header)
- `variants/V1_lowerCS/switchlingua_sentiment_v1_lowerCS_480.jsonl` (480)

## Research question
How does **code-switching intensity** (a generation control) affect the data? V1 changes ONE factor vs
the GEN-960 baseline profile: `cs_ratio ["50%","60%"] → ["70%","80%"]` (Arabic-dominant). Everything else
(Intrasentential, Expressive, age [18-25,26-40], Present, 9 topics, 3 intensities, M/F) held constant.

## 12-point validation (verified 2026-06-30)
| # | check | result |
|---|---|---|
| 1 | total rows | **480** (CSV == JSONL) |
| 2 | label balance | **160 / 160 / 160** (pos/neg/neu) |
| 3 | duplicates (within + cross-dedup vs GEN pool/960/V2a) | **0 / 0** |
| 4 | CS-valid (recomputed) | **480 / 480** |
| 5 | TaskValidator passed | **480 / 480** |
| 6 | quality ≥ 7.0 | **480 / 480** (range 7.0–8.65) |
| 7 | AR : EN ratio | **62.9 : 37.1**  (GEN-960 52.7 : 47.3) |
| 8 | CMI mean / median | **33.0 / 33.3**  (GEN-960 40.9) · hist {0-20:33, 20-40:303, 40-60:144} |
| 9 | length mean/median/p90 | 14.2 / 14 / 19  (GEN-960 mean 14.1) |
| 10 | vs GEN-960 | **AR +10.2pp · CMI −7.9 · length +0.1** |
| 11 | source / cs_ratio / age | gen_V1_lowerCS 480 · 70%:301 / 80%:179 · age 26-40:226 / 18-25:254 |
| 12 | dataset card | this file |

## Factor verdict
**Intended factor moved, cleanly isolated.** Arabic dominance **↑ (+10.2pp)** and CMI **↓ (−7.9)**, while
**length is unchanged** (+0.1) — confirming this variant isolates *code-switching intensity*, not register/length.

## Generation cost (the price of lower CS)
- CS-valid yield ≈ **30%** (vs ~43% at baseline ratios) — Arabic-dominant prompts produce more fully-Arabic
  (monolingual) sentences that the deterministic CS filter correctly rejects.
- **Negative** was the bottleneck (hardest label × low yield); reached 176 only after sweeping the full
  648-scenario genV1 space (no `tense: Past` expansion was needed in the end). 11 windows over several
  quota-days; network instability slowed several windows.

## Filters (every example passed all)
non-empty → TaskValidator passed → deterministic CS-valid → quality ≥ 7.0 → de-dup (within + cross vs
GEN-960, GEN pool incl. pilot_v1/daily_runs/240/480, and V2a).

## Schema
`text, label, topic, cs_ratio, cs_type, cs_function, tense, perspective, conversation_type, intensity,
ambiguity, scenario_id, task_validator_passed, cs_valid, cs_ar_ratio, cs_en_ratio, fluency, naturalness,
quality_score, validator_predicted_label, gender, age, education_level, source`

## Limitations
- Heterogeneous cs_ratio (70/80%) by design; age mixed (18-25/26-40); tense Present only.
- Generator-labeled, LLM-judged; no human verification.
- Companion variant **V2a (register)** documented as **partial/weak** (shorter outputs, but no CMI shift) —
  not scaled.

## Reproducibility
- Config: `experiments/switchlingua/config_sentiment_genV1_lowerCS.yaml`.
- Runner: `experiments/switchlingua/run_gen_sensitivity_pilot.py` (isolated; manifest `gen_sensitivity/V1_lowerCS/manifest.json`, 648/648 scenarios).
- Pipeline: Modified_Version SwitchLingua (FROZEN), `gpt-4o-mini`.

## Status
**NOT trained.** This is a generation-config sensitivity dataset; downstream tests (generated-only
primary_only, low-resource augmentation primary_only — no full_agentic) come **after** approval.
GEN 240/480/960 datasets are preserved and untouched.
