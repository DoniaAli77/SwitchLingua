# Dataset Card — Experiment C3 Sentiment (SwitchLingua-generated, 960)

Balanced Arabic–English code-switched **sentiment** training data for Multi-Agent BERT Experiment **C3**,
produced by the (frozen) Modified_Version SwitchLingua pipeline (`gpt-4o-mini`). Scaled up from the 480
set by **accumulation** (resume-safe, append-only) — not regenerated from zero. NER untouched; no
prompt/core/pipeline/filter changes.

## Files
- `merged/switchlingua_sentiment_train_960_320perlabel.csv`  (UTF-8-SIG, 960 rows + header)
- `merged/switchlingua_sentiment_train_960_320perlabel.jsonl` (960 rows)

## Integrity (verified 2026-06-21)
| check | result |
|---|---|
| total examples | **960** (CSV == JSONL) |
| label balance | **positive 320 / negative 320 / neutral 320** |
| duplicate normalized texts | **0** |
| CS-valid (recomputed `compute_true_cs_stats`) | **960 / 960** |
| TaskValidator passed | **960 / 960** |
| quality_score ≥ 7.0 | **960 / 960** (range 7.0 – 9.0) |
| empty metadata fields | none |

## Lineage (240 → 480 → 960)
| dataset | per label | total | preserved file |
|---|--:|--:|---|
| C (v1) | 80 | 240 | `switchlingua_sentiment_train_240_80perlabel.{csv,jsonl}` |
| C (scaled) | 160 | 480 | `switchlingua_sentiment_train_480_160perlabel.{csv,jsonl}` |
| **C3** | **320** | **960** | `switchlingua_sentiment_train_960_320perlabel.{csv,jsonl}` |
All earlier datasets are **preserved** (not overwritten). The 960 is a superset selection of the same pool.

## Config evolution (config-only changes; CS-validity fix kept throughout)
| config | cs_ratio | cs_type | age | tense | purpose |
|---|---|---|---|---|---|
| v3 | 50/60 | Intrasentential | 18-25 | Present | base accumulation |
| v4 | 50/60 | Intrasentential | 18-25, **26-40** | Present | expand space 324→648 |
| v5 | 50/60 | Intrasentential | 18-25, 26-40 | Present, **Past** | final negative top-up |

## Composition (the 960)
- **cs_ratio:** 50% → 477 · 60% → 381 · 70% → 102 (the 70% are legacy pilot_v1/early rows).
- **age:** 18-25 → 556 · 26-40 → 404.
- **source mix:** run_20260619 260 · run_20260614 171 · run_20260621 162 · run_20260613 147 · pilot_v1 99 · run_20260620 80 · run_20260616 38 · run_20260606 3.

## Filters applied (every example passed all)
non-empty text → **TaskValidator passed** → **deterministic CS-valid** → **quality ≥ 7.0** → **de-dup** (normalized text, across all sources).

## Schema (per row)
`text, label, topic, cs_ratio, cs_type, cs_function, tense, perspective, conversation_type, intensity,
ambiguity, scenario_id, task_validator_passed, cs_valid, cs_ar_ratio, cs_en_ratio, fluency, naturalness,
quality_score, validator_predicted_label, gender, age, education_level, source`

## Limitations
- **Mixed generation configs (heterogeneous cs_ratio / age / tense).** Acceptable for sentiment (label
  doesn't depend on these), but note it for any analysis of those factors. 70% Arabic rows are a small
  legacy minority (102/960).
- **Generator-labeled, LLM-judged** (target label + TaskValidator); no human verification of sentiment yet.
- Neutral remains the conceptually hardest class; negative had the lowest generation yield (drove the scaling effort).
- Single model / single language pair (gpt-4o-mini, Arabic-matrix + English).

## Reproducibility
- Configs: `experiments/switchlingua/config_sentiment_expC_{v3,v4,v5}.yaml`.
- Resume manifest: `completed_scenarios_v3.json` (666/1296 scenarios completed across v3→v5 spaces).
- Pipeline: Modified_Version SwitchLingua (FROZEN), `gpt-4o-mini`.
- Re-merge: `manage_sentiment_data.py merge --target-per-label 320 --out-name switchlingua_sentiment_train_960_320perlabel`.

## Status
**NOT yet used for training.** Awaiting approval to train Multi-Agent BERT Experiment C3.
EESA test (818) remains the held-out evaluation anchor.
