# Dataset Card — Experiment C Sentiment (SwitchLingua-generated)

**Generated Arabic–English code-switched sentiment training data** for Multi-Agent BERT Experiment C,
produced by the (frozen) Modified_Version SwitchLingua pipeline with `gpt-4o-mini`. Balanced, filtered,
de-duplicated. **NER untouched; no prompt/core/pipeline changes.**

## Files
- `merged/switchlingua_sentiment_train_merged.csv`  (UTF-8-SIG, 240 rows + header)
- `merged/switchlingua_sentiment_train_merged.jsonl` (240 rows)

## Integrity (verified 2026-06-13)
| check | result |
|---|---|
| total examples | **240** (CSV == JSONL) |
| label balance | **positive 80 / negative 80 / neutral 80** |
| duplicate normalized texts | **0** |
| CS-valid (recomputed via `compute_true_cs_stats`) | **240/240** (0 monolingual) |
| TaskValidator passed | **240/240** |
| quality_score ≥ 7.0 | **240/240** (range **7.0–8.4**) |
| empty metadata fields | **none** |

## Label balance
positive 80 · negative 80 · neutral 80  (down-sampled to the smallest label = neutral).
Pre-balance pool was positive 113 / negative 96 / neutral 80.

## Source mix
| source | rows in balanced set | config |
|---|--:|---|
| `run_20260613` (v3 window) | 141 | cs_ratio 50/60%, Intrasentential-only |
| `pilot_v1` | 94 | cs_ratio 70%, Intra+Intersentential |
| `run_20260606` (early daily) | 5 | cs_ratio 70%, Intra+Intersentential |

## cs_ratio mix (the heterogeneity)
| cs_ratio target | rows |
|---|--:|
| 70% | 99 |
| 60% | 72 |
| 50% | 69 |

## Topic mix
sports 35 · tech 33 · shopping 32 · medical 30 · education 30 · social 29 · business 22 · finance 21 · health 8.

## Filters applied (every example passed all)
1. **Non-empty text**
2. **TaskValidator passed** (the sentence expresses the target sentiment label)
3. **Deterministic CS-valid** — `compute_true_cs_stats.is_code_switched` (≥1 Arabic-script AND ≥1 Latin token)
4. **Quality ≥ 7.0** — per-sentence weighted score (fluency/naturalness/cs/socio-cultural)
5. **De-duplicated** by normalized text (cross-source)

## Schema (per row)
`text, label, topic, cs_ratio, cs_type, cs_function, tense, perspective, conversation_type, intensity,
ambiguity, scenario_id, task_validator_passed, cs_valid, cs_ar_ratio, cs_en_ratio, fluency, naturalness,
quality_score, validator_predicted_label, gender, age, education_level, source`

## Limitations
- **Mixed generation configs (heterogeneous cs_ratio).** 99/240 examples target 70% Arabic (from
  `pilot_v1` + 5 early rows); 141/240 target 50–60% (v3). The set is **not a clean single-config corpus**.
  Acceptable for sentiment classification (label does not depend on cs_ratio), but note it if analyzing
  CS-ratio effects.
- **Generator-labeled, LLM-judged.** Labels come from the generation target + TaskValidator (gpt-4o-mini),
  not human annotation. No human verification of sentiment correctness yet.
- **Neutral is the binding label** (only 80 available pre-balance) and the hardest class; neutrals can lean
  mildly aspirational (frozen-prompt limitation — see `CS_VALIDITY_DIAGNOSIS.md`).
- **Topic imbalance:** `health` is under-represented (8).
- **Single model / single language pair** (gpt-4o-mini, Arabic-matrix + English).
- **Empirical CS-validity note:** at scale, cs_ratio **50% (49% CS-valid) outperformed 60% (40%)** — opposite
  of the small v2 pilot; relevant if generating more data later.

## Provenance / reproducibility
- Configs: `experiments/switchlingua/config_sentiment_expC_v3.yaml` (v3), `config_sentiment_expC.yaml` (v1/pilot).
- Resume manifest (v3): `completed_scenarios_v3.json` (100/324 v3 scenarios completed).
- Pipeline: Modified_Version SwitchLingua (FROZEN), `gpt-4o-mini`.
- Re-merge: `manage_sentiment_data.py merge --target-per-label 80`.

## Status
**NOT yet used for training.** Awaiting approval to train Multi-Agent BERT Experiment C.
EESA test set remains the held-out evaluation anchor (Exp A reference).
