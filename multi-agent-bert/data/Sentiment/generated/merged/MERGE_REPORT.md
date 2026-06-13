# Experiment C — Sentiment Data MERGE Report

Combined **3** source(s) into one de-duplicated, label-balanced sentiment dataset. Pilot_v1 + daily runs; NER frozen/untouched; pipeline & prompts unchanged.

## Scenario coverage (resume manifest)
- requested (full design): **324** scenarios
- completed so far: **100**
- remaining: **224**
- observed yield: **2.89** kept examples / completed scenario

## Sources merged
| source | rows in | 
|---|--:|
| pilot_v1 | 114 |
| run_20260606 | 6 |
| run_20260613 | 170 |

## Merged dataset (pre-balance, cross-source de-duplicated)
- unique examples: **289**  (cross-source duplicates removed: **1**)

**By label:**
| label | count |
|---|--:|
| positive | 113 |
| negative | 96 |
| neutral | 80 |

**By topic:**
| topic | count |
|---|--:|
| tech | 46 |
| sports | 44 |
| education | 37 |
| shopping | 37 |
| medical | 35 |
| social | 32 |
| business | 26 |
| finance | 21 |
| health | 11 |

## Balanced training set (down-sampled to 80/label)
| label | count |
|---|--:|
| positive | 80 |
| negative | 80 |
| neutral | 80 |
| **TOTAL** | **240** |

## Cumulative funnel (aggregated across source stats.json)
- raw by label: {'neutral': 305, 'negative': 412, 'positive': 446}
- kept by label: {'positive': 113, 'neutral': 81, 'negative': 96}
- filtering loss by reason:

| reason | count |
|---|--:|
| empty_text | 0 |
| validator_failed | 28 |
| not_cs_valid | 745 |
| low_quality | 70 |
| duplicate | 0 |

## Kept by topic (cumulative)
| topic | kept |
|---|--:|
| tech | 46 |
| sports | 44 |
| education | 37 |
| shopping | 37 |
| medical | 35 |
| social | 32 |
| business | 27 |
| finance | 21 |
| health | 11 |

## Estimated remaining work (target = 80/label)
- current smallest label: **80** (need **0** more/label)
- estimated additional KEPT examples needed: **~0**
- estimated additional SCENARIOS to run: **~0** (at current yield; 224 scenarios remain in the 324-design — run across days under the daily quota)

## Outputs
- `merged/switchlingua_sentiment_train_merged.csv` / `.jsonl` — balanced, ready for Experiment C.
- per-row metadata: text, label, topic, cs_ratio, cs_type, cs_function, tense, perspective, conversation_type, intensity, ambiguity, scenario_id, task_validator_passed, cs_valid, cs_ar_ratio, cs_en_ratio, fluency, naturalness, quality_score, validator_predicted_label, gender, age, education_level, source.

## Notes
- Multi-Agent BERT is **NOT trained here** — accumulation only.
- Add more data: `manage_sentiment_data.py generate --max <N>` on a new day, then `merge` again.
- `on_execute.round` / `shared.style` remain inert; size grows by running more scenarios.