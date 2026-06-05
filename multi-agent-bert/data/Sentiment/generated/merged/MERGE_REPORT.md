# Experiment C — Sentiment Data MERGE Report

Combined **2** source(s) into one de-duplicated, label-balanced sentiment dataset. Pilot_v1 + daily runs; NER frozen/untouched; pipeline & prompts unchanged.

## Scenario coverage (resume manifest)
- requested (full design): **324** scenarios
- completed so far: **140**
- remaining: **184**
- observed yield: **0.85** kept examples / completed scenario

## Sources merged
| source | rows in | 
|---|--:|
| pilot_v1 | 114 |
| run_20260606 | 6 |

## Merged dataset (pre-balance, cross-source de-duplicated)
- unique examples: **119**  (cross-source duplicates removed: **1**)

**By label:**
| label | count |
|---|--:|
| positive | 38 |
| negative | 41 |
| neutral | 40 |

**By topic:**
| topic | count |
|---|--:|
| sports | 19 |
| shopping | 18 |
| social | 17 |
| tech | 14 |
| medical | 13 |
| education | 13 |
| finance | 11 |
| business | 10 |
| health | 4 |

## Balanced training set (down-sampled to 38/label)
| label | count |
|---|--:|
| positive | 38 |
| negative | 38 |
| neutral | 38 |
| **TOTAL** | **114** |

## Cumulative funnel (aggregated across source stats.json)
- raw by label: {'neutral': 201, 'negative': 240, 'positive': 249}
- kept by label: {'positive': 38, 'neutral': 41, 'negative': 41}
- filtering loss by reason:

| reason | count |
|---|--:|
| empty_text | 0 |
| validator_failed | 20 |
| not_cs_valid | 481 |
| low_quality | 39 |
| duplicate | 0 |

## Kept by topic (cumulative)
| topic | kept |
|---|--:|
| sports | 19 |
| shopping | 18 |
| social | 17 |
| tech | 14 |
| medical | 13 |
| education | 13 |
| finance | 11 |
| business | 11 |
| health | 4 |

## Estimated remaining work (target = 100/label)
- current smallest label: **38** (need **62** more/label)
- estimated additional KEPT examples needed: **~186**
- estimated additional SCENARIOS to run: **~219** (at current yield; 184 scenarios remain in the 324-design — run across days under the daily quota)

## Outputs
- `merged/switchlingua_sentiment_train_merged.csv` / `.jsonl` — balanced, ready for Experiment C.
- per-row metadata: text, label, topic, cs_ratio, cs_type, cs_function, tense, perspective, conversation_type, intensity, ambiguity, scenario_id, task_validator_passed, cs_valid, cs_ar_ratio, cs_en_ratio, fluency, naturalness, quality_score, validator_predicted_label, gender, age, education_level, source.

## Notes
- Multi-Agent BERT is **NOT trained here** — accumulation only.
- Add more data: `manage_sentiment_data.py generate --max <N>` on a new day, then `merge` again.
- `on_execute.round` / `shared.style` remain inert; size grows by running more scenarios.