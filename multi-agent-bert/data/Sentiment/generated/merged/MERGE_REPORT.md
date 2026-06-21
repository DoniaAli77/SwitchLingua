# Experiment C — Sentiment Data MERGE Report

Combined **8** source(s) into one de-duplicated, label-balanced sentiment dataset. Pilot_v1 + daily runs; NER frozen/untouched; pipeline & prompts unchanged.

## Scenario coverage (resume manifest)
- requested (full design): **1296** scenarios
- completed so far: **666**
- remaining: **630**
- observed yield: **1.67** kept examples / completed scenario

## Sources merged
| source | rows in | 
|---|--:|
| pilot_v1 | 114 |
| run_20260606 | 6 |
| run_20260613 | 170 |
| run_20260614 | 195 |
| run_20260616 | 42 |
| run_20260619 | 311 |
| run_20260620 | 92 |
| run_20260621 | 184 |

## Merged dataset (pre-balance, cross-source de-duplicated)
- unique examples: **1113**  (cross-source duplicates removed: **1**)

**By label:**
| label | count |
|---|--:|
| positive | 408 |
| negative | 328 |
| neutral | 377 |

**By topic:**
| topic | count |
|---|--:|
| social | 151 |
| tech | 143 |
| education | 134 |
| sports | 132 |
| medical | 130 |
| shopping | 122 |
| business | 105 |
| health | 99 |
| finance | 97 |

## Balanced training set (down-sampled to 320/label)
| label | count |
|---|--:|
| positive | 320 |
| negative | 320 |
| neutral | 320 |
| **TOTAL** | **960** |

## Cumulative funnel (aggregated across source stats.json)
- raw by label: {'neutral': 1282, 'negative': 1275, 'positive': 1313}
- kept by label: {'positive': 408, 'neutral': 378, 'negative': 328}
- filtering loss by reason:

| reason | count |
|---|--:|
| empty_text | 0 |
| validator_failed | 102 |
| not_cs_valid | 2399 |
| low_quality | 225 |
| duplicate | 0 |

## Kept by topic (cumulative)
| topic | kept |
|---|--:|
| social | 151 |
| tech | 143 |
| education | 134 |
| sports | 132 |
| medical | 130 |
| shopping | 122 |
| business | 106 |
| health | 99 |
| finance | 97 |

## Estimated remaining work (target = 320/label)
- current smallest label: **328** (need **0** more/label)
- estimated additional KEPT examples needed: **~0**
- estimated additional SCENARIOS to run: **~0** (at current yield; 630 scenarios remain in the 324-design — run across days under the daily quota)

## Outputs
- `merged/switchlingua_sentiment_train_merged.csv` / `.jsonl` — balanced, ready for Experiment C.
- per-row metadata: text, label, topic, cs_ratio, cs_type, cs_function, tense, perspective, conversation_type, intensity, ambiguity, scenario_id, task_validator_passed, cs_valid, cs_ar_ratio, cs_en_ratio, fluency, naturalness, quality_score, validator_predicted_label, gender, age, education_level, source.

## Notes
- Multi-Agent BERT is **NOT trained here** — accumulation only.
- Add more data: `manage_sentiment_data.py generate --max <N>` on a new day, then `merge` again.
- `on_execute.round` / `shared.style` remain inert; size grows by running more scenarios.