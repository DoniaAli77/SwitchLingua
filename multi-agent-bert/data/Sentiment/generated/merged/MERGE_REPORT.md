# Experiment C — Sentiment Data MERGE Report

Combined **5** source(s) into one de-duplicated, label-balanced sentiment dataset. Pilot_v1 + daily runs; NER frozen/untouched; pipeline & prompts unchanged.

## Scenario coverage (resume manifest)
- requested (full design): **324** scenarios
- completed so far: **255**
- remaining: **69**
- observed yield: **2.06** kept examples / completed scenario

## Sources merged
| source | rows in | 
|---|--:|
| pilot_v1 | 114 |
| run_20260606 | 6 |
| run_20260613 | 170 |
| run_20260614 | 195 |
| run_20260616 | 42 |

## Merged dataset (pre-balance, cross-source de-duplicated)
- unique examples: **526**  (cross-source duplicates removed: **1**)

**By label:**
| label | count |
|---|--:|
| positive | 199 |
| negative | 162 |
| neutral | 165 |

**By topic:**
| topic | count |
|---|--:|
| sports | 74 |
| tech | 74 |
| medical | 66 |
| shopping | 63 |
| social | 61 |
| education | 59 |
| business | 50 |
| finance | 44 |
| health | 35 |

## Balanced training set (down-sampled to 160/label)
| label | count |
|---|--:|
| positive | 160 |
| negative | 160 |
| neutral | 160 |
| **TOTAL** | **480** |

## Cumulative funnel (aggregated across source stats.json)
- raw by label: {'neutral': 592, 'negative': 650, 'positive': 684}
- kept by label: {'positive': 199, 'neutral': 166, 'negative': 162}
- filtering loss by reason:

| reason | count |
|---|--:|
| empty_text | 0 |
| validator_failed | 54 |
| not_cs_valid | 1201 |
| low_quality | 114 |
| duplicate | 0 |

## Kept by topic (cumulative)
| topic | kept |
|---|--:|
| sports | 74 |
| tech | 74 |
| medical | 66 |
| shopping | 63 |
| social | 61 |
| education | 59 |
| business | 51 |
| finance | 44 |
| health | 35 |

## Estimated remaining work (target = 160/label)
- current smallest label: **162** (need **0** more/label)
- estimated additional KEPT examples needed: **~0**
- estimated additional SCENARIOS to run: **~0** (at current yield; 69 scenarios remain in the 324-design — run across days under the daily quota)

## Outputs
- `merged/switchlingua_sentiment_train_merged.csv` / `.jsonl` — balanced, ready for Experiment C.
- per-row metadata: text, label, topic, cs_ratio, cs_type, cs_function, tense, perspective, conversation_type, intensity, ambiguity, scenario_id, task_validator_passed, cs_valid, cs_ar_ratio, cs_en_ratio, fluency, naturalness, quality_score, validator_predicted_label, gender, age, education_level, source.

## Notes
- Multi-Agent BERT is **NOT trained here** — accumulation only.
- Add more data: `manage_sentiment_data.py generate --max <N>` on a new day, then `merge` again.
- `on_execute.round` / `shared.style` remain inert; size grows by running more scenarios.