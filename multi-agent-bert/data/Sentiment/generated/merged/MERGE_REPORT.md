# Experiment C — Sentiment Data MERGE Report

Combined **1** source(s) into one de-duplicated, label-balanced sentiment dataset. Pilot_v1 + daily runs; NER frozen/untouched; pipeline & prompts unchanged.

## Scenario coverage (resume manifest)
- requested (full design): **324** scenarios
- completed so far: **130**
- remaining: **194**
- observed yield: **0.88** kept examples / completed scenario

## Sources merged
| source | rows in | 
|---|--:|
| pilot_v1 | 114 |

## Merged dataset (pre-balance, cross-source de-duplicated)
- unique examples: **114**  (cross-source duplicates removed: **0**)

**By label:**
| label | count |
|---|--:|
| positive | 38 |
| negative | 38 |
| neutral | 38 |

**By topic:**
| topic | count |
|---|--:|
| sports | 18 |
| shopping | 18 |
| social | 17 |
| medical | 13 |
| education | 12 |
| tech | 12 |
| finance | 11 |
| business | 9 |
| health | 4 |

## Balanced training set (down-sampled to 38/label)
| label | count |
|---|--:|
| positive | 38 |
| negative | 38 |
| neutral | 38 |
| **TOTAL** | **114** |

## Cumulative funnel (aggregated across source stats.json)
- raw by label: {'neutral': 176, 'negative': 226, 'positive': 238}
- kept by label: {'positive': 38, 'neutral': 38, 'negative': 38}
- filtering loss by reason:

| reason | count |
|---|--:|
| empty_text | 0 |
| validator_failed | 19 |
| not_cs_valid | 445 |
| low_quality | 32 |
| duplicate | 0 |

## Kept by topic (cumulative)
| topic | kept |
|---|--:|
| sports | 18 |
| shopping | 18 |
| social | 17 |
| medical | 13 |
| education | 12 |
| tech | 12 |
| finance | 11 |
| business | 9 |
| health | 4 |

## Estimated remaining work (target = 150/label)
- current smallest label: **38** (need **112** more/label)
- estimated additional KEPT examples needed: **~336**
- estimated additional SCENARIOS to run: **~383** (at current yield; 194 scenarios remain in the 324-design — run across days under the daily quota)

## Outputs
- `merged/switchlingua_sentiment_train_merged.csv` / `.jsonl` — balanced, ready for Experiment C.
- per-row metadata: text, label, topic, cs_ratio, cs_type, cs_function, tense, perspective, conversation_type, intensity, ambiguity, scenario_id, task_validator_passed, cs_valid, cs_ar_ratio, cs_en_ratio, fluency, naturalness, quality_score, validator_predicted_label, gender, age, education_level, source.

## Notes
- Multi-Agent BERT is **NOT trained here** — accumulation only.
- Add more data: `manage_sentiment_data.py generate --max <N>` on a new day, then `merge` again.
- `on_execute.round` / `shared.style` remain inert; size grows by running more scenarios.