# Pilot v2 (CS-validity fix) — Results

Config-only change: `cs_ratio: [50%,60%,70%]`, `cs_type: [Intrasentential]`. No prompt/NER/pipeline change. Isolated run — **not merged** into the training set.

## Run
- scenarios requested: **40** | completed: **0** | failed (429/other): **40**
- quality threshold: **7.0** (unchanged)

## Yield vs baseline
| metric | baseline (v1) | pilot v2 |
|---|--:|--:|
| CS-valid rate (of non-empty) | 30% | **0%** (0/0) |
| fully-Arabic share of failures | 99.6% | **0%** (0/0) |

## CS-valid rate by cs_ratio target
| cs_ratio | CS-valid |
|---|---|

## Filter funnel (instances)
| stage | count |
|---|--:|
| raw generated | 0 |
| kept (validator+CS-valid+quality>=7.0+dedup) | 0 |

**Filtering loss by reason:**
| reason | count |
|---|--:|
| empty_text | 0 |
| validator_failed | 0 |
| not_cs_valid | 0 |
| low_quality | 0 |
| duplicate | 0 |
| duplicates removed | 0 |

## Kept by label
| label | kept |
|---|--:|
| positive | 0 |
| negative | 0 |
| neutral | 0 |

## Kept by topic
| topic | kept |
|---|--:|

## Examples of newly-valid code-switched outputs

## Verdict
- CS-valid 30% → **0%** (NOT improved).
- **Not merged** into the main training set (per instruction; merge only if clearly better).
- Multi-Agent BERT not trained from this.