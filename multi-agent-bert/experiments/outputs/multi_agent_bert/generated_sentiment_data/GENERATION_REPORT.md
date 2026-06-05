# Experiment C — SwitchLingua Sentiment Generation (pilot)

Generated Arabic-English code-switched **sentiment** training data with the Modified_Version pipeline (`gpt-4o-mini`), then filtered for trainable, balanced data. **NER untouched (frozen); sentiment-only run.**

## Generation
- Config: `experiments/switchlingua/config_sentiment_expC.yaml` (324 sentiment scenarios by design).
- Scenarios run: **2** (failed: 0).
- Quality threshold (per-sentence weighted_score): **7.0**.

## Filter funnel (instances)
| stage | count |
|---|--:|
| raw generated instances | 11 |
| TaskValidator passed | 9 |
| + deterministic CS-valid | 4 |
| + quality >= 7.0 | 3 |
| + de-duplicated (KEPT) | 3 |

## Kept per label (pre-balance)
| label | kept |
|---|--:|
| positive | 0 |
| negative | 1 |
| neutral | 2 |

## Balanced training set (down-sampled to smallest label = 1/label)
| label | count |
|---|--:|
| positive | 0 |
| negative | 1 |
| neutral | 1 |
| **TOTAL** | **2** |

## Outputs
- `data/Sentiment/generated/switchlingua_sentiment_train_pilot.csv` / `.jsonl` — **balanced** training pilot.
- `data/Sentiment/generated/switchlingua_sentiment_kept_all.jsonl` — all kept (pre-balance).
- `data/Sentiment/generated/_raw_pipeline/raw_states.jsonl` — raw pipeline output.

## Examples (balanced set)
- [neutral/high] (q=7.3, ar%=45.45) بينما كنا نتسوق, we talked about the latest trends في الأزياء.
- [negative/low] (q=7.6, ar%=62.5) أحياناً أعتقد أن الاستثمار في السوق المالية ليس فكرة جيدة, especially when I see the fluct

## Notes / caveats
- **Neutral as factual/descriptive:** the generation prompt is frozen, so neutral-specific wording could not be injected via config (only `intensity`/`ambiguity` flow into sentiment task_constraints). `ambiguity: low` was used for cleaner labels; neutral quality is enforced post-hoc by the TaskValidator filter. Residual risk of mildly-polar 'neutral' remains — recommend a human spot-check before training.
- `on_execute.round` and `shared.style` are inert in the pipeline (not read by any code); dataset size is governed by the config Cartesian product, not those fields.
- Filtering is reproducible offline: `run_expC_sentiment_generation.py --filter-only`.
- **Multi-Agent BERT is NOT trained here** (data generation only).
