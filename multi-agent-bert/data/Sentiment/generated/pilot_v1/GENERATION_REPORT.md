# Experiment C — SwitchLingua Sentiment Generation (pilot)

Generated Arabic-English code-switched **sentiment** training data with the Modified_Version pipeline (`gpt-4o-mini`), then filtered for trainable, balanced data. **NER untouched (frozen); sentiment-only run.**

## Generation
- Config: `experiments/switchlingua/config_sentiment_expC.yaml` (324 sentiment scenarios by design).
- Scenarios attempted: **324**; **succeeded ~130**, **failed 194**.
- Quality threshold (per-sentence weighted_score): **7.0**.

> ⚠️ **PARTIAL RUN — daily request quota hit.** From ~scenario 130 onward, every scenario failed with
> OpenAI `429 RateLimitError: requests per day (RPD) Limit 10000, Used 10000`. The **modified pipeline is
> request-heavy** (per-sentence scoring = fluency/naturalness/cs/social *per sentence* + validator + the
> refinement loop ⇒ roughly 50–70 API requests per scenario), so the full 324-scenario run needs **~20k+
> requests — more than one day's 10k cap**. Only the first ~130 scenarios produced data (629 instances).
> This 114-sentence balanced pilot therefore reflects ~40% of the intended scenarios. To grow it: re-run
> after the daily quota resets (and/or across multiple days) and merge, or raise the account's RPD limit.

## Filter funnel (instances)
| stage | count |
|---|--:|
| raw generated instances | 629 |
| TaskValidator passed | 612 |
| + deterministic CS-valid | 172 |
| + quality >= 7.0 | 141 |
| + de-duplicated (KEPT) | 141 |

## Kept per label (pre-balance)
| label | kept |
|---|--:|
| positive | 54 |
| negative | 49 |
| neutral | 38 |

## Balanced training set (down-sampled to smallest label = 38/label)
| label | count |
|---|--:|
| positive | 38 |
| negative | 38 |
| neutral | 38 |
| **TOTAL** | **114** |

## Outputs
- `data/Sentiment/generated/switchlingua_sentiment_train_pilot.csv` / `.jsonl` — **balanced** training pilot.
- `data/Sentiment/generated/switchlingua_sentiment_kept_all.jsonl` — all kept (pre-balance).
- `data/Sentiment/generated/_raw_pipeline/raw_states.jsonl` — raw pipeline output.

## Examples (balanced set)
- [positive/medium] (q=8.0, ar%=70.59) أحب كيف أن دراسة الطب تساعدني على فهم الجسم البشري بشكل أفضل, and I feel so accomplished.
- [positive/low] (q=7.2, ar%=57.14) أنا أحب كيف أن الاستثمار في الأسهم يمكن أن يكون مفيدًا جدًا. It's really exciting to see m
- [positive/high] (q=7.6, ar%=63.63999999999999) عندما أسجل هدف, أشعر أنني في السماء, nothing beats that feeling!
- [neutral/low] (q=7.3, ar%=66.67) أعتقد أن التعليم مهم جدًا, and it opens many أبواب في المستقبل.
- [neutral/high] (q=7.0, ar%=50.0) أحيانًا أفكر في أن أكون لاعب محترف, and I know that practice is important.
- [positive/medium] (q=7.55, ar%=41.18) تجربتي في التعلم عن التمويل الشخصي رائعة. I’m excited to share my knowledge with my friend
- [negative/low] (q=7.3, ar%=63.160000000000004) أنا أشعر بالقلق لأنه في بعض الأحيان لا أستطيع التواصل مع أصدقائي, and it makes me feel rea
- [negative/medium] (q=7.0, ar%=42.11) الديون بتتزايد وأنا مش قادر أتحمل الضغط المالي, I wish I could find a way to manage it bet
- [negative/low] (q=7.0, ar%=35.709999999999994) أنا أستثمر وقتي في الدراسة، but I often wonder if it's worth it.

## Notes / caveats
- **Neutral as factual/descriptive:** the generation prompt is frozen, so neutral-specific wording could not be injected via config (only `intensity`/`ambiguity` flow into sentiment task_constraints). `ambiguity: low` was used for cleaner labels; neutral quality is enforced post-hoc by the TaskValidator filter. Residual risk of mildly-polar 'neutral' remains — recommend a human spot-check before training.
- `on_execute.round` and `shared.style` are inert in the pipeline (not read by any code); dataset size is governed by the config Cartesian product, not those fields.
- Filtering is reproducible offline: `run_expC_sentiment_generation.py --filter-only`.
- **Multi-Agent BERT is NOT trained here** (data generation only).
