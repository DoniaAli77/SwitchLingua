# Test 2 — TaskValidatorAgent Necessity & Effectiveness

Reference task-correctness: **Test 1 blind LLM judge** · Validator: **real TaskValidatorAgent (gpt-4o-mini)** · quality threshold (weighted score) = **7.0** · evaluated **115** sentences.

Policy **A: quality_only** (accept if quality>=threshold) vs **B: quality + TaskValidator** (accept if quality passes AND the real validator passes).

## Overall (vs reference)

| metric | A: quality_only | B: quality + validator |
|---|---|---|
| accepted | 86 (74.8%) | 62 (53.9%) |
| task-correct among accepted | 70.9% | 85.5% |
| **task-WRONG accepted (false accepts)** | **25** | **9** |
| false-accept rate (of wrong) | 78.1% | 28.1% |
| false-reject rate (of correct) | 26.5% | 36.1% |

## The validator as a task-correctness detector (vs reference)

- agreement 78.3% · precision 83.0% · recall 88.0%  (TP 73, FP 15, FN 10, TN 17)

## Per task

| task | policy | accepted | task-correct among accepted | task-wrong accepted | false-accept % |
|---|---|---|---|---|---|
| topic | quality_only | 33 | 100.0% | 0 | None% |
| topic | quality_plus_validator | 33 | 100.0% | 0 | None% |
| sentiment | quality_only | 22 | 77.3% | 5 | 45.5% |
| sentiment | quality_plus_validator | 22 | 77.3% | 5 | 45.5% |
| ner | quality_only | 31 | 35.5% | 20 | 95.2% |
| ner | quality_plus_validator | 7 | 42.9% | 4 | 19.0% |

## High-quality but task-wrong (quality_only accepts these)

**23** sentences pass quality AND have fluency>=8 & naturalness>=8 but are task-WRONG; the real validator caught **14** of them. Examples:

- sentiment (q=8.4, flu 8.0/nat 8.0, validator_passed=True): في الكلية, I often collaborate مع زملائي على مشاريع تقنية.
- sentiment (q=7.3, flu 9.0/nat 8.0, validator_passed=True): أنا أشجع فريقي المفضل في الدوري, and I hope they win this season.
- sentiment (q=7.0, flu 8.0/nat 8.0, validator_passed=True): في كل يوم، أمارس الرياضة قليلاً، because it helps me feel more energized.
- sentiment (q=7.1, flu 9.0/nat 8.0, validator_passed=True): أستمتع بتجربة الأجهزة الجديدة. They always seem to have features that make tasks more convenient.
- sentiment (q=7.0, flu 8.0/nat 8.0, validator_passed=True): أحب كيف أن التكنولوجيا تتطور بسرعة, because it offers new opportunities for learning.
- ner (q=7.3, flu 9.0/nat 8.0, validator_passed=False): أنا أحب ممارسة كرة السلة مع أصدقائي, especially when we play at the Los Angeles courts.
- ner (q=7.8, flu 8.0/nat 8.0, validator_passed=False): أنا أشرب الكثير من الماء كل يوم. When I visited Cairo, I learned عن أهمية hydration من خلال محاضرة ف
- ner (q=7.55, flu 9.0/nat 9.0, validator_passed=False): أنا أشجع فريق Manchester United في كرة القدم. في كل مباراة، أستمتع بمشاهدتهم يلعبون ضد Chelsea.

## Interpretation

- Quality-only admits **25** task-wrong sentences (false-accept 78.1% of all wrong) — they look fluent/natural, so quality cannot separate them.

- Adding the real TaskValidator cuts task-wrong accepted to **9** (false-accept 28.1%), raising precision from 70.9% to 85.5% — at a false-reject cost of 36.1%.

- The validator is imperfect (precision 83.0%, recall 88.0% vs the reference), so the gain is real but not total. Reference can be swapped for human labels via --labels.
