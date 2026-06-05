# CS-ratio Measurement Validation — PARTIAL (Test 4)

**Measurement-method comparison on a FIXED 30-sentence set — NOT an Original-vs-Modified generation comparison, and no generation pipeline was run.** Methods: (1) our deterministic counter `compute_true_cs_stats`, (2) original-style LLM-only counting (gpt-4o-mini, temperature 0.7, 3 repeats), (3) human counts (PENDING).

## Status
- **Human-reference accuracy metrics are PENDING** (manual token counts not yet filled).
- This partial run evaluates **LLM instability** and **deterministic-vs-LLM disagreement** only.
- **Final accuracy claims (MAE, detection accuracy, boundary error) require the manual human token counts.**

## Results (now)

| metric | value |
|---|---|
| n_sentences | 30 |
| llm_model | gpt-4o-mini |
| llm_temperature | 0.7 |
| llm_repeats | 3 |
| human_counts | PENDING (blank) |
| deterministic_variance | 0 (exact, reproducible, free) |
| mean_llm_std_arabic_count | 0.598 |
| mean_llm_std_arabic_ratio_pct | 2.317 |
| sentences_llm_repeats_disagree | 12/30 |
| det_vs_llm_is_cs_mismatch | 0/30 |
| mean_det_vs_llm_arabic_ratio_abs_diff_pct | 5.04 |
| monolingual_edge_cases | 2 |
| det_monolingual_correct | 2/2 |
| llm_monolingual_correct | 2/2 |

## LLM instability — the 3 repeats disagree on these
- `CS003` (modified/topic): ar_count std=1.247, ratio std=6.745%  | det ar%=57.14 vs llm_mean=54.12  | أحتاج إلى بعض النصائح حول كيفية إدارة وقتي. Can you sha
- `CS012` (modified/ner): ar_count std=0.816, ratio std=6.736%  | det ar%=100.0 vs llm_mean=95.24  | أحب الذهاب إلى صالة الألعاب الرياضية في شيكاغو, حيث ألت
- `CS014` (modified/ner/masked): ar_count std=1.247, ratio std=6.508%  | det ar%=81.25 vs llm_mean=77.7  | في دراستي في الجامعة، تعلمت عن الأمان السيبراني وكنت أق
- `CS029` (english_with_arabic_phrase): ar_count std=0.943, ratio std=4.832%  | det ar%=30.77 vs llm_mean=29.91  | The restaurant was excellent and honestly كان السعر منا
- `CS005` (modified/sentiment): ar_count std=0.816, ratio std=2.599%  | det ar%=91.67 vs llm_mean=74.74  | يساعدني هذا المجال على فهم الأسواق better و كيفية اتخاذ
- `CS028` (arabic_with_english_event): ar_count std=0.943, ratio std=1.57%  | det ar%=81.82 vs llm_mean=82.22  | أنا متحمس جداً لحضور World Cup هذا العام مع عائلتي وأصد

## Deterministic vs LLM disagreement
- `CS005` det(is_cs=True, ar%=91.67) vs llm(is_cs=True, ar%=74.74)  | يساعدني هذا المجال على فهم الأسواق better و كيفية 
- `CS030` det(is_cs=True, ar%=88.89) vs llm(is_cs=True, ar%=72.62)  | في عام 2024، اشترينا 3 أجهزة laptops بسعر $1500 لك

## Monolingual detection on controlled edge cases
- fully-Arabic / fully-English cases: deterministic correct **2/2**, LLM correct **2/2**.

## Method notes
- Deterministic counter = `compute_true_cs_stats` (Arabic vs Latin script token counts): **0 variance, no API call, exact and reproducible**. 'other' is a derived count (numbers/symbols) for the 3-way view.
- LLM ar_ratio = arabic/(arabic+english), matching the deterministic ratio definition.
- When human counts are filled, re-running adds: Arabic/English/other token-count MAE, ratio MAE, monolingual + code-switch detection accuracy, and boundary error rate vs the human reference.
