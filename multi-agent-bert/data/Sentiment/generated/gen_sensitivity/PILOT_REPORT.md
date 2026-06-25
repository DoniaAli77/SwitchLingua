# GEN sensitivity pilots — report

Generation-config **sensitivity** study (NOT EESA-tailored). Isolated pilots; cross-deduped vs GEN-960 and each other.

## Baseline GEN-960
- {'n': 960, 'ar_pct_mean': 52.7, 'en_pct_mean': 47.3, 'cmi_mean': 40.9, 'cmi_median': 42.1, 'cmi_hist': {'0-20': 17, '20-40': 297, '40-60': 646, '60-80': 0, '80-100': 0}, 'len_mean': 14.1, 'len_median': 14, 'len_p90': 18, 'quality_min': 7.0, 'quality_max': 9.0, 'label_dist': {'negative': 320, 'neutral': 320, 'positive': 320}}

## V1_lowerCS
1. attempted scenarios: 90 (completed 90, failed 0)
2. raw generated: 429
3. valid kept (filters): 88 | after cross-dedup: 88 | balanced pilot: 56
4. label distribution: {'negative': 16, 'positive': 20, 'neutral': 20}
5. CS-valid yield: 29.4%
6. AR:EN ratio: 62.3 : 37.7  (GEN-960 52.7:47.3)
7. CMI mean 32.9 (GEN-960 40.9) | hist {'0-20': 5, '20-40': 31, '40-60': 20, '60-80': 0, '80-100': 0}
8. length mean/median/p90: 14.4/13/18  (GEN-960 14.1/14/18)
9. quality range: 7.0–8.15
10. cross-dup removed: 0
11. vs GEN-960: AR +9.6pp, CMI -8.0, len +0.3
12. examples:
   - [negative/80%] ar%=46.51 cmi=46.51 len=43 | في بعض الأحيان، أشعر بأن المعلمين لا يقدمون ما يكفي من المساعدة، وهذا يجعلني أتساءل عن جود
   - [positive/70%] ar%=53.849999999999994 cmi=46.15 len=25 | أنا أشعر بسعادة كبيرة لأنني أدرس في جامعة مميزة وأتعلم أشياء جديدة كل يوم. It's really exc
   - [positive/70%] ar%=69.23 cmi=30.77 len=13 | أنا أعتقد أن الاستثمار في الأسهم فكرة رائعة, especially for young people مثلنا.
   - [neutral/80%] ar%=60.0 cmi=40.0 len=10 | عندما ألعب مع أصدقائي, we often play football في الحديقة.
   - [neutral/80%] ar%=44.440000000000005 cmi=44.44 len=9 | أنا أتحدث مع فريقي about the upcoming project deadlines.
   - [negative/80%] ar%=60.0 cmi=40.0 len=15 | عندما أذهب إلى العيادة, I often leave feeling more confused مما كنت عليه من قبل.
   - [negative/70%] ar%=73.68 cmi=26.32 len=19 | في الحقيقة, أحيانًا أشعر أن investments اللي أنا مستثمر فيها ما رح تجيب أي فائدة, which is
   - [positive/70%] ar%=92.31 cmi=7.69 len=12 | من خلال ممارسة mindfulness، أستطيع أن أكون أكثر سعادة وراحة في حياتي.
   - [neutral/70%] ar%=30.769999999999996 cmi=30.77 len=13 | التجارب السريرية مهمة جداً, and they play a crucial role in advancing medicine.
   - [neutral/80%] ar%=14.29 cmi=14.29 len=13 | في محاضراتنا, we discuss various medical cases that help us understand real-world applicat

## V2a_register
1. attempted scenarios: 70 (completed 37, failed 33)
2. raw generated: 184
3. valid kept (filters): 76 | after cross-dedup: 76 | balanced pilot: 49
4. label distribution: {'neutral': 20, 'negative': 9, 'positive': 20}
5. CS-valid yield: 52.2%
6. AR:EN ratio: 51.1 : 48.9  (GEN-960 52.7:47.3)
7. CMI mean 43.1 (GEN-960 40.9) | hist {'0-20': 0, '20-40': 9, '40-60': 40, '60-80': 0, '80-100': 0}
8. length mean/median/p90: 12.5/12/16  (GEN-960 14.1/14/18)
9. quality range: 7.0–8.7
10. cross-dup removed: 0
11. vs GEN-960: AR -1.6pp, CMI +2.2, len -1.6
12. examples:
   - [neutral/60%] ar%=58.330000000000005 cmi=41.67 len=12 | أحياناً أجد صعوبة في اختيار ما أريد, especially when everything looks nice.
   - [negative/60%] ar%=58.330000000000005 cmi=41.67 len=12 | أنا أواجه صعوبة في فهم بعض الدروس، which makes me feel frustrated.
   - [positive/60%] ar%=53.849999999999994 cmi=46.15 len=12 | أنا دائماً أحب أن أخرج مع أصدقائي, it's always so much fun!
   - [neutral/50%] ar%=54.55 cmi=45.45 len=10 | اليوم أنا في اجتماع مع الفريق، and we're discussing strategies.
   - [positive/60%] ar%=61.53999999999999 cmi=38.46 len=12 | عندما أسجل هدف، أشعر أنني في قمة السعادة! I can't stop smiling!
   - [positive/60%] ar%=42.86 cmi=42.86 len=14 | الدراسة في الكلية تجعلني أشعر بالتحدي, but I am excited about the opportunities ahead.
   - [neutral/60%] ar%=50.0 cmi=50.0 len=11 | أحب أن أشارك الأفكار مع الآخرين, it's interesting to exchange ideas.
   - [negative/50%] ar%=50.0 cmi=50.0 len=13 | أنا أشعر بالقلق بشأن مشاكلي الصحية الأخيرة، and it’s really hard to ignore.
   - [positive/50%] ar%=43.75 cmi=43.75 len=16 | أنا أستمتع بالتعلم في كل يوم جديد, it really opens up my mind to new possibilities!
   - [neutral/50%] ar%=45.45 cmi=45.45 len=11 | أفكر في كيفية تحسين الأداء، maybe we need a new approach.
