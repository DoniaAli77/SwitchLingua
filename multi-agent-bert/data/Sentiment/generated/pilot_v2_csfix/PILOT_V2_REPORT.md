# Pilot v2 (CS-validity fix) — Results

Config-only change: `cs_ratio: [50%,60%,70%]`, `cs_type: [Intrasentential]`. No prompt/NER/pipeline change. Isolated run — **not merged** into the training set.

## Run
- scenarios requested: **40** | completed: **6** | failed (429/other): **34**
- quality threshold: **7.0** (unchanged)

## Yield vs baseline
| metric | baseline (v1) | pilot v2 |
|---|--:|--:|
| CS-valid rate (of non-empty) | 30% | **43%** (12/28) |
| fully-Arabic share of failures | 99.6% | **100%** (16/16) |

## CS-valid rate by cs_ratio target
| cs_ratio | CS-valid |
|---|---|
| 60% | 7/10 (70%) |
| 70% | 5/18 (28%) |

## Filter funnel (instances)
| stage | count |
|---|--:|
| raw generated | 28 |
| kept (validator+CS-valid+quality>=7.0+dedup) | 10 |

**Filtering loss by reason:**
| reason | count |
|---|--:|
| empty_text | 0 |
| validator_failed | 0 |
| not_cs_valid | 16 |
| low_quality | 2 |
| duplicate | 0 |
| duplicates removed | 0 |

## Kept by label
| label | kept |
|---|--:|
| positive | 0 |
| negative | 7 |
| neutral | 3 |

## Kept by topic
| topic | kept |
|---|--:|
| tech | 6 |
| finance | 4 |

## Examples of newly-valid code-switched outputs
- [negative/60%/high] ar%=61.9 q=7.5 | أنا أشعر بالإحباط لأن التكنولوجيا الجديدة التي اشتريتها لا تعمل كما كان متوقعًا, I thought it w
- [negative/60%/high] ar%=52.629999999999995 q=7.15 | أحيانًا أشعر أن الأجهزة الحديثة تفشل في تحقيق ما نحتاجه, it's really frustrating when they don'
- [negative/60%/high] ar%=58.81999999999999 q=7.5 | أنا أشعر بالاستياء من المنتجات التكنولوجية التي تروج لها الشركات, they never live up to the hyp
- [neutral/70%/high] ar%=50.0 q=7.0 | أنا أستخدم التكنولوجيا الجديدة في دراستي, and I find it really helpful.
- [neutral/70%/high] ar%=60.0 q=7.2 | أنا أستخدم التطبيقات المختلفة لتنظيم وقتي, especially for my assignments.
- [neutral/70%/high] ar%=54.55 q=7.45 | في الجامعة, I often discuss technology trends مع أصدقائي, ونتبادل الأفكار.
- [negative/60%/medium] ar%=66.67 q=7.35 | أنا صراحة مش مرتاح مع الوضع المالي الحالي, it's really stressful.
- [negative/60%/medium] ar%=43.75 q=7.2 | الأمور المالية بدأت تأثر على حياتي اليومية, I just can't seem to manage my expenses.
- [negative/60%/medium] ar%=58.330000000000005 q=7.25 | الديون تتزايد وأنا مش قادر أتحملها، it feels like I'm غارق.
- [negative/60%/medium] ar%=50.0 q=7.2 | حاولت أن أستثمر ولكن النتائج كانت مخيبة للآمال. I feel like I've lost my money.

## Verdict
- CS-valid 30% → **43%** (IMPROVED).
- **Not merged** into the main training set (per instruction; merge only if clearly better).
- Multi-Agent BERT not trained from this.