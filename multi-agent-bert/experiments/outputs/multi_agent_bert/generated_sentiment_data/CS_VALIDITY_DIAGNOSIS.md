# CS-Validity Failure Diagnosis (Experiment C sentiment generation)

Scope: **690** generated instances across pilot_v1 + daily runs (690 non-empty). READ-ONLY diagnosis — no prompts/config changed.

## Headline
- Non-empty instances: **690** | CS-valid: **209** (30%) | CS-FAIL: **481** (70%)
- Failure breakdown: fully-Arabic **479**, fully-English **2**, no-letters **0**, empty **0**

## Q1–Q5 nature of failures
1. **Fully Arabic (zero English tokens):** 479 (100% of failures) — the dominant failure mode.
2. **Fully English (zero Arabic tokens):** 2 (0% of failures).
3. **Mostly-Arabic / too-few-English:** failures have **zero** English by definition; among the CS-VALID ones, **12/209** are *fragile* (exactly 1 English token) — i.e. one dropped word away from failing. Mean Arabic share of valid CS sentences = **56.3%**.
4. **English written in Arabic script:** 479/479 fully-Arabic failures contain **no Latin characters at all** (purely Arabic script). Where the sentiment clearly intends an English insertion, it was transliterated into Arabic letters → not counted as English. (See examples.)
5. **Is 70% Arabic too Arabic-heavy?** Mean Arabic share = **86.5%** over all non-empty, **56.3%** over valid CS — **above the 70% target**, so the matrix is overshooting Arabic, leaving little room for (or zero) English tokens in short single sentences.

## Q6 Is neutral more likely to go monolingual? / Q7 failure rate by label
| label | CS-fail rate |
|---|---|
| positive | 191/249 (77%) |
| negative | 165/240 (69%) |
| neutral | 125/201 (62%) |

## Q8 failure rate by topic
| topic | CS-fail rate |
|---|---|
| health | 51/64 (80%) |
| education | 67/87 (77%) |
| tech | 59/78 (76%) |
| medical | 54/75 (72%) |
| business | 47/67 (70%) |
| shopping | 66/97 (68%) |
| finance | 46/69 (67%) |
| social | 39/60 (65%) |
| sports | 52/93 (56%) |

## Q9 failure rate by cs_type / cs_function / intensity

**cs_type:**
| value | CS-fail rate |
|---|---|
| Intersentential | 260/350 (74%) |
| Intrasentential | 221/340 (65%) |

**cs_function:**
| value | CS-fail rate |
|---|---|
| Expressive | 481/690 (70%) |
_(only one value present in the data — no variation to compare)_

**intensity:**
| value | CS-fail rate |
|---|---|
| low | 184/249 (74%) |
| medium | 147/207 (71%) |
| high | 150/234 (64%) |

## Q10 — 10 not_cs_valid failure examples (with explanation)
1. [neutral/Intrasentential/high] ar=9 en=0 (fully Arabic, 0 English tokens)
   `اليوم رحت للتسوق مع صديقاتي, ولقينا بعض العروض المدهشة.`
2. [negative/Intrasentential/low] ar=14 en=0 (fully Arabic, 0 English tokens)
   `أنا أشعر أن الوضع المالي اليوم ليس جيداً، وأشعر أنني أواجه صعوبة في مصاريفي.`
3. [negative/Intrasentential/low] ar=10 en=0 (fully Arabic, 0 English tokens)
   `في الحقيقة, خطة التوفير الخاصة بي ليست كما كنت آمل.`
4. [negative/Intrasentential/low] ar=13 en=0 (fully Arabic, 0 English tokens)
   `أنا أكره كيف أن الديون تتزايد, وأشعر أنني أفقد السيطرة على أموري المالية.`
5. [negative/Intrasentential/low] ar=11 en=0 (fully Arabic, 0 English tokens)
   `بالرغم من محاولاتي, ميزانيتي دائماً تبدو غير كافية لتغطية جميع النفقات.`
6. [neutral/Intrasentential/high] ar=11 en=0 (fully Arabic, 0 English tokens)
   `عندما أكون في السوق, أشعر بالمتعة, حتى لو لم أشتري الكثير.`
7. [neutral/Intrasentential/high] ar=11 en=0 (fully Arabic, 0 English tokens)
   `في الكلية, انضممت إلى فريق كرة القدم لأنني أريد تحسين مهاراتي.`
8. [negative/Intrasentential/low] ar=13 en=0 (fully Arabic, 0 English tokens)
   `الأسعار ترتفع بشكل مستمر، وأتساءل إذا كنت قد اتخذت الخيارات الصحيحة في استثماراتي.`
9. [positive/Intersentential/high] ar=17 en=0 (fully Arabic, 0 English tokens)
   `أنا أدرس في الجامعة وأشعر أن التعليم يفتح لي آفاق جديدة. من الرائع كم أتعلم كل يوم!`
10. [positive/Intersentential/high] ar=9 en=0 (fully Arabic, 0 English tokens)
   `الأساتذة هنا يدعمونني بشكل كبير، وأنا ممتن جداً لتوجيهاتهم.`
## Diagnosis summary (root cause)
- **70% of generated sentences are 100% Arabic** (479/690), zero English tokens — that single mode IS the
  yield problem. Fully-English (2) and Arabic-script-English (0 Latin chars) are non-issues.
- When the model **does** code-switch, output is healthy (**56% Arabic** mean, only 6% fragile) — so the
  generator *can* code-switch; it just frequently emits pure Arabic instead.
- The `70%` Arabic target biases "predominantly Arabic" → tips many short single sentences to all-Arabic
  (mean Arabic over ALL = **86.5%**, far above 70%).
- **Intersentential** cs_type is worse (74% vs 65% Intra): switching *between* sentences makes a
  single-sentence instance monolingual by design — mismatched with a *per-sentence* CS-validity filter.
- Label myth busted: **neutral is the BEST** (62% fail), **positive the WORST** (77%) — neutral is not the
  monolingual culprit.

## Recommended SMALLEST fix (config-only — NOT applied; awaiting approval)
Ranked by leverage, all within the allowed "config only" lane (no prompt/NER/pipeline change):

1. **Add lower cs_ratio values (e.g., 60% and/or 50%).** Most direct config lever. Failures are all
   too-Arabic; valid sentences already sit at ~56% Arabic. A 50–60% target pushes more English tokens per
   sentence. `pre_execute.cs_ratio: ["70%","60%","50%"]` (also multiplies scenarios — adds diversity).
2. **Drop `Intersentential` from `cs_type`** (keep `Intrasentential` only). Intrasentential = switch *within*
   a sentence ⇒ both languages present per instance, which is exactly what per-sentence CS-validity wants.
   Removes the 74%-fail bucket.

**Do NOT:**
- **Change the filter threshold / CS-validity definition** — *not justified*. The filter is correctly
  rejecting genuinely monolingual sentences; loosening it would pollute a *code-switching* dataset with
  non-code-switched data, defeating the purpose. Quality threshold is a separate filter and is not the cause.
- **Change prompts** — out of scope (frozen).

**Honest caveat:** config-only changes may *partially* help, because the model sometimes ignores the ratio
and emits pure Arabic regardless. The highest-leverage fix is a **generation-instruction requiring ≥1
English insertion per sentence** (a scoped prompt change) — currently frozen. If 50–60% cs_ratio +
Intrasentential-only doesn't lift CS-valid yield enough, lifting the freeze for that one targeted prompt
line is the next step.
