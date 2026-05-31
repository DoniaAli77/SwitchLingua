# Test 1 — Error Analysis

Input: `task_aware_details.jsonl` · sentiment=40, ner=35, total=115

Goal: locate failures (prompt vs model vs evaluator artifact). No prompts changed.

## 1. Sentiment errors

Per-label accuracy: positive=100.0%, negative=100.0%, neutral=15.4%

Confusion matrix (rows = target, cols = blind predicted):

| target \ predicted | positive | negative | neutral |
|---|---|---|---|
| positive | 12 | 0 | 0 |
| negative | 0 | 15 | 0 |
| neutral | 11 | 0 | 2 |

**Incorrect sentiment examples:**
- target=`neutral` predicted=`positive` (flu 8.0/nat 8.0): في الكلية, I often collaborate مع زملائي على مشاريع تقنية.
- target=`neutral` predicted=`positive` (flu 8.0/nat 8.0): أعتقد أن المعرفة في finance مفيدة جداً في الحياة اليومية, especially when it comes to budgeting.
- target=`neutral` predicted=`positive` (flu 8.0/nat 8.0): أحب تناول الفواكه والخضروات, because they provide essential nutrients for my body.
- target=`neutral` predicted=`positive` (flu 9.0/nat 8.0): أنا أشجع فريقي المفضل في الدوري, and I hope they win this season.
- target=`neutral` predicted=`positive` (flu 8.0/nat 8.0): في كل يوم، أمارس الرياضة قليلاً، because it helps me feel more energized.
- target=`neutral` predicted=`positive` (flu 9.0/nat 8.0): أستمتع بتجربة الأجهزة الجديدة. They always seem to have features that make tasks more convenient.

## 2. NER errors

required: 2-3 entities incl. PER+ORG (read from source data). Failed 19/35.

Failure categories (a case can have several):

| category | count |
|---|---|
| no_entities | 0 |
| too_few | 7 |
| too_many | 2 |
| missing_PER | 9 |
| missing_ORG | 10 |

**Failed NER examples:**
- counts={'PER': 0, 'ORG': 0, 'LOC': 1} total=1 flags=['too_few', 'missing_PER', 'missing_ORG']: أنا أحب ممارسة كرة السلة مع أصدقائي, especially when we play at the Los Angeles courts.
- counts={'PER': 0, 'ORG': 1, 'LOC': 1} total=2 flags=['missing_PER']: أنا أشرب الكثير من الماء كل يوم. When I visited Cairo, I learned عن أهمية hydration من خلال محاضرة في جامعة ال...
- counts={'PER': 1, 'ORG': 2, 'LOC': 1} total=4 flags=['too_many']: أنا أعتقد أن health هو أهم شيء في الحياة. I often read articles from WebMD و Healthline التي تساعدني في فهم كي...
- counts={'PER': 2, 'ORG': 0, 'LOC': 0} total=2 flags=['missing_ORG']: أنا أشجع فريق Manchester United في كرة القدم. في كل مباراة، أستمتع بمشاهدتهم يلعبون ضد Chelsea.
- counts={'PER': 0, 'ORG': 0, 'LOC': 1} total=1 flags=['too_few', 'missing_PER', 'missing_ORG']: أحب الذهاب إلى صالة الألعاب الرياضية في شيكاغو, حيث ألتقي بأصدقائي هناك.
- counts={'PER': 0, 'ORG': 1, 'LOC': 0} total=1 flags=['too_few', 'missing_PER']: أنا أحاول أن أتناول طعام صحي. I found a great recipe on BBC Good Food that علمتني كيف أعد وجبات صحية.

## 3. Quality vs task mismatch

- mean fluency: correct=8.21 vs wrong=8.4
- mean naturalness: correct=8.15 vs wrong=8.37
- **fluency>=8 AND naturalness>=8 but task WRONG: 27 cases (90.0% of all wrong)** — quality scores do not separate task success.

**High-quality-but-wrong examples:**
- sentiment target=`neutral` flu 8.0/nat 8.0: في الكلية, I often collaborate مع زملائي على مشاريع تقنية.
- sentiment target=`neutral` flu 8.0/nat 8.0: أعتقد أن المعرفة في finance مفيدة جداً في الحياة اليومية, especially when it comes to budgeting.
- sentiment target=`neutral` flu 8.0/nat 8.0: أحب تناول الفواكه والخضروات, because they provide essential nutrients for my body.
- sentiment target=`neutral` flu 9.0/nat 8.0: أنا أشجع فريقي المفضل في الدوري, and I hope they win this season.
- sentiment target=`neutral` flu 8.0/nat 8.0: في كل يوم، أمارس الرياضة قليلاً، because it helps me feel more energized.
- sentiment target=`neutral` flu 9.0/nat 8.0: أستمتع بتجربة الأجهزة الجديدة. They always seem to have features that make tasks more convenient.

## 4. CS ratio control (target = 70% Arabic)

- within ±10: 26.1% · within ±15: 37.4% · within ±20: 55.7%
- mean abs error: 19.98 · median: 20.0

**High CS-validity but poor ratio control (|err|>20):**
- topic ar_ratio=41.67% (err 28.3): التواصل مع الزملاء أمر مهم، especially when we work on team projects.
- topic ar_ratio=38.46% (err 31.5): أحب العمل الجماعي في المشاريع. Collaborating with different people always brings fresh ideas.
- topic ar_ratio=40.0% (err 30.0): في عالم الأعمال اليوم، networking is really important to succeed.
- topic ar_ratio=30.769999999999996% (err 39.2): في عالم الأعمال اليوم، I feel like we need to be more innovative.
- topic ar_ratio=38.46% (err 31.5): التسويق الرقمي أصبح شيئاً أساسياً، and I really enjoy learning about new strategies.
- topic ar_ratio=41.67% (err 28.3): أحب التعلم عن الثقافات المختلفة. I think it really broadens my perspective.

## Where do the failures come from? (reading)

- **Sentiment**: check the confusion matrix — if errors cluster on one label (e.g. neutral<->positive), it is likely subjectivity / evaluator boundary, not pure generation failure.

- **NER**: if failures are dominated by `missing_PER`/`missing_ORG` or `too_few`, the model is under-producing required entities (prompt/model), not an evaluator artifact.

- **Quality mismatch**: high-quality-but-wrong cases prove quality scoring is blind to task success (motivates task-aware validation).

- **CS ratio**: low within-±10 shows the model cannot self-regulate the requested proportion (model limitation), motivating deterministic CS-ratio control.
