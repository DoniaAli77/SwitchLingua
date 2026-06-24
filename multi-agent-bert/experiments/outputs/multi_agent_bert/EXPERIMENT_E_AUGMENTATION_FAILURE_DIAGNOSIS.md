# Why SwitchLingua-960 Augmentation Did Not Help EESA — Diagnosis

Read-only dataset comparison (no training). Compares EESA train, EESA test, and
generated-960 across style, vocabulary, code-switching, and labels. Date: 2026-06-24.
Script: `scripts/diagnose_augmentation.py`.

## Profile table
| dataset | n | label dist (pos/neg/neu) | avg len | AR ratio | EN ratio | CMI | switches | **test-vocab coverage** |
|---|---|---|---|---|---|---|---|---|
| EESA train | 2,464 | 1092 / 594 / 778 | 12.3 | **0.73** | 0.27 | **23.6** | 1.58 | **0.490** |
| EESA test | 818 | 363 / 197 / 258 | 12.1 | 0.72 | 0.28 | 24.0 | 1.55 | 1.000 |
| **GEN-960** | 960 | **320 / 320 / 320** | 14.1 | **0.53** | **0.47** | **40.7** | 1.43 | **0.098** |

(CMI = code-mixing index; test-vocab coverage = fraction of EESA-test vocabulary
types present in the dataset.)

## The four mismatches (each pushes augmentation the wrong way)

### 1. Vocabulary barely overlaps the real test — the dominant issue
- EESA train covers **49%** of the EESA-test vocabulary; **GEN-960 covers only 9.8%.**
- Label-wise vocab coverage of EESA test: EESA-train ≈ **0.40–0.46**, GEN-960 ≈ **0.08–0.10**
  (Jaccard: train ~0.14–0.17 vs gen ~0.05).
- → The generated text shares almost no words with the real test, so it adds little
  usable signal for EESA and a lot of off-distribution vocabulary.

### 2. Different code-switching regime
- EESA is **Arabic-dominant with light English insertion** (AR 0.73 / EN 0.27, CMI ~24).
- GEN-960 is **near-balanced and heavily mixed** (AR 0.53 / EN 0.47, **CMI 40.7**).
- → The generated CS pattern (lots of English, much higher mixing) is unlike the
  real distribution the test comes from.

### 3. Different register / dialect (MSA-essay vs dialect-social-media)
Top Arabic words reveal the split:
- EESA: dialectal — `مش، اللي، بس، ده، الي، يا` (Egyptian/Levantine social media).
- GEN-960: formal MSA — `أشعر، أحب، عندما، إلى، أن، عن` (standard Arabic, with hamza/diacritics).
Top English words:
- EESA: `love, dislike, https, com` (platform artifacts, links, terse reactions).
- GEN-960: `really, feel, especially, it's` (fluent essay-style English).
- → Generated = clean MSA + fluent English essays; EESA = noisy dialectal social-media.

### 4. Label distribution mismatch
- EESA train **and** test are **positive-heavy** (~44% positive).
- GEN-960 is **perfectly balanced** (33% each).
- → Mixing balanced data into a pos-heavy task shifts the model's class prior away
  from the test distribution (dilutes the positive prior the test rewards).

## Examples
**Generated samples most unlike real EESA** (high OOV vs EESA vocab) — clean,
MSA+fluent-English, topic-themed, intra-sentential switching:
- `مؤخراً, I started using some innovative tools, وكنت متفاجئاً بمدى فعاليتها.`
- `أستمتع بقراءة الكتب، especially those that تتعلق بتطوير الذات.`
- `أحيانًا أذهب shopping مع صديقاتي after classes لتجديد خزانتي.`
- `أحياناً, I regret my investment choices, لأنني أشعر أنني فقدت بعض الفرص.`

**Real EESA-test samples not covered by generated style** (high OOV vs GEN vocab) —
terse, dialectal, noisy, platform artifacts:
- `شخصيتك مجانا free`
- `مين. مدهول عامل unlike`
- `22 الف غبي الي حطو unlike`
- `كتير هبل بلا طعمة clip !!!!`
- `Lyrics بربع جنيه`

## Conclusion — why augmentation gave no benefit
The generated data is effectively a **different genre**: fluent MSA-Arabic +
fluent-English **essay-style** code-switching, **balanced** ar/en with **high CMI**,
themed around topics (tools, investments, shopping, self-development). The real EESA
task is **terse, dialectal, noisy social-media** text that is **Arabic-dominant**
with light English insertions and platform artifacts (`unlike`, `clip`, `Lyrics`,
links), and **positive-heavy**.

Because the two distributions barely overlap (≈10% test-vocab coverage, opposite CS
regime, MSA vs dialect, balanced vs pos-heavy), adding 960 generated samples to
2,463 real samples **does not teach the model about the real test distribution** — it
adds off-distribution signal and shifts the class prior, so the controlled effect was
**slightly negative (−0.012 acc, within seed noise)** rather than helpful. This is
consistent with generated data being a decent *standalone* source (it carries real
sentiment signal, C1–C3 reach 0.59–0.67) but a poor *augmenter* of an already-strong
real-data model.

## Interpretation — a domain-compatibility finding (NOT a call to overfit to EESA)
This result should be read as a statement about **domain/register compatibility**,
not as a deficiency of SwitchLingua or a reason to redesign the generator for EESA:

- **SwitchLingua generation is config-controlled and worked as intended** — it
  produced valid, CS-valid sentiment data (960/960 CS-valid, TaskValidator-passed,
  quality ≥ 7). The data carries real sentiment signal.
- **As standalone training data it is useful and scales** (C1→C2→C3: 0.59 → 0.65 →
  ~0.67 mean; the multi-agent pipeline lifts it to 0.75).
- **Simple augmentation of the *full* real EESA train did not help** because the
  generated data and EESA occupy **different domains/registers** (MSA-essay,
  balanced/high-CMI, themed vs dialectal social-media, Arabic-dominant, pos-heavy) —
  not because the generation is wrong.
- **This is not a reason to tailor the generator to EESA.** Overfitting SwitchLingua
  to one corpus would defeat its purpose as a *general* CS data generator.
- **The real lesson:** augmentation benefit depends on (a) **domain compatibility**
  between generated and target data, and (b) **how much real labeled data is already
  available**. When the target already has thousands of in-domain labeled examples
  (full EESA), generic out-of-domain CS data adds little. Its value is expected to be
  largest in **low-resource** settings, where real labeled data is scarce — which is
  the next experiment to run (EESA 10% / 25% / 50% ± SwitchLingua-960).
