# Dataset Card — Generated TOPIC dataset (540, 9-class)

Balanced Arabic–English code-switched **topic-classification** training data (Experiment T), produced by
the Modified_Version SwitchLingua pipeline (`gpt-4o-mini`). **First dataset generated under the task-aware
pipeline defaults** (task-aware `meet_criteria` routing + `TASK_AWARE_ACCEPT=1` write-time task gating),
unlike the sentiment corpora (240/480/960/V1), which were generated under quality-only acceptance.

## Files
- `merged/switchlingua_topic_train_540_60perlabel.csv`  (UTF-8-SIG, 540 + header)
- `merged/switchlingua_topic_train_540_60perlabel.jsonl` (540)

## Integrity (verified 2026-07-14)
| check | result |
|---|---|
| total examples | **540** (CSV == JSONL) |
| label balance | **60 × 9 classes** (business, education, finance, health, medical, shopping, social, sports, tech) |
| duplicates (normalized text) | **0** |
| CS-valid (recomputed `compute_true_cs_stats`) | **540 / 540** |
| TaskValidator passed | **540 / 540** |
| quality ≥ 7.0 | **540 / 540** (range 7.0 – 9.0) |
| empty metadata fields | none |
| overlap with sentiment corpora | 1 sentence coincides with GEN-960 (cross-task; harmless for topic training — dedup if corpora are ever mixed) |

## Generation
- Config: `experiments/switchlingua/config_topic_expT_v1.yaml` — 9 labels × (cs_ratio [50,60] ×
  tense [Present,Past] × gender [M,F] × age [18-25,26-40] × education [College,Professional]) = **288 scenarios**,
  all completed (manifest `completed_scenarios_topic.json`, 288/288) in a single day, **0 API errors**.
- Tooling: `experiments/switchlingua/manage_topic_data.py` (resume-safe accumulation, isolated
  `data/Topic/generated/` tree, label-dynamic balancing).
- Yield: 629 kept from 288 scenarios (**~2.2 kept/scenario**, CS-valid ~57% — the best-yield track so far;
  topic generation is easier than sentiment and its validator passes nearly everything).
- Filters: non-empty → TaskValidator passed → deterministic CS-valid → quality ≥ 7.0 → dedup
  (+ write-time task-aware acceptance inside the pipeline).
- Pool before balancing: business 64 · education 63 · finance 68 · health 63 · medical 76 · shopping 69 ·
  social 85 · sports 67 · tech 74 (down-sampled to 60/label).

## Intrinsic profile (from `corpus_profile_summary.csv`)
| metric | TOPIC-540 | GEN-960 (sentiment baseline) |
|---|--:|--:|
| tokens | 7,639 | 13,523 |
| AR : EN token % | **50.7 : 49.3** | 53.3 : 46.7 |
| CMI mean | **42.1** | 40.7 |
| switch points mean | 1.19 | 1.43 |
| length mean | ~14.1 | 14.1 |
The topic corpus is the most balanced (AR≈EN) and most mixed (highest CMI) of all generated sets,
with slightly fewer switch points (longer single-switch clauses).

## Composition
- cs_ratio: 50% → 307 · 60% → 233 ; tense: Past 289 / Present 251 ; education: College 286 / Professional 254.
- Schema identical to the sentiment corpora (`text, label, topic, cs_ratio, cs_type, …, quality_score, source`).

## Limitations
- Generator-labeled, LLM-validated (no human verification). Topic validator historically **over-rejects**
  (~17% false-reject in Test 2), so the kept set skews conservative — acceptable for training data.
- Single model (gpt-4o-mini), single language pair, single-turn sentences.
- Label set aligned with the 9 config topics; verify mapping against ArEnTC class names before joint use.

## Status
**NOT yet used for training.** Intended for Multi-Agent BERT topic experiments (e.g., generated-only
transfer vs ArEnTC-trained baseline, and two-stage DAPT-style pretraining, mirroring the sentiment track).
