# NER Tag Coverage / Generalization (explanatory — does NOT change Test 1)

After the PER-focused prompt repair: does NER work for entity types beyond PER/ORG (adds PRODUCT, EVENT)?

Policy: English-only target entities, allow_code_switched_context, refiner OFF, gpt-4o-mini, English-only judge.

## Group 1 — single-tag capability

| variant | must | count | n | task-correct % (95% CI) | missing (req types) | disallowed | count-valid % | CS % | flu | nat |
|---|---|---|--:|--:|---|--:|--:|--:|--:|--:|
| single_LOC | LOC | 1-2 | 19 | 63.2 (41.0-80.9) | LOC:3/19 | 0 | 73.7 | 89.5 | 8.74 | 8.63 |
| single_PRODUCT | PRODUCT | 1-2 | 20 | 85.0 (64.0-94.8) | PRODUCT:3/20 | 0 | 100.0 | 100.0 | 8.55 | 8.65 |
| single_EVENT | EVENT | 1-2 | 19 | 84.2 (62.4-94.5) | EVENT:1/19 | 0 | 89.5 | 100.0 | 8.89 | 8.58 |

## Group 2 — pairwise constraints

| variant | must | count | n | task-correct % (95% CI) | missing (req types) | disallowed | count-valid % | CS % | flu | nat |
|---|---|---|--:|--:|---|--:|--:|--:|--:|--:|
| EVENT_LOC | EVENT+LOC | 2-3 | 18 | 83.3 (60.8-94.2) | EVENT:1/18;LOC:1/18 | 0 | 83.3 | 100.0 | 8.5 | 8.61 |
| PER_EVENT | PER+EVENT | 2-3 | 18 | 22.2 (9.0-45.2) | PER:12/18;EVENT:5/18 | 0 | 66.7 | 100.0 | 8.72 | 8.5 |

## How to read

- **Which tags generalize?** High task-correct + low missing for a type = the model produces that English-script entity type reliably. Likely order: ORG/LOC easy, PER moderate (post-repair), PRODUCT/EVENT unknown — this experiment measures them.

- **missing (req types)** shows per-type absence; a type with high 'missing' is the bottleneck in that variant.

- **disallowed** = entities the judge labeled outside the allowed set (should be ~0).

- Small n (18–20) → wide CIs and run-to-run variance; read this as coverage breadth, not precise rates. Does NOT change the main Test 1 NER number.
