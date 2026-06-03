# NER Tag Coverage / Generalization (explanatory — does NOT change Test 1)

After the PER-focused prompt repair: does NER work for entity types beyond PER/ORG (adds PRODUCT, EVENT)?

Policy: English-only target entities, allow_code_switched_context, refiner OFF, gpt-4o-mini, English-only judge.

## Group 1 — single-tag capability

| variant | must | count | n | task-correct % (95% CI) | missing (req types) | disallowed | count-valid % | CS % | flu | nat |
|---|---|---|--:|--:|---|--:|--:|--:|--:|--:|

## Group 2 — pairwise constraints

| variant | must | count | n | task-correct % (95% CI) | missing (req types) | disallowed | count-valid % | CS % | flu | nat |
|---|---|---|--:|--:|---|--:|--:|--:|--:|--:|
| PER_ORG | PER+ORG | 2-3 | 20 | 25.0 (11.2-46.9) | PER:12/20;ORG:5/20 | 0 | 55.0 | 100.0 | 8.5 | 8.3 |

## How to read

- **Which tags generalize?** High task-correct + low missing for a type = the model produces that English-script entity type reliably. Likely order: ORG/LOC easy, PER moderate (post-repair), PRODUCT/EVENT unknown — this experiment measures them.

- **missing (req types)** shows per-type absence; a type with high 'missing' is the bottleneck in that variant.

- **disallowed** = entities the judge labeled outside the allowed set (should be ~0).

- Small n (18–20) → wide CIs and run-to-run variance; read this as coverage breadth, not precise rates. Does NOT change the main Test 1 NER number.
