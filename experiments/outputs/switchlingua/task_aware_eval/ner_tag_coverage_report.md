# NER Tag Coverage / Generalization (explanatory — does NOT change Test 1)

After the PER-focused prompt repair: does NER work for entity types beyond PER/ORG (adds PRODUCT, EVENT)?

Policy: English-only target entities, allow_code_switched_context, refiner OFF, gpt-4o-mini, English-only judge.

## Group 1 — single-tag capability

| variant | must | count | n | task-correct % (95% CI) | missing (req types) | disallowed | count-valid % | CS % | flu | nat |
|---|---|---|--:|--:|---|--:|--:|--:|--:|--:|
| single_PER | PER | 1-2 | 20 | 30.0 (14.5-51.9) | PER:14/20 | 0 | 95.0 | 95.0 | 8.95 | 8.85 |
| single_ORG | ORG | 1-2 | 20 | 80.0 (58.4-91.9) | ORG:3/20 | 0 | 95.0 | 100.0 | 8.85 | 8.9 |
| single_LOC | LOC | 1-2 | 18 | 44.4 (24.6-66.3) | LOC:9/18 | 0 | 94.4 | 94.4 | 8.5 | 8.39 |
| single_PRODUCT | PRODUCT | 1-2 | 20 | 45.0 (25.8-65.8) | PRODUCT:11/20 | 0 | 95.0 | 90.0 | 9.05 | 8.75 |
| single_EVENT | EVENT | 1-2 | 18 | 38.9 (20.3-61.4) | EVENT:11/18 | 0 | 94.4 | 94.4 | 8.83 | 8.56 |

## Group 2 — pairwise constraints

| variant | must | count | n | task-correct % (95% CI) | missing (req types) | disallowed | count-valid % | CS % | flu | nat |
|---|---|---|--:|--:|---|--:|--:|--:|--:|--:|
| PER_ORG | PER+ORG | 2-3 | 9 | 55.6 (26.7-81.1) | PER:3/9;ORG:1/9 | 0 | 66.7 | 88.9 | 8.78 | 8.56 |
| PER_LOC | PER+LOC | 2-3 | 10 | 70.0 (39.7-89.2) | PER:3/10;LOC:2/10 | 0 | 100.0 | 100.0 | 8.7 | 8.4 |
| ORG_PRODUCT | ORG+PRODUCT | 2-3 | 13 | 46.2 (23.2-70.9) | ORG:3/13;PRODUCT:5/13 | 0 | 76.9 | 100.0 | 8.62 | 8.38 |
| EVENT_LOC | EVENT+LOC | 2-3 | 16 | 25.0 (10.2-49.5) | EVENT:11/16;LOC:2/16 | 0 | 87.5 | 87.5 | 8.44 | 8.44 |
| PER_EVENT | PER+EVENT | 2-3 | 9 | 11.1 (2.0-43.5) | PER:2/9;EVENT:7/9 | 0 | 66.7 | 100.0 | 8.44 | 8.22 |

## How to read

- **Which tags generalize?** High task-correct + low missing for a type = the model produces that English-script entity type reliably. Likely order: ORG/LOC easy, PER moderate (post-repair), PRODUCT/EVENT unknown — this experiment measures them.

- **missing (req types)** shows per-type absence; a type with high 'missing' is the bottleneck in that variant.

- **disallowed** = entities the judge labeled outside the allowed set (should be ~0).

- Small n (18–20) → wide CIs and run-to-run variance; read this as coverage breadth, not precise rates. Does NOT change the main Test 1 NER number.

## Findings (this run)
- **ORG generalizes best** — single_ORG 80% (missing 3/20). The model reliably emits English-script organizations.
- **EVENT is the hardest type** — single_EVENT 39% (missing 11/18), and EVENT drags down every pair it appears in:
  EVENT_LOC 25% (EVENT missing 11/16), PER_EVENT 11% (EVENT missing 7/9). The model rarely produces
  English-script event names.
- **LOC is only moderate (44%)** despite being intuitively easy — likely the same Arabic-script issue as PER:
  the model writes locations in Arabic ("القاهرة") rather than "Cairo", which the English-only policy rejects.
- **PER-only stays hard (30%) but PER-in-pairs is easier** (PER_LOC 70%, PER_ORG 56%) — consistent with the
  earlier non-monotonic finding (more required types/slots → the model actually includes the person).
- **disallowed_type_count = 0 everywhere** → failures are always a MISSING required type, never a wrong type.
- **Recurring root cause:** the model defaults to **Arabic-script** entity names for types with natural Arabic
  forms (PER, LOC, EVENT); the English-only target policy correctly rejects those → high "missing". ORG (and
  to a degree PRODUCT) are produced in Latin script more naturally, so they generalize better.

## Caveats
- **Pairwise group is under-powered** (n=9–16, below the 18 target — the generator returns fewer instances per
  scenario under tight 2-type constraints). Treat Group-2 point estimates as *suggestive*; the CIs are wide.
  Re-run with more topics/scenarios for firmer pairwise numbers if needed.
- Single-tag n=18–20 is also small. Everything is LLM-judged (blind extractor). Does NOT change the Test 1 number.
- Ranking is the robust signal (ORG best → EVENT worst), not the exact percentages.
