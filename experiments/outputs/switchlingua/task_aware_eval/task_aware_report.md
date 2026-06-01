# Test 1 — Task-Aware Generation Quality (automated, no humans yet)

**Source (pre-refinement, refiner OFF):** `C:\Users\Eng.Donia\Documents\matser\SwitchLingua\experiments\outputs\switchlingua\per_sentence\validation_raw\Arabic.jsonl`  
**Per task:** 40 sentences

## Results

| Task | n | Task-correct % | (method) | CS-valid % | CS-ratio MAE vs 70 | Fluency | Naturalness |
|------|--:|--:|:--|--:|--:|--:|--:|
| topic | 40 | 100.0 | blind relevance (LLM) | 100.0 | 23.27 | 8.35 | 8.07 |
| sentiment | 40 | 70.0 | blind re-classification (LLM) | 87.5 | 22.06 | 8.05 | 8.05 |
| ner | 35 | 62.9 | constraint-aware entity extraction (LLM JSON) + deterministic check | 97.1 | 13.84 | 8.4 | 8.54 |

## Notes on rigor

- **CS validity and CS-ratio are deterministic/objective** (compute_true_cs_stats; no LLM, no circularity).

- **Task correctness (sentiment/topic/NER) is automated by a BLIND gpt-4o-mini judge** not shown the target (sentiment = re-classification; topic = relevance; NER = entity extraction + deterministic constraint check). Less circular than the in-pipeline validator but still LLM-based; **human confirmation is pending** via `human_eval/consolidated_annotation_sheet.csv`.

- The English-only deterministic NER policy was deliberately NOT used (it ignores Arabic-script entities this task permits, giving unfairly low scores).

- **The NER judge is CONSTRAINT-AWARE**: allowed entity types, the min/max count range, the must-include types, and the script policy (Arabic/English/mixed) are generated dynamically from the sample's task constraints — nothing is hardcoded. The judge returns strict JSON; validation is deterministic (per-sentence fields: entity_counts, total_entities, missing_required_types, disallowed_types, count_valid, parse_error).

- Fluency/naturalness are the pipeline's own per-sentence judge scores (for reference).

- Sample reuses the fresh pre-refinement validation set; no regeneration.
