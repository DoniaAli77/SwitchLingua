# Consolidated Human Annotation — Guidelines

**One sheet, three purposes.** Your ratings support three evaluation claims at once:
(1) task-aware generation quality, (2) masking / per-sentence confirmation,
(3) code-switching validity & linguistic quality. You do **not** need to know which row
serves which purpose — just rate each sentence honestly.

**File:** `consolidated_annotation_sheet.csv` (open in Excel).
If two people annotate, each saves their own copy as `annotator1.csv`, `annotator2.csv`
in this folder, and writes their name/initials in the `annotator_id` column.

---

## Do NOT change these columns (they are metadata)
`sample_id, source_experiment, task, text, target_label, predicted_or_generated_label,
bio_tags_or_entities, pipeline_accepted, pipeline_score, masked_case`

- `task` — topic / sentiment / ner (tells you what the sentence was *supposed* to do).
- `target_label` — the intended label/constraint (e.g. sentiment = "positive";
  topic = "business"; ner = required entity types & count).
- The other metadata is for later analysis — ignore while rating.

---

## Columns YOU fill (every row)

| Column | How to fill |
|---|---|
| `annotator_id` | your name/initials |
| `is_code_switched_yes_no` | **yes** if the sentence genuinely mixes Arabic AND English; **no** if it is essentially all-Arabic or all-English (monolingual) |
| `label_or_task_correct_yes_no` | **yes** if the sentence actually achieves its task `target_label` (see per-task rules below); else **no** |
| `fluency_1_10` | grammar/wording correctness & smoothness — 1 = very bad … 10 = perfect |
| `naturalness_1_10` | does it sound like a real Arabic-English bilingual speaker? 1 … 10 |
| `overall_acceptable_yes_no` | **yes** if you would accept this sentence into a quality code-switching dataset; else **no** |
| `error_type` | main problem, one of: `none, monolingual, wrong_task_label, grammar, unnatural_mixing, wrong_entities, cultural, other` |
| `notes` | optional free text |

### Per-task rule for `label_or_task_correct_yes_no`
- **topic** → does the sentence clearly belong to the `target_label` topic?
- **sentiment** → does the sentence actually express the `target_label` sentiment
  (positive / negative / neutral)?
- **ner** → does the sentence contain the required entity types and count
  (see `target_label`)? Use the NER-specific columns below for detail.

---

## Columns YOU fill ONLY for `task = ner` rows (leave blank otherwise)

| Column | How to fill |
|---|---|
| `entities_correct_yes_no` | **yes** if the named entities present are correct & of the required types |
| `bio_valid_yes_no` | **yes** if entity spans are coherent (no broken/partial entity boundaries); a sentence-level proxy for valid BIO tagging |
| `boundary_correct_yes_no` | **yes** if each entity's start/end covers exactly the full entity (no missing/extra tokens) |

---

## Scales & tips
- **1–10**: 1–3 bad, 4–6 mediocre, 7–8 good, 9–10 excellent. Use the full range.
- Rate each sentence **on its own**, in the order shown (rows are shuffled on purpose).
- A sentence can be fluent yet **monolingual** (mark `is_code_switched=no`,
  `error_type=monolingual`) — that is an important failure to flag.
- A sentence can be fluent and code-switched but have the **wrong sentiment / topic /
  entities** (mark `label_or_task_correct=no`, `error_type=wrong_task_label` or `wrong_entities`).

## What this enables (for reference, not needed while rating)
- Task quality = mean `label_or_task_correct` per task.
- Masking confirmation = compare `masked_case=yes` vs `no` on your scores.
- CS validity = `is_code_switched` rate; pipeline agreement = `pipeline_accepted` vs your `overall_acceptable`.

## Note on two blank metadata columns
`predicted_or_generated_label` and `bio_tags_or_entities` are intentionally empty in this
version (the source sample had the task-validator off). They will be filled from Test 1
outputs later for the pipeline-agreement analysis. **You do not need them.**
