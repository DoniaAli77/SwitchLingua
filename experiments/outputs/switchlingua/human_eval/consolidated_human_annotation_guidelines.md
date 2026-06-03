# Consolidated Human Annotation — Guidelines

**One sheet, many claims.** Your ratings validate the remaining AI-judged results: task-aware generation
quality (topic/sentiment/NER), the masking claim, code-switch validity, and a small CS-ratio token-count
check. You do NOT need to know which row serves which purpose — just rate each sentence honestly.

**File:** `consolidated_human_annotation_sheet.csv` (open in Excel). If two people annotate, each saves a
copy as `annotator1.csv`, `annotator2.csv` in this folder.

## ⚠️ Most important rule
The sheet has **pre-filled columns** (the AI's labels, validator pass/fail, scores). **Rate each sentence
on its own first — do NOT let those columns influence you.** Ideally hide columns
`pipeline_task_correct_or_judge_label` … `masked_case` while annotating. The whole point is your
*independent* judgement.

---

## Columns YOU fill on EVERY row
| Column | How |
|---|---|
| `human_task_correct` (yes/no) | Does the sentence achieve its task for `target_label`? (topic = on that topic; sentiment = expresses that sentiment; NER = meets the entity requirement in `task_constraints`) |
| `human_cs_valid` (yes/no) | Does it genuinely MIX Arabic AND English? (no = essentially all one language) |
| `human_fluency_1_5` | Grammar/wording: 1 very bad … 5 perfect |
| `human_naturalness_1_5` | Sounds like a real Arabic-English bilingual? 1 … 5 |
| `human_overall_acceptable` (yes/no) | Would you accept it into a quality code-switching dataset? |
| `human_error_type` | one of: `none, monolingual, wrong_task_label, grammar, unnatural_mixing, wrong_entities, cultural, other` |
| `human_notes` | optional |

## Sentiment rows (task = sentiment)
| `human_sentiment_label` | your independent read: **positive / negative / neutral** |

Some sentiment rows are marked **SENTIMENT-DISPUTED** (the AI judge disagreed with the target). Your label
on these is what *resolves* the dispute — please read them carefully.

## NER rows (task = ner) — entities must be ENGLISH-script (Latin letters)
| Column | How |
|---|---|
| `human_entities_present` (yes/no) | Are there any named entities at all? |
| `required_entity_types_present` (yes/no) | Are the **required types** (see `task_constraints` → must) present? |
| `required_entities_english_script` (yes/no) | Are the **required** entities written in **English/Latin letters** (e.g. "Cairo", not "القاهرة")? |
| `human_ner_correct` (yes/no) | Overall: required types present AND in English script AND count in range? |

## CS-ratio subset (rows marked `[CS-RATIO]` in notes_for_annotator)
Count the word tokens by language (split on spaces; ignore punctuation):
| `human_arabic_token_count` · `human_english_token_count` · `human_other_token_count` |

Only fill these for the `[CS-RATIO]`-marked rows (~12). Leave blank otherwise.

---

## Scales & tips
- 1–5: 1–2 bad, 3 okay, 4 good, 5 excellent. Use the full range.
- A sentence can be fluent yet **monolingual** → `human_cs_valid = no`, `error_type = monolingual`.
- A sentence can be fluent + code-switched but **wrong task** (wrong sentiment / off-topic / missing
  English entity) → `human_task_correct = no`.
- Rate rows in the order shown (they are shuffled on purpose). Do not edit `sample_id` or `text`.

## What this enables (for reference, not needed while rating)
task correctness by task · CS validity · acceptability · AI-judge-vs-human and Validator-vs-human
agreement · masked-vs-control quality · neutral-sentiment dispute resolution · NER English-script
compliance · CS-ratio vs your manual token counts.
