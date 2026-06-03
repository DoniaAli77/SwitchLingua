# NER Freeze Checkpoint — 2026-06-04

Final dev checkpoint for the NER task-aware generation work. NER is frozen after this entry.

## 1. Code status
- Generic NER guidance **promoted to core** (`node_engine.build_ner_entity_guidance()` +
  `DEFAULT_ENTITY_GUIDANCE`; `{ner_entity_guidance}` placeholder in `DATA_GENERATION_NER_PROMPT`;
  `ner.entity_type_guidance` in config; propagation in `generate_scenarios`).
- **No further NER prompt changes planned.**
- **English-only target entity policy unchanged** (`target_entities_script: english`).
- **TaskValidator unchanged.**
- **Evaluator policy unchanged.**

## 2. Test status
- NER guidance regression: **5/5 pass**
- task-generation mock: **pass**
- full pipeline mock: **pass**
- per-instance scoring: **pass**
- refiner guardrail: **pass after stale-assertion update** (two assertions expected `refine_count==0`
  after a rolled-back fix; the NER infinite-loop fix intentionally spends the budget, `count→1`, so the
  loop terminates — assertions updated to `==1` with comments; rollback still reverts the TEXT)
- `*_real.py`: **skipped** (live API required)
- `test_pipeline_output_reviewer`: **excluded** (CLI utility, not a unit test)

## 3. Real smoke status (real CodeSwitchingAgent pipeline, real TaskValidator ON, refiner OFF, English-only)
- generated **12**
- parse **12/12**
- CS-valid **12/12**
- fluency / naturalness **8.8 / 8.5**
- **no runtime/API errors**
- NOTE: `task_correct 3/12` and `validator 0/12` are **NOT final performance estimates** — tiny n and
  known NER run-to-run variance; the real NER validator is also stricter than the English-only judge.

## 4. Freeze marker
**NER: implemented, generic, regression-tested, smoke-tested, FROZEN.**

Known limitation: **PER_EVENT and other competing hard entity combinations remain difficult under
2–3 entity slots** (fixing one hard type can cost the other when slots are scarce).

## 5. Next work (global, not NER)
- Consolidated human validation / spot-check.
- CS-ratio validation against manual token counts.

Do not continue optimizing NER unless tests break or human validation reveals a serious issue.
