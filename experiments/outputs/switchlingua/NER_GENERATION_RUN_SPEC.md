# Spec — NER generation run under the task-aware pipeline (proposal, NOT executed)

**Why:** the human-eval workbook could only supply **14** NER rows (vs 18 requested), covering only
PER+ORG and PER+ORG+LOC, with **no entity spans** and **no Arabic-script entities**. Those gaps are not
sampling problems — the data does not exist. This spec defines the run that would close them.

**Status:** nothing here has been implemented or run. NER is frozen; items marked **[UNFREEZE]** need
your explicit approval before I touch code.

---

## 1. The four gaps and what each requires

| # | Gap in current data | Root cause (verified) | Fix required |
|---|---|---|---|
| 1 | Only 14 accepted NER sentences | No NER dataset was ever generated; only eval artifacts | A generation run (§3) |
| 2 | Only PER+ORG and PER+ORG+LOC type-sets | Every NER scenario used `must_include_types: [PER, ORG]` | Config with per-type scenarios (§3) |
| 3 | `target_entities` empty everywhere | `annotations` is initialised to `[]` (`utils.py:134`) and never written. `_extract_english_ner_counts` **does build** `per_entities` / `org_entities` / `loc_entities` sets but returns **counts only** (`node_engine.py:344`) — the strings are computed then discarded | **[UNFREEZE]** small change: return + persist the entity strings (§2) |
| 4 | 0 Arabic-script entities | Frozen policy is English-only (`target_entities_script: english`); Arabic-script entities are counted as `arabic_script_ignored` | **Policy decision** (§4) |

---

## 2. [UNFREEZE] Minimal code change — persist entity spans

Contained, additive, and testable; does **not** alter any accept/reject decision.

1. `_extract_english_ner_counts(text, types)` → also return the entity strings it already computes
   (e.g. return `(counts, entities_by_type)`; keep the existing counts contract for current callers).
2. `_deterministic_ner_english_policy` → carry `entities_by_type` into its result dict.
3. `RunNERTaskValidatorAgent` → write the per-sentence entity strings into the record
   (natural home: the existing `annotations` field, or a new `entities_found` key).
4. `build_sentence_records` (`utils.py`) → copy that field onto each `sentence_record`.

**Risk:** low — read-only extraction data being surfaced, no gating logic touched.
**Tests to add:** extraction returns expected strings for a fixture sentence; records carry them
end-to-end in the mocked full-graph test; existing NER tests (`test_ner_guidance.py` 5/5) still pass.

*Alternative with no unfreeze:* run a **separate offline extractor** over the generated corpus after the
fact. Cheaper and zero risk to the pipeline, but the spans are then a post-hoc annotation rather than
pipeline output — acceptable if they are only for human evaluation, not training labels.

---

## 3. Generation config (`config_ner_expN_v1.yaml`)

Entity-type coverage as its own dimension — the thing the current data lacks:

| group | `must_include_types` | scenarios |
|---|---|---|
| single | `[PER]`, `[ORG]`, `[LOC]` | 3 |
| pairs | `[PER,ORG]`, `[PER,LOC]`, `[ORG,LOC]` | 3 |
| triple | `[PER,ORG,LOC]` | 1 |

× `cs_ratio [50%,60%]` × `cs_type [Intrasentential]` × topics × gender/age
(`min_entities`/`max_entities` scaled per group: 1–2 for single, 2–3 for pairs, 3–4 for the triple).

**Yield warning (important).** With `TASK_AWARE_ACCEPT=1` now the default, task-failing sentences are
**dropped**, and NER task-correctness is our weakest task (**~40%** English-only in Test 1; PER is the
bottleneck — tag coverage ORG ≈ 80%, LOC ≈ 44%, **PER ≈ 30%**). Expect roughly **0.5–1.0 kept/scenario**
versus 2.2 for topic — i.e. **2–4× more scenarios per accepted sentence**, and single-PER will be the
slowest cell. For a 18-row human-eval sample this is trivial (~40–60 scenarios); for a *training* set of
say 50/type-group × 7 groups = 350 sentences, budget **~500–700 scenarios ≈ 2–3 quota-days**.

---

## 4. Policy decision needed — Arabic-script entities

"≈half Arabic-script" is incompatible with today's English-only policy. Three options:

- **A. Keep English-only** (recommended for consistency): the human-eval sheet then documents that all
  entities are Latin-script by design. No code/policy change; comparable with all prior NER results.
- **B. Add a mixed-script arm** as a *config variant* (`target_entities_script: mixed` /
  `allow_code_switched_entities: true`), generated separately so the English-only results stay clean.
  This is the only way to get Arabic-script entities, and it needs validator support for scoring them.
- **C. Post-hoc**: keep generation English-only and have annotators mark Arabic-script entities they see.

**A or C requires no unfreeze; B does.**

---

## 5. Deliverables of the run
`data/NER/generated/` (isolated tree, mirroring Topic/Sentiment): `daily_runs/`, `merged/`,
`completed_scenarios_ner.json`, a balanced `switchlingua_ner_train_<N>.{csv,jsonl}` with
`text, entity_types, entities, …`, a `DATASET_CARD_NER_<N>.md`, and an intrinsic profile.
Tooling: `manage_ner_data.py`, cloned from `manage_topic_data.py` (which is label-dynamic already).

---

## 6. Decisions I need from you
1. **Scope:** just the 4 missing human-eval rows (~40–60 scenarios, quick), or a full NER training set (multi-day)?
2. **Spans:** [UNFREEZE] persist in-pipeline (§2), or offline post-hoc extractor (no unfreeze)?
3. **Script policy:** A (English-only), B (mixed-script arm), or C (annotator-marked)?

Nothing proceeds until these are answered.
