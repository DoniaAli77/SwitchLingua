# Topic Dataset Readiness — ARENTC V1 & V2 (ArEnTC)

Inspect / validate / normalize / export only. No fine-tuning, no full_agentic.
V1 and V2 kept fully separate. Original `.xlsx` untouched. Date: 2026-06-19.

Task: topic classification, 9 labels —
`business, education, finance, health, medical, shopping, social, sports, tech`.

---

## ARENTCV1

1–2. **Files found** (`data/Topic/ARENTCV1/`):
- train → `train_split.xlsx`
- dev → `dev_split.xlsx`
- test → `test_split.xlsx`
- (also present, not used: `combined_all_topics.xlsx`, `topic_distribution_counts.xlsx`)

3–4. **Columns:** `text`, `label` → text column = **`text`**, label column = **`label`**.

5. **Row counts:** train **73,976** · dev **10,569** · test **21,137**.

6–7. **Labels:** already lowercase; normalized defensively (strip+lower). All splits
contain **exactly the 9 expected labels** — no unexpected/extra labels. ✅

8. **Label distribution (balanced):**
- train: medical 8341 · social 8275 · education 8236 · tech 8218 · business 8202 · health 8198 · sports 8179 · finance 8171 · shopping 8156
- dev: 1167–1193 per label · test: 2330–2383 per label.

9. **Empty text rows:** 0 in all splits. ✅

10. **Duplicates within split (normalized text):** train **0** · dev **60** · test **0**.
    *(dev has 60 internal dup texts — left as-is per instructions; flag for decision.)*

11. **Leakage across splits (normalized text overlap):**
    train∩dev **0** · train∩test **0** · dev∩test **0**. ✅ no leakage.

12. **Code-switching status:** train 73,956 code-switched + 20 fully-Arabic ·
    dev 10,562 + 7 · test 21,134 + 3. → overwhelmingly code-switched; **30
    fully-Arabic rows total**, 0 fully-English.

13–14. **Exported JSONL** (`{"id","text","label"}`):
`data/Topic/processed/ARENTCV1/{train,dev,test}.jsonl` (73,976 / 10,569 / 21,137 lines).

---

## ARENTCV2  (no-fully-Arabic variant)

1–2. **Files found** (`data/Topic/ARENTCV2/`):
- train → `train_split_no_fully_arabic.xlsx`
- dev → `dev_split_no_fully_arabic.xlsx`
- test → `test_split_no_fully_arabic.xlsx`
- (also present, not used: `ArEnTC.xlsx`)

3–4. **Columns:** `text`, `label` → text = **`text`**, label = **`label`**.

5. **Row counts:** train **73,956** · dev **10,562** · test **21,134**
   (= V1 minus the ~30 fully-Arabic rows).

6–7. **Labels:** lowercase-normalized; all splits = **exactly the 9 labels**. ✅

8. **Label distribution (balanced):**
- train: medical 8340 · social 8275 · education 8235 · tech 8218 · business 8202 · health 8193 · sports 8179 · finance 8171 · shopping 8143
- dev: 1164–1191 · test: 2327–2383.

9. **Empty text rows:** 0. ✅

10. **Duplicates within split:** train **0** · dev **60** · test **0**.

11. **Leakage across splits:** train∩dev **0** · train∩test **0** · dev∩test **0**. ✅

12. **Code-switching status:** **100% code-switched** in every split (0 fully-Arabic,
    0 fully-English) — consistent with the "no_fully_arabic" filtering.

13–14. **Exported JSONL:**
`data/Topic/processed/ARENTCV2/{train,dev,test}.jsonl` (73,956 / 10,562 / 21,134 lines).

---

## Cross-version summary
| | ARENTCV1 | ARENTCV2 |
|---|---|---|
| train / dev / test | 73,976 / 10,569 / 21,137 | 73,956 / 10,562 / 21,134 |
| labels = 9 exact | ✅ | ✅ |
| balanced | ✅ (~8.2k/label train) | ✅ |
| empty text | 0 | 0 |
| dup within (train/dev/test) | 0 / 60 / 0 | 0 / 60 / 0 |
| cross-split leakage | none | none |
| code-switching | ~99.96% CS (30 fully-AR) | 100% CS |
| JSONL exported | ✅ | ✅ |

**Difference:** V2 is V1 with the ~30 fully-Arabic rows removed → V2 is purely
code-switched. Otherwise identical structure and balance.

## Flags / notes (no action taken — inspection only)
- **dev has 60 internal duplicate texts** in both versions (not cross-split
  leakage). Left untouched per "only inspect/normalize/export." Say the word to
  drop them if you want a deduped dev.
- Labels were **already lowercase** in the source; the Tech→tech / Business→business
  normalization was a no-op but applied defensively.
- JSONL uses `ensure_ascii=False` (Arabic preserved). IDs are
  `{VERSION}-{split}-{index:06d}`.
- **Git hygiene:** processed JSONL (~40 MB) is git-ignored; the source `.xlsx`
  (~20 MB) are **not** currently ignored — recommend adding `data/Topic/**/*.xlsx`
  to `.gitignore` so they aren't committed.

## Readiness verdict
Both ARENTCV1 and ARENTCV2 are **clean and ready** for topic fine-tuning/evaluation:
exact 9-label match, balanced, no empty text, no cross-split leakage, JSONL in the
`{"id","text","label"}` format `evaluate_pipeline.py` expects, and the pipeline
config already targets `topic_classification` with these 9 labels (see
`TOPIC_CLASSIFICATION_READINESS.md`). Next step (when you're ready): fine-tune a
topic primary checkpoint per version — **not done now, per instruction.**
