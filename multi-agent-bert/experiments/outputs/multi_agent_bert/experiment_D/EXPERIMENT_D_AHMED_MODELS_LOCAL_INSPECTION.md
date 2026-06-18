# Experiment D — Ahmed's models: local file inspection

Inspection only. No installs, no pipeline changes, no experiments run. Kept
separate from Experiments A/C. Date: 2026-06-18.

---

## 1. Location
**`C:\Users\Eng.Donia\Documents\matser\SwitchLingua\ahmed models`**
(POSIX: `/c/Users/Eng.Donia/Documents/matser/SwitchLingua/ahmed models`)

**Git safety: OK.** Both the `.hdf5` weights and the `.zip` archives are already
**git-ignored** and **untracked** (`git ls-files "ahmed models"` is empty). No risk
of committing large artifacts. (Recommend keeping it that way — see §6.)

---

## 2. The three models + files (what exists)

| File | M7_HBA | M17_HTH | M25_Ensemble |
|---|---|---|---|
| `params.json` (architecture) | ✅ 28 KB | ✅ 26 KB | ✅ 6 KB |
| `w.h5` (weights) | ✅ 39 MB | ✅ 45 MB | ⚠️ **16 KB** |
| `weights-saved.hdf5` (weights) | ✅ 117 MB | ✅ 135 MB | ⚠️ **28 KB** |
| `modelArchInfo.csv` | ✅ | ✅ | ✅ |
| `testErrorAnalysis.csv` (errors only) | ✅ 70 rows | ✅ 69 rows | ✅ 61 rows |
| `devErrorAnalysis.*`, `training_curve.png` | ✅ | ✅ | ✅ |
| **char vocab / tokenizer file** | ❌ | ❌ | ❌ |
| **feature `.npy` (model input)** | ❌ | ❌ | ❌ |
| **`.pkl` artifacts** | ❌ | ❌ | ❌ |

Top-level also has the original ZIPs (M7 147 MB, M17 167 MB, M25 70 KB) and the
shared notebook **`sentimentAnalysisTransFusionGCV2.ipynb`** (286 KB).

**M25_Ensemble is not a standalone model.** Its weights are tiny (16/28 KB) and the
notebook confirms it is *"TransFusion between M7 and M17 with average voting"* —
`EnsembleModelFusion(model1, model2)` loads **M7 and M17** and averages them. So
running M25 requires M7 + M17 to run first.

---

## 3. What the notebook tells us (the deciding part)

The notebook is complete code, but it shows the models **do not eat text** — they
eat **precomputed feature arrays loaded from `.npy` files we do not have**:

- **Word features (model input `input_1`, 1324-dim):** loaded via
  `np.load(testMemoFile)` where
  `testMemoFile = ".../x_test_Ar_AraBertTwitter_FP_DF_RLHT_LC_DNTF_DNEF_NEF-1324.npy"`.
  **These `.npy` files are absent from the repo.** They are built from
  **AraBERT-large-Twitter** (`aubmindlab/bert-large-arabertv02-twitter`, via `flair`
  `TransformerWordEmbeddings`) plus extra emotion/named-entity features (NEF/DNEF/DNTF).
  *(A separate newer variant also fuses OpenAI `text-embedding-3-large` at 4096-dim,
  but M7/M17 per their `modelArchInfo.csv` use the 1324-dim AraBERT version.)*
- **Char features (model input `char_input`):** the char vocabulary (`char2idx`) is
  **built at runtime from the dataset** inside `charEmbLayer(...)`, not loaded from a
  file. Reproducible only by re-running that step on the same data/tokenization.
- **Model loading is simple:** `model_from_json(params.json)` + `load_weights(.hdf5)`
  (needs the custom `SelfAttention` layer from the notebook registered).

So: **weights ✅ + architecture ✅ + code ✅, but the actual model inputs (the
feature `.npy`) ❌ and the char vocab is regenerated, not shipped.**

---

## 4. VERDICT — can we run inference locally from what's here?

**No, not from the downloaded files alone.** We have the model "brains" and the
code, but **the models' numeric inputs (the AraBERT-Twitter feature `.npy`) are
missing**, and the char vocab must be rebuilt from data. Loading weights without
those inputs gets us nowhere.

To make local inference possible we need **one** of these, in order of preference:

| Option | What Ahmed/we provide | Effort for us | Risk |
|---|---|---|---|
| **B (best)** | Ahmed runs his notebook and sends **prediction probabilities per sentence** (`text, pred, prob_pos, prob_neg, prob_neu`) | ~1 h: text-match to our 818 + adapter | low |
| **A** | Ahmed sends the **feature `.npy` files** (`x_test_…1324.npy`) + the row order/labels | medium: TF/Keras install + load + predict | medium (his test = ~1099, must align to our 818) |
| **C (worst)** | We **regenerate features** ourselves | high: install `flair` + download AraBERT-large-Twitter (~1.5 GB, SSL proxy) + reproduce preprocessing + NE/emotion features + char vocab | high (feature drift silently lowers accuracy) |

Note for all options: the comparison must be on **our 818-sample test**
(`eesa_sentiment_test.jsonl`), not Ahmed's ~1099-row test, or it won't be
comparable to the XLM-R numbers.

---

## 5. Recommendation
Local-only inference is **not feasible** with the current files (feature `.npy`
missing). Proceed with **Option B** — request prediction probabilities from Ahmed.
This needs nothing installed on our side, avoids the AraBERT/flair reproduction
risk, and plugs straight into the planned `PrecomputedPrimaryClassifier` adapter.
If Ahmed can only give the `.npy` features (Option A), that also works but adds a
TF/Keras inference step on our side.

**Next concrete step (when you're ready):** I draft the exact spec to send Ahmed —
the file format and that it must cover the 818 test sentences (or his full test set
*with text*, so we can match the 818 ourselves).

---

## 6. Git hygiene reminder
The large artifacts are currently ignored/untracked (good). To be safe against an
accidental `git add -A`, consider an explicit ignore rule for
`ahmed models/` (or `*.h5 *.hdf5 *.zip`). I did **not** modify `.gitignore` — flagging
only, per "inspect first."
