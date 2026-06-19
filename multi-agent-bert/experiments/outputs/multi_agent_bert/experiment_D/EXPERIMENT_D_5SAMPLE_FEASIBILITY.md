# Experiment D — 5-sample `testFlag=False` feasibility test (STOPPED)

Goal: check whether Ahmed's notebook can generate test features from **raw EESA
text** (no cached `.npy`) for a tiny 5-sample probe. Isolated env only. Date: 2026-06-19.
**Outcome: STOPPED before `model.predict` — required files missing / char vocab not
reproducible (your explicit stop conditions).**

## What works (already proven)
- Isolated env: Python 3.9 + **TensorFlow 2.15 + Keras 2.15** (`ahmed_inference/.venv_ahmed`), separate from the PyTorch pipeline. ✔
- **Model loads:** M7_HBA loads from `params.json` + weights via the extracted
  `SelfAttention` layer; expected inputs confirmed:
  `input_1 = (None, 116, 1343)`, `char_input = (None, 116, 15)`, output `(None, 3)`. ✔

## Does `testFlag=False` work? — NO (not standalone)
Tracing the notebook's `generateMemoArray` (`if not trainFlag`) path shows it
regenerates the **embeddings** fresh, but still depends on inputs we don't have:

The model input `input_1` (1343-dim/token) = **300 + 1024 + 19**:
| Block | Dim | Source in notebook | Available? |
|---|---|---|---|
| flair `WordEmbeddings('ar')` | 300 | downloaded by flair | ⚠️ needs flair + download |
| AraBERT-large-twitter | 1024 | `TransformerWordEmbeddings('aubmindlab/bert-large-arabertv02-twitter')` | ⚠️ ~1.5 GB download |
| **handcrafted features `x[2]`** | 19 | read from the **input data** (per-token), built earlier from a **lexicon** | ❌ **missing** |

Two hard blockers (both are your listed stop conditions):
1. **Required files missing.**
   - Our EESA test CSV has only `text, label` — it does **not** carry the 19-dim
     per-token feature vectors (`x[2]`) the path consumes.
   - Those 19-dim features are built from a specific lexicon
     **`ArabicLexiconMSA_EA2016_and_ArEnSA_lexicon…`** which is **not in the repo**
     (we only have a differently-named `EESA-Lexicon.csv`, unconfirmed match).
2. **Char vocab cannot be reproduced.** `char2idx` (299-char map) is built by
   iterating Ahmed's dataset in a specific order; the saved dict (`saveDictFile`
   pkl) is **not in the repo**. Rebuilding it without his exact data/order would
   produce different indices → the model's char embeddings would be meaningless.

## Stopped at
Before any download or `model.predict`, at the missing-required-file check —
exactly as instructed ("stop immediately if a required file is missing / char vocab
cannot be reproduced").

## Report fields
- **testFlag=False works?** No — it regenerates the 1324-dim embeddings from text,
  but the 19-dim handcrafted features and the char vocab are not available.
- **Resources required:** flair (+`WordEmbeddings('ar')`), AraBERT-large-twitter
  (~1.5 GB), the **missing** lexicon for the 19-dim features, and the **missing**
  `char2idx` mapping.
- **Final input shapes:** not reached (stopped pre-feature-build).
- **Prediction probabilities:** not reached.
- **Reliable enough to scale to 818?** **No.** Blocked by the missing lexicon /
  feature data and the non-reproducible char vocab — independent of sample count.

## What would unblock it (smallest asks to Ahmed)
We've already proven the TF env + model loading work locally, so the gap is purely
feature artifacts. Any **one** of these makes local inference possible:
1. **Precomputed test feature `.npy`** (`x_test…1343`) + his **`char2idx` pkl** →
   we load model + npy and predict locally (no flair/AraBERT/lexicon needed). OR
2. The **`char2idx` pkl + the exact lexicon file** + confirmation the 19-dim
   feature code is self-contained → we generate features locally. OR
3. **Predictions directly** (CSV: `text, pred, prob_pos, prob_neg, prob_neu`) →
   no inference on our side at all (simplest).

Kept ready for option 1/2: the isolated env and `ahmed_inference/custom_layers.py`.
