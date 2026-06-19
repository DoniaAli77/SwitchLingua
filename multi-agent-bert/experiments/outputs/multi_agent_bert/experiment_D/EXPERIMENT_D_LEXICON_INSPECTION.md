# Experiment D — EESA-Lexicon.csv vs notebook's expected lexicon

Inspection only, no inference. Date: 2026-06-19.

## 1. Exact path
`C:\Users\Eng.Donia\Documents\matser\SwitchLingua\multi-agent-bert\data\Sentiment\raw\EESA-Corpus-main\EESA-Lexicon.csv`

## 2. Columns
**2 columns, no header row:** `term , polarity`. 2,324 rows. Terms are Arabic and
English; polarity ∈ {positive, negative, …}.

## 3. Sample rows
| term | polarity |
|---|---|
| 5 نجوم | positive |
| a lot of vibes | positive |
| adore | positive |
| affecting | positive |
| amazing | positive |

## 4. Does Ahmed's notebook expect this format?
**Shape: roughly yes. Identity: no.** The notebook loads its lexicon with
`lexiconArray = readMainCSVFile(lexiconFile); lexiconArray = lexiconArray[1:]`
(i.e. a 2-column term/polarity CSV, **dropping the first row as a header**). Our
file is the same 2-column term→polarity shape, so it's *format-compatible*.
Caveat: our first row (`5 نجوم, positive`) is real data, not a header, so the
notebook's `[1:]` would silently drop one entry.

## 5. Can it generate the 19 features (`x[2]`)? — NO, not on its own
The 19-dim per-token vector
(`[emotion, text, arNum, enNum, others, arToken, enToken, LC, UC, MC, repeatedChar,
neutral, positive, negative, contextually, negationWord, intensifiedToken, NE,
compoundPhrase]`) needs **four** external resources, of which the sentiment lexicon
is only one:

| Resource (notebook variable / filename) | Role | Present? |
|---|---|---|
| `lexiconFile` = **`ArabicLexiconMSA_EA2016_and_ArEnSA_lexicon.csv`** | positive/negative term matching | ❌ (we have a *different* file, EESA-Lexicon.csv) |
| `negationFile` = **`wordFeatures-negationUpdated.csv`** | `negationWord` feature | ❌ missing |
| `intensFile` = **`wordFeatures-intensifyTokensUpdated.csv`** | `intensifiedToken` feature | ❌ missing |
| `nameEntityFile` (neFile) | `NE` feature | ❌ missing |

The text-only features (emoji, Arabic/English script, numbers, case, repeated
chars) *are* computable from raw text. But the sentiment (`positive/negative/
neutral/contextually`), `negationWord`, `intensifiedToken`, and `NE` features need
the four files above. **So EESA-Lexicon.csv alone cannot build the 19-dim vector**,
and even the lexicon-based features would **drift** because it isn't the lexicon the
models were trained on.

## 6. Does any lexicon filename in the notebook match this file?
**No.** Notebook expects `ArabicLexiconMSA_EA2016_and_ArEnSA_lexicon.csv`; we have
`EESA-Lexicon.csv`. They're clearly related (both EESA sentiment lexicons, same
2-column shape) but **not the same file** — different name, and the notebook's is a
*merged* MSA + EA2016 + ArEnSA lexicon, so almost certainly larger / different
coverage. Using ours as a substitute would change which tokens are tagged
positive/negative → different features → unfaithful predictions.

## 7. Is char2idx still missing?
**Yes, still missing.** The notebook never saves char2idx to a file (no
`saveDictFile(char2idx, …)`); it **rebuilds it at runtime** from the training data
`ArEnSAData-preFinal-NDCS3C-train-mixed.csv` via
`char2idx = {c: i+1 for i, c in enumerate(chars)}`. That training file is **not in
the repo**, and the mapping is order-sensitive, so the exact 299-char→index map the
models learned **cannot be reproduced** from what we have.

---

## Bottom line
EESA-Lexicon.csv is the *right type* of file and format-compatible, but it is **not
the same lexicon**, and it is only **1 of 4** missing feature resources — plus the
char vocab is still unreproducible. So it does **not** unblock local feature
generation. To run these models faithfully we still need, from Ahmed, **either**:
- the precomputed test feature `.npy` + his `char2idx`, **or**
- the 4 resource files (`ArabicLexiconMSA_EA2016_and_ArEnSA_lexicon.csv`,
  `wordFeatures-negationUpdated.csv`, `wordFeatures-intensifyTokensUpdated.csv`,
  the NE file) + `char2idx` + the training CSV, **or**
- the predictions directly (`text, pred, prob_pos, prob_neg, prob_neu`).
