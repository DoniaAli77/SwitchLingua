# Human Check — Masking Test (per-dimension)

## What annotators do
Open **human_check_sheet.csv** in Excel. For EACH sentence fill these columns:

1. **fluency_1to10** — is the grammar/wording correct & smooth? (1 = very bad … 10 = perfect)
2. **naturalness_1to10** — does it sound like a real Arabic-English bilingual speaker? (1…10)
3. **cultural_1to10** — is it culturally/socially appropriate & sensible? (1…10)
4. **is_real_codeswitch_yes_no** — does it genuinely MIX Arabic and English?
   - yes = mixes both  |  no = basically all Arabic or all English (monolingual)
5. **notes** — optional.

Do NOT change `id` or `sentence`. Rate each sentence on its own, in the order shown
(they are deliberately shuffled). If 2 people rate, each saves their own copy
(`annotator1.csv`, `annotator2.csv`) in this folder.

## Why
Each sentence is secretly either a machine-flagged weak ("MASKED") sentence or a
"good neighbour". The sheet hides which. Afterwards we check: do humans rate the
MASKED ones lower? If yes, masking is confirmed by people, not just the AI.
Rating the same 3 dimensions the AI uses also lets us check the AI's scoring is trustworthy.

## This sheet
- 50 sentences from 11 masking scenarios (13 MASKED, 37 neighbours)
- Bar = 7.0, seed = 42
