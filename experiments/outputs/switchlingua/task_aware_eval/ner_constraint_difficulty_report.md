# NER Constraint-Difficulty Analysis (explanatory — does NOT change the final Test 1 NER score)

Question: is low NER compliance a general model failure, or an artifact of requiring English-script **PER + ORG in every sentence**? Same English-only policy, gpt-4o-mini, refiner OFF, final English-only judge.

Sample: 20 NER sentences per variant.

| variant | must-include | count | n | task-correct % | missing_PER | missing_ORG | count-valid % | CS-valid % | fluency | naturalness |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| easy_ORG | ORG | 1-2 | 20 | 90.0 | 20 | 2 | 90.0 | 85.0 | 8.65 | 8.55 |
| medium_PER | PER | 1-2 | 20 | 35.0 | 13 | 10 | 90.0 | 95.0 | 8.4 | 8.25 |
| hard_PER_ORG | PER+ORG | 2-3 | 20 | 60.0 | 7 | 2 | 85.0 | 90.0 | 8.4 | 8.55 |

## How to read

- If **easy_ORG / medium_PER score much higher than hard_PER_ORG**, the low Test 1 NER (40%) is largely driven by the **strict PER+ORG-in-every-sentence config**, not a general NER failure.

- Compare **missing_PER vs missing_ORG**: if missing_PER stays high even in medium_PER, the model specifically struggles to produce **English-script PERSON** names (it tends to write person references in Arabic inside the Arabic-matrix sentence).

- The final Test 1 NER number (40%, hard_PER_ORG English-only) is unchanged; this only explains *why*.

## Findings (this run)
- **ORG is easy:** easy_ORG = 90% task-correct, missing_ORG only 2/20 → the model reliably produces
  English-script organization entities. So this is NOT a general NER failure.
- **PERSON is the bottleneck:** medium_PER = 35%, with 13/20 sentences missing a person entity even
  when PER is the only required type. The model struggles to put English-script person names into an
  Arabic-matrix sentence (it tends to render the person in Arabic).
- **Difficulty is non-monotonic:** hard_PER_ORG (60%) > medium_PER (35%). Explicitly requiring PER *and*
  ORG elicited more persons (missing_PER 7/20) than requiring PER alone (13/20). So low NER is NOT simply
  "too many constraints" — relaxing to PER-only scored worse.
- **Conclusion:** low NER compliance is driven by a specific weakness — generating **English-script
  PERSON entities** — not by general NER inability nor purely by config strictness. Relaxing the config
  would not fix it unless PER is dropped.
- **Variance caveat:** hard_PER_ORG here = 60% vs Test 1 = 40% (same config) → run-to-run generation
  variance on small samples. Exact NER % is noisy (~40–60%); the PER-bottleneck pattern is the robust
  finding. The final Test 1 number (40%) is unchanged by this explanatory analysis.
