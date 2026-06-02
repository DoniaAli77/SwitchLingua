# NER PER-focused Prompt Repair — before/after pilot

Variant prompt tested in the harness only (core prompt NOT modified). hard_PER_ORG config, English-only judge, gpt-4o-mini @0.7, refiner OFF.

N per arm: 50. CIs are Wilson 95%.

| arm | n | task-correct % (95% CI) | missing_PER | missing_ORG | count-valid % | CS-valid % | CS-ratio MAE vs70 | fluency | naturalness |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| current | 40 | 22.5 (12.3–37.5) | 28 | 8 | 47.5 | 95.0 | 18.0 | 8.32 | 8.4 |
| per_focused | 44 | 56.8 (42.2–70.3) | 11 | 8 | 75.0 | 90.9 | 15.95 | 8.61 | 8.52 |

## How to read

- **Real improvement?** Compare task-correct CIs: current (12.3, 37.5) vs per_focused (42.2, 70.3). If the CIs **overlap heavily**, the change is within run-to-run noise (recall same-config NER varies ~40–60%) — not a confirmed win.

- **Did PER improve?** missing_PER should drop in per_focused.

- **Side effects?** Watch CS-ratio MAE and naturalness — forcing English names can push text toward English (worse ratio) or read less naturally. A 'fix' that breaks the CS ratio is not a win.

- Still LLM-judged (blind judge); a confirmed win needs human spot-check. Core prompt unchanged; promote the variant only if it clearly wins without side effects.
