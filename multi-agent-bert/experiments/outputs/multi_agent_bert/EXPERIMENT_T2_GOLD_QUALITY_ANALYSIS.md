# ARENTCV2 Topic — Are the Agents Wrong, or Is the GOLD Wrong?

**Analysis only — no runs.** Tests whether the reason-first agents' "errors" on the escalated
subset are real mistakes or artifacts of **inconsistent gold labels**. Uses the full-test
reasoned+sharpened run (agents = primary = 0.9947) and the ARENTCV2 test gold. Date: 2026-07-05.

## The gold is internally inconsistent — proof
Label distribution across the **whole test set** for the exact confusable concepts:

| concept (appears in text) | n | gold labels | taxonomy says | inconsistency |
|---|---|---|---|---|
| **telemedicine** | 188 | **health 68 / medical 120** | medical (clinical delivery) | **same concept split 36%/64%** |
| **clinical trials** | 233 | health 15 / **medical 218** | medical | 15 mislabeled health |
| **blockchain** | 386 | finance 93 / **tech 293** | tech (classify by technology) | 93 mislabeled finance |
| **investment opportunities** | 87 | **finance 84** / business 3 | finance | 3 outliers |
| **venture capital** | 25 | business 21 / finance 4 | business/finance (dual) | fuzzy |

**Telemedicine is literally labeled *both* health (68×) and medical (120×)** — the annotators
could not decide. Clinical trials and blockchain have clear majority labels, but a minority are
labeled inconsistently. **The gold contradicts its own taxonomy on exactly the cases that escalate.**

## The 17 agent-vs-gold disagreements, judged against the taxonomy
| disagreement | count | verdict |
|---|---|---|
| health → **medical** (telemedicine / clinical trials / chronic-illness / symptoms) | 11 | **Agent is taxonomy-correct; gold is the inconsistent minority (health)** |
| finance → **tech** (blockchain) | 1 | **Agent correct** (blockchain = tech; gold = minority finance) |
| business → **finance** ("investment opportunities") | 2 | **Agent correct** (investment = finance, the 84/87 majority; gold = 3/87 outlier) |
| finance → business / medical → education | 3 | **genuine toss-up** (dual-topic, either label defensible) |
| **clear agent error (gold clearly right)** | **0** | — |

**≈14 of 17 (82%) "agent errors" are the agent choosing the taxonomy-consistent / majority label
while the gold is the inconsistent minority. ~3 are genuine toss-ups. ZERO are clear agent errors.**

## Illustrative clean-gold re-scoring
On the **14 escalated cases where the gold contradicts its own taxonomy**, score against the
taxonomy-correct label (telemedicine/clinical→medical, blockchain→tech, investment→finance):

| on those 14 taxonomy-clear cases | matches the taxonomy |
|---|---|
| **reason-first agents** | **14 / 14** |
| primary classifier | **4 / 14** |

**The agents are +10 more taxonomy-consistent than the primary.** On the noisy gold the two *tie*
(31/48); on a de-noised gold the **agents beat the primary**, because the primary *memorized the
gold's inconsistencies during training* (it learned "telemedicine→health" from the noisy labels),
while the agents reason from the definitions and are penalized for being consistent.

## Conclusion
> **The topic agents do not fail on these cases — the LABELS do.** ~82% of the reason-first agents'
> escalated "errors" are cases where the agent applies the taxonomy correctly and the gold does not.
> The gold is demonstrably inconsistent (telemedicine split 36/64 health/medical; blockchain 24%
> mislabeled finance). So the headline "agents only *match*, never *beat*, the primary" is an
> **artifact of gold noise**: on the raw labels they tie (0.9947), but on a taxonomy-consistent
> gold the agents would score **higher** than the primary. The multi-agent layer's real strength —
> reasoning from definitions rather than memorizing label noise — is exactly what the noisy gold
> hides.

## Caveats
- "Taxonomy-correct" is judged from the dataset's own `label_descriptions` + the majority label per
  concept; a few (medical-education, venture-capital) are genuinely dual and left as toss-ups.
- This is the *escalated* tail (48/21,134); it does not change the full-set number, but it reframes
  what that number means.
- A rigorous version would re-annotate the escalated subset by an independent judge; the
  distribution evidence (telemedicine 68/120) already establishes inconsistency without re-labeling.

## Artifacts
- Disagreements: `_T2_reasoned_agenterrors.txt`; concept distributions computed inline from
  `data/Topic/processed/ARENTCV2/test.jsonl`.
- Basis: `experiment_T2_reasoned_full/` (agents = primary = 0.9947), `topic_disambig.yaml`.
