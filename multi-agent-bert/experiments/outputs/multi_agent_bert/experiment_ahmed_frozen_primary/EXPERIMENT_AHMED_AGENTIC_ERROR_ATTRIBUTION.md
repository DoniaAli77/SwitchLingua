# Ahmed Frozen-Primary Full-Agentic — Error Attribution (84 escalated)

Why did the agentic layer not improve Ahmed (0.9254 → 0.9205), and which component is
weak? Analysis of the **84 escalated samples** from the threshold-0.7 run. Date: 2026-06-27.

> **Data.** Transition-level metrics are from the original clean threshold-0.7 run.
> The per-agent breakdown is from a **clean re-run** of the 84 escalated samples through
> the same pipeline (`scripts/ahmed_agentic_attribution.py`; 0 errors), which
> **reproduced the original transitions exactly** (C→C 48 / C→W 15 / W→C 11 / W→W 10),
> so the agent outputs are representative. No training, no generation.

## 1. Escalated subset
| metric | value |
|---|---|
| escalated | 84 / 818 |
| Ahmed correct / wrong | 63 / 21 |
| **Ahmed accuracy (escalated)** | **0.7500** |
| **final full_agentic accuracy (escalated)** | **0.7024** |

Transitions (consensus changed Ahmed's label on **27/84**): C→C 48 · **C→W 15** ·
**W→C 11** · W→W 10 · **net −4**. The agentic layer **lowers** accuracy on the hard
escalated cases (0.750 → 0.702).

## 2. correct→wrong (15) — what the agents broke
| ahmed (correct) → final (wrong) | count |
|---|---|
| **neutral → negative** | **7** |
| **neutral → positive** | **4** |
| negative → neutral | 2 |
| positive → neutral | 2 |

**11/15 breaks were genuinely-neutral predictions pushed to a polar label**, dominated
by **neutral→negative (7)**. Which agent matched the (wrong) final label:
**lexical 15/15 · logic 15/15 · contextual 13/15** — i.e. on every break, *all three
agents had already agreed on the wrong polar label* (the consensus just followed them).

## 3. wrong→correct (11) — what the agents fixed
7/11 fixes were cases where **Ahmed wrongly said "neutral"** and the agents correctly
pushed to polar. Which agent matched the (correct) final label:
**lexical 11/11 · logic 10/11 · contextual 11/11** — again all agents agreed.

So the agents carry a **polarity bias**: it *helps* when Ahmed's neutral was wrong (7)
but *hurts more* when Ahmed's neutral was correct (11) → net negative on neutral.

## 4. Agent-level accuracy on the escalated subset (n=84)
| component | accuracy |
|---|---|
| **Ahmed primary** | **0.7500** |
| LLMLexical | 0.7143 |
| **LLMLogic** | **0.6786 ← weakest** |
| Contextual | 0.7262 |
| final consensus | 0.7024 |

**Every individual agent is *less* accurate than Ahmed on these hard cases**, and the
**consensus (0.702) is below both Ahmed (0.750) and the best single agent (contextual
0.726)** — the voting *destroys* value rather than adding it. **Logic is the weakest
agent (0.679).**

## 5. Agreement analysis
| pattern | count |
|---|---|
| all 3 agents agree | **77 / 84 (92%)** |
| agents split | 7 / 84 |
| Ahmed correct but ≥1 agent disagrees | 17 |
| Ahmed wrong AND all 3 agents agree on the correct label | 10 |

**The three agents agree with each other 92% of the time** — they are **highly
correlated**, not independent (all are GPT-4o-mini reading the same text from three
prompt angles). So "three votes" is effectively **one correlated bloc**.

## 6. Diagnosis — which component is weak
1. **Consensus / override design is the primary culprit.** The three agents are
   (a) **individually weaker than Ahmed** (0.68–0.73 vs 0.75) and (b) **92% correlated**,
   yet together they carry weight 3 vs Ahmed's `w_primary = 1.0`. A **correlated bloc of
   weaker voters systematically outvotes a stronger primary** → consensus 0.702 < Ahmed
   0.750. The "≥1 of 3 agents" diversity the consensus assumes does not exist here.
2. **Shared polarity bias on neutral CS text.** All agents over-read sentiment on
   genuinely-neutral code-switched comments (neutral→negative the top failure). This is
   an *agent-prompt/model* property, equally present in all three.
3. **Logic is the single weakest agent (0.679)** — it never improves on Ahmed and is
   the lowest of the three.
4. **Not the router threshold** (0.7 correctly escalates the uncertain ~10%; §10 of the
   main report) and **not the primary**.
Net: the weakness is **the override rule giving a correlated bloc of weaker agents more
weight than a strong primary**, amplified by a shared neutral→polarity bias; the logic
agent is the worst contributor.

## 7. Suggested fixes (NOT run)
Ordered by how well they fit the evidence above:
1. **Treat the agents as correlated, not independent.** Because they agree 92% of the
   time, "override only if ≥2 agents agree" is **ineffective** (a majority almost always
   exists). Instead, **down-weight the agent bloc** (e.g. cap total agent weight, or use
   their *agreement* as low evidence) and/or **raise `w_primary` substantially**
   (≈3.0, comparable to the 3-agent bloc) so a strong primary is not outvoted.
2. **Never override a confident primary**: keep Ahmed whenever his confidence ≥ a cutoff
   — directly protects the 11 correct-neutral breaks.
3. **Gate the override on agent confidence** (only flip if agents are *highly* confident),
   so low-confidence polar guesses can't beat a neutral.
4. **Class-specific guard for neutral↔negative**: bias the tie-break toward the primary
   on neutral↔negative disagreements (the dominant confusion, 7 of 15 breaks).
5. **Drop or fix the logic agent** for this task (weakest at 0.679; contributes the
   bloc's wrong votes without adding accuracy).
6. (Threshold 0.6 would escalate fewer (41), reducing exposure, but doesn't fix the
   override behavior.)

## Artifacts
- `error_attribution/attribution_table.csv` / `.json` — per-sample table (84 rows) with
  Ahmed + lexical/logic/contextual/consensus/final labels & confidences, transitions,
  agree/disagree sets.
- `error_attribution/transition_table_from_original.csv` — transition table from the
  original clean run.
- `scripts/ahmed_agentic_attribution.py` — the capture script.
