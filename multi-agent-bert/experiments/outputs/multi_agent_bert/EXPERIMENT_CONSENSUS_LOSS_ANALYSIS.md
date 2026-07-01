# Consensus Loss — Counterfactual Contribution of Each Specialist (Designs A–G, G2, v3)

For every escalated sample and every design with per-agent captures, we ask: **did a
specialist hold the correct answer that the consensus then discarded because the other agents
stayed wrong?** This quantifies the *consensus loss* of the architecture — correct answers the
panel possessed but majority voting suppressed. **Analysis only — no runs, no new design.**
84 escalated samples per design. Date: 2026-07-01.

## Definitions
- **Suppressed-correct(agent)** = #{agent's label == true AND final != true}. The agent was
  right, the consensus output was wrong.
- **Lone-correct loss(agent)** = suppressed cases where that agent was the **only** correct
  voter — the textbook majority-voting failure (1 right vs ≥2 wrong).
- **Oracle-any-voter** = accuracy if we could always pick the correct label whenever *any*
  voter holds it. **Consensus loss = oracle − final** (recoverable correct answers thrown away).
- **Unrecoverable** = final wrong AND *no* voter had the truth = the information/model floor
  (the ~0.75 Bayes/model ceiling from the root-cause analysis), NOT an architecture loss.
- IntentGate is a **non-voting guard**; it is analysed separately (a veto cannot be outvoted).

## Per-design consensus loss
| design | N | final acc | **oracle-any-voter** | **recoverable loss** | unrecoverable (floor) |
|---|---|---|---|---|---|
| A semantic_v1 | 84 | 0.726 | 0.798 | **6** | 17 |
| B Pol+Ctx | 84 | 0.750 | 0.774 | 2 | 19 |
| C Lex+Pol+Ctx | 84 | 0.738 | 0.821 | **7** | 15 |
| D 4-agent | 84 | 0.762 | 0.821 | 5 | 15 |
| **E 4-vote** | 84 | 0.750 | **0.881** | **11** | 10 |
| F no-Lexical | 84 | 0.726 | 0.845 | 10 | 13 |
| **G IntentGate** | 84 | **0.786** | 0.833 | **4** | 14 |
| G2 selective | 84 | 0.774 | 0.833 | 5 | 14 |
| v3 pragmatic | 84 | 0.750* | 0.833 | 7 | 14 |

\*capture value (headline final = G). **The panel routinely contains 4–11 more correct answers
than the consensus emits** — i.e. an oracle selector would score **0.80–0.88** on the escalated
subset where the actual consensus scores ~0.75. That gap (≈5–13% of the subset) is the consensus
loss. **Design G has the lowest recoverable loss (4)** among the strong designs — the veto
recovers part of what majority voting would suppress. **Design E has the highest latent loss (11)**
— its four votes carry the most independent-but-suppressed correct signal.

## Per-specialist suppressed gains (aggregated across designs where the specialist VOTES)
| specialist | suppressed-correct (right, but final wrong) | of which **LONE-correct** (only this agent right) |
|---|---|---|
| **Lexical** | **25** | **17** |
| **Contextual** | **20** | **13** |
| **Intent** (as a voter, E/F) | **12** | **12 (100%)** |
| Polarity | 7 | 5 |
| Logic | 3 | 2 |

**The loss falls on the decorrelated signal, exactly.**
- **Lexical (25, 17 lone)** and **Contextual (20, 13 lone)** — the two agents that most often hold
  an *idiosyncratic* correct read — are the most suppressed. When they are uniquely right, the
  other two (correlated) agents are jointly wrong and outvote them.
- **Intent as a voter: 12 suppressed, all 12 lone → 100% of its correct contributions were
  discarded.** Its entire value is decorrelation (independent pragmatic reads), and majority
  voting destroyed *every one* of those reads when it disagreed with the bloc. **This is the
  mechanistic reason Design E (Intent-as-4th-vote) was a wash.**
- **Polarity (7) and Logic (3)** are the *least* suppressed — because they are the most
  correlated with the bloc, so they rarely dissent correctly in the first place.

→ **Majority voting systematically suppresses precisely the minority-correct, independent
information that an ensemble is supposed to exploit**, and preserves the redundant, correlated
votes that add least.

## The decisive contrast: vote vs veto
| Intent's pragmatic signal, entered as… | outcome |
|---|---|
| **a VOTE** (Design E/F) | **12 / 12 correct contributions suppressed** by the majority |
| **a VETO** (IntentGate, Design G/G2/v3) | **0 missed** (gate-correct-but-didn't-fire = 0); **4 helped** per design (2–3 hurt) |

The *same* decorrelated pragmatic signal is **100% destroyed as a vote** but **fully realised as
a veto** — because a non-voting guard cannot be outvoted. This is the single most important
finding of the consensus-loss analysis, and it explains every ranking we observed:
- **G > E**: the gate turns Intent's suppressed vote into an un-suppressible veto.
- **Component-up / system-flat (v3, F, G2)**: a single agent's improvement is *minority-correct*
  signal → it lands in the lone-correct bucket → majority voting suppresses it. Improving a
  member cannot help while the aggregation rule discards minority-correct voices.

## Interpretation — what "consensus loss" is, mechanistically
1. **Two error buckets.** Of ~19–23 final errors per design: **~⅔ unrecoverable** (no agent had
   the truth — the info/model floor) + **~⅓ recoverable** (a specialist had it; the architecture
   lost it). The consensus loss is that recoverable third, **4–11 samples**.
2. **The loss is structural, not stochastic.** It concentrates in lone-correct cases (Lexical 17,
   Contextual 13, Intent 12) — a direct consequence of correlated errors: when one agent is
   uniquely right, the correlated remainder is jointly wrong and outvotes it. Confidence-weighted
   consensus (Fix #2) helps at the margin but cannot rescue a minority-correct voice that the
   majority contradicts.
3. **The architecture is anti-ensemble in the regime that matters.** On the hard escalated subset,
   it discards the independent signal (Lexical/Contextual/Intent dissent) and keeps the redundant
   signal (Polarity/Logic) — the opposite of what an ensemble should do — so it collapses toward a
   single correlated opinion (the ~0.75 member ceiling) instead of exceeding it.

## Honesty / bounds
- **Oracle-any-voter overstates the *achievable* gain**: recovering it requires knowing *which*
  agent is right without the label. Because the correct voice is by construction the *minority*,
  any majority-like or confidence-weighted rule tends to suppress it, and the router/primary
  cannot identify it (`EXPERIMENT_AHMED_ROUTER_SELECTABILITY.md`). So the *realistic* recoverable
  fraction is smaller than 4–11.
- **But the lone-correct concentration is a real, addressable design property.** The vote-vs-veto
  contrast proves the fix is at the **aggregation layer** (gating / vetoes / learned per-case
  weighting that can elevate a dissenting specialist), **not** at the prompt layer — consistent
  with the root-cause analysis (prompts move members inside a fixed feasible set; they cannot
  change how the votes are combined).

## Bottom line
> **The architecture loses ~4–11 correct answers per 84 escalated samples (an oracle would score
> 0.80–0.88 vs the actual ~0.75), and this consensus loss falls almost entirely on the
> *decorrelated* specialists — Lexical (25/17-lone), Contextual (20/13-lone), and Intent
> (12/12-lone). Majority voting suppresses exactly the independent signal an ensemble should
> exploit. The IntentGate result proves the remedy: the identical pragmatic signal that is 100%
> suppressed as a vote is fully realised as a non-voting veto. The consensus loss is therefore an
> aggregation-rule problem, not a specialist-competence or prompt problem.**

## Artifacts
- Per-agent tables: each `experiment_ahmed_*/error_attribution/attribution_table.json`.
- Related: `EXPERIMENT_ESCALATED_CEILING_ROOT_CAUSE.md` (the floor / correlation),
  `EXPERIMENT_SENTIMENT_INTENT_GATE_ABLATION.md` (veto), `EXPERIMENT_SENTIMENT_INTENT_AGENT_ABLATION.md`
  (Intent-as-vote wash), `EXPERIMENT_AGENT_BEHAVIOR_COMPARISON.md` (correlation).
