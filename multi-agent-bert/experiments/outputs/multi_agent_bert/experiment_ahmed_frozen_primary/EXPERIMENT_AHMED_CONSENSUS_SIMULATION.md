# Ahmed Frozen-Primary — Offline Consensus-Rule Simulation

Can alternative consensus rules recover the accuracy the agentic layer lost on Ahmed?
**Offline simulation only** — re-scores the 84 escalated samples using the
already-captured agent outputs (`error_attribution/attribution_table.json`). **No new
OpenAI calls, no training, no generation.** The simulator faithfully replicates
`src/agents/consensus_agent.py` (confidence-weighted scoring + primary-aware vote +
non-positional tie-break) and **reproduces the current result exactly** (escalated
0.7024 / full 0.9205 / net −4), validating it.

Baselines: **Ahmed primary_only = 0.7500 (escalated) / 0.9254 (full test).**
Full-test accuracy = (694 non-escalated Ahmed-correct + correct-on-84) / 818.

## Results (84 escalated)
| rule | esc acc | full acc | W→C | C→W | net | overrides |
|---|---|---|---|---|---|---|
| 1. current (w_p=1, agents 1/1/1) | 0.7024 | 0.9205 | 11 | 15 | −4 | 27 |
| 2a. w_primary = 2 | 0.7143 | 0.9218 | 10 | 13 | −3 | 24 |
| 2b. w_primary = 3 | 0.7143 | 0.9218 | 10 | 13 | −3 | 24 |
| 2c. w_primary = 4 | 0.7262 | 0.9230 | 7 | 9 | −2 | 17 |
| 3a. agent-bloc, w_p=1 | 0.7024 | 0.9205 | 11 | 15 | −4 | 27 |
| 3b. agent-bloc, w_p=2 | 0.7262 | 0.9230 | 0 | 2 | −2 | 2 |
| **3c. agent-bloc, w_p=3** | **0.7500** | **0.9254** | **0** | **0** | **+0** | **0** |
| 4a. no-logic, w_p=1 | 0.7262 | 0.9230 | 11 | 13 | −2 | 25 |
| 4b. no-logic, w_p=2 | 0.7262 | 0.9230 | 11 | 13 | −2 | 25 |
| 4c. no-logic, w_p=3 | 0.7024 | 0.9205 | 4 | 8 | −4 | 12 |
| 5a. conservative: override only if all 3 agree vs Ahmed | 0.7143 | 0.9218 | 10 | 13 | −3 | 24 |
| 5b. conservative + agent conf ≥ 0.8 | 0.7262 | 0.9230 | 10 | 12 | −2 | 23 |
| 6. neutral guard (keep Ahmed-neutral unless 3 agree polar & conf ≥ 0.8) | 0.7143 | 0.9218 | 10 | 13 | −3 | 24 |

(Agent-bloc = treat the 3 correlated agents as one weight-1 voter using their majority
label; 3-way split ⇒ no agent vote.)

## Reading the results
- **No rule beats Ahmed primary_only.** Every rule's net is ≤ 0; the best any rule
  achieves is to **recover Ahmed exactly** — **agent-bloc with w_primary=3 (3c) → 0.9254,
  net 0, 0 overrides**. Recovering Ahmed means the optimal consensus is the one that
  **never overrides the primary** on these cases.
- **Higher primary weight helps monotonically but is not enough alone:** w=1→0.9205,
  w=2/3→0.9218, w=4→0.9230 — even at w=4 there are still 17 overrides and net −2; it
  never reaches Ahmed's 0.9254, because the 3 correlated agents still out-score a
  low-confidence-but-correct primary on the hard cases.
- **Treating the agents as a correlated bloc is the key lever.** Collapsing the 3
  agents to one vote and giving the primary weight 3 (3c) reaches Ahmed exactly with
  **0 overrides**, and even bloc+w=2 (3b) cuts overrides 27→2 and gets to 0.9230 — far
  more efficient than raising w_primary on three independent votes.
- **"Override only if ≥2/all agents agree" barely helps** (5a: −3) — confirming §5 of
  the attribution: the agents agree **92%** of the time and are *unanimously wrong* on
  the breaks, so requiring agreement does not stop the bad overrides. Adding an
  agent-confidence gate (5b) helps a little (−2).
- **Dropping the logic agent** helps slightly at w=1/2 (−2) but flips back to −4 at
  w=3 (a two-voter tie-break artifact) — not a reliable fix.
- **Neutral guard** (6) gives −3 — it stops some neutral→polar breaks but the agents
  are often unanimous + confident on the wrong polar label, so it leaks.

## Confusion (breaks) shrink with the fixes
Current breaks (15): neutral→negative 7, neutral→positive 4, negative→neutral 2,
positive→neutral 2. Under **w_primary=4** the neutral→negative breaks drop 7→5 and
total breaks 15→9; under **agent-bloc w_p=3** all breaks and all fixes go to 0 (no
overrides at all).

## Conclusion
1. **The consensus/override rule was indeed the weak component** — a better rule
   recovers the 0.0049 accuracy the current rule lost (0.9205 → 0.9254).
2. **But no rule extracts net-positive value from these agents on the escalated subset.**
   The best achievable equals the primary; the agents carry no usable signal beyond
   Ahmed on these hard, mostly-neutral code-switched cases (consistent with all three
   agents being weaker than Ahmed and 92% correlated).
3. **Best concrete fix = treat the agents as a correlated bloc and favor a strong
   primary** (agent-bloc + w_primary≈3), which is equivalent to *not overriding a strong
   primary*. Equivalently: **for a very strong primary, the agentic layer should be off
   (or strictly non-overriding).** Higher primary weight alone (w≈4) helps but is a
   weaker, partial version of the same idea.

This matches the primary-strength curve: at ~0.92 the primary is past the point where
the LLM agents can add value, so the right design choice is to protect it.

## Artifacts
- `error_attribution/consensus_simulation_results.json` — full per-rule results.
- `scripts/ahmed_consensus_simulation.py` — the offline simulator.
