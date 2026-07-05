# sequential_sentiment_v2 on the WEAK C3 Primary — Results + Parallel-vs-Sequential Verdict

Real run of forward-pragmatics v2 on the C3 generated primary (`sz960_seed456` transformer,
GPU), threshold 0.90, gpt-4o-mini temp 0, same EESA test / 231 escalated. Completes the
parallel-vs-sequential comparison on the regime where agents matter. Date: 2026-07-02.

## Track 3 (weak C3 primary) — full comparison

| system | accuracy | macro F1 | escalated acc | net vs primary | W→C / C→W | cost |
|---|---|---|---|---|---|---|
| primary_only (C3) | 0.6956 | 0.6830 | 125/231 = 0.541 | — | — | — |
| full_agentic (default trio) | 0.7543 | 0.7387 | — | — | — | — |
| **Design G (parallel)** ✅ | **0.7604** | **0.7469** | **178/231 = 0.771** | **+53** | **65 / 12** | $0.176 |
| **sequential v2** | 0.7531 | 0.7391 | 172/231 = 0.745 | **+47** | **75 / 28** | $0.125 |

## Two clean findings

### 1. v2 HELPS on the weak primary (+47) — the exact mirror of Ahmed (−11)
Same v2 pipeline: **−11 on strong Ahmed, +47 on weak C3.** This is the ceiling thesis made
undeniable — v2's aggressive intervention is *harmful* when the primary is already right on
its escalated subset (Ahmed, 0.75) and *helpful* when the primary is wrong there (C3, 0.54).
The architecture didn't change; the primary's headroom did. v2's forward-pragmatics
hypothesis — that pragmatic features pay off *where agents matter* — is **confirmed** on C3.

### 2. But parallel G still WINS (0.7604 vs 0.7531)
Head-to-head on the identical 231-escalated set:
- **G rescues 65 and breaks 12 (net +53).** v2 **rescues 75 and breaks 28 (net +47).**
- v2 is **more active** (103 changed decisions vs G's 77) and **higher-recall** (75 rescues
  vs 65) — but pays an **aggressiveness tax**: it breaks **28** correct primary calls vs G's
  **12**. The extra 10 rescues don't cover the extra 16 breakages.
- The difference is **G's IntentGate veto**: a non-voting guard that caps C→W damage while
  still capturing most rescues. v2 has no such brake — the same un-anchored aggressiveness
  that cratered on Ahmed here helps, but less efficiently than G's conservative veto.

**The 6-sample final-accuracy gap (622 vs 616) is small and likely within temp-0 noise; the
robust, interpretable difference is the churn: v2 breaks 2.3× as many correct answers as G.**

## Verdict on the whole parallel-vs-sequential question
- **On a strong primary:** parallel ≈ 0, sequential ≤ 0 (both at/through ceiling). Parallel
  (G, +2) > sequential (v1 −1, v2 −11).
- **On a weak primary:** both help a lot; **parallel (G, +53) ≥ sequential (v2, +47)**.
- **Conclusion:** across *both* regimes, the **parallel Design G is the winner or tied**, and
  its advantage is the **IntentGate veto's damage control**, not raw reasoning power.
  Sequential forward-pragmatics is a viable, cheaper ($0.125 vs $0.176) alternative that
  *works where it should*, but its lack of a veto makes it strictly less efficient than G.
  **Staged reasoning did not beat parallel voting-plus-veto in either regime.**

## What this implies for next steps
- The best system remains **Design G**. To push C3 higher, the lever is **reducing v2/G's
  C→W breakage** (a veto for v2, or a stronger base model), or a **better primary**, not more
  reasoning topology.
- A natural cheap add: give v2 an **IntentGate-style veto** (a v2.1) to cut its 28 breakages
  toward G's 12 — that is the single change most likely to make sequential competitive.

## Caveats
- Single temp-0 draw; the G-vs-v2 gap (6 samples) is within noise. Both gains over primary
  (+53, +47) are far outside noise (McNemar p≪0.001).
- Seed-456 is the best-dev C3 checkpoint (primary slightly high); relative comparisons are
  clean (same checkpoint for all three).
- `decided_by` / feature-usage not serialized; a cheap trace re-capture would show how v2's
  28 breakages split across the gate / sarcasm / implicit-stance features.

## Artifacts
- `experiment_seqv2_c3/seqv2_c3__{primary_only,full_pipeline}_*`, `…__llm_usage.json`
- Comparators: `EXPERIMENT_G_C3_RESULTS.md`, `EXPERIMENT_SEQUENTIAL_SENTIMENT_V2_AHMED_RESULTS.md`
