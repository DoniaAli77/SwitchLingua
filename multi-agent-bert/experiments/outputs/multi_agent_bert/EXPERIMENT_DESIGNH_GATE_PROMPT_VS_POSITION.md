# Design H — Is the Selective IntentGate's Benefit from its PROMPT or its POSITION?

The Selective IntentGate (G2) and the Polarity agent ask a substantially similar
question ("does the author express a stance, or is this platform/meta/mention?").
This ablation asks whether the gate's contribution comes from that **prompt
content** or from its **architectural position** as a non-voting, post-consensus
veto.

**Design H** = Lexical + Polarity + Contextual (3 voters, **no separate gate**),
where the Selective IntentGate's exact meta/mention-vs-stance criteria — including
the "absence of an explicit sentiment word is NOT enough for neutral" rule — are
merged verbatim into the Polarity agent's own system prompt. Everything else is
identical to G2: Ahmed precomputed primary, threshold 0.70, semantic_v1,
consensus w_primary 1.0, gpt-4.1-mini. Date: 2026-08-04.

## Result

| config | accuracy | macro F1 | escalated | net vs primary |
|---|---|---|---|---|
| primary_only | 0.9254 | 0.9207 | 63/84 | — |
| Design C (no gate) | 0.9266 | 0.9216 | 64/84 | +1 |
| **Design H (gate merged into Polarity)** | **0.9267** | **0.9217** | **64/84** | **+1** (13 W→C / 12 C→W) |
| Design G2 (gate as post-consensus veto) | 0.9303 | 0.9262 | 67/84 | +4 |

0 fallbacks.

## Finding

**Design H is indistinguishable from Design C** (macro-F1 0.9217 vs 0.9216;
identical 64/84 escalated). Merging the gate's criteria into a voting agent
recovers **none** of the gate's benefit. The same criteria are worth **+3
escalated samples** when applied as a post-consensus veto (G2) and **0** when
applied inside a vote (H).

**The gate's contribution is therefore positional, not content-based.**

## Mechanism
The gate's niche is the case where Polarity judges a text as meta/mention (voting
neutral, agreeing with the primary) but is outvoted by Lexical + Contextual. A
merged prompt makes Polarity *better informed* but leaves it a single vote, and in
confidence-weighted voting one voter cannot outvote two agreeing voters:

```
Polarity neutral @ 0.95  +  primary neutral @ 0.50  = 1.45
Lexical negative @ 0.85  +  Contextual negative @ 0.85 = 1.70   ← still wins
```

The veto escapes this arithmetic entirely by acting *after* the weighted sum. It
is also **asymmetric**: it can only block a move away from the primary, never force
a move toward its own view — so it behaves as a conservative brake, not a louder
voter.

## Consequences for the thesis
1. The gate should be described as a **conditional, one-directional veto**, not as
   an additional opinion detector or verification agent. Its value is structural.
2. The overlap between the gate's and Polarity's prompts is **not** redundancy that
   could be optimised away by merging — H demonstrates the merge is inert.
3. Related prior evidence: Designs E/F placed intent reasoning as a *voting* agent
   and were also outperformed by the non-voting gate designs, consistent with this
   result.

## Reproduce
- Variant: `lexical_polaritygate_contextual` (opt-in; default behaviour unchanged).
- Prompt: `SYSTEM_PROMPT_GATE_MERGED` in `src/prompts/polarity_prompt.py`
  (`get_system_prompt("gate_merged")`); `PolarityAgent(system_variant="gate_merged")`.
- Wiring: `evaluate_pipeline.py` (Design H branch; no `IntentGateAgent` constructed).
- Script: `scripts/ahmed_designH_merged_gate.py`.
- Artifacts: `experiment_ahmed_designH_merged_gate/{records.json,designH_report.txt}`.
