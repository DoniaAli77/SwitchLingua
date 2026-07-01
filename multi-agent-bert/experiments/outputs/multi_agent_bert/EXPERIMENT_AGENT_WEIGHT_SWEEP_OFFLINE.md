# Offline Per-Agent Static Weight Sweep — Exploratory

Why did we only sweep `w_primary` and not per-agent static weights, and does per-agent weighting
help? **Offline only — no OpenAI calls, no retraining, no generation.** IntentGate kept as a
non-voting veto. Date: 2026-07-01.

## Why only w_primary was tested before
1. **It was the only exposed ablation knob.** `--consensus_primary_weight` / `w_primary` is the
   single consensus weight surfaced as a CLI/config override (audit Fix #2); per-agent weights were
   never exposed as a flag.
2. **The design captures serialized *labels only*.** A faithful per-agent weight sweep needs the
   *confidence-weighted* consensus recomputed, i.e. per-agent **confidences** — and the Design G / v3
   captures store `lexical_label / polarity_label / contextual_label` but **not** their confidences
   (`ahmed_conf`, the primary's confidence, *is* stored, which is why w_primary was sweepable).
3. **The prompt/architecture redesigns were a *more expressive* form of reweighting.** Replacing the
   weak Logic agent with Polarity, refining Contextual, and adding the IntentGate change *what an
   agent contributes*, not merely *how much its vote counts* — a superset of static reweighting.

## Data availability (checked)
| capture | per-agent confidences? | usable for faithful sweep? |
|---|---|---|
| Design **G**, **v3** | **NO** (labels only; `*_conf` absent) | **No** — faithful sweep impossible |
| original frozen-primary (default prompts) | **YES** (`lexical_conf/logic_conf/contextual_conf`) | Yes — but it is a **different design** (Logic not Polarity, **no gate**, default prompts) |

Per the instruction: **the requested G/v3 sweep cannot be run faithfully** (no confidences). Below,
(A) a clearly-labelled **label-only proxy** on G, and (B) a **faithful** confidence-weighted sweep on
the one capture that has confidences (a different, weaker design) as a reference.

---

## A) Label-only proxy on Design G — [PROXY, agent confidences flattened to 1.0]
Consensus reconstructed from labels: each agent contributes `weight × 1.0` (real agent confidence
*unavailable*), the primary contributes `w_primary × ahmed_conf` (real), non-positional tie-break,
IntentGate veto re-applied. **Baseline (saved G): esc 66/84, full 0.9291, net +3** (capture draw,
within noise of headline G 0.9279/+2).

| config | full acc | macro F1 | esc | W→C | C→W | net | changed vs G | help / hurt |
|---|---|---|---|---|---|---|---|---|
| 1 equal (1,1,1,1) | 0.9291 | 0.9257 | 66/84 | 10 | 7 | +3 | 0 | 0/0 |
| 2 ctx-heavy (0.75,1,1.5,1) | 0.9291 | 0.9257 | 66/84 | 10 | 7 | +3 | 0 | 0/0 |
| 3 pol+ctx-heavy (0.75,1.25,1.5,1) | 0.9291 | 0.9257 | 66/84 | 10 | 7 | +3 | 0 | 0/0 |
| 4 primary-protected (1,1,1.25,1.25) | 0.9291 | 0.9257 | 66/84 | 10 | 7 | +3 | 0 | 0/0 |
| 5 lexical-light (0.5,1.25,1.5,1) | 0.9291 | 0.9257 | 66/84 | 10 | 7 | +3 | 0 | 0/0 |
| **full grid (144 configs)** | — | — | best **66/84** | — | — | — | **0 configs beat baseline** | — |

**Every one of the 144 grid configs and all 5 presets produced the identical result — zero changed
decisions.** On G's escalated set, the agent+primary votes are already so aligned/decisive that no
weight in L∈[0.5,1] × P∈[0.75,1.25] × C∈[1,2] × Pr∈[0.75,1.5] flips a single case. Structural reason:
the agents are ~84% correlated (they mostly agree, so reweighting an agreeing bloc changes nothing),
and the neutral-protection domain is already handled by the gate. **Per-agent static weighting has no
headroom on G.** (Caveat: the proxy flattens agent confidence; but the 0/144 result plus the
correlation structure make it very unlikely a faithful G sweep would differ materially.)

## B) Faithful confidence-weighted sweep — original capture (different, weaker design)
Real per-agent confidences (`lexical_conf/logic_conf/contextual_conf`) + `ahmed_conf`, no gate,
default prompts. **Baseline (equal weights): esc 59/84, full 0.9205, net −4** (reproduces the
original run). Here reweighting **does** bite:

| config | full acc | macro F1 | esc | W→C | C→W | net | changed | help / hurt |
|---|---|---|---|---|---|---|---|---|
| 1 equal (1,1,1,1) | 0.9205 | 0.9153 | 59/84 | 11 | 15 | −4 | 0 | 0/0 |
| 2 ctx-heavy (0.75,1,1.5,1) | 0.9230 | 0.9183 | 61/84 | 11 | 13 | **−2** | 2 | **2/0** |
| 3 pol+ctx-heavy (0.75,1.25,1.5,1) | 0.9230 | 0.9183 | 61/84 | 11 | 13 | −2 | 2 | 2/0 |
| 4 primary-protected (1,1,1.25,1.25) | 0.9230 | 0.9183 | 61/84 | 11 | 13 | −2 | 2 | 2/0 |
| 5 lexical-light (0.5,1.25,1.5,1) | 0.9230 | 0.9183 | 61/84 | 11 | 13 | −2 | 2 | 2/0 |
| **full grid (144)** | best **0.9242** | — | best **62/84** | — | — | best **−1** | — | **123/144 beat baseline** |

Best grid configs: **high Contextual (C=2.0), low Lexical (0.5–0.75)**, primary 0.75–1.25 →
esc 62/84, full 0.9242, net −1. **The signal is clear and consistent: up-weight Contextual (the
strongest agent), down-weight Lexical (the over-reader).** The 2 clean gains (help 2 / hurt 0) are
harmful-override breaks recovered by letting Contextual outweigh a wrong Lexical+Polarity read.

**But even the best faithful reweighting (net −1, 0.9242) is still below primary_only (0.9254) and far
below G (0.9279 / +2).** Static reweighting improves a *weak, uncalibrated* consensus (−4 → −1) but
cannot reach what the role-refined prompts + IntentGate already achieve.

---

## Interpretation
- **Per-agent static weighting is a real lever — but only on a weak/uncalibrated consensus, and its
  ceiling is below what we already have.** On the original consensus it recovers −4 → −1 by favouring
  Contextual and discounting Lexical; on G it does **nothing** (0/144), because the prompt/agent
  redesigns already rebalanced the panel (replaced weak Logic→Polarity, refined Contextual, added the
  gate) — a *more expressive* reweighting that G has already banked.
- **The direction the sweep prefers is exactly the direction the whole design line took:** more weight
  to Contextual, less to Lexical. This cross-validates the redesign choices — and shows they
  subsumed static reweighting rather than leaving it on the table.
- **This answers the original question:** we didn't sweep per-agent weights earlier because (a) the
  knob wasn't exposed, (b) G/v3 didn't serialize agent confidences, and (c) the redesigns were the
  chosen, stronger mechanism. Now that we've checked: on the strong-primary lead (G) there is **no
  weighting headroom**; on a weak consensus it helps but plateaus below G.

## Honesty / limitations (do not overclaim)
- **A is a proxy** (agent confidences flattened to 1.0); it should not be read as a faithful G sweep,
  though the 0/144 + correlation structure make a null result very likely faithfully too.
- **B is faithful but a different design** (default prompts, Logic not Polarity, no gate); its numbers
  do **not** transfer to G in absolute terms — only the *direction* (Contextual↑, Lexical↓) does.
- **B's grid is in-sample (test set).** "123/144 beat baseline" and the best config (C=2.0, L=0.5) are
  **test-optimized → overfit risk.** **Do not promote any swept weight as final.** Per the
  instruction, any candidate weighting must be **validated on dev or on the C3 primary** before use.
- Consistent with the aggregation-rescoring and root-cause findings: on the strong primary the system
  is at its aggregation ceiling; the only regime where reweighting could pay off is the **weak C3
  primary**, where the panel carries the decision.

## Conclusion
> **Per-agent static weighting has no headroom on the strong-primary lead (G): 0 of 144 grid configs
> changed a single decision. On a weak/uncalibrated consensus it does help (−4 → −1) by up-weighting
> Contextual and down-weighting Lexical — the same direction as our prompt redesigns — but it plateaus
> below primary_only and well below G. So static reweighting is a weaker, subsumed version of the
> agent-role work already done; it is not a new lever on the strong primary. The faithful sweep is
> only possible on a different, weaker capture, and its best weights are test-optimized — not to be
> promoted without dev/C3 validation.**

## Artifacts
- Sweep computed inline from `experiment_ahmed_designG_intent_gate/error_attribution/` (labels; proxy)
  and `experiment_ahmed_frozen_primary/error_attribution/` (with confidences; faithful).
- Basis: `EXPERIMENT_CONSENSUS_INVESTIGATION_SUMMARY.md`, `EXPERIMENT_G_OFFLINE_CONSENSUS_RESCORING.md`,
  `EXPERIMENT_AHMED_CONSENSUS_SIMULATION.md` (w_primary), `EXPERIMENT_CONSENSUS_LOSS_ANALYSIS.md`.
