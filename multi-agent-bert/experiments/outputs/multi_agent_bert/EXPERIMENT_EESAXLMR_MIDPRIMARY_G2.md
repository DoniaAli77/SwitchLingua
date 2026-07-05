# G2 @ gpt-4.1-mini on the EESA-trained XLM-R Primary (E0) — the Missing Mid Point

Fills the gap: agents on a **real XLM-R fine-tuned on EESA** (the E0 checkpoint, your own
recipe) — not the external Ahmed predictions, not the generated-data C3. G2 (selective gate),
semantic_v1, threshold 0.90, gpt-4.1-mini, EESA test (818). Date: 2026-07-05.

## Result — significant agent gain at mid strength

| | accuracy | macro F1 | escalated acc |
|---|---|---|---|
| E0 primary_only | 0.8533 | 0.8409 | — |
| **E0 + G2 agents** | **0.8826** | **0.8726** | 129/169 = 0.763 |

- **+0.0293 accuracy** (698 → 722 / 818), **net +24** (W→C 40, C→W 16).
- Escalation 169/818 (20.7%) at th 0.90; primary was only **0.621** on that escalated subset.
- **McNemar χ² ≈ 9.4, p ≈ 0.002 — statistically SIGNIFICANT.** (Unlike the strong primary.)

## The primary-strength curve — now three real points, all with G2 @ gpt-4.1-mini

| primary | what it is | standalone | primary-on-escalated | **+G2 agents** | gain | net | significant |
|---|---|---|---|---|---|---|---|
| **C3** | XLM-R on *generated* data (weak) | 0.6956 | 0.541 | 0.7665 | **+0.071** | +58 | p≪0.001 ✅ |
| **E0** | XLM-R on *EESA* (mid) | 0.8533 | 0.621 | **0.8826** | **+0.029** | +24 | p≈0.002 ✅ |
| **Ahmed** | external precomputed (strong) | 0.9254 | 0.750 | 0.9303 | +0.005 | +4 | p≈0.37 ✗ |

**This is the cleanest single demonstration of the whole thesis**, and it uses your own trained
model for the mid point:
1. **As the primary strengthens (0.70 → 0.85 → 0.93), the agent gain shrinks monotonically**
   (+0.071 → +0.029 → +0.005) and the significance fades (p≪0.001 → p≈0.002 → not significant).
2. **The agents lift the escalated subset to ~0.76–0.79 *regardless* of primary strength**
   (C3 0.766, E0 0.763, Ahmed 0.786) — a stable **agent ceiling ~0.77**.
3. So the gain is fully explained by **how far the primary sits below that ceiling on its own
   escalated subset**: C3 0.54 (huge room), E0 0.62 (moderate room), Ahmed 0.75 (almost none).
   This is the root-cause equation — **gain ≈ (agent_ceiling − primary_on_escalated) ×
   escalation_rate** — now confirmed empirically at three strengths, not just predicted.

## Why this point matters for the thesis
- It is the **most honest "XLM-R primary" result**: a real XLM-R you fine-tuned on real EESA
  data, evaluated on EESA test, with the agent stack on top — and the agents give a **real,
  significant +2.9 points**.
- It converts the earlier *estimated* "EESA-XLM-R +0.027" (a formula guess) into a **measured
  +0.029, significant** — the estimate was essentially exact.
- Together with C3 and Ahmed it gives a **3-point monotone curve** that visually and
  statistically tells the story: *the multi-agent layer's value is real and grows as the base
  classifier weakens.*

## Caveats
- Single temp-0 draw (agents) and single seed (E0 checkpoint). The +24 net is well outside
  noise (p≈0.002); a multi-seed E0 + multi-draw agent pass would tighten error bars.
- Threshold 0.90 chosen to match C3 and to escalate a meaningful 20.7% where the primary is
  weak (0.62); other thresholds trade escalation size vs cost.
- Only G2 (selective gate) run here; on a mid primary G-vs-G2 was not separately tested (on C3
  they tied; on Ahmed G2 edged G).

## Artifacts
- `experiment_G2_eesaXLMR_41mini/*` (checkpoint `expE0_eesa_only_adafactor_xlmr`).
- Curve comparators: `EXPERIMENT_C3_GPT41_GATE_BUNDLE.md`, `EXPERIMENT_GPT41_GATE_ABLATION.md`,
  `EXPERIMENT_ESCALATED_CEILING_ROOT_CAUSE.md` (the equation this confirms).
