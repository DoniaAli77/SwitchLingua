# C3 (Weak Primary) at gpt-4.1-mini — G vs G2 Bundle

Stronger-model + gate-choice on the weak C3 generated primary (`sz960_seed456` XLM-R, GPU),
threshold 0.90, 231 escalated. Tests (a) whether the 4o-mini→4.1-mini upgrade compounds on the
weak primary, and (b) whether the strong-primary gate/model-strength interaction (G2>G at
4.1-mini) replicates here. Date: 2026-07-02.

## Results

| config | full acc | macro F1 | escalated acc | net vs primary (W→C / C→W) |
|---|---|---|---|---|
| primary_only (C3) | 0.6956 | 0.6830 | 0.541 | — |
| G @ 4o-mini | 0.7604 | 0.7469 | 0.771 | +53 |
| v2 @ 4o-mini | 0.7531 | 0.7391 | 0.745 | +47 |
| **G @ 4.1-mini** ✅ | **0.7677** | **0.7527** | **0.797** | **+59** (71 / 12) |
| G2 @ 4.1-mini | 0.7665 | 0.7509 | 0.792 | +58 (71 / 13) |

## Finding 1 — the stronger model COMPOUNDS on the weak primary
G@4o-mini +53 → **G@4.1-mini +59** (full acc 0.7604 → 0.7677, new best C3). Contrast Ahmed,
where 4.1-mini only nudged +2→+3 (non-significant). The stronger model's recoverable slice is
**much larger on the weak primary**, so the upgrade pays off here and not on the strong one —
exactly the ceiling model's prediction. McNemar on C3 (b=71, c=12): χ² ≈ 42, **p ≪ 0.001 —
highly significant** (unlike every Ahmed result).

## Finding 2 — the gate interaction does NOT replicate on the weak primary
On Ahmed, G2 (selective gate) **beat** G (full gate) at 4.1-mini (0.9303 vs 0.9291), because
near the ceiling each over-veto of a correct polar is costly. On C3, **G ≈ G2** (0.7677 vs
0.7665; 184 vs 183 escalated correct — 1 sample). Reason: on the weak primary the agents
rescue **71** primary errors and both gates keep C→W tiny (12–13), so the full-vs-selective
gate distinction is **second-order** — swamped by the huge W→C. The selective gate's advantage
was **strong-primary-specific**; on the weak primary the gate choice barely matters.

## Consolidated best configs
- **Weak primary (C3): G @ gpt-4.1-mini = 0.7677 / 0.7527** (full gate; selective gives no edge).
- **Strong primary (Ahmed): G2 @ gpt-4.1-mini = 0.9303 / 0.9262** (selective gate; full gate
  over-vetoes).
- **Unifying rule:** the right gate depends on the regime — selective helps only where the
  system is near ceiling (strong primary); on a weak primary either gate is fine and the win
  comes from the agents fixing the primary's many errors.

## Caveats
- Single temp-0 draw; C3 gains (+58/+59) are far outside noise; G-vs-G2 gap (1 sample) is within it.
- Seed-456 is the best-dev C3 checkpoint (primary_only 0.6956 slightly high); relative
  comparisons are clean (same checkpoint).

## Artifacts
- `experiment_G_c3_41mini/*`, `experiment_G2_c3_41mini/*`
- Comparators: `EXPERIMENT_G_C3_RESULTS.md`, `EXPERIMENT_GPT41_GATE_ABLATION.md` (the Ahmed
  gate interaction this tests for replication).
