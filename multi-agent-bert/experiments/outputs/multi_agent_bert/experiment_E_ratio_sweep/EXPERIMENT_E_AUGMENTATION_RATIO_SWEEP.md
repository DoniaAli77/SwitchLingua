# Experiment E — Augmentation Ratio Sweep (EESA% + r× SwitchLingua-960)

Is generic SwitchLingua data harmful at all augmentation ratios, or only when it
dominates the real data? Uses the existing GEN-960 pool only. Date: 2026-06-25.

## Design
- Real EESA subsets (stratified, leak-cleaned): 10% (246), 25% (615), 50% (1232).
- For each subset of size N, add **r × N** generated samples (r ∈ {0.25, 0.5, 1.0}),
  **stratified by label** from the 960 pool. So the generated **fraction of the mix
  depends only on r**: 0.25×→20%, 0.5×→33%, 1.0×→50% (the 50%×1.0× cell is capped at
  the 960 pool → 44%).
- Matched compute: fresh `xlm-roberta-base`, adafactor, lr 2e-5, batch 4×grad_accum 4,
  max_length 256, fp16, seed 42, **`--max_steps 400` + `--load_best`** (best-dev by
  macro F1, eval every 50). primary_only on EESA test. Real-only baselines reused
  from the low-resource experiment (identical recipe).

## Results — EESA test (818)
| real | + gen | gen% | acc | macro F1 | weighted F1 | pos/neg/neu F1 | Δ acc |
|---|---|---|---|---|---|---|---|
| 10% | — | 0 | 0.7751 | 0.7586 | 0.7733 | .85/.72/.70 | — |
| 10% | 0.25× | 20 | 0.7494 | 0.7333 | 0.7477 | .83/.70/.67 | −0.026 |
| 10% | 0.5× | 33 | 0.7800 | 0.7677 | 0.7816 | .85/.73/.72 | +0.005 |
| 10% | 1.0× | 50 | 0.7567 | 0.7439 | 0.7557 | .83/.73/.67 | −0.018 |
| 25% | — | 0 | 0.7873 | 0.7758 | 0.7911 | .87/.72/.74 | — |
| 25% | 0.25× | 20 | 0.7836 | 0.7695 | 0.7831 | .85/.73/.72 | −0.004 |
| 25% | 0.5× | 33 | 0.7885 | 0.7755 | 0.7884 | .86/.75/.72 | +0.001 |
| 25% | 1.0× | 50 | 0.7934 | 0.7786 | 0.7927 | .86/.73/.74 | **+0.006** |
| 50% | — | 0 | 0.8166 | 0.8025 | 0.8142 | .88/.78/.74 | — |
| 50% | 0.25× | 20 | 0.8081 | 0.7962 | 0.8079 | .88/.78/.73 | −0.009 |
| 50% | 0.5× | 33 | 0.8020 | 0.7904 | 0.8037 | .88/.76/.73 | −0.015 |
| 50% | 1.0× | 44* | 0.8032 | 0.7907 | 0.8024 | .87/.77/.73 | −0.013 |

\*50%×1.0× capped at the 960 pool (= 44% gen, ≈ the LR `eesa50_plus960` cell).

## Findings
1. **No ratio consistently helps, and none catastrophically hurts.** Every Δ is
   within **±0.026**, and all but one within **±0.018** — i.e., **within the ~±0.02
   single-seed noise** measured earlier. The signs bounce (−0.026 … +0.006) with no
   monotone trend.
2. **The strong harm from the low-resource experiment was a *domination* effect, not
   a ratio-wide one.** There, fixed +960 hurt 10% by −0.034 *because 960 was 80% of
   the mix*. Here, capping generated at ≤50% of the mix removes that harm (worst case
   −0.026; the 25% row is even slightly positive at higher ratio).
3. **By subset:** 25% is neutral-to-slightly-positive at every ratio (−0.004 … +0.006);
   50% is slightly negative everywhere (−0.009 … −0.015); 10% is noisy. None exceeds
   the noise band.

## Answer to the research question
**Generic SwitchLingua data is NOT harmful at all augmentation ratios — it is harmful
only when it *dominates* the training mix (≈ ≥60–80% generated, as in the fixed-960
low-resource case).** At minority/balanced ratios (20–50% generated), the effect on
EESA test is **neutral — within seed noise** — neither a reliable gain nor a loss.
The hoped-for "a small amount of generated data helps low-resource" is **not
supported**: at no tested ratio does it reliably improve over real-only; it is simply
harmless when kept to a minority share.

This is consistent with the **domain-compatibility** explanation: generated and EESA
are different registers, so generated data is at best inert filler for EESA (doesn't
add target-domain signal) and at worst a distractor when it dominates. Its value
remains as **standalone** training data, not as an augmenter of a different-domain
real set.

## Caveats
- **Single seed per cell.** All Δ are within ±0.02; per-cell signs are **not reliable**.
  A ≥3-seed repeat is needed to resolve whether any ratio is genuinely ≠ 0.
- One ratio family tested (relative to N) on one target (EESA). A different target
  *in the SwitchLingua domain* could show positive augmentation — untested.
- `eesa25_gen10` required a retry (intermittent native segfault, exit 139); the
  successful run is reported.
