# Seed-Stability Check — C2 (480) vs C3 (960) generated sentiment

Tests whether the single-run "480 > 960" gap is real or seed variance. 3 seeds each
(42 = original C2/C3; 123, 456 = new). Fresh `xlm-roberta-base` per run, identical
recipe (adafactor, eff batch 16, max_length 256, 4 epochs), EESA dev/test,
primary_only only. Date: 2026-06-21.

## Per-run results (EESA test, 818)
| size | seed | dev acc | dev macF1 | test acc | test macF1 | test wF1 | pos F1 | neg F1 | neu F1 | pred dist p/n/neu |
|---|---|---|---|---|---|---|---|---|---|---|
| 480 | 42 | 0.6577 | 0.6484 | 0.6491 | 0.6382 | 0.6553 | 0.735 | 0.576 | 0.604 | 290/213/315 |
| 480 | 123 | 0.6430 | 0.6266 | 0.6308 | 0.6121 | 0.6356 | 0.734 | 0.507 | 0.595 | 280/154/384 |
| 480 | 456 | 0.6589 | 0.6497 | 0.6699 | 0.6532 | 0.6732 | 0.765 | 0.577 | 0.618 | 322/181/315 |
| 960 | 42 | 0.6455 | 0.6398 | 0.6381 | 0.6322 | 0.6454 | 0.709 | 0.588 | 0.600 | 263/276/279 |
| 960 | 123 | 0.6760 | 0.6646 | 0.6748 | 0.6624 | 0.6816 | 0.768 | 0.588 | 0.631 | 280/191/347 |
| 960 | 456 | 0.6980 | 0.6820 | 0.6956 | 0.6830 | 0.6971 | 0.780 | 0.661 | 0.609 | 332/251/235 |

(seed-42 rows are the original C2/C3 runs; 123/456 are the new seed runs under
`experiment_seed_stability/`, checkpoints under `checkpoints/seed_stability/`.)

## Summary (mean ± std, n = 3)
| size | test accuracy | test macro F1 | test range (acc) |
|---|---|---|---|
| 480 | **0.6500 ± 0.0160** | 0.6345 ± 0.0170 | 0.6308 – 0.6699 |
| 960 | **0.6695 ± 0.0238** | 0.6592 ± 0.0209 | 0.6381 – 0.6956 |

Mean difference (960 − 480): **+0.0195 acc / +0.0247 macro F1**.

## Verdict
1. **The single-run "480 > 960" was a seed artifact.** C3's seed-42 run (0.6381)
   was the *low outlier* of the 960 group; the other two 960 seeds (0.6748, 0.6956)
   beat every 480 seed.
2. **Does 960 improve, match, or underperform 480?** On average it **improves** —
   960 trends ~+0.02 acc / +0.025 macro F1 above 480, the opposite direction from
   the single-run impression. So 960 does **not** plateau or regress.
3. **Is the difference meaningful?** **Within seed variance.** Per-seed std is
   ~0.016–0.024 and the test-acc ranges overlap (480: 0.631–0.670; 960: 0.638–0.696);
   the +0.02 mean gap is ~1 std. At n = 3 this is **directionally favorable to 960
   but not statistically conclusive** (a t-test would not reach significance).

## Takeaways
- **480 vs 960 is mostly seed noise**; 960 is at least as good as 480, probably
  slightly better. The earlier "plateau/regression" finding (C3 report) is
  **retracted** — it was a single-seed artifact.
- **Methodology for the dataset paper:** report **mean ± std over ≥3 seeds**;
  single-run comparisons at this dataset scale are unreliable (std ~±0.02, range
  ~0.04–0.06). For a firm 480-vs-960 claim, use ≥5 seeds.
- Outputs in separate seed folders; no C2/C3 checkpoints overwritten; no
  full_agentic; no Ahmed models.
