# Generated topic corpora: corpus x budget 2x2 on Silver-1163

One recipe throughout; only the training corpus and --max_steps differ.
primary_only evaluation, Silver used for final evaluation only. 3 seeds, no
significance claimed.

## Accuracy grid (mean +/- SD over seeds 42/43/44)

| corpus | 136 steps | 272 steps |
|---|---|---|
| Topic-540 | 0.6145 +/- 0.0004 | 0.6151 +/- 0.0158 |
| Topic-1080 | 0.6065 +/- 0.0099 | 0.6280 +/- 0.0057 |

Epochs per cell: 540@136 = 4.0 | 540@272 = 8.0 | 1080@136 = 2.0 | 1080@272 = 4.0

## Per-run detail

| corpus | budget | seed | steps | epoch | acc | macro-F1 | weighted-F1 | #classes | top share |
|---|--:|--:|---|--:|--:|--:|--:|--:|--:|
| Topic-540 | 136 | 42 | 136/136 | 4.0 | 0.6148 | 0.5429 | 0.5946 | 9 | 31.7% |
| Topic-540 | 136 | 43 | 136/136 | 4.0 | 0.6148 | 0.5575 | 0.6036 | 9 | 28.5% |
| Topic-540 | 136 | 44 | 136/136 | 4.0 | 0.6139 | 0.5415 | 0.5938 | 9 | 29.8% |
| Topic-540 | 272 | 42 | 272/272 | 8.0 | 0.5985 | 0.5214 | 0.5812 | 9 | 29.2% |
| Topic-540 | 272 | 43 | 272/272 | 8.0 | 0.6363 | 0.5810 | 0.6271 | 9 | 32.1% |
| Topic-540 | 272 | 44 | 272/272 | 8.0 | 0.6105 | 0.5396 | 0.5965 | 9 | 29.6% |
| Topic-1080 | 136 | 42 | 136/136 | 2.0 | 0.6053 | 0.5371 | 0.5909 | 9 | 27.8% |
| Topic-1080 | 136 | 43 | 136/136 | 2.0 | 0.5950 | 0.5387 | 0.5914 | 9 | 24.7% |
| Topic-1080 | 136 | 44 | 136/136 | 2.0 | 0.6191 | 0.5419 | 0.5957 | 9 | 27.9% |
| Topic-1080 | 272 | 42 | 272/272 | 4.0 | 0.6311 | 0.5785 | 0.6256 | 9 | 26.0% |
| Topic-1080 | 272 | 43 | 272/272 | 4.0 | 0.6199 | 0.5692 | 0.6209 | 9 | 24.9% |
| Topic-1080 | 272 | 44 | 272/272 | 4.0 | 0.6328 | 0.5672 | 0.6202 | 9 | 27.0% |

## Mean +/- SD, all metrics

| cell | acc | macro-F1 | weighted-F1 |
|---|---|---|---|
| Topic-540 @ 136 | 0.6145 +/- 0.0004 | 0.5473 +/- 0.0073 | 0.5973 +/- 0.0044 |
| Topic-540 @ 272 | 0.6151 +/- 0.0158 | 0.5473 +/- 0.0249 | 0.6016 +/- 0.0191 |
| Topic-1080 @ 136 | 0.6065 +/- 0.0099 | 0.5393 +/- 0.0020 | 0.5927 +/- 0.0021 |
| Topic-1080 @ 272 | 0.6280 +/- 0.0057 | 0.5716 +/- 0.0049 | 0.6223 +/- 0.0024 |

## A. EPOCH-MATCHED: 1080@272 (4.0 ep) - 540@136 (4.0 ep)

| seed | d acc | d macro-F1 | d weighted-F1 |
|--:|--:|--:|--:|
| 42 | +0.0163 | +0.0356 | +0.0310 |
| 43 | +0.0052 | +0.0116 | +0.0173 |
| 44 | +0.0189 | +0.0257 | +0.0264 |
| **mean** | **+0.0135** | **+0.0243** | **+0.0249** |

- signs: acc consistent, macro-F1 consistent, weighted-F1 consistent

## B. STEP-MATCHED: 1080@272 - 540@272

| seed | d acc | d macro-F1 | d weighted-F1 |
|--:|--:|--:|--:|
| 42 | +0.0327 | +0.0571 | +0.0444 |
| 43 | -0.0163 | -0.0118 | -0.0062 |
| 44 | +0.0224 | +0.0276 | +0.0237 |
| **mean** | **+0.0129** | **+0.0243** | **+0.0206** |

- signs: acc MIXED, macro-F1 MIXED, weighted-F1 MIXED

## C. BUDGET: 1080@272 - 1080@136

| seed | d acc | d macro-F1 | d weighted-F1 |
|--:|--:|--:|--:|
| 42 | +0.0258 | +0.0414 | +0.0348 |
| 43 | +0.0249 | +0.0305 | +0.0295 |
| 44 | +0.0138 | +0.0253 | +0.0245 |
| **mean** | **+0.0215** | **+0.0324** | **+0.0296** |

- signs: acc consistent, macro-F1 consistent, weighted-F1 consistent

## Collapse check (all 12 runs)

- runs with <9 predicted classes or top share >=50%: NONE
- top-share range across all runs: 24.7% - 32.1%

Reference: Original-180 corpus-draw sd at 136 steps = 0.0111 (9-run subset grid);
Topic-180 grand mean 0.6190, Topic-360 0.6022.