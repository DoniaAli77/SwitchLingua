# Topic-540 compute control: 272 steps (8 epochs) vs 136 steps (4 epochs)

Same fixed Topic-540 corpus, model seeds 42/43/44, identical recipe; training
duration is the only difference. No dev set / load_best off in BOTH arms (the only
topic dev set available is ArEnTC, excluded by this study line), so the paired
difference isolates duration. Silver-1163 used for final evaluation only.

## Step verification

| seed | steps reached | final epoch | reached 272 |
|--:|---|--:|---|
| 42 | 272/272 | 8.0 | YES |
| 43 | 272/272 | 8.0 | YES |
| 44 | 272/272 | 8.0 | YES |

## Per-run, Silver-1163

| budget | seed | acc | macro-F1 | weighted-F1 | #pred classes | top share | collapsed |
|---|--:|--:|--:|--:|--:|--:|---|
| 136 steps | 42 | 0.6148 | 0.5429 | 0.5946 | 9 | 31.7% | no |
| 136 steps | 43 | 0.6148 | 0.5575 | 0.6036 | 9 | 28.5% | no |
| 136 steps | 44 | 0.6139 | 0.5415 | 0.5938 | 9 | 29.8% | no |
| 272 steps | 42 | 0.5985 | 0.5214 | 0.5812 | 9 | 29.2% | no |
| 272 steps | 43 | 0.6363 | 0.5810 | 0.6271 | 9 | 32.1% | no |
| 272 steps | 44 | 0.6105 | 0.5396 | 0.5965 | 9 | 29.6% | no |

## Mean +/- SD

| budget | acc | macro-F1 | weighted-F1 |
|---|---|---|---|
| 136 steps | 0.6145 +/- 0.0004 | 0.5473 +/- 0.0073 | 0.5973 +/- 0.0044 |
| 272 steps | 0.6151 +/- 0.0158 | 0.5473 +/- 0.0249 | 0.6016 +/- 0.0191 |

## Paired differences (272 - 136), same model seed

| seed | d acc | d macro-F1 | d weighted-F1 |
|--:|--:|--:|--:|
| 42 | -0.0163 | -0.0215 | -0.0134 |
| 43 | +0.0215 | +0.0234 | +0.0235 |
| 44 | -0.0034 | -0.0019 | +0.0027 |
| **mean** | **+0.0006** | **+0.0000** | **+0.0043** |

- acc: -0.0163 +0.0215 -0.0034 -> MIXED
- macro_f1: -0.0215 +0.0234 -0.0019 -> MIXED
- weighted_f1: -0.0134 +0.0235 +0.0027 -> MIXED

## Per-class F1 (mean over seeds)

| class | 136 steps | 272 steps | delta |
|---|--:|--:|--:|
| business | 0.367 | 0.447 | +0.079 |
| education | 0.368 | 0.414 | +0.045 |
| health | 0.493 | 0.400 | -0.093 |
| shopping | 0.490 | 0.530 | +0.040 |
| medical | 0.576 | 0.552 | -0.025 |
| sports | 0.628 | 0.607 | -0.021 |
| tech | 0.738 | 0.746 | +0.008 |
| finance | 0.775 | 0.756 | -0.018 |
| social | 0.490 | 0.474 | -0.016 |

## Checks

- all runs reached 272 steps: YES
- collapsed runs at 272 steps: NONE
- reference: Original-180 corpus-draw sd at 136 steps = 0.0111 (9-run grid)