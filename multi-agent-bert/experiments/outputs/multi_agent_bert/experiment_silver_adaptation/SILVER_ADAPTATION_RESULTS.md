# Silver topic-adaptation experiment (secondary, exploratory)

Does pretraining XLM-R on the generated Topic-540 corpus improve adaptation to the
limited Silver-labelled target domain? Three systems x model seeds 42/43/44, evaluated
on the frozen 300-row Silver hybrid test split.

EXPLORATORY: labels are multi-LLM consensus Silver labels, not human gold, and the split
is not completely video-disjoint - one sports-dominant video contributes a separated
17-row test block. No statistical significance is claimed from three seeds.

## Checkpoint provenance and reproduction check

The per-seed Topic-540 136-step checkpoints had been deleted by the matched-compute
script. They were regenerated with the identical recipe and seed, then verified against
the recorded Silver-1163 accuracies:

| seed | recorded | regenerated | match |
|--:|--:|--:|---|
| 42 | 0.6148 | 0.6148 | YES |
| 43 | 0.6148 | 0.6148 | YES |
| 44 | 0.6139 | 0.6139 | YES |

## Optimizer steps

| system | seed | expected | observed | match |
|---|--:|--:|--:|---|
| A. G-only (Topic-540) | 42 | 136 | 136 | YES |
| A. G-only (Topic-540) | 43 | 136 | 136 | YES |
| A. G-only (Topic-540) | 44 | 136 | 136 | YES |
| B. S-only (Silver 860) | 42 | 216 | 216 | YES |
| B. S-only (Silver 860) | 43 | 216 | 216 | YES |
| B. S-only (Silver 860) | 44 | 216 | 216 | YES |
| C. G->S (two-stage) | 42 | 216 | 216 | YES |
| C. G->S (two-stage) | 43 | 216 | 216 | YES |
| C. G->S (two-stage) | 44 | 216 | 216 | YES |

## Headline metrics, 300-row Silver hybrid test

| system | seed | accuracy | macro-F1 | weighted-F1 | #classes predicted | all 9 | top class | top share |
|---|--:|--:|--:|--:|--:|---|---|--:|
| A. G-only (Topic-540) | 42 | 0.6733 | 0.5977 | 0.6601 | 9 | yes | tech | 33.3% |
| A. G-only (Topic-540) | 43 | 0.6267 | 0.5428 | 0.6234 | 9 | yes | tech | 27.3% |
| A. G-only (Topic-540) | 44 | 0.6467 | 0.5556 | 0.6316 | 9 | yes | tech | 30.0% |
| B. S-only (Silver 860) | 42 | 0.6033 | 0.4547 | 0.5585 | 8 | NO | tech | 33.0% |
| B. S-only (Silver 860) | 43 | 0.4333 | 0.1413 | 0.2952 | 3 | NO | tech | 70.3% |
| B. S-only (Silver 860) | 44 | 0.6367 | 0.5466 | 0.6121 | 8 | NO | tech | 29.3% |
| C. G->S (two-stage) | 42 | 0.7767 | 0.7596 | 0.7783 | 9 | yes | tech | 26.3% |
| C. G->S (two-stage) | 43 | 0.7467 | 0.7120 | 0.7499 | 9 | yes | tech | 24.0% |
| C. G->S (two-stage) | 44 | 0.7133 | 0.6804 | 0.7144 | 9 | yes | tech | 24.3% |

## Mean +/- sample SD over seeds 42/43/44

| system | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| A. G-only (Topic-540) | 0.6489 +/- 0.0234 | 0.5654 +/- 0.0287 | 0.6383 +/- 0.0193 |
| B. S-only (Silver 860) | 0.5578 +/- 0.1091 | 0.3808 +/- 0.2125 | 0.4886 +/- 0.1696 |
| C. G->S (two-stage) | 0.7456 +/- 0.0317 | 0.7173 +/- 0.0399 | 0.7475 +/- 0.0321 |
| majority baseline (always tech) | 0.2467 | 0.0440 | - |

## PRIMARY CONTRAST: paired same-seed G->S minus S-only

| seed | d accuracy | d macro-F1 | d weighted-F1 |
|--:|--:|--:|--:|
| 42 | +0.1733 | +0.3049 | +0.2199 |
| 43 | +0.3133 | +0.5707 | +0.4546 |
| 44 | +0.0767 | +0.1338 | +0.1023 |
| **mean** | **+0.1878** | **+0.3365** | **+0.2589** |

- macro-F1 differences: +0.3049 +0.5707 +0.1338 -> CONSISTENT positive

**CONCLUSION: generated pretraining provided a consistent adaptation benefit.** All
three paired macro-F1 differences are positive. Three seeds; no significance claimed.

## Per-class precision / recall / F1

### A. G-only (Topic-540)

| class | support | P s42 | R s42 | F1 s42 | P s43 | R s43 | F1 s43 | P s44 | R s44 | F1 s44 | mean F1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| business | 37 | 0.773 | 0.459 | 0.576 | 0.778 | 0.378 | 0.509 | 0.789 | 0.405 | 0.536 | 0.540 |
| education | 18 | 1.000 | 0.222 | 0.364 | 0.444 | 0.222 | 0.296 | 0.500 | 0.278 | 0.357 | 0.339 |
| health | 22 | 0.727 | 0.364 | 0.485 | 0.727 | 0.364 | 0.485 | 1.000 | 0.182 | 0.308 | 0.426 |
| shopping | 20 | 0.458 | 0.550 | 0.500 | 0.300 | 0.450 | 0.360 | 0.409 | 0.450 | 0.429 | 0.430 |
| medical | 21 | 0.500 | 0.619 | 0.553 | 0.382 | 0.619 | 0.473 | 0.366 | 0.714 | 0.484 | 0.503 |
| sports | 17 | 0.571 | 0.706 | 0.632 | 0.667 | 0.588 | 0.625 | 0.688 | 0.647 | 0.667 | 0.641 |
| tech | 74 | 0.620 | 0.838 | 0.713 | 0.671 | 0.743 | 0.705 | 0.667 | 0.811 | 0.732 | 0.716 |
| finance | 67 | 0.843 | 0.881 | 0.861 | 0.881 | 0.881 | 0.881 | 0.819 | 0.881 | 0.849 | 0.864 |
| social | 24 | 0.727 | 0.667 | 0.696 | 0.471 | 0.667 | 0.552 | 0.615 | 0.667 | 0.640 | 0.629 |

### B. S-only (Silver 860)

| class | support | P s42 | R s42 | F1 s42 | P s43 | R s43 | F1 s43 | P s44 | R s44 | F1 s44 | mean F1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| business | 37 | 0.615 | 0.432 | 0.508 | 0.000 | 0.000 | 0.000 | 0.564 | 0.595 | 0.579 | 0.362 |
| education | 18 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.444 | 0.667 | 0.533 | 0.178 |
| health | 22 | 0.302 | 0.727 | 0.427 | 0.000 | 0.000 | 0.000 | 0.333 | 0.500 | 0.400 | 0.276 |
| shopping | 20 | 0.500 | 0.400 | 0.444 | 0.000 | 0.000 | 0.000 | 0.533 | 0.400 | 0.457 | 0.301 |
| medical | 21 | 0.600 | 0.429 | 0.500 | 0.000 | 0.000 | 0.000 | 0.818 | 0.429 | 0.562 | 0.354 |
| sports | 17 | 0.900 | 0.529 | 0.667 | 0.000 | 0.000 | 0.000 | 0.929 | 0.765 | 0.839 | 0.502 |
| tech | 74 | 0.626 | 0.838 | 0.717 | 0.327 | 0.932 | 0.484 | 0.636 | 0.757 | 0.691 | 0.631 |
| finance | 67 | 0.762 | 0.910 | 0.830 | 0.693 | 0.910 | 0.787 | 0.822 | 0.896 | 0.857 | 0.825 |
| social | 24 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### C. G->S (two-stage)

| class | support | P s42 | R s42 | F1 s42 | P s43 | R s43 | F1 s43 | P s44 | R s44 | F1 s44 | mean F1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| business | 37 | 0.635 | 0.892 | 0.742 | 0.633 | 0.838 | 0.721 | 0.564 | 0.838 | 0.674 | 0.712 |
| education | 18 | 0.733 | 0.611 | 0.667 | 0.647 | 0.611 | 0.629 | 0.688 | 0.611 | 0.647 | 0.647 |
| health | 22 | 0.722 | 0.591 | 0.650 | 0.600 | 0.545 | 0.571 | 0.714 | 0.455 | 0.556 | 0.592 |
| shopping | 20 | 0.714 | 0.750 | 0.732 | 0.500 | 0.650 | 0.565 | 0.579 | 0.550 | 0.564 | 0.620 |
| medical | 21 | 0.696 | 0.762 | 0.727 | 0.812 | 0.619 | 0.703 | 0.778 | 0.667 | 0.718 | 0.716 |
| sports | 17 | 1.000 | 0.824 | 0.903 | 0.882 | 0.882 | 0.882 | 0.722 | 0.765 | 0.743 | 0.843 |
| tech | 74 | 0.772 | 0.824 | 0.797 | 0.792 | 0.770 | 0.781 | 0.753 | 0.743 | 0.748 | 0.776 |
| finance | 67 | 0.945 | 0.776 | 0.852 | 0.949 | 0.836 | 0.889 | 0.897 | 0.776 | 0.832 | 0.858 |
| social | 24 | 0.783 | 0.750 | 0.766 | 0.667 | 0.667 | 0.667 | 0.586 | 0.708 | 0.642 | 0.691 |

## Predicted-class counts

| system | seed | business | education | health | shopping | medical | sports | tech | finance | social |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A. G-only (Topic-540) | 42 | 22 | 4 | 11 | 24 | 26 | 21 | 100 | 70 | 22 |
| A. G-only (Topic-540) | 43 | 18 | 9 | 11 | 30 | 34 | 15 | 82 | 67 | 34 |
| A. G-only (Topic-540) | 44 | 19 | 10 | 4 | 22 | 41 | 16 | 90 | 72 | 26 |
| B. S-only (Silver 860) | 42 | 26 | 1 | 53 | 16 | 15 | 10 | 99 | 80 | 0 |
| B. S-only (Silver 860) | 43 | 1 | 0 | 0 | 0 | 0 | 0 | 211 | 88 | 0 |
| B. S-only (Silver 860) | 44 | 39 | 27 | 33 | 15 | 11 | 14 | 88 | 73 | 0 |
| C. G->S (two-stage) | 42 | 52 | 15 | 18 | 21 | 23 | 14 | 79 | 55 | 23 |
| C. G->S (two-stage) | 43 | 49 | 17 | 20 | 26 | 16 | 17 | 72 | 59 | 24 |
| C. G->S (two-stage) | 44 | 55 | 16 | 14 | 19 | 18 | 18 | 73 | 58 | 29 |

## Confusion matrices (rows = true, columns = predicted)

### A. G-only (Topic-540) - seed 42

| true \ pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| business | 17 | 0 | 0 | 3 | 0 | 2 | 8 | 7 | 0 |
| education | 1 | 4 | 0 | 0 | 4 | 0 | 9 | 0 | 0 |
| health | 0 | 0 | 8 | 2 | 5 | 0 | 7 | 0 | 0 |
| shopping | 0 | 0 | 2 | 11 | 2 | 2 | 2 | 0 | 1 |
| medical | 0 | 0 | 0 | 1 | 13 | 2 | 3 | 1 | 1 |
| sports | 0 | 0 | 0 | 1 | 0 | 12 | 2 | 1 | 1 |
| tech | 2 | 0 | 0 | 3 | 1 | 3 | 62 | 2 | 1 |
| finance | 0 | 0 | 0 | 1 | 1 | 0 | 4 | 59 | 2 |
| social | 2 | 0 | 1 | 2 | 0 | 0 | 3 | 0 | 16 |

### A. G-only (Topic-540) - seed 43

| true \ pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| business | 14 | 0 | 0 | 6 | 0 | 0 | 10 | 5 | 2 |
| education | 1 | 4 | 0 | 0 | 4 | 1 | 7 | 0 | 1 |
| health | 0 | 0 | 8 | 2 | 6 | 0 | 2 | 0 | 4 |
| shopping | 0 | 1 | 2 | 9 | 6 | 0 | 0 | 0 | 2 |
| medical | 0 | 1 | 0 | 4 | 13 | 2 | 0 | 0 | 1 |
| sports | 0 | 0 | 0 | 2 | 0 | 10 | 2 | 1 | 2 |
| tech | 2 | 2 | 0 | 5 | 3 | 1 | 55 | 2 | 4 |
| finance | 1 | 1 | 0 | 0 | 1 | 1 | 2 | 59 | 2 |
| social | 0 | 0 | 1 | 2 | 1 | 0 | 4 | 0 | 16 |

### A. G-only (Topic-540) - seed 44

| true \ pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| business | 15 | 1 | 0 | 3 | 1 | 1 | 9 | 6 | 1 |
| education | 1 | 5 | 0 | 0 | 6 | 1 | 4 | 1 | 0 |
| health | 0 | 0 | 4 | 0 | 10 | 0 | 6 | 0 | 2 |
| shopping | 0 | 0 | 0 | 9 | 5 | 1 | 3 | 0 | 2 |
| medical | 0 | 0 | 0 | 1 | 15 | 1 | 2 | 1 | 1 |
| sports | 0 | 1 | 0 | 1 | 0 | 11 | 1 | 2 | 1 |
| tech | 2 | 2 | 0 | 3 | 1 | 1 | 60 | 3 | 2 |
| finance | 0 | 1 | 0 | 2 | 2 | 0 | 2 | 59 | 1 |
| social | 1 | 0 | 0 | 3 | 1 | 0 | 3 | 0 | 16 |

### B. S-only (Silver 860) - seed 42

| true \ pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| business | 16 | 0 | 0 | 0 | 0 | 1 | 7 | 13 | 0 |
| education | 1 | 0 | 3 | 1 | 6 | 0 | 7 | 0 | 0 |
| health | 0 | 0 | 16 | 2 | 0 | 0 | 4 | 0 | 0 |
| shopping | 1 | 0 | 8 | 8 | 0 | 0 | 3 | 0 | 0 |
| medical | 2 | 0 | 7 | 1 | 9 | 0 | 2 | 0 | 0 |
| sports | 2 | 0 | 3 | 0 | 0 | 9 | 2 | 1 | 0 |
| tech | 4 | 0 | 4 | 0 | 0 | 0 | 62 | 4 | 0 |
| finance | 0 | 1 | 3 | 0 | 0 | 0 | 2 | 61 | 0 |
| social | 0 | 0 | 9 | 4 | 0 | 0 | 10 | 1 | 0 |

### B. S-only (Silver 860) - seed 43

| true \ pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| business | 0 | 0 | 0 | 0 | 0 | 0 | 21 | 16 | 0 |
| education | 0 | 0 | 0 | 0 | 0 | 0 | 18 | 0 | 0 |
| health | 0 | 0 | 0 | 0 | 0 | 0 | 22 | 0 | 0 |
| shopping | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 1 | 0 |
| medical | 1 | 0 | 0 | 0 | 0 | 0 | 18 | 2 | 0 |
| sports | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 2 | 0 |
| tech | 0 | 0 | 0 | 0 | 0 | 0 | 69 | 5 | 0 |
| finance | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 61 | 0 |
| social | 0 | 0 | 0 | 0 | 0 | 0 | 23 | 1 | 0 |

### B. S-only (Silver 860) - seed 44

| true \ pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| business | 22 | 0 | 0 | 0 | 0 | 1 | 7 | 7 | 0 |
| education | 1 | 12 | 3 | 0 | 0 | 0 | 1 | 1 | 0 |
| health | 0 | 0 | 11 | 3 | 1 | 0 | 7 | 0 | 0 |
| shopping | 2 | 1 | 6 | 8 | 0 | 0 | 3 | 0 | 0 |
| medical | 1 | 4 | 5 | 0 | 9 | 0 | 2 | 0 | 0 |
| sports | 1 | 1 | 0 | 0 | 0 | 13 | 1 | 1 | 0 |
| tech | 6 | 4 | 4 | 1 | 0 | 0 | 56 | 3 | 0 |
| finance | 1 | 1 | 1 | 0 | 0 | 0 | 4 | 60 | 0 |
| social | 5 | 4 | 3 | 3 | 1 | 0 | 7 | 1 | 0 |

### C. G->S (two-stage) - seed 42

| true \ pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| business | 33 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 |
| education | 1 | 11 | 1 | 0 | 2 | 0 | 3 | 0 | 0 |
| health | 0 | 0 | 13 | 3 | 0 | 0 | 5 | 0 | 1 |
| shopping | 0 | 1 | 2 | 15 | 0 | 0 | 1 | 0 | 1 |
| medical | 0 | 1 | 1 | 1 | 16 | 0 | 1 | 0 | 1 |
| sports | 0 | 0 | 0 | 0 | 0 | 14 | 1 | 1 | 1 |
| tech | 7 | 1 | 0 | 0 | 4 | 0 | 61 | 1 | 0 |
| finance | 11 | 1 | 0 | 0 | 1 | 0 | 1 | 52 | 1 |
| social | 0 | 0 | 1 | 2 | 0 | 0 | 3 | 0 | 18 |

### C. G->S (two-stage) - seed 43

| true \ pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| business | 31 | 0 | 0 | 1 | 0 | 0 | 4 | 1 | 0 |
| education | 1 | 11 | 1 | 0 | 1 | 1 | 3 | 0 | 0 |
| health | 0 | 0 | 12 | 4 | 0 | 0 | 3 | 0 | 3 |
| shopping | 0 | 2 | 3 | 13 | 0 | 0 | 1 | 0 | 1 |
| medical | 1 | 1 | 1 | 2 | 13 | 1 | 1 | 0 | 1 |
| sports | 0 | 0 | 0 | 1 | 0 | 15 | 0 | 1 | 0 |
| tech | 6 | 2 | 3 | 2 | 2 | 0 | 57 | 1 | 1 |
| finance | 8 | 0 | 0 | 0 | 0 | 0 | 1 | 56 | 2 |
| social | 2 | 1 | 0 | 3 | 0 | 0 | 2 | 0 | 16 |

### C. G->S (two-stage) - seed 44

| true \ pred | business | education | health | shopping | medical | sports | tech | finance | social |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| business | 31 | 0 | 0 | 0 | 0 | 1 | 4 | 1 | 0 |
| education | 1 | 11 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| health | 0 | 0 | 10 | 3 | 1 | 0 | 6 | 0 | 2 |
| shopping | 1 | 0 | 2 | 11 | 0 | 1 | 2 | 0 | 3 |
| medical | 1 | 2 | 0 | 0 | 14 | 1 | 1 | 1 | 1 |
| sports | 0 | 0 | 0 | 1 | 0 | 13 | 1 | 1 | 1 |
| tech | 9 | 2 | 0 | 1 | 1 | 1 | 55 | 2 | 3 |
| finance | 12 | 0 | 0 | 0 | 1 | 0 | 1 | 52 | 1 |
| social | 0 | 1 | 1 | 3 | 0 | 0 | 2 | 0 | 17 |