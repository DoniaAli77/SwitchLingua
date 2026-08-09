# Topic-180 subset x model-seed variance at 136 steps

3 nested 180-subsets x 3 model seeds = 9 runs, identical recipe, primary_only on Silver-1163.

## Accuracy grid

| subset \ model seed | 42 | 43 | 44 | row mean | row sd |
|---|--:|--:|--:|--:|--:|
| topic_180_seed42 | 0.6148 | 0.6148 | 0.6320 | **0.6205** | 0.0081 |
| topic_180_seed43 | 0.6277 | 0.6294 | 0.6380 | **0.6317** | 0.0045 |
| topic_180_seed44 | 0.5942 | 0.6019 | 0.6182 | **0.6048** | 0.0100 |
| **col mean** | 0.6122 | 0.6154 | 0.6294 | | |

## Variance decomposition (accuracy)

| source | sd |
|---|--:|
| model seed, within subset (mean of row sds) | 0.0076 |
| **subset, across subset means** | **0.0111** |
| all 9 runs pooled | 0.0136 |

Grand mean over the 9 runs: **0.6190** (min 0.5942, max 0.6380)

## Macro-F1 grid

| subset \ model seed | 42 | 43 | 44 | row mean |
|---|--:|--:|--:|--:|
| topic_180_seed42 | 0.5476 | 0.5468 | 0.5680 | **0.5541** |
| topic_180_seed43 | 0.5517 | 0.5718 | 0.5785 | **0.5673** |
| topic_180_seed44 | 0.5027 | 0.5248 | 0.5459 | **0.5244** |

## Collapse check (all 9)

- runs predicting <9 classes or with top share >=50%: NONE
- top-share range: 27.6% - 34.5%

## Same 136-step budget, other corpora

| corpus | acc | sd |
|---|--:|--:|
| Original-180 (9 runs, 3 subsets) | 0.6190 | 0.0136 |
| Topic-360 (3 subsets x 1 seed) | 0.6022 | 0.0028 |
| Topic-540 (1 corpus x 3 seeds) | 0.6145 | 0.0004 |
| ArabicDominant-180 (1 corpus x 3 seeds) | 0.6065 | 0.0025 |