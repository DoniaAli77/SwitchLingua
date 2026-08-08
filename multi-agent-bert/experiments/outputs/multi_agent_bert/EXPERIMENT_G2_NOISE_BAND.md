# Temperature-0 Run-to-Run Noise Band (Design G2, Ahmed)

The identical Design-G2 configuration was executed five times to quantify residual
non-determinism in gpt-4.1-mini at temperature 0, so that small differences between
configurations elsewhere in this work can be judged against a measured baseline.
Config held constant: Ahmed precomputed primary, selective IntentGate, threshold
0.70, semantic_v1, consensus w_primary 1.0, gpt-4.1-mini, 818 samples / 84
escalated. Date: 2026-08-04.

## Results

| Run | Accuracy | Macro F1 | Escalated |
|---|---|---|---|
| run 1 (canonical, registered) | 0.9303 | 0.9262 | 67/84 |
| run 2 (fresh) | 0.9315 | 0.9277 | 68/84 |
| run 3 | 0.9291 | 0.9251 | 66/84 |
| run 4 | 0.9315 | 0.9277 | 68/84 |
| run 5 | 0.9303 | 0.9262 | 67/84 |
| **Mean (n=5)** | **0.9306** | **0.9266** | **67.2/84** |
| **Range** | 0.9291–0.9315 | 0.9251–0.9277 | 66–68 |
| **Stdev** | 0.0010 | 0.0011 | 0.84 |

**Noise band: ±2 escalated samples ≈ ±0.0026 macro F1.**

Note the 734 non-escalated predictions are identical across all runs (they come
from the frozen precomputed primary); all variance originates in the ~84 escalated
samples where the LLM agents run.

## Application to reported comparisons

| Comparison | Gap | Verdict |
|---|---|---|
| Parallel vs sequential (Designs C, G, G2) | ≤1 sample | within noise |
| Agent-order sweep (independent and collaborative) | ≤1 sample | within noise |
| Collaborative vs independent chain framing | ≤2 samples | within noise |
| Design H (64/84) vs Design G2 (mean 67.2) | 3.2 samples ≈ 3.8σ | **outside noise — real** |
| Design H (64/84) vs Design C (64/84) | 0 samples | identical |

The Design-H finding (the Selective IntentGate's benefit is positional, not
prompt-based) is therefore statistically supported: Design H falls two samples
below the *lowest* G2 run observed and ~3.8 standard deviations below the mean.

## Suggested thesis wording
> Repeated execution of an identical pipeline configuration (n = 5) yielded
> accuracies of 0.9291–0.9315 (mean 0.9306, SD 0.0010), corresponding to a
> variation of ±2 of the 84 escalated samples. This reflects residual
> non-determinism in the language model at temperature 0. Differences of this
> magnitude between configurations are therefore not interpreted as meaningful,
> and the G2 configuration is reported as 0.9306 (range 0.9291–0.9315).

## Reproduce
`scripts/ahmed_g2_repeat_runs.py`; artifacts in
`experiment_ahmed_g2_repeats/{records.json,g2_repeats_report.txt}`.
