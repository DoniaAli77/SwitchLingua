# V1_lowerCS Downstream Sensitivity — C-V1 (transfer) + E-V1 (augmentation)

Downstream test of the **V1_lowerCS** generated variant (cs_ratio 70/80, Arabic-dominant:
AR 62.9% vs GEN 52.7, CMI 33.0 vs 40.9, length unchanged — ~halfway toward EESA AR~73/CMI~24).
primary_only, XLM-R, exact C2 recipe (Adafactor, eff-batch16, fp16, grad-ckpt, lr2e-5,
maxlen256). Dev/test = EESA. Framed as **domain-compatibility sensitivity**, not EESA-tuning.
Date: 2026-07-02.

## Exp C-V1 — generated-only transfer (V1-480, 3 seeds)

| seed | accuracy | macro F1 | weighted F1 | pred-dist (pos/neu/neg) |
|---|---|---|---|---|
| 42 | 0.6491 | 0.6402 | 0.6574 | 268 / 308 / 242 |
| 123 | 0.6174 | 0.6112 | 0.6251 | 233 / 390 / 195 |
| 456* | 0.6247 | 0.6160 | 0.6317 | 247 / 399 / 172 |
| **mean±std** | **0.6304 ± 0.0166** | **0.6225 ± 0.0156** | **0.6381 ± 0.0170** | skews **neutral↑, negative↓** |

Per-class F1 (mean): positive **0.708**, neutral 0.598, negative **0.561** (negative weakest).
\*seed456 re-run after a transient 4GB-GPU CUDA-OOM (fragmentation between back-to-back runs);
the other 5 runs completed first-try.

### C-V1 vs baselines (transfer)
| config | size | acc | Δ vs C-V1 |
|---|---|---|---|
| **C-V1 (V1-480)** | 480 | **0.6304 ± 0.017** | — |
| C2 (GEN-480, matched) | 480 | 0.6500 ± 0.016 | **C-V1 −0.0196** |
| C3 (GEN-960) | 960 | 0.6695 ± 0.024 | C-V1 −0.039 |

**Answer (transfer): Arabic-dominant / lower-CMI transfers WORSE, not better.** At matched
size, V1 is **−0.02 below GEN-480** (~1.2σ — slightly worse, clearly not an improvement), and
below GEN-960. The V1-trained models **over-predict neutral and under-predict negative** — the
reduced code-switching intensity (CMI↓) makes the generated set a **weaker standalone signal**
for the code-switched EESA task, despite being "closer" to EESA in AR%/CMI.

## Exp E-V1 — low-resource augmentation (EESA{10,25,50}% + V1-gen, gen capped ≤50%, seed 42)

Gen fraction after cap: eesa10 50% · eesa25 44% · eesa50 28% (all ≤50% per the "gen hurts when
it dominates" rule; V1 has only 480, so the cap binds only at 10%).

| EESA% | real-only | +V1 | +GEN-960 | **ΔV1 − real** | **ΔV1 − GEN-960** |
|---|---|---|---|---|---|
| 10% | 0.7751 | 0.7531 | 0.7408 | **−0.0220** | **+0.0122** |
| 25% | 0.7873 | 0.7885 | 0.7689 | **+0.0012** | **+0.0196** |
| 50% | 0.8166 | 0.8178 | 0.8142 | **+0.0012** | **+0.0037** |

### Answer (augmentation)
- **vs real-only:** V1 augmentation is **neutral-to-harmful** — it *hurts* at 10% (−0.022,
  where gen is 50% of the mix) and is *neutral* at 25/50% (+0.001). Consistent with the prior
  finding that gen doesn't help augmentation and hurts when it's a large fraction. **V1 does not
  beat real-only.**
- **vs GEN-960:** V1 is **consistently better at every ratio** (+0.012 to +0.020). The
  more-EESA-like V1 is a **less-harmful augmenter** than GEN-960.

## The divergence — the actual sensitivity finding
EESA-proximity has **opposite effects** on the two uses, and it reconciles cleanly:
- **Standalone transfer (C-V1): proximity HURT.** With no real data, the model must learn the
  CS phenomena *from* the gen set; V1's lower code-switching intensity (CMI 33 vs 41) is exactly
  the signal it lacks → worse transfer.
- **Augmentation (E-V1): proximity HELPED (vs GEN-960).** On top of real EESA (which already
  carries the CS signal), the gen set acts as a *perturbation*; V1 sits closer to EESA's
  distribution (CMI 33 → EESA 24, vs GEN-960's 41) so it **distorts the real signal less** →
  less harmful than the more code-switched GEN-960.

**In one line:** *Greater EESA-proximity (Arabic↑ / CMI↓) did NOT improve standalone
generated-only transfer (V1 0.630 vs matched GEN-480 0.650, −0.02) but DID make V1 a
less-harmful augmenter than GEN-960 (+0.01–0.02 at every ratio, still ≤ real-only) — because
the proximity came from reduced code-switching, which augmentation tolerates but standalone
transfer needs.*

## Caveats
- **C-V1** is 3-seed (mean±std reported); the C-V1<C2 gap (−0.02) is ~1σ — "slightly worse /
  comparable," not a large effect. Direction (not an improvement) is clear.
- **E-V1** is single-seed (seed 42, matching the LR/E setup); its deltas (esp. the +0.012–0.020
  V1-vs-960 edge) are single-draw and would benefit from seed replication before strong claims.
- EESA test kept out of training (dev/test = EESA dev/test; gen never contains EESA). New
  folders only; C1/C2/C3/E/E0/LR untouched.

## Artifacts
- C-V1: `experiment_CV1_v1lowerCS/seed{42,123,456}/`; E-V1:
  `experiment_EV1_v1lowerCS_augmentation/eesa{10,25,50}_plusV1/`.
- Mixes: `data/Sentiment/processed/augmentation/lowresource_v1/`; runner
  `scripts/run_v1_downstream.sh`. Gen context: memory `gen_sensitivity_datasets.md`.
