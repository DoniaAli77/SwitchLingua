# Experiment E — Augmentation: Consolidated Summary

Consolidation of all augmentation experiments. No new training/data. Date: 2026-06-25.

---

## 1. Objective
Test whether **generic SwitchLingua-generated sentiment data improves real EESA
sentiment classification when used as augmentation** — i.e., does adding generated
code-switched sentiment samples to real EESA training data raise accuracy/macro F1
on the real EESA test set? This is distinct from *standalone* generated-only training
(Experiments C1–C3); here the generated data is **mixed into real EESA train**.

All runs: fresh `xlm-roberta-base`, Adafactor, fp16, max_length 256, primary_only,
evaluated on the real EESA test (818). EESA test = positive-heavy (363/197/258).

---

## 2. Experiments covered
| ID | What | Role |
|---|---|---|
| **E0** | EESA-only, Adafactor | matched control (no generated) |
| **E3** | EESA + GEN-960, Adafactor | full-data augmentation |
| **LR** | EESA 10/25/50% ± fixed GEN-960 | low-resource augmentation |
| **Ratio sweep** | EESA 10/25/50% + r×N generated (r=0.25/0.5/1.0) | mixture-ratio dependence |
| **Diagnosis** | EESA train/test vs GEN-960 profiling | why augmentation behaves this way |

(Matched-compute note: LR and the ratio sweep use `--max_steps 400 + --load_best`
because matched-*epochs* starved the small subsets — the 10% real-only baseline
otherwise collapsed to all-positive, acc 0.444.)

---

## 3. Main quantitative results

### Full-data matched comparison (the clean control)
| system | optimizer | gen data | accuracy | macro F1 |
|---|---|---|---|---|
| EESA-only reference | AdamW | no | 0.8240 | 0.8088 |
| **E0: EESA-only** | Adafactor | no | **0.8533** | **0.8409** |
| E3: EESA + GEN-960 | Adafactor | yes | 0.8411 | 0.8294 |

- Optimizer effect (AdamW→Adafactor): **+0.029 acc**.
- Augmentation effect (E0→E3, both Adafactor): **−0.012 acc / −0.012 macro F1**.
- **Conclusion: augmentation did not improve the full real-data model** (slight
  negative; E3's apparent gain over the old 0.8240 baseline was the optimizer, not
  the generated data).

### Low-resource, fixed +GEN-960
| EESA real | gen % of mix | real-only | +GEN-960 | Δ acc |
|---|---|---|---|---|
| 10% (246) | 80% | 0.7751 | 0.7408 | **−0.034** |
| 25% (615) | 61% | 0.7873 | 0.7689 | **−0.018** |
| 50% (1232) | 44% | 0.8166 | 0.8142 | **−0.002** |

- **Conclusion: fixed +960 hurts most when the generated data dominates the mix**
  (harm monotone in generated fraction; ≈0 once generated is a minority).

### Ratio sweep (generated scaled to subset size; gen share = 20/33/50%)
| Δ acc vs real-only | +0.25× (20%) | +0.5× (33%) | +1.0× (50%) |
|---|---|---|---|
| 10% EESA | −0.026 | +0.005 | −0.018 |
| 25% EESA | −0.004 | +0.001 | +0.006 |
| 50% EESA | −0.009 | −0.015 | −0.013 |

- At 20–50% generated share, all changes are **within ±0.026 (mostly ±0.018) — i.e.
  within single-seed noise (~±0.02)**.
- **No ratio reliably improves** performance.
- The **strong harm disappears** when generated data is not dominant (capping it at
  ≤50% of the mix removes the −0.034 seen at 80%).
- **Conclusion: generic generated data is mostly neutral as minority augmentation,
  harmful when dominant, and not reliably helpful at any tested ratio.**

---

## 4. Diagnosis — domain mismatch
Profiling EESA train/test vs GEN-960 (`EXPERIMENT_E_AUGMENTATION_FAILURE_DIAGNOSIS.md`):

| property | EESA (train/test) | GEN-960 |
|---|---|---|
| Arabic / English token ratio | ~0.73 / 0.27 | ~0.53 / 0.47 |
| code-mixing index (CMI) | ~24 | ~41 |
| register | dialectal, noisy social-media (with platform artifacts: `unlike`, `clip`, links) | cleaner, more formal MSA + fluent English, topic-themed |
| label prior | positive-heavy (~44%) | balanced (33/33/33) |
| **vocab coverage of EESA test** | 0.49 (train) | **0.098** |

- EESA is **Arabic-dominant, lightly mixed, noisy social-media style**.
- GEN-960 is **more balanced Arabic-English, higher CMI, cleaner/more formal**.
- **Vocabulary overlap with EESA test is very low (~10%).**
- → The generated data adds **off-domain signal**, not target-domain signal, so it
  cannot improve the EESA model and can distract it when it dominates the mix.

---

## 5. Final interpretation
- **SwitchLingua-generated data is useful as standalone synthetic training data.** It
  carries genuine sentiment signal and is config-controlled, CS-valid, and quality-filtered.
- **It scales positively in generated-only training:** 240 → 480 → 960 (≈0.59 → 0.65
  → 0.67 mean accuracy on EESA test, 3-seed).
- **The multi-agent pipeline strongly improves generated-only C3** (selected 960
  checkpoint: 0.696 → 0.754 with full_agentic, +0.06).
- **But generic generated data does not improve a strong *or* low-resource EESA model
  through naive augmentation** — it is neutral as a minority and harmful when it
  dominates the training mix.
- **Augmentation usefulness depends on (a) domain compatibility between generated and
  target data, and (b) the mixture ratio** — not on the absolute amount of generated
  data, and not a reason to redesign the generic generator for one corpus.

---

## 6. Caveats
- **Most augmentation ratio cells are single-seed.** All observed Δ are within ±0.02.
- **Small effects within ±0.02 should not be over-interpreted** (per-cell signs are
  not statistically reliable).
- **A multi-seed (≥3) ratio sweep would be needed for statistical confirmation** of
  any genuinely non-zero small effect.
- The conclusion concerns the **current generic GEN-960 configuration vs EESA**, not
  all possible SwitchLingua configurations or all possible target domains. A target
  *within* the SwitchLingua-generated domain was not tested and could differ.

---

## 7. Thesis-ready wording
> We evaluated whether generic SwitchLingua-generated code-switched sentiment data
> improves a real Arabic–English sentiment classifier (EESA) when used as data
> augmentation. Under matched optimisation, adding the generated data did not improve
> the full real-data model (0.853 → 0.841 accuracy), and across low-resource subsets
> (10–50% of real EESA) and a range of mixture ratios the effect remained within
> single-seed noise — generic generated data was neutral as a minority component and
> harmful only when it dominated the training mixture. A distributional analysis
> attributes this to a domain/register mismatch: the generated text is more balanced
> in its Arabic–English mixing, more formal, and shares only ~10% of its vocabulary
> with the EESA test set, whereas EESA is Arabic-dominant, dialectal, and noisy.
> These findings indicate that the value of SwitchLingua data lies in *standalone*
> training (where it scales positively and is further improved by the multi-agent
> pipeline) rather than in naive augmentation of a different-domain real corpus, and
> that augmentation benefit is governed by domain compatibility and mixture ratio
> rather than by the quantity of generated data alone.

---

## Source reports
- `experiment_E0_eesa_only_adafactor/EXPERIMENT_E0_EESA_ONLY_ADAFACTOR_REPORT.md`
- `experiment_E3_eesa_plus_switchlingua960/EXPERIMENT_E3_EESA_PLUS_SWITCHLINGUA960_REPORT.md`
- `experiment_LR_lowresource_augmentation/EXPERIMENT_LR_LOWRESOURCE_AUGMENTATION_REPORT.md`
- `experiment_E_ratio_sweep/EXPERIMENT_E_AUGMENTATION_RATIO_SWEEP.md`
- `EXPERIMENT_E_AUGMENTATION_FAILURE_DIAGNOSIS.md`
