# Results — slide-ready tables

All numbers from this project. Dataset = EESA sentiment test (818) unless noted.
Primary = XLM-R (`xlm-roberta-base`). LLM agents = GPT-4o-mini.

---

## Slide 1 — Headline: does the multi-agent system help?
EESA test, XLM-R primary.

| System | Accuracy | Macro F1 |
|---|---|---|
| Primary only (BERT baseline) | 0.8240 | 0.8088 |
| **Multi-agent, best (full_agentic, threshold 0.9)** | **0.8509** | **0.8401** |
| Δ (gain from agents) | **+0.0269** | **+0.0313** |

**Takeaway:** escalating only low-confidence cases to the agent panel lifts
accuracy ~2.7 pts over the BERT baseline.

---

## Slide 2 — Ablation 2×2 (threshold 0.8)
full_agentic, EESA test. Escalated = 109/818 (13.3%). Clean run (0 errors).
`w_p` = primary vote weight · `sig` = primary-signal prompt block · anchor% =
agent↔primary agreement on escalated samples.

| Cell | w_p | sig | Accuracy | Macro F1 | neg F1 | anchor% | cost |
|---|---|---|---|---|---|---|---|
| A (original) | 0 | off | 0.8460 | 0.8331 | 0.808 | 62.4 | $0.057 |
| B (+primary vote) | 1.0 | off | 0.8447 | 0.8316 | 0.806 | 62.4 | $0.057 |
| C (+primary signal) | 0 | on | 0.8435 | 0.8312 | 0.806 | 65.1 | $0.064 |
| D (both) | 1.0 | on | 0.8423 | 0.8300 | 0.804 | 66.1 | $0.064 |

**Takeaway:** all four within noise on accuracy; the primary-signal block raises
copying (anchor 62→66%) with no accuracy gain.

---

## Slide 3 — Ablation 2×2 (threshold 0.9)
full_agentic, EESA test. Escalated = 190/818 (23.2%). Clean run (0 errors).
escAcc = accuracy on escalated subset · net = correct−wrong overrides vs primary.

| Cell | w_p | sig | Accuracy | Macro F1 | escAcc | anchor% | net | cost |
|---|---|---|---|---|---|---|---|---|
| A (original) | 0 | off | 0.8496 | 0.8386 | 0.668 | 63.7 | +21 | $0.099 |
| **B (+primary vote)** | 1.0 | off | **0.8509** | **0.8401** | 0.674 | 64.7 | +22 | $0.099 |
| C (+primary signal) | 0 | on | 0.8472 | 0.8369 | 0.658 | 67.4 | +19 | $0.111 |
| D (both) | 1.0 | on | 0.8496 | 0.8394 | 0.668 | 70.0 | +21 | $0.111 |

**Takeaway:** best cell is B (primary vote on, signal off); the signal block again
only raises anchoring (up to 70%) without helping accuracy.

---

## Slide 4 — Threshold 0.8 vs 0.9 (anchoring effect)
Agreement of agents with the primary on escalated samples.

| Cell | anchor% @0.8 | anchor% @0.9 |
|---|---|---|
| A original | 62.4 | 63.7 |
| B primary vote | 62.4 | 64.7 |
| C primary signal | 65.1 | 67.4 |
| D both | 66.1 | 70.0 |
| **Signal-on lift (avg)** | **+3.2** | **+4.5** |

**Takeaway:** showing agents the primary's label makes them copy it more, and the
effect grows with escalation — without improving accuracy. Hence: keep it off.

---

## Slide 5 — Design decisions (locked)
| Component | Decision | Why |
|---|---|---|
| Primary vote in consensus | **ON** (w_primary = 1.0) | +0.064 acc on weak agents (paper_style); neutral/slightly-better on strong agents |
| Primary-signal prompt block | **OFF** | only adds anchoring, no accuracy gain, costs more |
| Router threshold | per-run knob | best result at 0.9; 0.8 acceptable at ~half the cost |

---

## Slide 6 — Experiment A vs Experiment C (data source)
Both XLM-R fine-tuned, evaluated on EESA test. **Separate experiments — do not
treat as a controlled comparison.**

| | Exp A (reference) | Exp C (pilot) |
|---|---|---|
| Train data | 2,464 real EESA | 240 SwitchLingua-generated |
| Optimizer | AdamW | Adafactor |
| primary_only accuracy | 0.8240 | 0.5905 |
| primary_only macro F1 | 0.8088 | 0.5619 |
| primary_only weighted F1 | — | 0.5838 |

**Takeaway:** 240 generated samples transfer to ~59% on real EESA (well above the
33% 3-class chance, below the 2,464-sample real-data model). Gap is confounded by
**both** 10× less data **and** the optimizer change → transfer *signal*, not a
clean effect size.

---

## Slide 7 — Experiment C details (generated-240 → EESA test)
Per-class F1 (primary_only):

| Class | Precision | Recall | F1 | support |
|---|---|---|---|---|
| positive | 0.657 | 0.755 | 0.703 | 363 |
| negative | 0.577 | 0.457 | 0.510 | 197 |
| neutral | 0.486 | 0.461 | 0.473 | 258 |

Confusion matrix (rows = true, cols = predicted):

| true ↓ / pred → | positive | negative | neutral |
|---|---|---|---|
| **positive** | 274 | 22 | 67 |
| **negative** | 48 | 90 | 59 |
| **neutral** | 95 | 44 | 119 |

Training: Adafactor, eff. batch 16, lr 2e-5, 4 epochs, 202 s, train_loss 0.893.
Dev (EESA dev) by epoch: 0.444 → 0.546 → 0.531 → **0.628** acc.

**Takeaway:** positives transfer best; the model over-predicts positive (neutral
and negative leak into it).

---

## Slide 8 — Cost summary (paid runs)
| Run | LLM calls | Cost |
|---|---|---|
| 2×2 ablation @0.8 (4 cells) | ~1,744 | ~$0.24 |
| 2×2 ablation @0.9 (4 cells) | ~3,040 | ~$0.42 |
| Exp C full_agentic (stopped) | ~190 | ~$0.04 |
| **Total** | — | **~$0.70** |

(primary_only and fine-tuning run locally — no API cost.)

---

## Slide 9 — Caveats (footnote slide)
- Experiment C is a **transfer pilot**, not size-matched (2,464 vs 240) and uses a
  different optimizer (Adafactor vs AdamW) → not a clean generated-vs-real result.
- Experiment C **full_agentic was stopped** (weak primary escalated ~95% at 0.9);
  no final agentic metric — re-run at threshold 0.6–0.7 for a usable number.
- Ablation accuracy differences are within GPT run-to-run noise (~±0.003); the
  anchoring metric is the clearer differentiator.
- All sentiment numbers; topic/NER paths not yet evaluated.
