# Real-LLM Threshold Sweep (GPT-4o-mini)

Tests whether routing **more** escalated samples to the real LLM agents (raising
the router threshold) improves on the threshold-0.6 pilot. Same EESA test, same
mBERT/XLM-R checkpoints, same OpenAI gpt-4o-mini client. No retraining, no router
logic change, no SwitchLingua change. Date: 2026-06-09.

## Method (controlled)
- Ran `full_agentic` natively at **threshold 0.9** per model (escalates ~23%,
  not all rows). Because the agents are threshold-independent and run at
  temperature 0, a sample's agent verdict is identical at any threshold;
  thresholds **0.7 / 0.8** are derived exactly from the same 0.9-run calls
  (their escalated sets are strict subsets), and **0.6** reuses the pilot.
- Validated by determinism: a model's 0.6-pilot escalated predictions should
  match its 0.9-run predictions on the overlap.

## ⚠ Data-quality note — internet disconnect during the mBERT runs
The mBERT 0.9 runs were hit by an **internet outage** mid-run: **82 then 129
OpenAI connection errors**, which truncated agent calls (only **417** then
**220** of the ~736 expected) and forced affected samples to **fall back to the
primary**. Determinism vs the clean 0.6 pilot fell to **74% / 57%**. **mBERT
0.7/0.8/0.9 are therefore NOT clean** and are reported only as conservative lower
bounds (fallback-to-primary biases them *toward* primary_only). XLM-R completed
cleanly (741/760 calls, **100%** determinism on the 0.6 overlap).

---

## XLM-R — clean sweep (headline)

| Threshold | esc % | final acc | macro F1 | weighted F1 | primAcc on esc | LLM-agent acc on esc | W→C | C→W | net | neg F1 | neu F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.6 (pilot) | 5.0 | 0.8399 | 0.8264 | 0.8392 | 0.415 | 0.732 | 15 | 2 | +13 | 0.792 | 0.781 |
| 0.7 | 7.5 | 0.8423 | 0.8284 | 0.8411 | 0.443 | 0.689 | 20 | 5 | +15 | 0.796 | 0.781 |
| 0.8 | 13.3 | **0.8460** | 0.8328 | 0.8446 | 0.495 | 0.661 | 26 | 8 | +18 | 0.806 | 0.784 |
| 0.9 † | 23.2 | 0.8447 | **0.8330** | 0.8423 | 0.558 | 0.647 | 41 | 24 | +17 | **0.827** | 0.770 |

† XLM-R 0.9 had a minor ~5-call cluster (741/760); 0.6–0.8 rows are clean.

Baselines: XLM-R primary_only = **0.8240 / 0.8088**.

**Reading:** raising the threshold keeps helping and **does not hurt** — accuracy
peaks at 0.8 (0.8460) and macro F1 plateaus 0.8–0.9 (~0.833). The agents stay
**well above** the primary on the escalated subset at every threshold (0.65–0.73
vs 0.42–0.56), and **net is positive throughout** (+13…+18). The gain is driven
by **negative-class recovery** (neg F1 0.792 → 0.827 as more is escalated);
neutral is flat-to-slightly-down. This is the **opposite** of the mock sweep,
which declined monotonically (net −8 → −68).

---

## mBERT — clean 0.6 only; 0.7/0.8/0.9 contaminated (re-run needed)

| Threshold | acc | macro F1 | status |
|---|---|---|---|
| 0.6 (pilot, clean) | **0.8166** | **0.8038** | clean |
| 0.7 / 0.8 (derived) | — | — | unreliable (truncated 0.9 run) |
| 0.9 (native, attempt 1) | 0.8215 | 0.8095 | contaminated (82 conn errors, 417/736 calls) |
| 0.9 (native, attempt 2) | 0.8178 | 0.8048 | contaminated worse (129 conn errors, 220/736 calls) |

Baseline: mBERT primary_only = **0.7971 / 0.7833**.

Even contaminated, both mBERT 0.9 attempts (0.818–0.822) sit **above**
primary_only (0.797) and near/above the clean 0.6 pilot (0.817). Since
contamination drags the result *toward* primary, the **clean** mBERT 0.9 is
expected to be **≥ 0.822** — i.e. mBERT qualitatively matches XLM-R (more
escalation does not hurt), but precise clean numbers await a re-run on a stable
connection.

---

## Cost / errors

| Run | OpenAI calls | cost | connection errors |
|---|---|---|---|
| XLM-R 0.9 (clean) | 741 | $0.0936 | ~5 |
| mBERT 0.9 attempt 1 | 417 | $0.0525 | 82 |
| mBERT 0.9 attempt 2 | 220 | $0.0277 | 129 |
| 0.6 pilot (both, prior) | 350 | $0.044 | 1 |

Parse errors: **0** across all runs (JSON mode holds). The connection errors were
an internet outage, not a model/JSON/proxy issue.

---

## Conclusion
- **With real LLM agents, raising the escalation threshold does not hurt and
  modestly helps**, plateauing around 0.8–0.9 — cleanly demonstrated on XLM-R
  (acc 0.8399→0.8460, macro F1 0.8264→0.8330) and directionally on mBERT.
- The mechanism is consistent with the pilot: agents beat the primary on the
  low-confidence escalated slice, recovering the **negative** class.
- This validates the escalation design and stands in direct contrast to the mock
  sweep's monotonic decline — the agents, not the routing, were the prior
  bottleneck.

## Open item (blocked — checkpoints deleted)
A clean-connection re-run of **mBERT 0.7/0.8/0.9** was attempted (2026-06-10) and
**failed**: the entire `experiments/checkpoints/` directory (both `eesa_mbert`
and `eesa_xlm_roberta_base`) has been **deleted** (untracked local files, not in
git). The fine-tuned models no longer exist on disk, so no new real-transformer
evaluation can run until they are regenerated.

- All previously saved predictions/metrics are intact — the **XLM-R clean sweep
  and both 0.6 pilots above are unaffected**.
- To finish the clean mBERT high-threshold rows we must first **re-fine-tune
  mBERT** (`scripts/finetune_transformer_classifier.py`, seed 42, ~10 min, free
  local GPU), which yields a *new* checkpoint — so to stay self-consistent the
  mBERT primary_only / 0.6 / 0.9 would all be re-derived on the new checkpoint
  (do not mix the old-checkpoint 0.6 pilot with a new-checkpoint 0.9).
- **Action needed before any future paid real-primary re-eval** (e.g. validating
  the consensus fixes): regenerate and **back up** both checkpoints
  (mBERT ~10 min, XLM-R ~41 min). Connectivity itself is fine (probe succeeded).
