# Multi-Agent BERT — Full Project Status Report (A → Z)

Consolidated report of everything built and measured to date. Date: 2026-06-13.
Scope: the **Multi-Agent BERT** track (sentiment classification primary focus;
NER path and topic phase noted where relevant).

---

## 1. Executive summary

A stateful **multi-agent text-classification system**: a fast **primary
transformer classifier** handles confident inputs, and **only low-confidence
("escalated") inputs** are routed to a panel of **specialist agents** that
deliberate and reach a **consensus**. The goal is BERT-level throughput with
LLM-level judgement on the hard cases, at controlled cost.

Status:
- **Architecture** implemented end-to-end (primary → router → specialists →
  consensus → explainability), three run modes, plus a separate NER path.
- **Four audit-driven correctness fixes** implemented, **897 tests passing**.
- **Two controlled 2×2 ablations** (thresholds 0.8 and 0.9) completed; the fix
  defaults are empirically justified.
- **Experiment A** (full-EESA-trained XLM-R) is the strong reference.
- **Experiment C** (240 generated-data transfer pilot) completed for primary_only.
- A real hardware constraint was characterised: XLM-R full fine-tuning barely fits
  the local 4 GB Windows GPU.

---

## 2. System architecture

### 2.1 High-level flow

```
                         ┌───────────────────────────────────────────────┐
                         │                  INPUT TEXT                     │
                         └───────────────────────┬───────────────────────┘
                                                 │
                                 ┌───────────────▼───────────────┐
                                 │      PRIMARY CLASSIFIER         │   XLM-R / mBERT
                                 │  (transformer or mock)          │   → label + confidence
                                 │  primary_model_output           │     + probabilities
                                 └───────────────┬───────────────┘
                                                 │
                                 ┌───────────────▼───────────────┐
                                 │            ROUTER              │   confidence ≥ threshold?
                                 │  decision = accept_primary /   │
                                 │             escalate           │
                                 └───────┬───────────────┬───────┘
                           accept_primary │               │ escalate
                          (FAST PATH)     │               │ (SLOW PATH)
                                          │               │
                  ┌───────────────────────▼──┐   ┌────────▼─────────────────────────────────┐
                  │   ExplainabilityAgent     │   │           SPECIALIST PANEL                │
                  │  (short template explain) │   │  ┌─────────────┐ ┌─────────┐ ┌─────────┐ │
                  │   → final = primary        │   │  │  Lexical    │ │  Logic  │ │Contextual│ │
                  └───────────────────────────┘   │  │  agent      │ │  agent  │ │  agent   │ │
                                                   │  └──────┬──────┘ └────┬────┘ └────┬─────┘ │
                                                   │         └──────────┬──┴───────────┘       │
                                                   │            (optional) DeliberationAgent   │
                                                   │                     │                     │
                                                   │            ┌────────▼─────────┐           │
                                                   │            │  CONSENSUS AGENT │           │  primary-aware
                                                   │            │  votes + primary │           │  weighted vote
                                                   │            │  → final_output  │           │  (Fix #2)
                                                   │            └────────┬─────────┘           │
                                                   │            ExplainabilityAgent (full)     │
                                                   └───────────────────┬──────────────────────┘
                                                                       │
                                            ┌──────────────────────────▼──────────────────────────┐
                                            │   FINAL OUTPUT: label, confidence, explanation,       │
                                            │   full audit history (every stage logged)             │
                                            └───────────────────────────────────────────────────────┘
```

### 2.2 Components (code map)
| Component | File | Role |
|---|---|---|
| Primary (transformer) | `src/models/primary_transformer_classifier.py` | XLM-R/mBERT inference → label, confidence, probs |
| Primary (mock) | `src/models/mock_primary_classifier.py` | deterministic stand-in for tests |
| Router | `src/pipeline/router.py` | threshold gate → `accept_primary` / `escalate` |
| Orchestrator | `src/pipeline/orchestrator.py` | wires stages, per-mode agent selection, error capture |
| Lexical agent | `lexical_agent.py` / `llm_lexical_agent.py` | keyword/regex vs LLM evidence |
| Logic agent | `logic_agent.py` / `llm_logic_agent.py` | rule/negation vs LLM reasoning |
| Contextual agent | `contextual_agent.py` / `transformer_contextual_agent.py` | LLM vs transformer context read |
| Deliberation agent | `deliberation_agent.py` | optional cross-talk before consensus (full_agentic) |
| Consensus agent | `consensus_agent.py` | primary-aware weighted vote → final label |
| Explainability | `explainability_agent.py` / `llm_explainability_agent.py` | rationale + audit |
| NER path | `ner_*_agent.py` | separate sequence-labeling panel (System B) |
| Prompts | `src/prompts/` incl. `_primary_block.py`, `_abstain.py` | task-config-driven, generic |

### 2.3 Run modes
| Mode | Path | Specialists | Use |
|---|---|---|---|
| **primary_only** | primary → final | none (router skipped) | BERT baseline / throughput |
| **paper_style** | primary → router → (escalate) deterministic lexical+logic+contextual → consensus | non-LLM (reference BERT framework) | weak-agent regime |
| **full_agentic** | primary → router → (escalate) LLM lexical+logic+contextual → (opt) deliberation → consensus | LLM-backed (GPT-4o-mini) | strong-agent regime |

---

## 3. Audit-driven correctness fixes

A prompt/logic audit produced four fixes (all implemented, default as noted):

| # | Fix | What it does | Default | Evidence |
|---|---|---|---|---|
| 1 | **Generic prompts** | prompts driven by task_config, no hardcoded sentiment/topic labels | on | enables topic reuse |
| 1b | **Abstain fallback** | no-vote/tie no longer silently picks `labels[0]` | on | removes label-0 bias |
| 2 | **Primary-aware consensus** | consensus weights the primary's vote (`w_primary`, confidence-scaled) | **on, w_primary=1.0** | paper_style **+0.064 acc**; neutral→protective for full_agentic |
| 3 | **Primary-signal prompt block** | agents can be shown the primary's prediction | **off** | induces anchoring without accuracy gain (see §4) |

Seam flags (no default change): `--consensus_primary_weight {0|1.0}`,
`--agents_use_primary_signal`.

---

## 4. Ablation results — Fix #2 × Fix #3 (2×2)

XLM-R, full_agentic, EESA test (818), gpt-4o-mini. **Both runs clean** (0 connection
/ 0 parse errors). **anchor%** = escalated final == primary_only agreement (higher =
more copying).

### Threshold 0.8 (escalation 109/818)
| Cell | w_p | sig | acc | macro F1 | anchor% | cost |
|---|---|---|---|---|---|---|
| A original | 0 | off | 0.8460 | 0.8331 | 62.4 | $0.057 |
| B Fix2 | 1.0 | off | 0.8447 | 0.8316 | 62.4 | $0.057 |
| C Fix3 | 0 | on | 0.8435 | 0.8312 | 65.1 | $0.064 |
| D both | 1.0 | on | 0.8423 | 0.8300 | 66.1 | $0.064 |

### Threshold 0.9 (escalation 190/818)
| Cell | w_p | sig | acc | macro F1 | anchor% | cost |
|---|---|---|---|---|---|---|
| A original | 0 | off | 0.8496 | 0.8386 | 63.7 | $0.099 |
| **B Fix2** | 1.0 | off | **0.8509** | **0.8401** | 64.7 | $0.099 |
| C Fix3 | 0 | on | 0.8472 | 0.8369 | 67.4 | $0.111 |
| D both | 1.0 | on | 0.8496 | 0.8394 | 70.0 | $0.111 |

**Conclusions (robust across 0.8 & 0.9):**
- Accuracy differences are **within GPT run-to-run noise** (spread ≈ 0.004).
- **Fix #3 induces anchoring** — signal-ON raises copy-rate +3–7 pts (70% at 0.9) —
  **without accuracy benefit** and at higher cost → **keep OFF** for strong agents.
- **Fix #2** is neutral→slightly-protective for full_agentic (B edges ahead at 0.9)
  and a **big win for paper_style (+0.064)** → **keep ON, w_primary=1.0**.

---

## 5. Final sentiment decision (locked, 2026-06-13)
- **Fix #2 ON, w_primary=1.0** · **Fix #3 OFF** (= shipped defaults; no code change).
- Router unchanged; threshold is a per-run knob (config default 0.6).
- **Best setting:** XLM-R + full_agentic + gpt-4o-mini + threshold 0.9 + w_p 1.0 +
  signal off → **0.8509 acc / 0.8401 macro F1**.
- **Practical:** threshold 0.8 acceptable at ~half the cost (0.8447 / 0.8316).
- Details: `SENTIMENT_FINAL_STATUS.md`.

---

## 6. Experiment A — full-EESA-trained XLM-R (strong reference)
Train = real EESA train (**2,464**), AdamW. EESA test:
| setting | accuracy | macro F1 |
|---|---|---|
| primary_only | 0.8240 | 0.8088 |
| full_agentic best (th 0.9) | 0.8509 | 0.8401 |

---

## 7. Experiment C — generated-240 transfer pilot (SEPARATE)
Train = **240 SwitchLingua-generated** samples (80/80/80); dev/test = real EESA;
XLM-R **fine-tuned** (not from scratch). Full report:
`experiment_C/EXPERIMENT_C_SWITCHLINGUA_240_REPORT.md`.

| stage | accuracy | macro F1 | weighted F1 |
|---|---|---|---|
| dev (epoch 4) | 0.6284 | 0.6038 | 0.6258 |
| **primary_only (EESA test)** | **0.5905** | **0.5619** | **0.5838** |

Per-class F1 (test): positive **0.703** · negative **0.510** · neutral **0.473**.
Confusion: main error is neutral↔positive.

- **full_agentic: stopped, no metric** — weak/uncalibrated primary escalated **~95 %**
  at threshold 0.9 (vs 23 % for the EESA model); projected ~80 min / ~$0.45.
  Needs a lower threshold (0.6–0.7) to be meaningful.
- **Caveats:** fine-tune not scratch; **not size-matched** (2,464 vs 240); optimizer
  differs (**Adafactor** here vs **AdamW** in A) → **pilot transfer, not a clean
  effect**. Δ vs A primary_only = −0.23 acc (data + size + optimizer confounded).
- **For a fair comparison:** re-run EESA-2,464 *and* an EESA-240 subset with the
  same `--optim adafactor`.

---

## 8. Environment / hardware learnings
- **XLM-R full fine-tuning barely fits a 4 GB Windows GPU.** Peak memory = model +
  optimizer + activations, **independent of dataset size** (240 vs 2,464 changes
  only step count).
- Failure modes seen: GPU OOM (AdamW floor > usable VRAM), GPU OOM via fragmentation
  (no `expandable_segments` on Windows), CPU intermittent segfault (libiomp/MKL).
- **Working recipe:** clean GPU (after restart) + **Adafactor** + batch 4/grad_accum 4
  + fp16 + gradient-checkpointing + `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`.
- Inference (primary_only, full_agentic) fits comfortably; only **training** is tight.
- Fixed a save-order bug in `finetune_transformer_classifier.py` (model moved to CPU
  before the final dev predict → device mismatch; now restores device).

---

## 9. Test suite
**897 tests passing**, fully offline (mock LLM + mock/real primary). Covers
orchestrator paths, per-mode agent selection, router gating, consensus tie-breaks,
abstain fallback, prompt genericity, and NER path.

---

## 10. Open items / next steps
| Item | Priority | Notes |
|---|---|---|
| Fair generated-vs-real comparison | high | EESA-2,464 + EESA-240 with Adafactor |
| Exp C full_agentic at threshold 0.6–0.7 | medium | meaningful escalation rate |
| Router / per-task threshold + margin | medium | mainly a **topic-phase** enabler |
| Revisit Fix #3 for **topic** | medium | weak 9-way agents may benefit |
| Confidence calibration (M3) | low | agent self-confidences uncalibrated |
| NER fixed-mode refinement loop bug | known | guardrail/counter interaction |

---

## 11. Artifact index
- `SENTIMENT_FINAL_STATUS.md` — locked sentiment decision
- `ablation_2x2/ABLATION_2x2_RESULTS.md` — threshold 0.8
- `ablation_2x2_th0.9/ABLATION_2x2_th0.9_RESULTS.md` — threshold 0.9
- `agent_prompt_audit/` — audit + per-fix changelogs, threshold-ablation plan
- `experiment_C/EXPERIMENT_C_SWITCHLINGUA_240_REPORT.md` — transfer pilot
- `experiments/checkpoints/` — `eesa_xlm_roberta_base` (A), `expC_switchlingua_xlmr_240` (C)
