# Project Status — A to Z (2026-06-14)

Two parallel tracks share the SwitchLingua generation pipeline:
- **Track 1 — SwitchLingua thesis** (per-sentence vs aggregate scoring). **Code FROZEN.**
- **Track 2 — Multi-Agent BERT Experiment C** (does SwitchLingua-*generated* sentiment data transfer to real EESA?). **Active.**

---

## 0. TL;DR
- **SwitchLingua:** contribution = *per-sentence scoring as a detection/routing mechanism*. Masking shown (35.6% at calibrated bar 7), refiner improves caught sentences (+0.60), TaskValidator helps NER, deterministic CS-counter beats LLM-only (40% self-disagreement). Code frozen; **human annotation pending** to finalize.
- **Multi-Agent BERT Exp C:** trained xlm-roberta on **240 synthetic** sentences → **59.0% acc / 0.562 macro-F1** on real EESA test vs **83.1%** for an EESA-trained model. Diagnosed as **expected weak synthetic transfer, not a bug**. Now **scaling 240 → 480** (160/label); after 2 windows the pool is pos 181 / neu 156 / **neg 147** — 13 short of target, paused on daily quota.

---

## 1. SwitchLingua thesis (Track 1)

### Claim
Per-sentence scoring catches a weak sentence that **aggregate** (scenario-average) scoring hides ("masking"), routes it to the refiner, and thus yields cleaner code-switched data. **System B** (Modified) = contribution; **System C** (Original) = control. Model: gpt-4o-mini.

### Architecture (System B vs C)
```
                         ┌─────────────────────────── SYSTEM B (Modified, contribution) ───────────────────────────┐
 config (pre_execute)    │  Generate N sentences                                                                    │
   → generate_scenarios  │        │                                                                                 │
   (Cartesian product)   │        ▼                                                                                 │
        │                │  PER-SENTENCE scoring  ── fluency · naturalness · cs_ratio · socio-cultural (each /10)   │
        ▼                │        │           └→ deterministic CS-ratio counter (compute_true_cs_stats, 0 variance) │
   scenario {topic,      │        ▼                                                                                 │
   label, cs_ratio,      │  TaskValidatorAgent  (topic / sentiment / NER  — is the task actually satisfied?)        │
   cs_type, intensity…}  │        │                                                                                 │
                         │        ▼                                                                                 │
                         │  meet_criteria?  per sentence: weighted_score ≥ bar AND task_passed                      │
                         │        │  no → Refiner (targeted + task-aware) → re-validate + re-score → rollback if worse│
                         │        ▼                                                                                 │
                         │  AcceptanceAgent → JSONL (per-sentence records)                                          │
                         └──────────────────────────────────────────────────────────────────────────────────────┘
   SYSTEM C (control): same generate, but AGGREGATE (scenario-mean) score + generic refiner, topic only.
```
Key difference: **B decides per sentence; C decides on the scenario average** → C "masks" a single weak sentence inside an otherwise-good batch.

### Results scorecard
| Claim | Evidence | Verdict |
|---|---|---|
| Per-sentence **catches** what aggregate hides | **35.6%** masking at bar 7 (101 scen); 41.5% (54 scen) | ✅ |
| Refining a caught sentence **improves** it | **+0.60**, 79/87 (90.8%), p≈0 | ✅ |
| Task-aware generation valid | topic **100%**, sentiment **70%**, NER **40%** (English-only) | 🟡 mixed |
| TaskValidator cuts task-wrong accepts | precision **70.9%→85.5%**, 25→9 wrong (NER-concentrated) | ✅ (NER) / ❌ (sentiment) |
| Deterministic CS-counter > LLM-only | LLM self-disagrees **40%** of sentences; deterministic **0 variance** | 🟡 reproducibility shown; human accuracy pending |
| Hits requested 70% CS ratio | MAE ≈ 14–23 pts off | ❌ off-target |
| Quality alone detects task failure | fluency/nat ~8 even when task wrong | ❌ (motivates validator) |
| Your refiner rewrites *better* than original's | tie, p=0.53 | ❌ not supported |
| Masking at default bar 8 | 0% (scores packed ~7) | ❌ (must calibrate bar ~7) |

**Defensible thesis statement:** the contribution is *detection/routing* (what gets refined), **not** a better refiner (rewrite step is a tie, p=0.53).

### Status
**Code FROZEN.** Pending = human annotation: (a) BLIND sheet (86 rows) → `analyze_consolidated_human_eval.py`; (b) CS-ratio manual token counts (30 rows) → `run_csratio_partial_validation.py --reuse-llm`. Full detail: `experiments/outputs/switchlingua/{masking_defense/REPORT.md, FINAL_STATUS.md}`.

---

## 2. Multi-Agent BERT — Experiment C (Track 2)

### Goal
Can sentiment data **generated** by SwitchLingua train a classifier that transfers to **real** Arabic-English code-switched data (EESA)? Test anchor = EESA test (818). Reference = Exp A (train on real EESA).

### Architecture & data flow
```
 SwitchLingua pipeline (gpt-4o-mini)         FILTER chain (every example)            ACCUMULATE
 ┌────────────────────────────┐   raw   ┌───────────────────────────────┐   kept  ┌────────────────────┐
 │ config_sentiment_expC*.yaml│────────▶│ non-empty                     │────────▶│ pilot_v1 (frozen)  │
 │ generate_scenarios → run   │ ~4.7/scn│ → TaskValidator passed        │         │ + daily_runs/      │
 └────────────────────────────┘         │ → deterministic CS-valid      │         │ + manifest (resume)│
                                         │ → quality ≥ 7.0               │         └─────────┬──────────┘
                                         │ → de-dup (normalized text)    │                   │ merge + balance
                                         └───────────────────────────────┘                   ▼
                                                                              ┌──────────────────────────────┐
                                                                              │ balanced train set (CSV/JSONL)│
                                                                              └───────────────┬──────────────┘
                                                                                              ▼  fine-tune xlm-roberta-base
                                                                              ┌──────────────────────────────┐
 EESA test (818, real) ───────────────────────────────────────── evaluate ──│ primary_only classifier       │
                                                                              └──────────────────────────────┘
 (agentic modes paper_style / full_agentic available but NOT used yet — primary still weak)
```

### Data-generation journey
```
240 (80/label) ──train──▶ 59% acc on EESA ──"looks bad"──▶ DIAGNOSE CS-validity
   │                                                            │
   │  v1 config: cs_ratio 70%, Intra+Inter                      ▼
   │                                            30% CS-valid; 99.6% of failures = fully-Arabic
   │                                            (70% Arabic too heavy; Intersentential breaks per-sentence)
   ▼  config-only FIX (no prompt/NER/filter change)
 v2 pilot: cs_ratio[50,60,70] Intra-only ──▶ 43% CS-valid (60% best in tiny pilot)
   ▼
 v3 accumulation: cs_ratio[50,60] Intra-only ──▶ at scale 50%≈49% / 60%≈40% CS-valid
   ▼
 SCALING 240 → 480 (160/label): window1 + window2 done → pool pos181/neu156/neg147 (paused on quota)
```

### Dataset 240 (accepted, frozen)
- `multi-agent-bert/data/Sentiment/generated/merged/switchlingua_sentiment_train_240_80perlabel.{csv,jsonl}`
- 80/80/80; **0 dups; 240/240 CS-valid; 240/240 validator-passed; quality 7.0–8.4.**
- Source mix: run_20260613 141 · pilot_v1 94 · run_20260606 5. cs_ratio mix: 70% 99 / 60% 72 / 50% 69.
- Card: `merged/DATASET_CARD_EXP_C.md`.

### Training (Exp C)
- Base `xlm-roberta-base`, fine-tuned (not from scratch). 240 samples, 4 epochs, Adafactor, fp16, eff. batch 16, final train loss 0.89. Checkpoint `experiments/checkpoints/expC_switchlingua_xlmr_240/`.

### Results — EESA test (818), primary_only
| metric | Exp C (240 synthetic) | Exp A (EESA real, 2,464) |
|---|--:|--:|
| accuracy | **0.590** | 0.831 |
| macro F1 | **0.562** | 0.819 |
| weighted F1 | ≈0.584 | 0.831 |
| majority baseline | 0.444 | — |

Per-class (Exp C): positive P0.66/R0.75/F0.70 · negative P0.58/R0.46/F0.51 · neutral P0.49/R0.46/F0.47.

Confusion matrix (rows=true, cols=pred):
```
            posi   nega   neut
positive    274     22     67
negative     48     90     59
neutral      95     44    119
```
**Not collapsed** (predicts all 3). Errors concentrate on negative/neutral leaking to positive/neutral; wrong cases are genuinely hard (sarcasm, promo, slang).

### Diagnosis verdict: **A — expected weak synthetic transfer, NOT a bug**
- Label maps consistent (pos0/neg1/neu2 in checkpoint == EESA reference); predictions string-aligned. No mismatch.
- +15 pp over majority baseline = real learning; no class collapse; sensible confusion.
- 24-pp gap vs Exp A explained by confounds: 10× smaller train (240 vs 2,464), synthetic→noisy-real domain shift, optimizer differs (Adafactor vs AdamW). For a fair read: run an **EESA-240 subset** control.
- Minor real bugs flagged (not fixed): fine-tune script's post-train eval crashes on a CPU/CUDA device mismatch (→ no dev_metrics.json); no train/dev split. Neither affects the 59%.

### Scaling 240 → 480 (current)
| label | pool (pre-balance) | target 160 |
|---|--:|--:|
| positive | **181** | ✅ |
| neutral | **156** | +4 |
| negative | **147** | **+13** |
| total | 484 | |
- 2 windows done (config v3 [50,60] Intra-only). Manifest **229/324** (95 remain).
- **Paused: daily quota exhausted** (window 2 had 48× 429). Need a small window 3 (`--max ~40`) after the rolling quota frees to finish negative+13/neutral+4, then `merge --target-per-label 160 --out-name switchlingua_sentiment_train_480_160perlabel`.
- 240 dataset preserved (snapshot); nothing overwritten.

### EESA reference baselines (Exp A)
xlm-roberta-base dev **0.831 / 0.819**; mBERT dev 0.807 / 0.790.

---

## 3. Key decisions log
- SwitchLingua frozen (2026-06-05); no prompt/core/NER changes since.
- Exp C uses the modified pipeline as a *data generator*; filters = validator + deterministic CS-valid + quality≥7 + dedup.
- CS-validity fix is **config-only** (cs_ratio + cs_type); filter never loosened (correctly rejects monolingual).
- Accumulation is resume-safe (`scenario_id` manifest), append-only, never overwrites accepted data.
- Management-script changes only (`--config`, `--manifest`, `--out-name`); SwitchLingua untouched.

## 4. Pending / next steps
1. **Exp C scaling:** window 3 (`--max 40`) after quota frees → build balanced 480 + `DATASET_CARD_EXP_C_480.md`, then retrain & re-evaluate.
2. **Fair-comparison control:** EESA-240 subset, same Adafactor setup.
3. **SwitchLingua:** human BLIND annotations + CS-ratio token counts → finalize Tests 1/2/4.
4. (Later) fix fine-tune post-train eval device bug; consider agentic modes once primary is stronger.

## 5. Key file index
- SwitchLingua: `experiments/outputs/switchlingua/masking_defense/REPORT.md`, `FINAL_STATUS.md`, `human_eval/`, `csratio/`.
- Exp C data + cards: `multi-agent-bert/data/Sentiment/generated/{merged/, pilot_v1/, daily_runs/, completed_scenarios_v3.json}`.
- Exp C training/eval: `multi-agent-bert/experiments/checkpoints/expC_switchlingua_xlmr_240/`, `…/experiment_C/EXPERIMENT_C_SWITCHLINGUA_240_REPORT.md`, `…/CS_VALIDITY_DIAGNOSIS.md`.
- Tooling: `experiments/switchlingua/{manage_sentiment_data.py, config_sentiment_expC*.yaml, run_pilot_csfix.py, analyze_cs_validity.py}`.
