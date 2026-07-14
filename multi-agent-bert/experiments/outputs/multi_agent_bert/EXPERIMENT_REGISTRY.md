# Multi-Agent BERT — Master Experiment Registry (Sentiment + Topic)

Authoritative, chronological catalogue of every experiment conducted on the
**Multi-Agent BERT** track, compiled for the thesis. Each entry uses a fixed
structure. Numbers are taken verbatim from the source reports; where a later run
corrected an earlier reading, the corrected version is kept and the evolution is
noted. Dataset = EESA sentiment test (818) unless stated. Primary = XLM-R
(`xlm-roberta-base`) unless stated. LLM agents = GPT-4o-mini unless stated.

**Global architecture (shared by all sentiment/topic experiments).** A fast
**primary transformer classifier** predicts a label + confidence + probabilities;
a **router** compares confidence to a threshold and either accepts the primary
(fast path) or **escalates** the sample to a **panel of specialist agents** that
deliberate and reach a **consensus** (slow path). Three run modes:
**primary_only** (no agents), **paper_style** (deterministic non-LLM agents),
**full_agentic** (LLM-backed agents, optional deliberation). Consensus is a
confidence-weighted vote with a config-gated **primary-aware** term
(`w_primary`). Code map: `src/models/…`, `src/pipeline/{router,orchestrator}.py`,
`src/agents/…`, `src/prompts/…`, `evaluate_pipeline.py`.

**The single unifying result** (established empirically and used throughout):
> **Δaccuracy ≈ (agent-ceiling − primary-accuracy-on-escalated) × escalation-rate**,
> with the agent ceiling on hard code-switched cases fixed at **~0.75**. The agentic
> layer therefore **helps when the primary is weak** on its escalated subset,
> is **neutral near parity**, and **slightly hurts a very strong primary**.

---

# PART I — EXPERIMENTS (chronological)

---

## EXPERIMENT_A_EESA_MBERT
**Experiment Name:** Experiment A — EESA real-data reference baseline (mBERT)
**Chronological Order:** 1 (2026-06-06)
**Research Question:** What accuracy does a primary transformer reach on EESA when
real in-domain labelled code-switched data is available?
**Motivation:** Establish the upper-reference "real data" bar before testing
generated-data transfer (Experiment C).
**Hypothesis:** A fine-tuned multilingual encoder reaches ~0.80 on EESA test.
**Baseline:** None (this is a founding baseline).
**Architecture:** primary_only / paper_style / full_agentic with mock LLM;
deterministic lexical+logic+contextual agents; router threshold as configured.
**Implementation Changes:** Fine-tune `scripts/finetune_transformer_classifier.py`;
checkpoint `experiments/checkpoints/eesa_mbert/`.
**Dataset:** EESA official splits — train 2,464 (1,092/594/778), dev 818, test 818
(363 pos / 197 neg / 258 neu). 100% code-switched.
**Model:** primary = `bert-base-multilingual-cased` (mBERT); LLM = mock.
**Parameters:** 4 epochs, batch 16, lr 2e-5, max_len 256, seed 42, GTX 1650 Ti 4 GB.
**Evaluation Metrics:** primary_only **acc 0.7971 · macro F1 0.7833 · wF1 0.7973**
(pos F1 0.862 / neg 0.731 / neu 0.757). Dev acc 0.8068. paper_style = full_agentic
= 0.7958 / 0.7788 (escalation 5.75%, escalated-only acc 0.4043).
**Main Findings:** Strong on positive (majority) class; negative recall weakest
(0.69, 48 negatives → neutral). With the mock agents, escalation **slightly hurts**.
**Strengths:** Clean real-data reference; fast (~10.6 min).
**Weaknesses:** Negative class weak; mock agents worse than the primary they override.
**Why It Succeeded/Failed:** Founding baseline — no comparator.
**Decision:** **Historical** (superseded as the reference by the XLM-R run).
**Influence on Later Work:** Motivated (i) using a stronger encoder (XLM-R), (ii)
replacing mock agents with real LLMs (the pilot), and (iii) revisiting agent value
on a *weaker* primary (Experiment C).

---

## EXPERIMENT_A_EESA_XLMR
**Experiment Name:** Experiment A — EESA real-data reference baseline (XLM-RoBERTa)
**Chronological Order:** 2 (2026-06-06)
**Research Question:** Does a stronger multilingual encoder (XLM-R) beat mBERT as the
EESA reference primary?
**Motivation:** mBERT's negative recall was weak; XLM-R is ~2.5× larger and stronger
on code-switched text.
**Hypothesis:** XLM-R improves all three classes, especially negative.
**Baseline:** mBERT Experiment A (0.7971 / 0.7833).
**Architecture:** Identical pipeline; base model swapped.
**Implementation Changes:** Added `--grad_accum` / `--gradient_checkpointing`;
checkpoint `experiments/checkpoints/eesa_xlm_roberta_base/`.
**Dataset:** EESA (same as above).
**Model:** primary = `xlm-roberta-base` (~270M); LLM = mock.
**Parameters:** 4 epochs, per-device batch 8 × grad_accum 2 (eff 16), fp16 + grad
checkpointing (required to fit 4 GB), lr 2e-5, max_len 256, seed 42. ~41 min.
**Evaluation Metrics:** primary_only **acc 0.8240 · macro F1 0.8088 · wF1 0.8232**
(pos F1 0.897 / neg 0.768 / neu 0.762). Dev acc 0.8313. paper_style 0.8142 /
full_agentic (mock) 0.8130 (escalation ~5%, agents slightly hurt).
**Main Findings:** XLM-R wins every aggregate (+0.027 acc over mBERT); biggest gain =
negative recall 0.690→0.756; cuts neg→neu confusion 48→36.
**Strengths:** Strongest of our trained EESA primaries at this stage; recovers the
minority class.
**Weaknesses:** ~4× slower; mock agents still hurt.
**Why It Succeeded/Failed:** Larger, better-pretrained CS encoder.
**Decision:** **Current best (trained primary, reference)** — the canonical
"Experiment A" primary used throughout ("0.8240").
**Influence on Later Work:** Became the primary in the real-LLM pilot/sweep, the
consensus 2×2 ablations, and the augmentation controls.

---

## EXPERIMENT_REAL_LLM_PILOT
**Experiment Name:** Real-LLM full_agentic pilot (GPT-4o-mini)
**Chronological Order:** 3 (2026-06-06)
**Research Question:** Do *real* LLM specialist agents help on the escalated subset,
where the mock agents hurt?
**Motivation:** Mock full_agentic collapsed the minority classes and lost to the
primary; the agents (not the routing) were suspected as the bottleneck.
**Hypothesis:** Capable LLM agents beat the primary on the low-confidence subset.
**Baseline:** primary_only + mock full_agentic (both mBERT and XLM-R).
**Architecture:** full_agentic with 4 LLM agents (lexical/logic/contextual/
explainability), deliberation off, JSON-mode structured output.
**Implementation Changes:** `--llm_client openai --llm_model gpt-4o-mini`, JSON mode
(`response_format=json_object`). Default client stays mock.
**Dataset:** EESA test (818).
**Model:** primary = mBERT and XLM-R; LLM = gpt-4o-mini (temp 0).
**Parameters:** router threshold **0.6**.
**Evaluation Metrics:**
- mBERT: primary 0.7971/0.7833 → **real-LLM 0.8166/0.8038** (+0.0195 acc); escalated
  47, primary-on-esc 0.4255 → agents 0.7660, **W→C 21 / C→W 5 / net +16**.
- XLM-R: primary 0.8240/0.8088 → **real-LLM 0.8399/0.8264** (+0.0159 acc); escalated
  41, primary-on-esc 0.4146 → agents 0.7317, **W→C 15 / C→W 2 / net +13**.
- Cost total $0.0438 (350 calls), 0 parse errors; mock comparators net −1 / −8.
**Main Findings:** Real LLM agents **nearly double** agent accuracy on the escalated
slice and turn net strongly positive; the gain concentrates on the **negative** class
(the primary's weak spot). **First result where the multi-agent system beats the
fine-tuned primary on EESA.**
**Strengths:** Cheap (~$0.02/run), minority-class recovery, JSON mode removes parse
errors.
**Weaknesses:** Single threshold (0.6); deliberation off.
**Why It Succeeded/Failed:** The escalated subset carries recoverable signal, and a
capable agent reads it where the mock could not.
**Decision:** **Accepted** — validates the escalation design with real agents.
**Influence on Later Work:** Directly motivated the real-LLM threshold sweep and the
consensus-fix program.

---

## EXPERIMENT_REAL_LLM_THRESHOLD_SWEEP
**Experiment Name:** Real-LLM threshold sweep (GPT-4o-mini)
**Chronological Order:** 4 (2026-06-09)
**Research Question:** Does routing *more* samples to the LLM agents (higher threshold)
improve over the 0.6 pilot?
**Motivation:** The mock sweep declined monotonically; test whether real agents behave
oppositely.
**Hypothesis:** Higher escalation keeps helping until the agents run out of headroom.
**Baseline:** primary_only; the 0.6 pilot.
**Architecture:** full_agentic at native threshold 0.9; 0.7/0.8 derived exactly as
strict subsets (agents are threshold-independent at temp 0); 0.6 reused.
**Implementation Changes:** None (routing knob only).
**Dataset:** EESA test (818).
**Model:** primary = XLM-R (clean) and mBERT (contaminated by an internet outage);
LLM = gpt-4o-mini.
**Parameters:** thresholds 0.6 / 0.7 / 0.8 / 0.9.
**Evaluation Metrics (XLM-R, clean):**
| thr | esc% | acc | macroF1 | prim-on-esc | agent-on-esc | W→C | C→W | net |
|---|---|---|---|---|---|---|---|---|
| 0.6 | 5.0 | 0.8399 | 0.8264 | 0.415 | 0.732 | 15 | 2 | +13 |
| 0.7 | 7.5 | 0.8423 | 0.8284 | 0.443 | 0.689 | 20 | 5 | +15 |
| 0.8 | 13.3 | **0.8460** | 0.8328 | 0.495 | 0.661 | 26 | 8 | +18 |
| 0.9 | 23.2 | 0.8447 | **0.8330** | 0.558 | 0.647 | 41 | 24 | +17 |

mBERT: only 0.6 clean (0.8166/0.8038); 0.9 attempts contaminated (82/129 connection
errors, checkpoints later deleted → clean high-threshold mBERT never completed).
**Main Findings:** Raising the threshold **does not hurt** and modestly helps (acc peaks
0.8460 at 0.8, macro F1 plateaus ~0.833). Agents stay well above the primary on the
escalated subset at every threshold; gain driven by **negative-class recovery**
(neg F1 0.792→0.827). Opposite of the mock sweep.
**Strengths:** Confirms agents (not routing) were the bottleneck.
**Weaknesses:** mBERT rows contaminated; parse errors 0 but connection outage.
**Why It Succeeded/Failed:** Real agents > primary on the escalated slice throughout.
**Decision:** **Accepted** — escalation design validated across thresholds.
**Influence on Later Work:** Set threshold 0.9 as the default for the consensus 2×2
ablations; exposed the checkpoint-backup requirement.

---

## EXPERIMENT_AGENT_PROMPT_AUDIT_AND_FIXES
**Experiment Name:** Prompt/logic audit → four correctness fixes
**Chronological Order:** 5 (~2026-06-11→06-13)
**Research Question:** Are there correctness defects in the prompt/consensus logic that
bias results independent of model quality?
**Motivation:** Audit before scaling; ensure prompts are generic (topic-reusable) and
consensus is unbiased.
**Hypothesis:** Removing hardcoded labels, label-0 bias, and adding a primary-aware
vote will improve or protect accuracy without harm.
**Baseline:** XLM-R full_agentic (original prompts).
**Architecture:** Four fixes: (1) **generic task-config-driven prompts** (no hardcoded
sentiment/topic labels); (1b) **abstain/no-vote fallback** (no silent `labels[0]`);
(2) **primary-aware consensus** (`w_primary`, confidence-scaled, non-positional
tie-break); (3) optional **primary-signal prompt block** (agents shown the primary's
label), config-gated, default off.
**Implementation Changes:** `src/prompts/{_primary_block,_abstain}.py`;
`src/agents/consensus_agent.py`; seams `--consensus_primary_weight {0|1.0}`,
`--agents_use_primary_signal`. 897 offline tests pass.
**Dataset:** EESA test (818).
**Model:** XLM-R primary; gpt-4o-mini.
**Parameters:** `w_primary` ∈ {0, 1.0}; signal ∈ {off, on}.
**Evaluation Metrics:** Fix #2 gives **+0.064 acc on paper_style** (weak agents) and is
neutral→protective on full_agentic. Fix #3 induces **anchoring** (copy-rate up) with no
accuracy gain.
**Main Findings:** Generic prompts + abstain remove structural bias; primary-aware
consensus is a big win for weak agents; the primary-signal block only adds copying.
**Strengths:** Makes the framework topic-reusable and unbiased.
**Weaknesses:** Fix #3 default-off decision needs the 2×2 evidence (next).
**Why It Succeeded/Failed:** Anchoring is a real, measurable failure mode.
**Decision:** **Accepted** — Fix #1/1b/2 on; Fix #3 off (pending 2×2 confirmation).
**Influence on Later Work:** Defined the 2×2 ablation and the locked sentiment defaults.

---

## EXPERIMENT_ABLATION_2x2_TH08 / _TH09
**Experiment Name:** Consensus 2×2 ablation (Fix #2 × Fix #3) at thresholds 0.8 and 0.9
**Chronological Order:** 6 (2026-06-13)
**Research Question:** Do the primary-vote (Fix #2) and primary-signal (Fix #3) fixes
help, hurt, or wash on the strong XLM-R primary?
**Motivation:** Justify the shipped defaults empirically.
**Hypothesis:** Fix #2 neutral/protective; Fix #3 raises anchoring without accuracy.
**Baseline:** Cell A (w_p 0, signal off).
**Architecture:** full_agentic, 4 cells (w_p ∈{0,1.0} × signal ∈{off,on}).
**Implementation Changes:** None beyond the seam flags.
**Dataset:** EESA test (818).
**Model:** XLM-R; gpt-4o-mini (both runs clean, 0 errors).
**Parameters:** thresholds 0.8 (esc 109) and 0.9 (esc 190).
**Evaluation Metrics:**
- **@0.8:** A 0.8460/0.8331 (anchor 62.4) · B 0.8447/0.8316 (62.4) · C 0.8435/0.8312
  (65.1) · D 0.8423/0.8300 (66.1). Cost $0.057–0.064.
- **@0.9:** A 0.8496/0.8386 (63.7, net +21) · **B 0.8509/0.8401 (64.7, net +22, BEST)**
  · C 0.8472/0.8369 (67.4) · D 0.8496/0.8394 (70.0). Cost $0.099–0.111.
**Main Findings:** Accuracy differences within GPT run-to-run noise (~±0.004);
**signal-ON raises copy-rate +3–7 pts without accuracy gain**; Fix #2 edges ahead at 0.9
and is a big win for weak agents. **Best sentiment setting = 0.8509 / 0.8401** (cell B).
**Strengths:** Two clean, controlled ablations; anchoring is the clean differentiator.
**Weaknesses:** Accuracy spread noise-level; escalation flooding at very low thresholds
untested here.
**Why It Succeeded/Failed:** Consensus is at parity with a strong primary → topology
tweaks wash.
**Decision:** **Accepted / locked (2026-06-13):** Fix #2 ON (`w_primary=1.0`), Fix #3
OFF. Practical low-cost point = threshold 0.8 (0.8447/0.8316, ~half the cost).
**Influence on Later Work:** These defaults carry into every subsequent sentiment run;
the "topology washes on a strong primary" theme recurs through the agent-design ladder.

---

## EXPERIMENT_C_SWITCHLINGUA_240
**Experiment Name:** Experiment C1 — generated-240 transfer pilot
**Chronological Order:** 7 (mid-June 2026)
**Research Question:** Does task-aware SwitchLingua-generated sentiment data transfer to
real EESA when used as the *only* training data?
**Motivation:** Test the dataset's standalone value vs the real-data reference.
**Hypothesis:** Generated data transfers above 3-class chance (0.33) but below real data.
**Baseline:** Experiment A XLM-R (0.8240, but AdamW; note confound).
**Architecture:** primary_only (fine-tune, not scratch).
**Implementation Changes:** Train on 240 generated (80/80/80); dev/test = real EESA.
**Dataset:** 240 SwitchLingua-generated (standalone); EESA dev/test.
**Model:** XLM-R fine-tuned; Adafactor.
**Parameters:** eff batch 16, lr 2e-5, 4 epochs (202 s).
**Evaluation Metrics:** dev (ep4) 0.6284; **primary_only test 0.5905 / 0.5619 / wF1
0.5838** (pos F1 0.703 / neg 0.510 / neu 0.473; over-predicts positive). full_agentic
**stopped, no metric** (weak primary escalated ~95% at 0.9).
**Main Findings:** 240 generated samples transfer to ~59% on real EESA — well above
chance, below the 2,464-real model. Gap confounded by **10× less data + optimizer
(Adafactor vs A's AdamW)** → transfer *signal*, not a clean effect size.
**Strengths:** Demonstrates real transferable sentiment signal in the generated data.
**Weaknesses:** Not size-matched, optimizer differs, agentic stopped.
**Why It Succeeded/Failed:** Genuine signal but small/off-optimizer.
**Decision:** **Historical / superseded** by the 3-seed C2/C3 scaling study.
**Influence on Later Work:** Motivated size scaling (480, 960), a matched-optimizer
control (E0), and a lower agentic threshold for weak primaries.

---

## EXPERIMENT_C2_480 / C3_960 + SEED_STABILITY
**Experiment Name:** Generated-only scaling (C2=480, C3=960) with 3-seed stability
**Chronological Order:** 8 (2026-06-21)
**Research Question:** Does generated-only training scale with size (240→480→960)?
**Motivation:** A single-run "480 > 960" reading needed a seed check.
**Hypothesis:** More generated data helps, monotonically.
**Baseline:** C1-240 (0.5905).
**Architecture:** primary_only, fresh XLM-R per run, identical recipe.
**Implementation Changes:** 3 seeds each (42/123/456); `experiment_seed_stability/`.
**Dataset:** SwitchLingua-generated 480 / 960 (standalone); EESA dev/test.
**Model:** XLM-R; Adafactor, eff batch 16, max_len 256, 4 epochs.
**Parameters:** seeds 42/123/456.
**Evaluation Metrics:**
- **C2-480:** 0.6500 ± 0.0160 acc / 0.6345 ± 0.0170 macroF1 (range 0.6308–0.6699).
- **C3-960:** 0.6695 ± 0.0238 acc / 0.6592 ± 0.0209 macroF1 (range 0.6381–0.6956).
- Mean diff 960−480 = **+0.0195 acc / +0.0247 macroF1** (~1 std, n=3).
**Main Findings:** **Scales positively 240→480→960** (~0.59→0.65→0.67). The single-run
"480 > 960" was a **seed artifact** (C3 seed-42 was the low outlier); the earlier
plateau/regression claim is **retracted**. Difference within seed variance but
directionally favouring 960.
**Strengths:** Corrects a false conclusion; sets a methodology (report mean ± std ≥3
seeds).
**Weaknesses:** n=3, not statistically conclusive (needs ≥5 seeds for a firm claim).
**Why It Succeeded/Failed:** Real scaling signal; single-seed reads unreliable at this
scale (std ~±0.02).
**Decision:** **Accepted** (scaling); the 480>960 reading **superseded/retracted**.
**Influence on Later Work:** Selected the best-dev 960 checkpoint (seed 456, dev macroF1
0.6820) for the weak-primary agentic runs.

---

## EXPERIMENT_C3_960_FULL_AGENTIC_SEED456
**Experiment Name:** C3-960 full_agentic on the selected seed-456 checkpoint
**Chronological Order:** 9 (2026-06-23)
**Research Question:** How much does the multi-agent pipeline rescue a **weak generated**
primary?
**Motivation:** Weak primary ⇒ maximum agent headroom (per the strength curve).
**Hypothesis:** Large positive rescue (far above the strong-primary regime).
**Baseline:** C3-960 seed-456 primary_only (0.6956).
**Architecture:** full_agentic (default trio), threshold 0.9, Fix-2 on, signal off.
**Implementation Changes:** None (uses the shipped defaults).
**Dataset:** EESA test (818); primary trained on 960 generated.
**Model:** XLM-R (seed 456, real transformer, GPU); gpt-4o-mini.
**Parameters:** threshold 0.9, w_primary 1.0.
**Evaluation Metrics:** primary 0.6956/0.6830 → **full_agentic 0.7543/0.7387/wF1 0.7515**
(**Δ +0.0587 / +0.0557**). Escalated 231 (28.2%), escalated-only 0.749, **W→C 71 /
C→W 23 / net +48**. Cost ~$0.12. Every class improved.
**Main Findings:** **Largest agent rescue of any experiment** — the strong-rescue end of
the primary-strength curve. full_agentic (0.754) closes much of the gap to the real-EESA
primary (0.824) at ~$0.12.
**Strengths:** Demonstrates the agentic layer's genuine value where the primary is weak.
**Weaknesses:** seed-456 is the best-dev checkpoint (favourable-case), not a 3-seed mean.
**Why It Succeeded/Failed:** Primary-on-escalated (0.54) far below the ~0.75 agent ceiling.
**Decision:** **Accepted / Current best (weak-primary regime, gpt-4o-mini default trio).**
**Influence on Later Work:** Became the anchor the later Design-G and sequential runs had
to beat on the weak primary; the "decisive C3 check" for every strong-primary design.

---

## EXPERIMENT_E0_EESA_ONLY_ADAFACTOR
**Experiment Name:** E0 — EESA-only, Adafactor (matched augmentation control)
**Chronological Order:** 10 (2026-06-25)
**Research Question:** What is the EESA-only accuracy under the *matched* optimizer
(Adafactor) used by the augmentation runs?
**Motivation:** The original A reference used AdamW; augmentation must be compared under a
matched recipe to isolate the data effect.
**Hypothesis:** Adafactor ≥ AdamW on this task.
**Baseline:** A XLM-R AdamW (0.8240).
**Architecture:** primary_only.
**Implementation Changes:** Optimizer AdamW → Adafactor.
**Dataset:** EESA train (2,463–2,464); EESA test.
**Model:** fresh XLM-R; Adafactor, fp16, max_len 256.
**Parameters:** matched to the augmentation runs.
**Evaluation Metrics:** **acc 0.8533 · macro F1 0.8409** (best trained primary).
**Main Findings:** Switching AdamW→Adafactor adds **+0.029 acc** (0.8240→0.8533) — the
apparent "augmentation gain" in earlier readings was the optimizer, not the data.
**Strengths:** Our strongest trained primary; the clean augmentation control.
**Weaknesses:** Single recipe.
**Why It Succeeded/Failed:** Better optimizer for this fine-tune.
**Decision:** **Current best (our trained primary).**
**Influence on Later Work:** The E0 vs E3 contrast is the core of the augmentation verdict.

---

## EXPERIMENT_E3_EESA_PLUS_GEN960 / LR / RATIO_SWEEP / DIAGNOSIS
**Experiment Name:** Experiment E — augmentation (generated data mixed into real EESA)
**Chronological Order:** 11 (2026-06-25)
**Research Question:** Does generic generated data improve a real EESA model when used as
**augmentation** (mixed into real train)?
**Motivation:** Distinct from standalone C-series; tests the augmentation use-case.
**Hypothesis:** Generated data helps, at least in low-resource settings.
**Baseline:** E0 (0.8533) for full data; real-only subsets for low-resource.
**Architecture:** primary_only, fresh XLM-R, Adafactor, matched compute.
**Implementation Changes:** E3 = EESA+GEN-960; LR = EESA 10/25/50% ± fixed GEN-960;
ratio sweep = EESA 10/25/50% + r×N generated (r=0.25/0.5/1.0). Matched-compute uses
`--max_steps 400 + --load_best`.
**Dataset:** EESA train + SwitchLingua GEN-960; EESA test.
**Model:** XLM-R; Adafactor.
**Parameters:** mixture ratio; subset size; seeds mostly single.
**Evaluation Metrics:**
- **Full data:** E0 0.8533/0.8409 vs **E3 0.8411/0.8294 → −0.012 acc** (augmentation
  did not help).
- **Low-resource +GEN-960:** 10% real −0.034 (80% gen) · 25% −0.018 · 50% −0.002
  (harm monotone in generated fraction).
- **Ratio sweep (gen share 20/33/50%):** all Δ within ±0.026 (mostly ±0.018) — single-seed
  noise; no ratio reliably helps.
- **Diagnosis:** EESA CMI ~24, Arabic-dominant, dialectal noisy social-media; GEN-960 CMI
  ~41, balanced AR-EN, cleaner MSA; **vocab overlap with EESA test only ~10% (0.098)**.
**Main Findings:** Generic generated data is **neutral as a minority augmenter, harmful
when dominant, not reliably helpful** at any ratio — because of a **domain/register
mismatch**, not a generation defect. Its value is **standalone** (C-series), not
augmentation.
**Strengths:** Clean matched control + distributional diagnosis; a publishable negative.
**Weaknesses:** Most ratio cells single-seed (±0.02 noise; needs ≥3 seeds for firm small
effects).
**Why It Succeeded/Failed:** Off-domain signal cannot improve a different-domain target.
**Decision:** **Rejected as an augmentation method** (kept as a documented negative);
generated data **accepted** as standalone training data.
**Influence on Later Work:** Confirms the C-series (standalone) as the correct use of the
generated data; feeds the thesis dataset chapter.

---

## EXPERIMENT_T1_ARENTCV1_TOPIC / EXPERIMENT_T2_ARENTCV2_TOPIC
**Experiment Name:** Topic classification (9-class) on ARENTCV1 and ARENTCV2
**Chronological Order:** 12 (2026-06-20)
**Research Question:** Does the framework generalise to a second task (topic), and do the
agents help on a near-perfect primary?
**Motivation:** Test prompt genericity (Fix #1) on a different task; extend the strength
curve to the near-perfect end.
**Hypothesis:** Topic is easy (~0.99); agents ≈ noise.
**Baseline:** primary_only.
**Architecture:** primary_only + full_agentic (threshold 0.9, Fix-2 on, signal off).
**Implementation Changes:** Generic prompts reused unchanged (no sentiment labels).
**Dataset:** ARENTCV1 (train 73,976 / dev 10,569 / test 21,137; ~99.96% CS) and ARENTCV2
(73,956 / 10,562 / 21,134; 100% CS). 9 balanced labels
(business…tech).
**Model:** fresh XLM-R; gpt-4o-mini.
**Parameters:** 3 epochs, batch 16, max_len 64, Adafactor, fp16, seed 42 (~5 h each).
**Evaluation Metrics:**
- **T1:** primary 0.9946/0.9946/0.9946; full_agentic **0.9947/0.9948 (Δ +0.0001)**;
  escalated 63 (0.3%), escalated-only 0.540, **W→C 14 / C→W 11 / net +3**.
- **T2:** primary 0.9947/0.9947/0.9947; full_agentic **0.9944 (Δ −0.0003)**; escalated 48
  (0.2%), escalated-only 0.521, **W→C 7 / C→W 13 / net −6**.
- Main residual confusion both: health↔medical.
**Main Findings:** V1 ≈ V2 (the ~30 fully-Arabic rows immaterial). On a near-perfect
primary the agents are **noise-level** (net +3 / −6, ±0.0003). **Use primary_only for
topic.** Confirms the near-perfect end of the strength curve.
**Strengths:** Proves prompt genericity/task-transfer; extends the strength curve.
**Weaknesses:** Task too easy to stress the agents; escalated sets tiny/ambiguous.
**Why It Succeeded/Failed:** Primary already at ceiling → no headroom.
**Decision:** **Accepted** (primary_only recommended); agents **rejected** for topic.
**Influence on Later Work:** Anchors the "near-perfect primary → agents ≈ 0" data point in
the primary-strength curve.

---

## EXPERIMENT_AHMED_MODEL_BASELINE
**Experiment Name:** Ahmed external EESA sentiment baseline
**Chronological Order:** 13 (2026-06-27)
**Research Question:** What is the strongest EESA sentiment model on record, and can it be
used as a frozen primary?
**Motivation:** Ahmed supplied test predictions from a stronger architecture; establish the
true ceiling.
**Hypothesis:** Ahmed's model exceeds our best trained primary.
**Baseline:** E0 (0.8533).
**Architecture:** External model (char-CNN + BiLSTM + AraBERT features + sentiment
lexicon/hints); scored from provided predictions, not reproduced locally.
**Implementation Changes:** Text-aligned `ahmed_eesa_test_predictions_aligned.csv`
(818/818 aligned); `PrecomputedPrimaryClassifier` adapter for frozen-primary use.
**Dataset:** EESA test (818; 197 neg / 258 neu / 363 pos — same split).
**Model:** Ahmed's TF/Keras model (external).
**Parameters:** `tag2idx={neg:0,neu:1,pos:2}`; confidence max ~0.864.
**Evaluation Metrics:** **acc 0.9254 · macro F1 0.9207 · wF1 0.9254** (pos F1 0.9534 /
neg 0.9138 / neu 0.8948; reproduces Ahmed's reported numbers exactly). Residual confusion
neutral↔negative.
**Main Findings:** **Strongest EESA sentiment model on record** — **+0.072 acc / +0.080
macro F1 over E0**. External (not our pipeline) but pluggable as a frozen primary.
**Strengths:** True ceiling reference; well calibrated per-class.
**Weaknesses:** Evaluated from provided predictions, not reproduced from raw text; test
predictions only (no train/dev) → inference-time integration only.
**Why It Succeeded/Failed:** Engineered features + lexicon + sentiment-hint fine-tuning.
**Decision:** **Current best (any EESA sentiment model);** external reference.
**Influence on Later Work:** Became the **frozen strong primary** for the entire
agent-decomposition ladder (Designs A–G, v3, sequential, GPT-4.1-mini).

---

## EXPERIMENT_AHMED_FROZEN_PRIMARY_FULL_AGENTIC
**Experiment Name:** Ahmed frozen-primary + full_agentic (default prompts)
**Chronological Order:** 14 (2026-06-27)
**Research Question:** Does the agentic layer add value on top of a very strong (0.9254)
primary?
**Motivation:** Test the strong-primary end of the curve with a real frozen primary.
**Hypothesis:** Little/no gain; possibly slight harm.
**Baseline:** Ahmed primary_only (0.9254).
**Architecture:** full_agentic (default lexical+logic+contextual trio), Fix-2 consensus on.
**Implementation Changes:** `PrecomputedPrimaryClassifier` (no TF/Keras in pipeline).
**Dataset:** EESA test (818).
**Model:** frozen Ahmed primary; gpt-4o-mini.
**Parameters:** **threshold 0.7** (0.9 invalid — Ahmed's confidence max 0.864 → 100%
escalation; thresholds must be **calibrated per primary**).
**Evaluation Metrics:** primary 0.9254/0.9207 → **full_agentic 0.9205/0.9153 (Δ −0.0049)**.
Escalated 84 (10.3%), **W→C 11 / C→W 15 / net −4**, cost ~$0.043.
**Main Findings:** The agentic layer **slightly hurts** a very strong primary — the agents
break more correct predictions than they fix. **Router thresholds must be calibrated per
primary** (probability scales differ; XLM-R peaks ~1.0, Ahmed ~0.86).
**Strengths:** Cleanly establishes the strong-primary null/slight-harm point.
**Weaknesses:** Default (non-refined) prompts; per-agent labels captured only for Ahmed.
**Why It Succeeded/Failed:** Primary-on-escalated (0.75) at/above the ~0.75 agent ceiling.
**Decision:** **Historical** (starting point that the redesign ladder tries to beat).
**Influence on Later Work:** Motivated the entire agent role/decomposition redesign and the
per-primary threshold rule.

---

## EXPERIMENT_AGENT_BEHAVIOR_COMPARISON
**Experiment Name:** Agent behaviour across experiments — the primary-strength curve
**Chronological Order:** 15 (2026-06-27)
**Research Question:** When does the agentic layer help vs hurt, across all three
full_agentic sentiment experiments?
**Motivation:** Unify C3 / EESA / Ahmed into one rule.
**Hypothesis:** Direction is set by primary strength relative to a fixed agent ceiling.
**Baseline:** primary_only for each.
**Architecture:** Analysis only (saved predictions); per-agent labels for Ahmed only.
**Implementation Changes:** None.
**Dataset:** EESA test (three primaries).
**Model:** three primaries; gpt-4o-mini agents.
**Parameters:** thresholds per primary (C3/EESA 0.9, Ahmed 0.7).
**Evaluation Metrics (escalated subset):**
| experiment | esc | prim-on-esc | final-on-esc | W→C | C→W | net | full Δ |
|---|---|---|---|---|---|---|---|
| C3-960 | 231 | 0.5411 | 0.7489 | 71 | 23 | +48 | +0.059 |
| EESA XLM-R | 190 | 0.5579 | 0.6737 | 42 | 20 | +22 | +0.027 |
| Ahmed | 84 | 0.7500 | 0.7024 | 11 | 15 | −4 | −0.005 |

Ahmed agent diversity: all-3-agree 92%, pairwise 93–96%; per-agent 0.68–0.73 (all below
the 0.75 primary).
**Main Findings:** Final consensus accuracy on hard cases is **roughly constant ~0.67–0.75
(the "agent ceiling")**; the **primary's** accuracy on the escalated subset is what varies
→ agents help below the ceiling, hurt above it. The three agents behave as **one correlated
bloc** — amplifying the rescue (weak primary) or the damage (strong primary).
**Strengths:** The single evidence-based rule for the whole project.
**Weaknesses:** A/B/D (diversity/per-agent/correlation) measured for Ahmed only; C3/EESA
inferred.
**Why It Succeeded/Failed:** Relative accuracy, not correlation, sets the sign.
**Decision:** **Accepted** — foundational cross-experiment rule.
**Influence on Later Work:** Frames every later design as "improve the rescue where the
primary is weak; reduce harm where it is strong."

---

## EXPERIMENT_SENTIMENT_SEMANTIC_V1 (Design A)
**Experiment Name:** Role-refined `semantic_v1` sentiment prompts
**Chronological Order:** 16 (2026-06-30)
**Research Question:** Does refining the three specialists' prompts (distinct reasoning
modes) decorrelate the panel and reduce surface-cue literalism?
**Motivation:** Linguistic error analysis found surface-cue over-reading, target confusion,
description-vs-evaluation, and missed implicit sentiment; the trio reasoned alike (92%).
**Hypothesis:** Distinct roles → less agreement, fewer literalism breaks, smaller agentic
gap.
**Baseline:** Ahmed full_agentic default prompts (0.9205, net −4).
**Architecture:** Lexical (evidence + mention-vs-express) + Logic (target attribution +
negation) + Contextual (pragmatics/sarcasm/holistic). Generic, dataset-agnostic.
**Implementation Changes:** `--sentiment_prompt_variant semantic_v1`;
`EXPERIMENT_SENTIMENT_PROMPT_SEMANTIC_V1_CHANGELOG.md`.
**Dataset:** EESA test (818), Ahmed frozen primary.
**Model:** frozen Ahmed; gpt-4o-mini (temp 0).
**Parameters:** threshold 0.7, w_primary 1.0, signal off.
**Evaluation Metrics:** **acc 0.9230 / macroF1 0.9183 / wF1 0.9228 (net −2)**; escalated-only
0.7262; W→C 12 / C→W 14; neutral→negative breaks 7→5. Agreement 91.7%→**84.5%** (pairwise
down); logic 0.679→0.690, contextual 0.726→0.750 (best); cost $0.0498/336 calls.
**Main Findings:** Decorrelation achieved (agreement −7.2 pts); **half the agentic gap
closed (−4→−2)**; the two recovered breaks were exactly the targeted literalism patterns
(plot-description, third-party "dislike"). But still **below primary_only** — the remaining
deficit is the **agent-ceiling**, not prompt wording.
**Strengths:** Mechanistic improvement (decorrelation + targeted fixes).
**Weaknesses:** Longer prompt (+cost); still net-negative; 1 over-conservatism regression.
**Why It Succeeded/Failed:** Reduces literalism but cannot beat a parity-strength primary.
**Decision:** **Superseded** (kept as the base prompt variant carried into C–G).
**Influence on Later Work:** Baseline prompts for the whole decomposition ladder; motivated
replacing the weak Logic agent (Polarity redesign).

---

## EXPERIMENT_SENTIMENT_POLARITY_REDESIGN_AND_DESIGN_ABLATION (A/B/C/D)
**Experiment Name:** Specialist-decomposition ablation — Polarity agent (A/B/C/D)
**Chronological Order:** 17 (2026-06-30 → 07-01)
**Research Question:** Is Lexical+Logic+Contextual the right split, or does a dedicated
**Polarity** agent (replacing weak Logic) do better?
**Motivation:** Logic measured weakest (0.690) and most redundant (~0.89 with Lexical).
**Hypothesis:** Replacing Logic with a disciplined Polarity decider (Design C) helps most
with least regression risk; adding it as a 4th (D) over-powers the bloc.
**Baseline:** Ahmed primary_only 0.9254; semantic_v1 (A) 0.9230.
**Architecture:** A = Lex+Logic+Ctx; **B = Pol+Ctx**; **C = Lex+Pol+Ctx** (Logic→Polarity,
consensus unchanged); **D = Lex+Logic+Ctx+Pol** (4 votes).
**Implementation Changes:** `polarity_agent.py`, `_sentiment_variant.py`;
`--sentiment_agent_variant lexical_polarity_contextual`;
`EXPERIMENT_SENTIMENT_POLARITY_AGENT_CHANGELOG.md`.
**Dataset:** EESA test (818), Ahmed frozen primary.
**Model:** frozen Ahmed; gpt-4o-mini (temp 0); semantic_v1 prompts.
**Parameters:** threshold 0.7, w_primary 1.0.
**Evaluation Metrics:**
| design | acc | macroF1 | esc-acc | W→C | C→W | net | breaks | cost |
|---|---|---|---|---|---|---|---|---|
| A Lex+Logic+Ctx | 0.9230 | 0.9183 | 0.726 | 12 | 14 | −2 | 14 | $0.050 |
| B Pol+Ctx | 0.9254 | 0.9212 | 0.750 | 10 | 10 | 0 | 10 | $0.035 |
| **C Lex+Pol+Ctx** | **0.9267** | **0.9226** | **0.762** | 12 | 11 | **+1** | 11 | $0.049 |
| D 4-agent | 0.9254 | 0.9211 | 0.750 | 11 | 11 | 0 | 11 | $0.064 |

Polarity 0.738 (>> Logic 0.690, biggest per-agent jump +0.048); Lex↔Pol 0.833 (lowest
redundancy in the study); all-3-agree 81.0% (C).
**Main Findings:** **C is the first agentic config to exceed Ahmed primary_only** (0.9267 >
0.9254, net +1); B is the "do no harm" config (net 0, cheapest, but no rescues); D is
dominated (= B's net at highest cost). Ranking **C > B > D > A**.
**Strengths:** Shows the bottleneck was the **specialist set**, not only prompt wording;
C needs no consensus change.
**Weaknesses:** Gain is +1 sample (+0.0013), within ±1–2 temp-0 noise; C3 regime untested
at this point.
**Why It Succeeded/Failed:** Polarity is a better, more independent sentiment specialist.
**Decision:** **C = lead variant** (opt-in, not yet default); **A worst; D retired.**
**Influence on Later Work:** C becomes the base trio for Designs E/F/G and v3.

---

## EXPERIMENT_SENTIMENT_INTENT_AGENT_ABLATION (Design E)
**Experiment Name:** Design E — Intent as a 4th voting agent (Lex+Intent+Pol+Ctx)
**Chronological Order:** 18 (2026-07-01)
**Research Question:** Does adding a narrow Intent/stance agent improve Design C?
**Motivation:** Intent could catch artifact/spotting false-positives.
**Hypothesis:** Intent adds diversity; may reduce harmful overrides.
**Baseline:** Design C (0.9267, +1).
**Architecture:** 4 votes (Intent in the polarity/4th slot; existing 4-slot consensus).
**Implementation Changes:** `intent_agent.py`, `intent_prompt.py`.
**Dataset/Model/Parameters:** Ahmed frozen, gpt-4o-mini, threshold 0.7, w_p 1.0.
**Evaluation Metrics:** **acc 0.9267 / macroF1 0.9227 / net +1**; W→C 11 / C→W 10 (one fewer
harmful override than C); breaks 10 (fewest, tied B); esc-acc 0.762; Intent weakest agent
(0.714) but **most decorrelated** (Intent↔Lexical 0.679); all-4-agree 66.7%; **cost $0.064
(+30% over C)**.
**Main Findings:** E **ties C** on accuracy/net; Intent adds **diversity, not accuracy** —
it kills artifact/spotting positives (good) but flattens implicit excitement/reactions
(bad) → a wash.
**Strengths:** Legitimate diversity-adding specialist; fewest breaks.
**Weaknesses:** +30% cost for no robust gain; double-edged neutral lean.
**Why It Succeeded/Failed:** Neutral-leaning vote washes on a strong primary.
**Decision:** **Keep C as lead; E opt-in secondary (not promoted).**
**Influence on Later Work:** Showed Intent is better cast as a **gate** than a vote →
Designs F/G.

---

## EXPERIMENT_SENTIMENT_DESIGN_F (Intent+Polarity+Contextual, remove Lexical)
**Experiment Name:** Design F — remove Lexical (Intent+Pol+Ctx)
**Chronological Order:** 19 (2026-07-01)
**Research Question:** Does Lexical contribute genuine evidence, or mainly surface-cue bias?
**Motivation:** Test whether the explicit-evidence agent can be dropped.
**Hypothesis:** Dropping Lexical may reduce literalism.
**Baseline:** Design C (+1).
**Architecture:** Intent + Polarity + Contextual (Lexical abstains), 3 active votes.
**Implementation Changes:** `--sentiment_agent_variant intent_polarity_contextual`.
**Dataset/Model/Parameters:** Ahmed frozen, gpt-4o-mini, threshold 0.7.
**Evaluation Metrics:** **acc 0.9218 / macroF1 0.9180 / net −3** (worst in the study, below
A); esc-acc 0.7143; C→W 14; **positive→neutral 2→5, negative→neutral 3→4** (over-
neutralization); Contextual still strongest (0.750); cost $0.049.
**Main Findings:** **Lexical is load-bearing** — removing it causes over-neutralization;
without an explicit-evidence anchor the three neutral-leaning deciders lose polar cases.
Confirms **C = Lexical + Polarity + Contextual** is the right decomposition. Maps to
Ahmed's own design (lexicon/hints always help; his rule-based polarity + intent-as-branch).
**Strengths:** Decisive negative that pins down the necessary agent set.
**Weaknesses:** N/A (a deliberate ablation).
**Why It Succeeded/Failed:** No evidence anchor → collapse to neutral.
**Decision:** **Rejected** (retired as a production candidate).
**Influence on Later Work:** Confirms C's composition; supports Intent-as-gate (G).

---

## EXPERIMENT_SENTIMENT_INTENT_GATE (Design G)
**Experiment Name:** Design G — IntentGate (Lex+Pol+Ctx + non-voting gate)
**Chronological Order:** 20 (2026-07-01)
**Research Question:** Does using Intent as a **non-voting veto** (not a vote) beat Design C?
**Motivation:** Design E showed the same pragmatic signal is 12/12 suppressed as a vote but
0 missed as a veto; a domain-restricted, non-overriding veto is the one aggregation lever
that ever helped.
**Hypothesis:** The gate fixes the persistent "unlike/dislike" meta-comment cluster without
over-neutralizing.
**Baseline:** Design C (0.9267, +1).
**Architecture:** C trio + IntentGate (4th agent, **consensus weight 0**); guard: if agents
overrode a **neutral** primary but the gate sides with the primary (no expressed opinion),
**block** the override. Never forces a flip.
**Implementation Changes:** `consensus_agent.py` (`intent_gate`);
`--sentiment_agent_variant lexical_polarity_contextual_intent_gate`.
**Dataset/Model/Parameters:** Ahmed frozen, gpt-4o-mini, threshold 0.7, w_p 1.0.
**Evaluation Metrics:** **acc 0.9279 / macroF1 0.9242 / wF1 0.9279 (net +2, best in study)**;
esc-acc 0.7738; **W→C 10 / C→W 8** (lowest harmful overrides); breaks 8 (fewest);
neutral→negative 4→2. Gate: neutral on 44/84, **6 interventions (4 helped, 2 hurt)**; gate
guard precision 4/6 = 67%. Cost $0.0636 / 420 calls.
**Main Findings:** **Best design** — fixes the meta-comment cluster ("who disliked" =
question about others = neutral) that **no prior design solved**, without F's
over-neutralization. Realizes Ahmed's annotation rule / rule-based inference gate as a guard.
**Strengths:** Reduces harmful overrides while keeping most rescues; principled mechanism.
**Weaknesses:** +2/818 = +0.0025 (noise-adjacent, ±1-sample); can hurt when the primary is
**wrong-neutral** on genuinely evaluative text (2 cases).
**Why It Succeeded/Failed:** A veto cannot be outvoted by the correlated bloc.
**Decision:** **Adopt G as the lead sentiment configuration** (pending the C3 check).
**Influence on Later Work:** The lead everything else is measured against; the gate becomes
the object of the GPT-4.1-mini gate ablation.

---

## EXPERIMENT_SENTIMENT_SELECTIVE_INTENT_GATE (Design G2)
**Experiment Name:** Design G2 — selective IntentGate (platform/meta only)
**Chronological Order:** 21 (2026-07-01)
**Research Question:** Can a selective gate recover G's 2 hurt cases without losing its 4
useful platform blocks (and reach 0.930)?
**Motivation:** G's 2 own-goals were an implicit insult and a fan cheer.
**Hypothesis:** Gate only on genuine platform/meta → fewer hurts, same helps.
**Baseline:** Design G (0.9279, +2).
**Architecture:** G with a prompt-only refinement (`SYSTEM_PROMPT_SELECTIVE`): return
neutral only for genuine platform/meta/mention; return polar for expressed implicit stance.
**Implementation Changes:** `intent_prompt.py` selective prompt;
`--sentiment_agent_variant lexical_polarity_contextual_selective_gate`.
**Dataset/Model/Parameters:** Ahmed frozen, gpt-4o-mini, threshold 0.7.
**Evaluation Metrics:** **acc 0.9279 / net +2 — ties G exactly** (same 59/818 wrong, esc
0.7738; macroF1 0.9245 vs 0.9242). Gate fires 23/84 (vs 44), interventions 5 (helped 4,
hurt 1). Recovered 1 of 2 hurt cases (`00021` insult) but **lost 3 of 4 platform blocks**
(`00203/00245` indignant unlike-questions; `00320` emoji spotting). Cost $0.0641.
**Main Findings:** Selective gate is **more precise but nets zero** — recovering implicit
insults and protecting affective platform-questions are in **direct tension** (both are
"affective questions", labelled neutral by convention). Empirically confirms the gap
analysis: 0.930 is at the noise ceiling with gpt-4o-mini.
**Strengths:** Clean demonstration of the tension; more precise firing.
**Weaknesses:** No net gain; lost platform protections; more complex prompt.
**Why It Succeeded/Failed:** The boundary is genuinely ambiguous for a single-prompt LLM.
**Decision:** **Retire G2 (at gpt-4o-mini); keep G** (simpler, safer). *(Note: G2 is later
**revived and wins** under the stronger gpt-4.1-mini — see the gate ablation.)*
**Influence on Later Work:** Motivated the gap analysis and the "gate aggressiveness must
scale inversely with model strength" finding.

---

## EXPERIMENT_SENTIMENT_PRAGMATIC_CONTEXTUAL_V3
**Experiment Name:** Design v3 — Pragmatic-Reasoner Contextual (in-place upgrade)
**Chronological Order:** 22 (2026-07-01)
**Research Question:** Does upgrading only the Contextual agent to an explicit pragmatic
reasoner (speech act, mention-vs-use, implicature, description-vs-evaluation) beat G?
**Motivation:** Contextual is the strongest, most independent agent; sharpen it.
**Hypothesis:** A better Contextual improves the final.
**Baseline:** Design G (0.9279, +2).
**Architecture:** G's agent set with the Contextual prompt swapped
(`semantic_v3_pragmatic_contextual`); one vote, schema unchanged.
**Implementation Changes:** `contextual_prompt.py` (`SYSTEM_PROMPT_PRAGMATIC`).
**Dataset/Model/Parameters:** Ahmed frozen, gpt-4o-mini, threshold 0.7.
**Evaluation Metrics:** **acc 0.9279 / net +2 — ties G exactly** (59/818 wrong).
**Contextual agent improved 0.7381 → 0.7619** (now strongest, above the primary), but the
final did not move: neutral→negative 2→1 gained, **negative→neutral 3→4 lost** (one-for-one
trade). Gate interventions 7 (hurt 3). Cost $0.0668.
**Main Findings:** The pragmatic reasoner **worked as an agent upgrade but not as a system
upgrade** — a better single vote is diluted/outvoted on a near-ceiling strong primary and
offset by one new over-neutralization (`00113` "wtf"). Consistent with the gap analysis and
"conservation of difficulty."
**Strengths:** Genuinely better Contextual agent (may matter more on a weak primary).
**Weaknesses:** No system gain; longer prompt; one F-like regression.
**Why It Succeeded/Failed:** Component gains don't propagate at the strong-primary ceiling.
**Decision:** **Keep G as lead; v3 opt-in** (better Contextual, revisit on C3).
**Influence on Later Work:** Reinforced that the decisive lever is the **weak primary**, not
more strong-primary agent work.

---

## EXPERIMENT_CONSENSUS_INVESTIGATION (loss / aggregation / rescoring / weight-sweep / specialization)
**Experiment Name:** Consensus/aggregation investigation on the Ahmed escalated subset
**Chronological Order:** 23 (2026-07-01)
**Research Question:** Is consensus a bottleneck, and can any *simple* aggregation rule beat
Design G?
**Motivation:** The panel often holds the correct answer in one agent that the vote discards.
**Hypothesis:** A better fusion rule (neutral guards, w_primary retune, minority-trust,
role-priority) recovers the suppressed-correct answers.
**Baseline:** Design G capture (net +3 on the capture draw; ~0.929).
**Architecture:** Offline simulation on saved G labels (rules fixed before scoring; true
labels never used to fire).
**Implementation Changes:** None (analysis; label-only captures).
**Dataset/Model:** Ahmed escalated subset (~84); labels from each design's attribution table.
**Parameters:** w_primary ∈ {1.0, 1.5, 2.0, 3.0}; various guard/override rules.
**Evaluation Metrics:**
- **Suppressed-correct** (agent right, final wrong): Lexical 25, Contextual 20, Intent-as-
  voter 12 (100% lone), Polarity 7, Logic 3 — concentrated on decorrelated agents.
- **Oracle-any-voter** upper bound per design: A 0.798, C 0.821, **E 0.881**, F 0.845,
  **G 0.833 (lowest recoverable loss = 4)**, G2 0.833, v3 0.833 — panel holds **4–11 more
  correct answers/84** than the vote emits; ~⅔ of errors are **unrecoverable** (info floor).
- **Lone-dissent precision:** Contextual 0.58, Intent 0.46, Lexical 0.37, Polarity 0.33
  (only Contextual beats a coin flip).
- **Offline rules vs G:** Contextual neutral guard **tie (0 fires)**; Lexical cue protection
  **−1**; combined **−1**; w_primary 1–2 **−1**, w=3 **= primary_only (net 0)**; minority-
  trust Ctx **−2**, Lex **−2**, both **−4**; role-priority ≡ minority-trust (−2).
- Agent specialization: agents emit identical labels on **81% escalated / ~96% overall**.
**Main Findings:** **Consensus is a genuine bottleneck, but simple aggregation is exhaustively
ruled out** — everything ties or loses. The **only** aggregation change that ever helped is
the **domain-restricted, non-overriding veto (IntentGate)**, already in G. Next safe direction
needs **new information** (calibrated per-agent confidence — not currently stored — or
genuinely decorrelated/heterogeneous models / dev-trained selector), all risky given ~80
samples and test-only captures.
**Strengths:** Closes off a large family of tempting fixes cheaply/offline.
**Weaknesses:** Confidence-aware/learned methods **cannot even be simulated** (labels-only
captures).
**Why It Succeeded/Failed:** A vote can be outvoted by a correlated bloc; a veto cannot.
**Decision:** **Rejected** (all simple re-fusion rules); **IntentGate retained**; learned/
calibrated/heterogeneous methods = open.
**Influence on Later Work:** Motivated the sequential architecture (route by intermediate
property to remove consensus loss) and the "capture confidences next time" data-hygiene rule.

---

## EXPERIMENT_SEQUENTIAL_SENTIMENT_V1 (Ahmed)
**Experiment Name:** Sequential v1 — staged reasoning (Intent → Polarity → Pragmatic →
deterministic controller)
**Chronological Order:** 24 (2026-07-01)
**Research Question:** Does staged (conditional) reasoning beat parallel voting-plus-veto?
**Motivation:** Parallel agents are correlated generalists (81% identical); a conjunctive
pipeline (like SwitchLingua's quality agents / Ahmed's rule-based sequence) may decorrelate
the *combined* output and remove consensus loss.
**Hypothesis:** Staged reasoning ≥ parallel where headroom exists.
**Baseline:** Ahmed primary_only (0.9254); Design G (+2).
**Architecture:** Stage 1 Intent/opinion-existence → Stage 2 Polarity → Stage 3 Pragmatic
verifier → Stage 4 deterministic controller (gate + resolver-trust made explicit).
**Implementation Changes:** `--sentiment_agent_variant sequential_sentiment_v1`;
`orchestrator.py`; `EXPERIMENT_SEQUENTIAL_SENTIMENT_V1_IMPLEMENTATION_CHANGELOG.md`.
**Dataset/Model/Parameters:** Ahmed frozen, gpt-4o-mini (temp 0), threshold 0.7.
**Evaluation Metrics:** **acc 0.9242 / macroF1 0.9195 / net −1** (756/818 vs primary 757);
esc-acc 0.7381; **W→C 13 / C→W 14** (27/84 interventions — very active). Faithful re-capture:
0 coercions/retries; `decided_by` = polarity_kept 58, intent_no_opinion 23, pragmatic_revision
**3**, fallback 0 → **net 0, esc 0.7500 (= primary-on-escalated)**. Cost $0.051 / 336 calls.
**Main Findings:** The staged pipeline **reproduces the primary's escalated accuracy** (net 0
to −1) — a clean, informative negative. It restructures *how* the decision is reached but the
"pragmatic/sarcasm" stage barely fires (3/84); the no-opinion gate is the only active lever.
Same ceiling, reached a different way.
**Strengths:** Confirms the ceiling from a genuinely different, more-active architecture.
**Weaknesses:** Inert Stage 3; ~33% headline retry rate (benign format jitter).
**Why It Succeeded/Failed:** Strong-primary escalated subset is at the info floor.
**Decision:** **Rejected on the strong primary** (does not beat G/primary); informative
negative.
**Influence on Later Work:** Set up v2 (forward-pragmatics) to test whether *un-anchoring* the
pragmatic stage helps.

---

## EXPERIMENT_SEQUENTIAL_SENTIMENT_V2 (Ahmed)
**Experiment Name:** Sequential v2 — forward-pragmatics (Intent → Pragmatic FEATURES →
feature-aware Polarity → controller)
**Chronological Order:** 25 (2026-07-01)
**Research Question:** Does letting pragmatic features **drive** (not just review) the label
help?
**Motivation:** v1's anchoring made Stage 3 inert; v2 removes the confirmation anchor.
**Hypothesis:** Forward pragmatics adds signal (design's stated risk: cascade over-calling).
**Baseline:** primary_only (0.9254); v1 (−1); Design G (+2).
**Architecture:** Pragmatic features (`implicit_stance`, `sarcasm_or_irony`) feed a
feature-aware Polarity resolver.
**Implementation Changes:** `--sentiment_agent_variant sequential_sentiment_v2`;
`EXPERIMENT_SEQUENTIAL_SENTIMENT_V2_FORWARD_PRAGMATICS_DESIGN.md`.
**Dataset/Model/Parameters:** Ahmed frozen, gpt-4o-mini, threshold 0.7.
**Evaluation Metrics:** **acc 0.9120 / macroF1 0.9076 / net −11** (746/818); esc-acc collapses
to **0.6190** (below the agent ceiling); **W→C 8 / C→W 19** (31/84 changed); 10 of 19 breaks
are correct-neutral → wrong-polar (over-calling); cost $0.045.
**Main Findings:** **v2 is substantially worse** — removing the anchor unleashed noisy
pragmatic features to over-call polarity on a subset with no recoverable signal. v1 (inert,
≈0) and v2 (active, −11) **bracket** the finding: on the strong primary, net effect **scales
with how much the layer intervenes**. Strong-primary ceiling now confirmed from **five**
directions.
**Strengths:** Clean, large-margin negative; mechanistically confirms the cascade risk.
**Weaknesses:** Uninformative about the design's actual hypothesis (which needs a weak primary).
**Why It Succeeded/Failed:** Aggressive intervention amplifies harm where the primary is right.
**Decision:** **Rejected on the strong primary**; hypothesis deferred to C3.
**Influence on Later Work:** Made the C3 run of v2 the decisive test of forward-pragmatics.

---

## EXPERIMENT_G_C3_RESULTS
**Experiment Name:** Design G on the weak C3 generated primary — the decisive check
**Chronological Order:** 26 (2026-07-02)
**Research Question:** Do the parallel improvements (Polarity, IntentGate, semantic_v1)
transfer to the regime where the agents matter, and does the neutral-protecting gate erode
the weak-primary rescue?
**Motivation:** Every strong-primary design deferred to this test.
**Hypothesis:** G helps a lot on C3 (per the root-cause equation, predicted +0.059).
**Baseline:** C3 primary_only (0.6956); original C3 full_agentic (0.7543).
**Architecture:** Design G (Lex+Pol+Ctx + IntentGate), semantic_v1.
**Implementation Changes:** `--primary_model precomputed`? No — real XLM-R seed-456 on GPU.
**Dataset/Model/Parameters:** EESA test (818); C3 `sz960_seed456`; gpt-4o-mini; **threshold
0.90**, w_p 1.0.
**Evaluation Metrics:** primary 0.6956/0.6830 → **G 0.7604/0.7469** (**+0.065 acc / +0.064
macroF1**; 622/818 vs 569). vs original full_agentic +0.006/+0.008. Escalated 231, esc-acc
**0.5411 → 0.7706**, **W→C 65 / C→W 12 / net +53**; McNemar χ²≈36.5, **p ≪ 0.001**. Cost
$0.176 / 1155 calls.
**Main Findings:** **The agents clearly help where the primary is weak** — the mirror image of
Ahmed, exactly as predicted. The gate does **not** erode the rescue. **Design G is the parallel
ceiling on C3 (0.7604 / esc 0.7706)** and the anchor the sequential runs must beat.
**Strengths:** Live, highly significant confirmation of the whole ceiling thesis.
**Weaknesses:** Single temp-0 draw; seed-456 primary slightly high; gate/decided-by counts not
serialized.
**Why It Succeeded/Failed:** Primary-on-escalated (0.54) far below the ~0.77 agent ceiling.
**Decision:** **Accepted — Current best (weak-primary, gpt-4o-mini).**
**Influence on Later Work:** Establishes the target the sequential-v2 C3 run and any 4.1-mini
C3 run must exceed.

---

## EXPERIMENT_SEQV2_C3_RESULTS
**Experiment Name:** Sequential v2 on the weak C3 primary — parallel-vs-sequential verdict
**Chronological Order:** 27 (2026-07-02)
**Research Question:** Does forward-pragmatics (v2) beat parallel G where the agents carry the
decision?
**Motivation:** Completes the parallel-vs-sequential comparison on the regime that matters.
**Hypothesis:** v2 helps on C3 (mirror of its Ahmed harm); does it beat G?
**Baseline:** C3 primary_only (0.6956); Design G on C3 (0.7604, +53).
**Architecture:** Sequential v2 (forward-pragmatics).
**Implementation Changes:** None new.
**Dataset/Model/Parameters:** EESA test; C3 seed-456 (GPU); gpt-4o-mini; threshold 0.90.
**Evaluation Metrics:** **v2 0.7531 / 0.7391, net +47** (esc 172/231 = 0.745; **W→C 75 / C→W
28**, 103 changed); vs **G 0.7604 / 0.7469, net +53** (esc 178/231 = 0.771; W→C 65 / C→W 12,
77 changed). Cost v2 $0.125 vs G $0.176.
**Main Findings:** **v2 helps on the weak primary (+47) — exact mirror of its Ahmed −11**,
confirming the ceiling thesis and v2's forward-pragmatics hypothesis. **But parallel G still
wins** (0.7604 vs 0.7531): v2 is more active/higher-recall (75 vs 65 rescues) but pays an
**aggressiveness tax** (breaks 28 vs G's 12). G's advantage is the **IntentGate veto's damage
control**, not raw reasoning. Staged reasoning did **not** beat parallel-voting-plus-veto in
either regime.
**Strengths:** Settles the whole parallel-vs-sequential question across both regimes.
**Weaknesses:** 6-sample final gap within temp-0 noise; decided-by not serialized.
**Why It Succeeded/Failed:** No veto brake on v2's aggressiveness.
**Decision:** **G remains best;** v2 = viable, cheaper, but strictly less efficient
(candidate "v2.1 with a veto" flagged).
**Influence on Later Work:** Locks Design G as the lead architecture; shifts remaining levers to
**model quality** and **primary quality**.

---

## EXPERIMENT_G_STRONGER_MODEL_DIAGNOSTIC
**Experiment Name:** Stronger-model diagnostic — 18 G-Ahmed failures at gpt-4.1-mini
**Chronological Order:** 28 (2026-07-02)
**Research Question:** Does a stronger base model fix the residual strong-primary failures, or
is the ceiling purely informational?
**Motivation:** Isolate model capability from noise on the exact failing cases.
**Hypothesis:** A stronger model recovers some compliance/obscured-cue failures.
**Baseline:** gpt-4o-mini control re-run on the same 18.
**Architecture:** Full G pipeline, only the 18 escalated failures re-decided.
**Implementation Changes:** `--llm_model gpt-4.1-mini`; `scripts/ahmed_G_failure_diagnostic.py`.
**Dataset/Model/Parameters:** Ahmed frozen; 18 failing escalated cases; temp 0.
**Evaluation Metrics:** gpt-4o-mini control **0/18** (deterministic → noise floor 0);
**gpt-4.1-mini 4/18** (all real, none noise). Fixed: `00240`/`00298` platform-meta "dislike";
`00542` misspelled praise; `00642` implicit "bravo". 14 remain (deep cultural implicit insults:
`00008` slur-question, `00182` "you are no one", …). Cost ~$0.03.
**Main Findings:** **First evidence the strong-primary ceiling is partly a MODEL ceiling and
moves cheaply** — a $0.03 swap recovers the compliance/obscured-cue slice (not the deepest
cultural slice). You cannot prompt past it (v1/v2/semantic_v1 failed) but a stronger model
recovers some.
**Strengths:** Cleanly separates recoverable (compliance) vs floor (cultural) errors.
**Weaknesses:** Only shows fixes on *failing* cases; net requires a full run.
**Why It Succeeded/Failed:** Better model applies existing rules more faithfully.
**Decision:** **Accepted diagnostic** → motivates the full G@4.1-mini run.
**Influence on Later Work:** Directly triggered the full G@4.1-mini run and the whole 4.1-mini
line.

---

## EXPERIMENT_G_AHMED_GPT41MINI_RESULTS
**Experiment Name:** Full Design G on Ahmed at gpt-4.1-mini
**Chronological Order:** 29 (2026-07-02)
**Research Question:** Does the stronger model's net effect (after re-deciding all 84 escalated)
beat G@4o-mini and reach 0.930?
**Motivation:** The diagnostic showed 4/18 fixed but a full run also re-rolls the 66 correct.
**Hypothesis:** Small net gain, partly offset by new errors.
**Baseline:** G@4o-mini (0.9279, +2); primary_only (0.9254).
**Architecture:** Design G, gpt-4.1-mini, all 84 escalated.
**Implementation Changes:** `--llm_model gpt-4.1-mini`.
**Dataset/Model/Parameters:** Ahmed frozen, threshold 0.7, w_p 1.0, semantic_v1.
**Evaluation Metrics:** **acc 0.9291 / macroF1 0.9248 / net +3** (760/818, esc 66/84 = 0.786;
W→C 10 / C→W 7). Best full-set Ahmed number, **+1 sample over G@4o-mini**. McNemar vs
primary_only χ²≈0.53, **p ≈ 0.47 — NOT significant**; **does not reach 0.930**. Cost ~$0.13.
**Main Findings:** A stronger model gives a **genuine but tiny bump** (0.9279 → 0.9291,
non-significant); the 4/18 diagnostic became only +1 net because it also **broke ~3 of the 66
previously-correct** cases. Model quality nudges the ceiling; topology doesn't; the weak primary
is where gains live.
**Strengths:** Best strong-primary point estimate so far; honest significance test.
**Weaknesses:** Not significant; not 0.930; cost table lacks 4.1-mini pricing.
**Why It Succeeded/Failed:** Stronger model re-rolls every decision (recall gains partly cancel).
**Decision:** **Accepted (best-yet, non-significant);** next lever is the gate + the weak primary.
**Influence on Later Work:** Triggered the "why it broke" per-agent trace and the gate ablation.

---

## EXPERIMENT_WHY_STRONGER_MODEL_BROKE_CASES
**Experiment Name:** Per-agent diagnosis of the 5 cases 4.1-mini broke that 4o-mini got right
**Chronological Order:** 30 (2026-07-02)
**Research Question:** What mechanism makes the stronger model break previously-correct cases?
**Motivation:** G@4.1-mini was a 5-fixed/5-broken wash on two axes.
**Hypothesis:** The gate and neutral-lean prompts interact badly with a more compliant model.
**Baseline:** G@4o-mini per-agent trace.
**Architecture:** Analysis + ~$0.01 per-agent re-capture.
**Implementation Changes:** `scripts/ahmed_G_broken_why.py`.
**Dataset/Model:** 5 broken escalated cases, both models.
**Evaluation Metrics:** Three mechanisms: **M1 — stronger GATE over-vetoes "dislike" as meta**
(00045, 00100: all 3 voters correctly negative, gate blocked → neutral); **M2 — stronger AGENTS
over-neutralize implicit negatives** (00041, 00127, following the neutral-lean guidance more
faithfully); **M3 — stronger AGENTS over-read described content** (00362 Breaking Bad plot →
negative).
**Main Findings:** The stronger model **didn't lose capability — it shifted its default bias
toward neutral and toward meta-vetoing**; 4o-mini's wins on these ambiguous polar-gold cases
were **non-compliance that happened to match the gold** (right by accident). The one actionable
lever is the **gate**: it becomes a **net liability under a stronger model** (one-directional
veto). Predicts G may do better **without** the gate or with the **selective** gate (G2).
**Strengths:** Turns a confusing wash into a testable prediction.
**Weaknesses:** 5 cases; per-agent capture.
**Why It Succeeded/Failed:** Gate tuned for a weaker model over-fires on a stronger one.
**Decision:** **Accepted diagnosis** → motivates the 4.1-mini gate ablation.
**Influence on Later Work:** Directly yields the gate-ablation experiment and the "gate
aggressiveness ∝ 1/model-strength" law.

---

## EXPERIMENT_G41_WASH_DIAGNOSIS + SEMANTIC_V2_DISAMBIG (negative)
**Experiment Name:** Wash diagnosis → general disambiguation prompt (`semantic_v2_disambig`)
**Chronological Order:** 31 (2026-07-02)
**Research Question:** Can a general (non-dataset) prompt that replaces "platform word → neutral"
with a report/endorse/attack **relationship** rule fix the 4.1-mini wash?
**Motivation:** The wash is two ambiguous axes encoded as directional shortcuts.
**Hypothesis:** Disambiguating the axis converts the coin-flip into a decision.
**Baseline:** G@4o-mini (+2); G@4.1-mini semantic_v1 (+3).
**Architecture:** Design G, gpt-4.1-mini, disambiguation prompt in Lexical/Polarity/Intent
(+ description-vs-evaluation in Contextual).
**Implementation Changes:** `semantic_v2_disambig` in the prompt modules.
**Dataset/Model/Parameters:** Ahmed frozen, threshold 0.7.
**Evaluation Metrics:** **acc 0.9242 / macroF1 0.9196 / net −1** (esc 62/84 = 0.738). vs
G@4o-mini: **fixed 4, broken 8 → net −4 escalated** — below plain G and below primary_only. The
target ambiguous cases (00041/00045 endorse/attack, 00362 description) were **NOT fixed**; extra
instruction **broke more elsewhere** (over-instruction noise).
**Main Findings:** **Prompt disambiguation is NOT the lever** — the two axes are ambiguous because
the underlying pragmatic judgment is genuinely hard for the model, not because the prompt lacked
the distinction. **Fourth prompt/topology intervention to fail on the strong primary** (semantic_v1
≈0, v3 ≈0, sequential ≤0, disambig −1 to −4): **you cannot prompt-engineer past the ceiling.**
**Strengths:** Decisive refutation of the prompt-fix hypothesis.
**Weaknesses:** N/A (a clean negative).
**Why It Succeeded/Failed:** Naming a hard distinction doesn't grant the capability to make it;
more clauses hurt calibration.
**Decision:** **Rejected — revert to `semantic_v1`** (keep disambig opt-in as a documented
negative).
**Influence on Later Work:** Settles the ceiling story; points the only remaining strong-primary
lever at the **gate** (next).

---

## EXPERIMENT_GPT41_GATE_ABLATION (G / G2 / C at gpt-4.1-mini)
**Experiment Name:** GPT-4.1-mini gate ablation — full gate (G) vs selective gate (G2) vs no gate (C)
**Chronological Order:** 32 (2026-07-02)
**Research Question:** Under a stronger model, does the IntentGate over-veto — and which gate
setting is best?
**Motivation:** The "why-broke" trace predicted the full gate is a liability at 4.1-mini.
**Hypothesis:** Selective gate (G2) is the sweet spot; no gate loses neutral protection.
**Baseline:** primary_only (0.9254); G@4.1-mini full gate (0.9291, +3).
**Architecture:** Three configs at gpt-4.1-mini, semantic_v1.
**Implementation Changes:** None (variant flags only).
**Dataset/Model/Parameters:** Ahmed frozen, threshold 0.7, w_p 1.0.
**Evaluation Metrics:**
| config | acc | macroF1 | esc-acc | net (W→C/C→W) |
|---|---|---|---|---|
| primary_only | 0.9254 | 0.9207 | 0.750 | — |
| G full gate @4.1 | 0.9291 | 0.9248 | 0.786 | +3 |
| **G2 selective @4.1** ✅ | **0.9303** | **0.9262** | **0.798** | **+4 (12/8)** |
| C no gate @4.1 | 0.9266 | 0.9216 | 0.762 | +1 (13/12) |

**Main Findings:** **G2 @ gpt-4.1-mini = 0.9303 is the new best** and the **first to cross 0.930**
(best macro F1 too). Full gate over-vetoes; no gate recovers those but loses neutral protection
(7 neutral→polar breakages); **selective gate is the sweet spot** (neutral protection kept, meta-
veto restricted). New reusable law: **gate aggressiveness must scale INVERSELY with model
strength.** Significance: McNemar vs primary_only χ²≈0.8, **p ≈ 0.37 — still NOT significant**;
G2 vs G@4.1 = +1 sample.
**Strengths:** Best configuration on record; a genuine mechanism (gate × model-strength).
**Weaknesses:** Non-significant vs primary_only; single temp-0 draw.
**Why It Succeeded/Failed:** A stronger model's more aggressive meta-detection needs a lighter veto.
**Decision:** **Current best (strong primary) = G2 @ gpt-4.1-mini (0.9303 / 0.9262).** No-gate
discarded. **Stop Ahmed tuning** (all gains non-significant). Use G2@4.1-mini as the config when C3
is next run.
**Influence on Later Work:** Final strong-primary configuration; recommended next = G@/G2@4.1-mini
on **C3** (larger recoverable slice).

---

# PART II — MASTER TABLES

## Table 1 — Complete chronological timeline
| Order | Experiment | Purpose | Status |
|---|---|---|---|
| 1 | A — EESA mBERT | real-data reference primary | Historical |
| 2 | A — EESA XLM-R | stronger reference primary (0.8240) | Reference (best trained, this phase) |
| 3 | Real-LLM pilot @0.6 | do real LLM agents help? | Accepted |
| 4 | Real-LLM threshold sweep | does more escalation help? | Accepted |
| 5 | Prompt audit + 4 fixes | correctness (generic/abstain/primary-vote/signal) | Accepted (Fix1/1b/2 on, 3 off) |
| 6 | Consensus 2×2 @0.8/@0.9 | justify defaults → best 0.8509 | Accepted / locked |
| 7 | C1 — generated 240 | standalone transfer pilot | Superseded |
| 8 | C2/C3 + seed stability | generated scaling (retract 480>960) | Accepted; 480>960 retracted |
| 9 | C3-960 full_agentic (seed456) | weak-primary rescue (+0.059) | Current best (weak, 4o-mini) |
| 10 | E0 — EESA-only Adafactor | matched control (0.8533) | Current best (trained primary) |
| 11 | E3/LR/ratio/diagnosis | augmentation study | Rejected (augmentation) |
| 12 | T1/T2 — topic (9-class) | task transfer + near-perfect end | Accepted (primary_only) |
| 13 | Ahmed baseline | external ceiling (0.9254) | Current best (any EESA model) |
| 14 | Ahmed frozen + full_agentic | strong-primary agentic (net −4) | Historical |
| 15 | Agent behaviour comparison | the primary-strength curve | Accepted (foundational) |
| 16 | semantic_v1 (Design A) | prompt refinement (net −2) | Superseded |
| 17 | Polarity ablation A/B/C/D | decomposition (C = 0.9267, +1) | C lead; A/D retired |
| 18 | Intent voter (Design E) | 4th vote (ties C) | E opt-in |
| 19 | Design F (drop Lexical) | is Lexical needed? (net −3) | Rejected |
| 20 | IntentGate (Design G) | non-voting veto (0.9279, +2) | Lead (4o-mini) |
| 21 | Selective gate (G2) | recover gate hurts (ties G) | Retired@4o-mini (revived@4.1) |
| 22 | Pragmatic Contextual v3 | upgrade Contextual (ties G) | Opt-in |
| 23 | Consensus investigation | can simple fusion beat G? (no) | Rejected (simple fusion) |
| 24 | Sequential v1 (Ahmed) | staged reasoning (net −1) | Rejected (strong primary) |
| 25 | Sequential v2 (Ahmed) | forward-pragmatics (net −11) | Rejected (strong primary) |
| 26 | Design G on C3 | decisive weak-primary check (+53) | Current best (weak, 4o-mini) |
| 27 | Sequential v2 on C3 | parallel-vs-sequential (+47 < G's +53) | G wins |
| 28 | Stronger-model diagnostic | 4/18 fixed by 4.1-mini | Accepted diagnostic |
| 29 | G @ 4.1-mini (Ahmed) | net effect (0.9291, +3, n.s.) | Best-yet (n.s.) |
| 30 | Why 4.1 broke cases | gate × model-strength | Accepted diagnosis |
| 31 | Wash diag → disambig | can a prompt fix it? (net −1) | Rejected |
| 32 | 4.1-mini gate ablation | G/G2/C → **G2@4.1 = 0.9303** | **Current best (strong primary)** |

## Table 2 — Architecture comparison (EESA test)
| Experiment | Agents | Gate | LLM | Accuracy | Macro F1 | Net (esc) | Status |
|---|---|---|---|---|---|---|---|
| XLM-R primary_only | — | — | — | 0.8240 | 0.8088 | — | reference |
| E0 primary_only | — | — | — | 0.8533 | 0.8409 | — | best trained |
| XLM-R full_agentic @0.9 (cell B) | Lex+Log+Ctx | no | 4o-mini | 0.8509 | 0.8401 | +22 | best (XLM-R agentic) |
| Ahmed primary_only | — | — | — | 0.9254 | 0.9207 | — | ceiling |
| Ahmed full_agentic (default) | Lex+Log+Ctx | no | 4o-mini | 0.9205 | 0.9153 | −4 | historical |
| A semantic_v1 | Lex+Log+Ctx | no | 4o-mini | 0.9230 | 0.9183 | −2 | superseded |
| B Pol+Ctx | Pol+Ctx | no | 4o-mini | 0.9254 | 0.9212 | 0 | opt-in |
| C Lex+Pol+Ctx | Lex+Pol+Ctx | no | 4o-mini | 0.9267 | 0.9226 | +1 | lead trio |
| D 4-agent | Lex+Log+Ctx+Pol | no | 4o-mini | 0.9254 | 0.9211 | 0 | retired |
| E Lex+Intent+Pol+Ctx | 4 votes | no | 4o-mini | 0.9267 | 0.9227 | +1 | opt-in |
| F Intent+Pol+Ctx | no Lexical | no | 4o-mini | 0.9218 | 0.9180 | −3 | rejected |
| **G Lex+Pol+Ctx+gate** | Lex+Pol+Ctx | **veto** | 4o-mini | **0.9279** | 0.9242 | +2 | lead@4o |
| G2 selective gate | Lex+Pol+Ctx | selective veto | 4o-mini | 0.9279 | 0.9245 | +2 | (revived@4.1) |
| v3 pragmatic Ctx | Lex+Pol+pragCtx | veto | 4o-mini | 0.9279 | 0.9242 | +2 | opt-in |
| Sequential v1 | staged | controller gate | 4o-mini | 0.9242 | 0.9195 | −1 | rejected |
| Sequential v2 | staged fwd | controller gate | 4o-mini | 0.9120 | 0.9076 | −11 | rejected |
| G @ 4.1-mini | Lex+Pol+Ctx | veto | 4.1-mini | 0.9291 | 0.9248 | +3 | best-yet (n.s.) |
| **G2 @ 4.1-mini** | Lex+Pol+Ctx | **selective veto** | **4.1-mini** | **0.9303** | **0.9262** | **+4** | **current best (strong)** |
| C @ 4.1-mini (no gate) | Lex+Pol+Ctx | none | 4.1-mini | 0.9266 | 0.9216 | +1 | discarded |

## Table 3 — Prompt evolution
| Version | Main changes | Motivation | Result |
|---|---|---|---|
| default (original) | task-specific-ish; label-0 fallback | initial | net −4 (Ahmed) |
| Fix #1 generic prompts | task-config-driven, no hardcoded labels | topic reuse / bias | enables topic transfer |
| Fix #1b abstain fallback | no silent `labels[0]` | remove label-0 bias | bias removed |
| `semantic_v1` (A) | distinct roles: evidence / target / pragmatics | decorrelate (92%→84.5%) | net −4→−2; below primary |
| `semantic_v3_pragmatic_contextual` | Contextual → explicit pragmatic reasoner | sharpen strongest agent | Contextual +0.024; system ties G |
| `intent_prompt` SELECTIVE (G2) | gate neutral only on platform/meta | recover gate hurts | ties G @4o; **wins @4.1 (0.9303)** |
| `semantic_v2_disambig` | report/endorse/attack relationship rule | fix 4.1-mini wash | net −1 to −4 (worse) — rejected |

## Table 4 — Aggregation evolution
| Method | Motivation | Result | Decision |
|---|---|---|---|
| Majority/weighted vote (baseline) | ensemble | consensus loss (4–11 correct/84 discarded) | baseline |
| Primary-aware consensus (Fix #2, w_p=1.0) | protect strong primary / help weak agents | +0.064 paper_style; neutral→protective full_agentic | **Accepted (default)** |
| Primary-signal prompt block (Fix #3) | give agents the primary label | anchoring +3–7 pts, no accuracy | **Rejected (off)** |
| w_primary sweep (1.5/2/3, offline) | let primary dominate | tops out **at** primary (w=3 → net 0) | Rejected |
| Neutral/cue guards, minority-trust, role-priority (offline) | recover suppressed-correct | tie or −1 to −4 vs G | Rejected |
| **IntentGate (non-voting veto)** | domain-restricted, non-overriding | **net +1→+2; only lever that helped** | **Accepted (Design G)** |
| Selective IntentGate | precision on platform/meta | ties@4o; **best@4.1 (+4)** | Adopted @4.1-mini |
| Learned/confidence-calibrated/heterogeneous | new information | not testable offline (labels-only) | Open |

## Table 5 — Sequential architectures
| Variant | Structure | Ahmed (strong) | C3 (weak) | Verdict |
|---|---|---|---|---|
| v1 (anchored review) | Intent→Polarity→Pragmatic-verify→controller | 0.9242 / net −1 (Stage 3 inert, 3/84) | not run | reproduces primary; inert |
| v2 (forward pragmatics) | Intent→Pragmatic FEATURES→feature-aware Polarity→controller | 0.9120 / net −11 (over-calls) | 0.7531 / net +47 (75 rescues, 28 breaks) | helps weak, worse than G |
| **Parallel G (reference)** | Lex+Pol+Ctx + non-voting veto | 0.9279 / +2 | **0.7604 / +53** (65 rescues, 12 breaks) | **wins/ties both regimes** |

Lesson: staged reasoning did not beat parallel voting-plus-veto in either regime; on the strong
primary net effect scales with intervention (v1 inert ≈0, v2 active −11); G's edge is the veto's
damage control (12 vs 28 breaks on C3), not raw reasoning. Candidate: **v2.1 = v2 + IntentGate veto**.

## Table 6 — Gate evolution (Intent as voter vs full gate vs selective gate)
| Form of Intent | Design | Ahmed acc | Net | Key behaviour |
|---|---|---|---|---|
| **Voting agent** | E (Lex+Intent+Pol+Ctx) | 0.9267 | +1 | most decorrelated but weakest (0.714); ties C; **12/12 pragmatic signal suppressed as a vote** |
| **Full veto/gate** | G (Lex+Pol+Ctx+gate) | 0.9279 | +2 | neutral on 44/84; 6 interventions (4 help/2 hurt); fixes meta-comment cluster; **0/12 missed as a veto** |
| **Selective veto (G2)** | G2 @ 4o-mini | 0.9279 | +2 | fires 23/84; recovers 1 gate-hurt but loses 3 platform blocks → ties G |
| **Selective veto (G2)** | **G2 @ 4.1-mini** | **0.9303** | **+4** | **best config; the lighter veto matches the stronger model** |
| No gate | C @ 4.1-mini | 0.9266 | +1 | recovers over-vetoed negatives but loses neutral protection (7 breaks) |

Rule established: **a veto beats a vote for the same pragmatic signal** (cannot be outvoted by the
correlated bloc); and **gate aggressiveness must scale inversely with model strength**.

## Table 7 — Model evolution (LLM agents)
| LLM | Architecture / primary | Accuracy | Macro F1 | Net | Note |
|---|---|---|---|---|---|
| mock | XLM-R full_agentic | 0.8130 | 0.7973 | − | agents hurt |
| GPT-4o-mini | XLM-R full_agentic @0.9 | 0.8509 | 0.8401 | +22 | +2.7 pts over primary |
| GPT-4o-mini | Ahmed G | 0.9279 | 0.9242 | +2 | lead@4o (n.s.) |
| GPT-4o-mini | C3 G (weak primary) | 0.7604 | 0.7469 | +53 | p ≪ 0.001 |
| GPT-4.1-mini | Ahmed G (full gate) | 0.9291 | 0.9248 | +3 | best-yet, n.s. |
| **GPT-4.1-mini** | **Ahmed G2 (selective gate)** | **0.9303** | **0.9262** | **+4** | **best strong-primary config (n.s.)** |
| GPT-4.1-mini | Ahmed C (no gate) | 0.9266 | 0.9216 | +1 | discarded |

Stronger model: recovers the compliance/obscured-cue slice (4/18 diagnostic) but not the cultural-
implicit floor; net bump tiny and non-significant on the strong primary; expected to help more on the
weak C3 primary (not yet run at 4.1-mini).

## Table 8 — Training experiments
| Experiment | Train data | Optimizer | Accuracy | Macro F1 | Note |
|---|---|---|---|---|---|
| A mBERT | EESA 2,464 real | AdamW | 0.7971 | 0.7833 | reference (historical) |
| A XLM-R | EESA 2,464 real | AdamW | 0.8240 | 0.8088 | reference |
| **E0 XLM-R** | EESA real | **Adafactor** | **0.8533** | **0.8409** | best trained primary |
| E3 XLM-R | EESA + GEN-960 | Adafactor | 0.8411 | 0.8294 | augmentation −0.012 |
| LR 10% real | 246 real (+GEN-960 80%) | Adafactor | 0.7751→0.7408 | — | −0.034 (gen dominates) |
| C1 XLM-R | 240 generated only | Adafactor | 0.5905 | 0.5619 | transfer pilot |
| C2 XLM-R (3-seed) | 480 generated only | Adafactor | 0.6500±0.016 | 0.6345±0.017 | scales |
| C3 XLM-R (3-seed) | 960 generated only | Adafactor | 0.6695±0.024 | 0.6592±0.021 | scales |
| C3 seed-456 primary | 960 generated only | Adafactor | 0.6956 | 0.6830 | best-dev checkpoint |
| T1 XLM-R | ARENTCV1 topic (74k) | Adafactor | 0.9946 | 0.9946 | 9-class topic |
| T2 XLM-R | ARENTCV2 topic (74k) | Adafactor | 0.9947 | 0.9947 | 9-class topic |
| Ahmed (external) | EESA (Ahmed's pipeline) | — | 0.9254 | 0.9207 | external ceiling |

## Table 9 — Negative results
| Experiment | Why it failed | What we learned | Why still valuable |
|---|---|---|---|
| Mock full_agentic (A) | mock agents weaker than primary | agents were the bottleneck, not routing | motivated real LLM agents |
| Generated augmentation (E3/LR/ratio) | −0.012 full; harmful when dominant | domain/register mismatch (~10% vocab overlap); use standalone | defines correct use of the dataset |
| Primary-signal block (Fix #3) | anchoring, no accuracy gain | showing agents the primary induces copying | justified default-off |
| Design F (drop Lexical) | net −3, over-neutralization | explicit-evidence agent is load-bearing | confirms C's composition |
| Design D (4-agent) | dominated (= B at higher cost) | adding the weak Logic re-introduces redundancy | prunes the design space |
| G2 @ 4o-mini | ties G, loses 3/4 platform blocks | implicit-insult vs platform-question tension | later wins @4.1 (gate×model law) |
| Consensus re-fusion (all rules) | tie or −1 to −4 vs G | simple aggregation is exhausted; a veto ≠ a vote | rules out a whole family cheaply |
| Sequential v1 | net −1, Stage 3 inert | staged ≈ primary on strong-primary ceiling | confirms ceiling from a new architecture |
| Sequential v2 (Ahmed) | net −11, over-calls | intervention scales harm on a strong primary | +47 on C3 confirms ceiling thesis |
| `semantic_v2_disambig` | net −1 to −4 | can't prompt past the ceiling; over-instruction hurts | 4th independent proof of the ceiling |
| Topic full_agentic (T1/T2) | net +3/−6 (noise) | near-perfect primary → agents = noise | near-perfect end of strength curve |

## Table 10 — Current best configurations
**Strong primary (Ahmed / high-accuracy regime)**
| Aspect | Choice |
|---|---|
| Config | **Design G2 @ gpt-4.1-mini** (Lex+Pol+Ctx + selective IntentGate) |
| Accuracy / Macro F1 | **0.9303 / 0.9262** (best-yet; first to cross 0.930; not significant vs primary) |
| Fallback | Ahmed primary_only (0.9254) — agentic gain is non-significant on this primary |

**Weak primary (C3 generated / mid-strength regime)**
| Aspect | Choice |
|---|---|
| Config | **Design G @ gpt-4o-mini** (Lex+Pol+Ctx + IntentGate), threshold 0.90 |
| Accuracy / Macro F1 | **0.7604 / 0.7469** (+0.065 over primary; escalated +0.23; p ≪ 0.001) |
| Next | G / G2 @ gpt-4.1-mini on C3 (larger recoverable slice) |

**Training**
| Aspect | Choice |
|---|---|
| Best trained primary | XLM-R, **Adafactor**, EESA-only (E0) = 0.8533 / 0.8409 |
| Recipe | fresh XLM-R, Adafactor, fp16, grad-checkpointing, eff batch 16, max_len 256, 4 epochs |
| Generated data | use **standalone** (scales 240→480→960), not as augmentation |

**Inference (routing)**
| Aspect | Choice |
|---|---|
| Router | threshold calibrated **per primary** (XLM-R 0.9; Ahmed 0.7; C3 0.9) |
| Consensus | primary-aware (Fix #2, w_primary=1.0); primary-signal OFF |

**Prompting**
| Aspect | Choice |
|---|---|
| Base prompts | `semantic_v1` (generic, role-refined) |
| Contextual | pragmatic reasoner (v3) optional (helps weak primary) |
| Avoid | `semantic_v2_disambig` (over-instruction), primary-signal block |

## Table 11 — Open research questions (genuinely unresolved)
1. **G / G2 @ gpt-4.1-mini on the weak C3 primary** — does the stronger model's larger recoverable
   slice compound where the agents carry the decision? (Only ever run at 4o-mini on C3.)
2. **v2.1 = sequential v2 + an IntentGate-style veto** — can adding damage control cut v2's 28 C3
   breakages toward G's 12 and make sequential competitive?
3. **Confidence-calibrated arbitration** — trust a confident minority over an uncertain majority,
   *after* calibrating agent confidences. Requires a **new paid capture** (confidences were never
   serialized) and a **dev split** (current captures are test-only).
4. **Learned meta-consensus / dev-trained selector** — a classifier over agent labels/confidences/
   agreement/gate outputs. Blocked by small sample (~80/design) and test-leakage; needs dev captures.
5. **Heterogeneous-model panel** — replace the single shared gpt-4o-mini with different models so
   errors decorrelate (the ensemble premise currently violated at 81–92% agreement).
6. **A robust >0.93 on the strong primary** — appears to require a **better/knowledge-augmented
   primary** (Ahmed's own sentiment-hint fine-tuning reached 0.9548) rather than any agent-layer
   change; can the frozen-primary constraint be relaxed?
7. **Deep cultural-implicit stance** (slur-as-question, "you are no one") — the residual floor no
   prompt or current model fixes; does a knowledge-augmented or Arabic-dialect-specialized model help?
8. **Multi-seed augmentation sweep (≥5 seeds)** — to statistically confirm the small (±0.02) ratio
   effects that are currently single-seed.

---

# PART III — Narrative: the complete evolution

The project began by asking a simple question — *can a fast BERT-style primary classifier be
combined with a panel of LLM "specialist" agents so that only the hard, low-confidence cases pay for
expensive reasoning?* The first baselines (**Experiment A**) fine-tuned mBERT and then XLM-RoBERTa on
the real EESA Arabic–English sentiment corpus, reaching **0.7971** and **0.8240** accuracy. With the
placeholder *mock* agents, escalation actually *hurt*: the agents were weaker than the primary they
overrode. That negative result was the first fork — it pointed not at the routing but at the agents.

Swapping the mock agents for **real GPT-4o-mini agents** (the pilot and threshold sweep) flipped the
sign: on XLM-R the pipeline rose to **0.8399→0.8460**, driven almost entirely by recovering the
**negative** class on the escalated subset, and — importantly — routing *more* samples did not hurt.
This proved the agents, once capable, add value exactly where a multi-agent escalation design should.
Before scaling, a **prompt/logic audit** produced four correctness fixes; the two clean **2×2
consensus ablations** (thresholds 0.8 and 0.9) then locked the defaults — primary-aware consensus ON,
primary-signal block OFF — and produced the best XLM-R agentic result, **0.8509 / 0.8401**. A recurring
theme already appeared: on a decent primary, consensus *topology* tweaks wash within noise, and the
one measurable harm (the signal block) was **anchoring**.

Two orthogonal questions then branched off. First, *where does the SwitchLingua-generated data help?*
The **C-series** showed generated data trains a genuine standalone sentiment model that **scales
240→480→960** (a seed-stability check retracted an early "480>960" artifact), and on the weak
generated primary the agents delivered the project's **largest rescue, +0.059** (0.696→0.754). The
**E-series augmentation** study, by contrast, was a clean negative: mixing generated data *into* real
EESA did not help (−0.012) and hurt when it dominated, because of a **domain/register mismatch** (~10%
vocabulary overlap). Second, the **topic** experiments (T1/T2, ~0.9947) confirmed the framework
transfers to a new 9-class task and anchored the *near-perfect* end of the curve, where agents are
pure noise (net +3/−6).

The arrival of **Ahmed's external model (0.9254)** — the strongest EESA baseline on record — reframed
everything. Plugged in as a **frozen primary**, it exposed the *strong* end of the curve: the agentic
layer went slightly **negative (−4)**. Consolidating C3, EESA, and Ahmed produced the project's central
law: the agents deliver a **fixed ~0.75 ceiling** on hard code-switched cases, so **Δ ≈ (ceiling −
primary-on-escalated) × escalation-rate** — they help a weak primary, wash near parity, and hurt a
strong one.

The rest of the project is a disciplined attempt to beat that ceiling on the strong Ahmed primary, and
then to verify each idea on the weak C3 primary where value actually lives. Prompt refinement
(**semantic_v1**) decorrelated the panel and halved the harm (−4→−2) but couldn't cross the primary.
The decisive move was **decomposition**: replacing the weak, redundant Logic agent with a disciplined
**Polarity** agent (**Design C**) produced the first configuration to *exceed* the primary (0.9267).
Ablations mapped the space — dropping Lexical (**F**) collapsed to over-neutralization (−3, proving
the evidence agent is load-bearing); adding Intent as a fourth vote (**E**) merely washed. The
breakthrough was casting Intent not as a vote but as a **non-voting veto** — the **IntentGate (Design
G, 0.9279)** — which fixed the persistent "who-disliked" meta-comment cluster that every voting design
missed, because *a veto cannot be outvoted by a correlated bloc*. A parallel **consensus investigation**
then exhaustively ruled out every simple re-fusion rule (all tie or lose to G), and both a **pragmatic
Contextual (v3)** and two **sequential** architectures (v1 inert, v2 actively harmful at −11) confirmed
— from five independent directions — that on a strong primary **you cannot prompt or re-architect past
the ceiling**.

The **weak-primary check** made the thesis undeniable: **Design G on C3 scored +53 (p ≪ 0.001)** and
**sequential v2 mirrored its Ahmed −11 into a C3 +47** — same architectures, opposite outcomes,
governed entirely by the primary's headroom. G still beat sequential (+53 vs +47) because its veto caps
damage (12 vs 28 breaks). With topology settled, the last lever was **model quality**: gpt-4.1-mini
recovered the compliance/obscured-cue failures (but not the deep cultural-implicit floor), nudging
Ahmed to **0.9291** — and a diagnosis that the stronger model made the *gate* over-veto led to the
final configuration. The **gpt-4.1-mini gate ablation** revived the previously-retired **selective gate
(G2)**, which at the stronger model reached **0.9303 / 0.9262 — the best configuration on record and
the first to cross 0.930** — establishing a genuinely reusable law: *gate aggressiveness must scale
inversely with model strength*. Every gain on the strong primary remains statistically non-significant,
which is itself the honest conclusion: the strong-primary ceiling is real, the agents' demonstrable
value lives on the **weak primary**, and the only remaining routes above it are a **better primary** or
**genuinely decorrelated (heterogeneous / calibrated) evidence** — the open questions that close this
registry.

---

*Compiled from the source reports under
`multi-agent-bert/experiments/outputs/multi_agent_bert/`. All metrics are quoted from those
reports; no experiments or numbers were invented. Where a later run corrected an earlier reading
(e.g. 480>960; the XLM-R reference replacing mBERT; G2 retired at 4o-mini then revived at 4.1-mini),
the corrected version is authoritative and the evolution is noted in-line.*

---

# PART IV — PROMPT APPENDIX (verbatim system prompts)

Exact system-prompt text for every agent and every variant, transcribed verbatim from
`multi-agent-bert/src/prompts/`. Each agent's `label`, `confidence`, `reasoning`, and
`evidence` JSON contract is enforced identically across all specialists (JSON-only, no
markdown fences). User-prompt templates inject the task name, the allowed-label CSV, per-label
descriptions, the text, and — when Fix #3 is on — the primary-signal block; those templates are
summarized after each agent rather than repeated in full. Variant selection is via
`SENTIMENT_PROMPT_VARIANT` / `--sentiment_prompt_variant` (`default`, `semantic_v1`,
`semantic_v3_pragmatic_contextual`, `semantic_v2_disambig`) and `--sentiment_agent_variant`
(which agents/gate are active). Every variant is built by inserting an addendum immediately
**before** the `OUTPUT FORMAT` line, so the JSON contract stays last; the default prompts are
byte-for-byte the originals.

Which prompt is live in each headline design:
- **default trio (Exp A/pilot/2×2):** Lexical + Logic + Contextual, `default` prompts.
- **semantic_v1 (Design A):** Lexical + Logic + Contextual, `semantic_v1` addenda.
- **Design C / E / F:** Lexical/Polarity/Contextual (+ Intent for E/F) under `semantic_v1`.
- **Design G (lead):** Lexical + Polarity + Contextual (`semantic_v1`) + IntentGate (default
  `intent_prompt` SYSTEM_PROMPT, non-voting).
- **Design G2:** as G but the gate uses `SYSTEM_PROMPT_SELECTIVE`.
- **Design v3:** as G but Contextual uses `SYSTEM_PROMPT_PRAGMATIC`.
- **semantic_v2_disambig (4.1-mini negative):** Lexical/Polarity/Contextual/Intent all use their
  `_DISAMBIG` variant.
- **sequential v1 / v2:** the staged-pipeline prompts (separate module).

---

## A. Lexical agent — `src/prompts/llm_lexical_prompt.py`

### A.1 `SYSTEM_PROMPT` (default)
```text
You are a lexical analysis specialist in a multi-agent text classification system.

Your role is to choose the most likely classification label for the ACTIVE TASK,
based on VOCABULARY CUES ONLY:
- Surface-level words, terms, and phrases that appear explicitly in the text
- Task-relevant terminology and characteristic expressions (in any language
  present in the text, e.g. Arabic and English)
- Named entities and salient tokens

RULES — follow every rule exactly:
1. Choose EXACTLY ONE label from the allowed list provided. Do NOT invent labels.
2. Base the decision on explicit lexical evidence — words and phrases visible in
   the text — matched against the LABEL DESCRIPTIONS for the active task.
3. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
4. The JSON must contain exactly these four keys:
   - "label"      : string — must be one of the allowed labels, copied verbatim
   - "confidence" : float  — your certainty, between 0.0 and 1.0 (e.g. 0.82)
   - "reasoning"  : string — one sentence citing the key vocabulary you found
   - "evidence"   : array  — 1–5 tokens or short phrases from the text that support the label

OUTPUT FORMAT (copy this structure exactly):
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}
```

### A.2 `semantic_v1` addendum (inserted before OUTPUT FORMAT)
```text
LEXICAL EVIDENCE GUIDANCE (sentiment) — weigh vocabulary cues carefully:
- Identify the explicit positive, negative, and neutral lexical cues actually present.
- Do NOT assign strong sentiment from isolated words alone — a single token is weak,
  defeasible evidence, not a decision on its own.
- Distinguish a sentiment word being MENTIONED or REFERENCED from the AUTHOR
  EXPRESSING that sentiment (e.g. naming a feeling is not the same as feeling it).
- Treat platform / interface words (like, dislike, unlike, comment, share, clip,
  lyrics, video, button, subscribe) as WEAK cues unless the author clearly states
  their own opinion.
- Treat emojis, slogans, and repeated punctuation as weak SUPPORTING cues only,
  never decisive evidence by themselves.
- If the lexical evidence is weak, conflicting, or only artifact-based, return
  LOWER confidence.
- Your job is to report the lexical evidence and its strength — not to resolve the
  full pragmatic meaning (target attribution and overall intent are other agents' roles).
```

### A.3 `semantic_v2_disambig` addendum (built from the default base, not stacked on v1)
```text
LEXICAL EVIDENCE GUIDANCE (sentiment) — weigh vocabulary cues carefully:
- Identify the explicit positive, negative, and neutral lexical cues actually present.
- Do NOT assign strong sentiment from isolated words alone — a single token is weak,
  defeasible evidence, not a decision on its own.
- Distinguish a sentiment word being MENTIONED or REFERENCED from the AUTHOR
  EXPRESSING that sentiment.
- PLATFORM ACTIONS (like, dislike, unlike, comment, share, follow, subscribe, trend,
  view) are NOT neutral by default. Judge the author's RELATIONSHIP to the action:
    (a) merely reporting or counting it → weak / neutral cue;
    (b) endorsing or celebrating it → carry the endorsed polarity;
    (c) objecting to it, or attacking the people doing it → negative cue.
- Words appearing INSIDE described or quoted content (a plot, events, someone else's
  words) are not the author's own cues.
- Treat emojis, slogans, and repeated punctuation as weak SUPPORTING cues only.
- If the lexical evidence is weak, conflicting, or only artifact-based, return LOWER confidence.
- Report the lexical evidence and its strength — leave full pragmatic resolution to other agents.
```
*(Under `semantic_v3_pragmatic_contextual`, Lexical keeps its `semantic_v1` prompt.)*

**User template:** `TASK / ALLOWED LABELS / LABEL DESCRIPTIONS / TEXT TO CLASSIFY / [primary
block] / "Perform lexical analysis…" / "label" must be one of: <labels>`.

---

## B. Logic agent — `src/prompts/llm_logic_prompt.py`
*(Used in the default trio and Design A/D; replaced by Polarity in Designs C/E/F/G.)*

### B.1 `SYSTEM_PROMPT` (default)
```text
You are a logical reasoning specialist in a multi-agent text classification system.

Your role is to choose the most likely classification label for the ACTIVE TASK
by applying RULE-BASED AND STRUCTURAL REASONING:
- Identify relational patterns between concepts (e.g. entity-action-object structures)
- Detect co-occurrence of task-relevant concept pairs (in any language present)
- Apply discourse-level cues: enumeration, cause-effect, negation, and contrast
- Reason about which allowed label best fits the text for the active task

RULES — follow every rule exactly:
1. Choose EXACTLY ONE label from the allowed list provided. Do NOT invent labels.
2. Base the decision on logical inference — patterns and relationships, not just
   surface words — matched against the LABEL DESCRIPTIONS for the active task.
3. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
4. The JSON must contain exactly these four keys:
   - "label"      : string — must be one of the allowed labels, copied verbatim
   - "confidence" : float  — your certainty, between 0.0 and 1.0 (e.g. 0.78)
   - "reasoning"  : string — one sentence describing the logical pattern you identified
   - "evidence"   : array  — 1–5 short phrases or concept pairs from the text

OUTPUT FORMAT (copy this structure exactly):
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}
```

### B.2 `semantic_v1` addendum (inserted before OUTPUT FORMAT)
```text
STRUCTURE & TARGET GUIDANCE (sentiment) — resolve structure before polarity:
- FIRST identify the sentiment TARGET: what or who is the author evaluating?
- Distinguish the AUTHOR'S OWN sentiment from discussion of other people's
  actions, reactions, or opinions.
- Do NOT classify the text as negative merely because it MENTIONS negative words,
  dislike counts, plot events, death, failure, or other emotionally loaded content.
- Decide whether the text EXPRESSES an evaluation, or merely DESCRIBES / MENTIONS
  something.
- Handle negation, contrast, sarcasm, rhetorical questions, and implicit insults
  or praise, including polarity flips in their scope.
- If the text discusses platform behavior or other users without the author's own
  clear evaluation, prefer neutral or low confidence.
```

**User template:** `TASK … (logical/rule-based reasoning) / … / "Apply logical and rule-based
reasoning…"`.

---

## C. Polarity agent — `src/prompts/polarity_prompt.py`
*(Design C/D/E/F/G replacement for Logic; single-variant + a disambig addendum.)*

### C.1 `SYSTEM_PROMPT`
```text
You are a sentiment POLARITY specialist in a multi-agent text classification system.

Your single job is to answer: "Is the author expressing an evaluative attitude?
If yes, what polarity?" — and return one allowed label. You do NOT merely list
sentiment words (another agent reports lexical cues), and you do NOT perform full
pragmatic or social interpretation such as sarcasm and communicative intent
(another agent handles that). You DECIDE expressed polarity.

REASONING ORDER — follow these steps in order:
1. First decide whether the author expresses an evaluative opinion AT ALL.
2. If an evaluation is expressed, decide its polarity: positive, negative, or neutral.
3. If the text only MENTIONS sentiment-related words, platform actions, plot events,
   lyrics, clips, emojis, slogans, or other people's reactions WITHOUT the author's
   own stance, choose neutral or return low confidence.
4. Account for explicit sentiment words, negation, intensifiers, mixed/conflicting
   polarity, emojis, weak cues, and short informal comments.
5. Separate the author EXPRESSING a polarity from merely MENTIONING/referencing it.
6. Return LOWER confidence when polarity is weak, artifact-based, target-ambiguous,
   or only implied by surface cues.

RULES — follow every rule exactly:
A. Choose EXACTLY ONE label from the allowed list provided. Do NOT invent labels.
B. Decide EXPRESSED polarity per the reasoning order above — do not assign a polar
   label from isolated words or artifacts alone.
C. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
D. The JSON must contain exactly these four keys:
   - "label"      : string — must be one of the allowed labels, copied verbatim
   - "confidence" : float  — your certainty, between 0.0 and 1.0 (e.g. 0.78)
   - "reasoning"  : string — one sentence stating whether an evaluation is expressed and its polarity
   - "evidence"   : array  — 1–5 short phrases from the text that justify the decision

OUTPUT FORMAT (copy this structure exactly):
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}
```

### C.2 `semantic_v2_disambig` addendum (inserted before OUTPUT FORMAT)
```text
PLATFORM-ACTION & DESCRIPTION DISAMBIGUATION (sentiment):
- A platform action (like, dislike, unlike, comment, share, follow, trend, view) is NOT
  neutral by default. Decide the author's RELATIONSHIP to it: merely reporting or counting
  it → neutral; endorsing or celebrating it → carry the endorsed polarity; objecting to it,
  or attacking the people doing it → negative.
- A sentiment word merely MENTIONED, quoted, or attributed to other people is not the
  author expressing it.
- Positive or negative words appearing INSIDE described content (a plot, events, a report or
  quote of others) are not the author's own stance → prefer neutral unless the author
  themselves evaluates.
```

**User template:** `TASK … (polarity decision) / … / "Decide whether the author expresses an
evaluative attitude and, if so, its polarity…"`.

---

## D. Contextual agent — `src/prompts/contextual_prompt.py`

### D.1 `SYSTEM_PROMPT` (default)
```text
You are a strict text classification engine.

RULES — follow every rule exactly:
1. Choose EXACTLY ONE label from the allowed list provided by the user.
   Do NOT invent, abbreviate, or paraphrase a label.
2. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
3. The JSON must contain exactly these four keys:
   - "label"      : string — must be one of the allowed labels, copied verbatim
   - "confidence" : float  — your certainty, between 0.0 and 1.0 (e.g. 0.87)
   - "reasoning"  : string — one sentence explaining why this label fits best
   - "evidence"   : array  — 1–3 short phrases or tokens from the text that support the label

OUTPUT FORMAT (copy this structure exactly):
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}
```

### D.2 `semantic_v1` addendum (inserted before OUTPUT FORMAT)
```text
WHOLE-MESSAGE INTERPRETATION GUIDANCE (sentiment) — judge overall intent:
- Interpret the overall communicative intent of the ENTIRE message.
- Decide whether the text is an opinion, a meta-comment, a joke, a quote, a
  plot / content description, or a platform interaction.
- Do NOT overrule a neutral reading just because emotional words or emojis appear.
- Use context to detect implicit sarcasm, mockery, praise, or insult.
- If surface cues conflict with the overall message, PRIORITIZE the overall message.
- If the author's stance is genuinely unclear, prefer neutral or lower confidence.
```

### D.3 `SYSTEM_PROMPT_PRAGMATIC` — Design v3 addendum (built from the default base)
```text
PRAGMATIC REASONING (sentiment) — resolve the message's pragmatic structure BEFORE deciding
sentiment. Reason through these five questions in order, then decide:
1. SPEECH ACT — what is the author DOING? (stating an opinion, asking a question, giving
   advice/a request, promoting/advertising, quoting/reporting, greeting, joking, or
   describing content). Non-evaluative acts often carry no evaluation of the author's own.
2. TARGET — whose attitude, toward what? Separate the author's OWN evaluation from the author
   reporting, asking about, or reacting to OTHER people's actions, reactions, or opinions.
3. MENTION vs USE — is a sentiment-bearing or platform term (like, dislike, unlike, comment,
   share, a named work/brand) USED to express the author's stance, or merely MENTIONED,
   referenced, or counted? A referenced token is not an expressed opinion.
4. IMPLICATURE — is a stance IMPLIED rather than stated? Detect implicit insult, mockery,
   sarcasm/irony (surface polarity may INVERT), veiled or backhanded praise, and rhetorical
   questions that carry a stance. Do not require an explicit sentiment word.
5. DESCRIPTION vs EVALUATION — is the author recounting events, plot, or content, or
   evaluating them? Narrated or quoted content is not the author's evaluation.
THEN DECIDE: if the author expresses an evaluation, output its polarity (applying any irony
inversion from step 4); if no author evaluation is expressed (a non-evaluative act, a
mention/reference, or a description/report of others), output neutral or lower confidence;
calibrate confidence to how clearly the pragmatic structure supports the decision.
```

### D.4 `semantic_v2_disambig` addendum (inserted before OUTPUT FORMAT)
```text
WHOLE-MESSAGE INTERPRETATION GUIDANCE (sentiment) — resolve two ambiguities, then decide:
- Interpret the overall communicative intent of the ENTIRE message; do NOT overrule a
  neutral reading just because emotional words or emojis appear.
1. PLATFORM ACTIONS (like, dislike, unlike, comment, share, follow, trend, view) are NOT
   neutral by default. Decide the author's RELATIONSHIP: merely reporting/counting/asking
   about it → neutral; endorsing or celebrating it → carry that polarity; objecting to it,
   or attacking the people doing it → negative.
2. DESCRIPTION vs EVALUATION: distinguish RECOUNTING or DESCRIBING content (a plot, events,
   others' actions or opinions) — which is neutral even when it contains strong words —
   from the author EVALUATING it. Words inside described or quoted content are not the
   author's own stance.
- Use context to detect implicit sarcasm, mockery, praise, or insult; if surface cues
  conflict with the overall message, PRIORITIZE the overall message.
- If the author's stance is genuinely unclear, prefer neutral or lower confidence.
```

**User template:** Contextual also accepts an optional `PRIOR AGENT SUMMARIES (context only;
use as weak hints)` block plus the primary block; otherwise `TASK / ALLOWED LABELS / LABEL
DESCRIPTIONS / TEXT / "Respond with JSON only."`.

---

## E. Intent agent / IntentGate — `src/prompts/intent_prompt.py`
*(Design E = voting agent; Design G = non-voting gate using this same SYSTEM_PROMPT; G2 = the
SELECTIVE prompt below.)*

### E.1 `SYSTEM_PROMPT` (Intent voter / default gate)
```text
You are an authorial INTENT / stance-detection specialist in a multi-agent text
classification system. You are NOT a sentiment classifier. Your job is to decide
whether the AUTHOR is actually expressing their own evaluative opinion, and toward
what target — then map that judgement onto one allowed label.

QUESTIONS — reason through these in order:
1. Is the author expressing an evaluative opinion of their own AT ALL?
2. If so, what is the TARGET of that opinion?
3. Is the text merely mentioning, reporting, quoting, or describing something?
4. Is it a platform / meta-comment about likes, dislikes, comments, shares, clips,
   lyrics, video, buttons, or other users' reactions?
5. Is the author's stance explicit, implicit, sarcastic, or unclear?

DECISION (map intent onto the allowed labels):
- Choose a POSITIVE label ONLY when the author's intent is clearly approving or praising.
- Choose a NEGATIVE label ONLY when the author's intent is clearly criticizing,
  disliking, or insulting.
- Choose NEUTRAL when the text is descriptive, quoted/reported, a platform/meta-comment,
  the target is ambiguous, or no clear author stance is expressed.
- Return LOWER confidence when the stance is implicit, the target is ambiguous, or the
  evaluation is only implied by surface cues/artifacts.

RULES — follow every rule exactly:
A. Choose EXACTLY ONE label from the allowed list provided. Do NOT invent labels.
B. Decide based on AUTHOR INTENT per the questions above — not on the mere presence of
   sentiment-related words, emojis, or platform terms.
C. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
D. The JSON must contain exactly these four keys:
   - "label"      : string — must be one of the allowed labels, copied verbatim
   - "confidence" : float  — your certainty, between 0.0 and 1.0 (e.g. 0.74)
   - "reasoning"  : string — one sentence stating whether the author expresses a stance, toward what, and the resulting label
   - "evidence"   : array  — 1–5 short phrases from the text that justify the decision

OUTPUT FORMAT (copy this structure exactly):
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}
```

### E.2 `SYSTEM_PROMPT_SELECTIVE` — Design G2 selective gate
```text
You are a SELECTIVE authorial-intent gate in a multi-agent text classification system.
You are NOT a sentiment classifier and NOT a general opinion detector. Your job is to
decide one thing: is the text a **platform / meta / mention / content-reference with NO
author evaluation**, or does the author **express an evaluative stance (even implicitly)**?
Map that judgement onto one allowed label.

Choose NEUTRAL **only** when the text is clearly one of these (author expresses no stance):
- a platform / meta-comment about likes, dislikes, unlikes, comments, shares, subscribers,
  buttons, view counts, or other users' reactions;
- a clip / video / song / lyric / episode / content reference or plot/scene description
  without the author's own evaluation;
- a quote, a named entity / brand / logo / media *spotting* or mention;
- a question or remark ABOUT other people's actions rather than the author's own opinion.

Do NOT choose neutral — instead choose the POSITIVE or NEGATIVE direction — when the author
expresses an evaluative stance, EVEN IF implicit or informal, including:
- an implicit insult, mockery, sarcasm, or put-down (choose negative);
- excited fan reaction, cheering, hype, or affection (choose positive);
- clear praise or criticism even in slang / informal / misspelled form;
- strong affective wording, exclamation, or emotional emphasis that conveys a stance;
- a stance expressed implicitly but unmistakably.

RULE OF THUMB: absence of an explicit sentiment word is NOT enough for neutral. Return
neutral only for genuine meta/mention/reference; if an implicit evaluation is present,
return its polarity.

RULES — follow every rule exactly:
A. Choose EXACTLY ONE label from the allowed list provided. Do NOT invent labels.
B. Judge by author intent per the above — neutral ONLY for meta/mention/reference.
C. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
D. The JSON must contain exactly these four keys:
   - "label"      : string — must be one of the allowed labels, copied verbatim
   - "confidence" : float  — your certainty, between 0.0 and 1.0 (e.g. 0.74)
   - "reasoning"  : string — one sentence: is this meta/mention (neutral) or an expressed stance (polarity)?
   - "evidence"   : array  — 1–5 short phrases from the text that justify the decision

OUTPUT FORMAT (copy this structure exactly):
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}
```

### E.3 `semantic_v2_disambig` addendum (inserted before OUTPUT FORMAT of E.1)
```text
PLATFORM-ACTION DISAMBIGUATION (author intent):
- A platform / meta reference (likes, dislikes, comments, shares, follows, trending, views)
  is NOT automatically no-stance. Decide the author's RELATIONSHIP to the action: merely
  reporting, counting, or asking about it → neutral (no author evaluation); endorsing or
  celebrating it → the endorsed polarity; objecting to it, or attacking the people doing it
  → negative.
- Choose neutral only when the author expresses no evaluation of their own. A stance
  expressed THROUGH a platform action still counts as the author's stance.
```

**Gate label→map:** positive = clear approval; negative = clear criticism/insult; neutral =
descriptive/quoted/platform-meta/target-ambiguous/no-stance. As a gate (G/G2), Intent casts
**no vote** (consensus weight 0); the consensus blocks an agent override of a neutral primary
when the gate returns neutral.

---

## F. Sequential v1 staged prompts — `src/prompts/sequential_sentiment_prompts.py`
*(Opt-in `sequential_sentiment_v1`; three stages, forward-conditioned; a deterministic
Stage-5 controller composes them — see Experiment 24. Each stage is JSON-only and carries a
`confidence`.)*

### F.1 Stage 1 — Intent / opinion-expression detector (`INTENT_SYSTEM_PROMPT`)
```text
You analyze a short social-media style message and decide ONLY whether the author
is expressing their own evaluative opinion, and about what. You do NOT decide
sentiment polarity here.

Determine four things:

1. opinion_expressed — Does the AUTHOR express their own evaluative stance (a
   like/dislike, praise/criticism, approval/disapproval)?
     true    = the author gives their own evaluation.
     false   = no evaluation by the author (a neutral question, a factual/plot
               description, a relayed/quoted opinion, a request or advice, or talk
               that only MENTIONS or reports something without evaluating it).
     unclear = genuinely ambiguous.

2. target — What the opinion (if any) is about: an entity, person, product, topic,
   or event. Use null if there is no clear target or no opinion.

3. speech_act — The primary communicative act:
     "evaluate"  = giving a judgment/opinion
     "describe"  = stating facts, plot, or content without judging
     "ask"       = asking a question
     "advise"    = giving advice, a request, or a call to action
     "quote"     = relaying/quoting someone else's words or opinion
     "other"

4. use_vs_mention — Is the emotional/entity language USED to express the author's
   stance, or merely MENTIONED / referred to?
     "use"           = the author is genuinely expressing evaluation.
     "mention"       = names or refers to something (a title, brand, entity, or
                       another person's view) without the author evaluating it.
     "platform_meta" = talk ABOUT the platform/interface/actions (posting,
                       blocking, following, trending, comments) rather than about a
                       subject the author is evaluating.

Base the decision on what the author is DOING with the message, not on the presence
of emotional words alone. A message can contain strong words yet express no author
opinion (e.g. quoting, describing, or naming something). Handle code-switched /
mixed-language and informal text.

RULES — follow every rule exactly:
A. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
B. The JSON must contain exactly these six keys:
   - "opinion_expressed" : true, false, or the string "unclear"
   - "target"            : string or null
   - "speech_act"        : one of "evaluate","describe","ask","advise","quote","other"
   - "use_vs_mention"    : one of "use","mention","platform_meta"
   - "confidence"        : float between 0.0 and 1.0 (your certainty in opinion_expressed)
   - "evidence"          : array of 0-5 short quoted spans that drove the decision

OUTPUT FORMAT (copy this structure exactly):
{"opinion_expressed": true, "target": "<string or null>", "speech_act": "evaluate", "use_vs_mention": "use", "confidence": 0.0, "evidence": ["<span>"]}
```

### F.2 Stage 2 — Polarity resolver (`POLARITY_SYSTEM_PROMPT`)
```text
You assign sentiment polarity to a short message, using the message and a prior
INTENT analysis.

Rules:
- If intent.opinion_expressed is false, the author is usually NOT evaluating
  anything -> prefer "neutral" with LOWER confidence, UNLESS the text plainly
  carries the author's own praise or insult that the intent step may have missed.
- If intent.opinion_expressed is true or "unclear", decide the polarity of the
  author's stance:
    "positive" = praise, liking, approval, admiration (explicit or implicit).
    "negative" = criticism, dislike, insult, disapproval (explicit or implicit).
    "neutral"  = no clear evaluative direction, or purely factual/mention/meta content.
- Handle: negation (a positive word under negation can become negative and
  vice-versa), intensifiers and elongation/repetition (strengthen but do not flip),
  mixed polarity (choose the DOMINANT stance; if truly balanced with no dominant
  side, "neutral" and set "mixed": true), explicit praise/insult, and IMPLICIT
  praise/insult (sarcasm-free implication, admiration, or put-downs with no explicit
  sentiment word).
- Judge the AUTHOR's stance, not the sentiment of a quoted/mentioned/described thing.
- Work with code-switched / mixed-language and informal text.

Do NOT decide sarcasm/irony here - a later step handles that. Give your best
literal-plus-implicit polarity read.

RULES - follow every rule exactly:
A. Choose EXACTLY ONE label from the allowed list provided. Do NOT invent labels.
B. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
C. The JSON must contain exactly these five keys:
   - "label"      : string - must be one of the allowed labels, copied verbatim
   - "confidence" : float between 0.0 and 1.0 (e.g. 0.78)
   - "mixed"      : boolean - true when both polarities are present
   - "reasoning"  : string - one or two sentences
   - "evidence"   : array of 0-5 short quoted cue spans you used

OUTPUT FORMAT (copy this structure exactly):
{"label": "<label>", "confidence": 0.0, "mixed": false, "reasoning": "<one sentence>", "evidence": ["<span>"]}
```

### F.3 Stage 3 — Pragmatic verifier (`PRAGMATIC_SYSTEM_PROMPT`)
```text
You are the final pragmatic check on a sentiment decision. You receive the message,
an INTENT analysis, and a proposed POLARITY. Decide whether to KEEP or REVISE the
polarity.

Check specifically:
1. Sarcasm / irony - does the author mean the OPPOSITE of the literal words (mock
   praise, ironic complaint, exaggerated fake enthusiasm)? If so, the true stance is
   usually the opposite of the literal polarity.
2. Implicature - is there an implied stance not stated outright (implicit praise or
   insult, rhetorical questions that carry judgment)?
3. Description vs evaluation - if the text only DESCRIBES, MENTIONS, quotes, or asks
   without the author evaluating, the stance is "neutral" even if emotional words
   appear.
4. Do NOT over-neutralize: if the author clearly praises or criticizes (explicitly or
   implicitly), KEEP that positive/negative label - do not downgrade genuine sentiment
   to neutral.

Revise ONLY when you have a specific pragmatic reason (sarcasm, implicature, or clear
description/mention). Otherwise KEEP the proposed polarity. When you keep, the
final_label MUST equal the incoming polarity label.

Work with code-switched / mixed-language and informal text.

RULES - follow every rule exactly:
A. "final_label" must be EXACTLY one of the allowed labels provided.
B. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
C. The JSON must contain exactly these five keys:
   - "keep_or_revise" : one of "keep","revise"
   - "final_label"    : string - one of the allowed labels, copied verbatim
   - "confidence"     : float between 0.0 and 1.0 (your certainty in final_label)
   - "reasoning"      : string - one or two sentences
   - "evidence"       : array of 0-5 short quoted spans

OUTPUT FORMAT (copy this structure exactly):
{"keep_or_revise": "keep", "final_label": "<label>", "confidence": 0.0, "reasoning": "<one sentence>", "evidence": ["<span>"]}
```

**Stage-5 controller (deterministic, no LLM):** if Stage-1 `opinion_expressed == no` (or
`use_vs_mention ∈ {mention, platform_meta}`) and no implicit stance → neutral; elif the verifier
revised → the revision; elif opinion present with cues → Stage-2 polarity; else Stage-2 polarity
or neutral. (`decided_by` observed on Ahmed: polarity_kept 58, intent_no_opinion 23,
pragmatic_revision 3, fallback 0.)

---

## G. Sequential v2 forward-pragmatics prompts — `src/prompts/sequential_sentiment_v2_prompts.py`
*(Opt-in `sequential_sentiment_v2`; pragmatics moved upstream as structured features, polarity
decided once and feature-aware, no prior label shown.)*

### G.1 Stage 1 — Intent (lean opinion-existence) (`INTENT_V2_SYSTEM_PROMPT`)
```text
You decide ONE thing: does the AUTHOR express their own evaluative opinion in this short
message? You do NOT assign sentiment polarity and you do NOT analyze sarcasm here.

  opinion_expressed:
    true    = the author gives their own evaluation (like/dislike, praise/criticism,
              approval/disapproval).
    false   = no author evaluation (a neutral question, a factual/plot description, a
              relayed/quoted opinion, a request or advice, or a bare mention/reference).
    unclear = genuinely ambiguous.

Base it on what the author is DOING with the message, not on the presence of emotional
words alone. Handle code-switched / mixed-language and informal text.

RULES:
A. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
B. Exactly these keys:
   - "opinion_expressed" : true, false, or the string "unclear"
   - "target"            : string or null (coarsely, what the message is about)
   - "confidence"        : float 0.0-1.0 (certainty in opinion_expressed)
   - "evidence"          : array of 0-5 short quoted spans

OUTPUT FORMAT:
{"opinion_expressed": true, "target": "<string or null>", "confidence": 0.0, "evidence": ["<span>"]}
```

### G.2 Stage 2 — Pragmatic feature extractor (no label) (`PRAGMATIC_FEATURES_SYSTEM_PROMPT`)
```text
You extract structured PRAGMATIC FEATURES of a short message for a downstream sentiment
decision. You do NOT output a sentiment label (positive/negative/neutral) here — only
features. A later step will decide the polarity using your features.

Determine each feature:
- speech_act: evaluate | describe | ask | advise | quote | other
- target: what any stance is about (string, or null)
- target_attribution: author | other | none  (is the stance the AUTHOR's own, someone
  ELSE's that is being reported/quoted, or NONE)
- use_vs_mention: use | mention | platform_meta
    use           = emotional/entity language is USED to express the author's evaluation
    mention       = something is named/referred to (a title, brand, entity, or another
                    person's view) without the author evaluating it
    platform_meta = talk ABOUT the platform/interface/actions (likes, blocks, follows,
                    comments, shares, trending, clips, lyrics, buttons, other users)
- platform_meta: true|false  (true iff the message is platform/meta as above)
- description_vs_evaluation: evaluation | description | mixed
- sarcasm_or_irony: true|false  (does the author mean the OPPOSITE of the literal words:
  mock praise, ironic complaint, exaggerated fake enthusiasm)
- implicit_stance: positive | negative | none  (a stance IMPLIED but not stated outright,
  e.g. implicit praise/insult, a rhetorical question carrying judgment)
- stance_strength: none | weak | moderate | strong

Judge the AUTHOR, not the sentiment of a quoted/mentioned/described thing. Base it on what
the author is DOING, not on isolated emotional words. Handle code-switched / informal text.

RULES:
A. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
B. Do NOT include any positive/negative/neutral label. Features only.
C. Exactly these keys:
   "speech_act", "target", "target_attribution", "use_vs_mention", "platform_meta",
   "description_vs_evaluation", "sarcasm_or_irony", "implicit_stance", "stance_strength",
   "confidence" (float 0.0-1.0), "evidence" (array of 0-5 spans)

OUTPUT FORMAT:
{"speech_act": "evaluate", "target": "<string or null>", "target_attribution": "author", "use_vs_mention": "use", "platform_meta": false, "description_vs_evaluation": "evaluation", "sarcasm_or_irony": false, "implicit_stance": "none", "stance_strength": "moderate", "confidence": 0.0, "evidence": ["<span>"]}
```

### G.3 Stage 3 — Feature-aware polarity resolver (`POLARITY_RESOLVER_SYSTEM_PROMPT`)
```text
You assign the final sentiment polarity, using the message, an INTENT judgment, and a set
of PRAGMATIC FEATURES. This is the ONLY step that outputs a label. You are NOT reviewing or
ratifying a previous label — decide fresh, informed by the features.

Use the features as evidence:
- If the features indicate no author evaluation (opinion_expressed false, use_vs_mention
  mention/platform_meta, description_vs_evaluation description, implicit_stance none) →
  usually "neutral".
- If sarcasm_or_irony is true, the intended stance is typically the OPPOSITE of the literal
  wording — resolve to the INTENDED polarity, not the literal one.
- Otherwise resolve positive/negative/neutral from the author's expressed or implicit
  stance, weighting stance_strength; handle negation, intensifiers, and mixed polarity
  (choose the DOMINANT side; if truly balanced with no dominant side → neutral).

Judge the AUTHOR's stance, not the sentiment of a quoted/mentioned/described thing. You may
disagree with a feature if the text plainly contradicts it. Handle code-switched / informal
text.

RULES:
A. Choose EXACTLY ONE label from the allowed list provided. Do NOT invent labels.
B. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
C. Exactly these keys:
   - "label"         : string - one of the allowed labels, copied verbatim
   - "confidence"    : float 0.0-1.0
   - "used_features" : array - which pragmatic feature names drove the decision (may be [])
   - "reasoning"     : string - one or two sentences
   - "evidence"      : array of 0-5 short quoted spans

OUTPUT FORMAT:
{"label": "<label>", "confidence": 0.0, "used_features": ["sarcasm_or_irony"], "reasoning": "<one sentence>", "evidence": ["<span>"]}
```

---

## H. Deliberation agent — `src/prompts/deliberation_prompt.py`
*(Optional cross-talk before consensus in full_agentic; deliberation stayed off in the reported
sentiment runs.)*

### H.1 `SYSTEM_PROMPT`
```text
You are a deliberation engine reviewing the outputs of multiple specialist text classification agents.

RULES — follow every rule exactly:
1. Read the agent votes below and determine whether they agree or conflict.
2. Choose EXACTLY ONE label from the allowed list as your recommendation.
   Do NOT invent, abbreviate, or paraphrase a label.
3. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
4. The JSON must contain exactly these four keys:
   - "recommended_label" : string — must be one of the allowed labels, copied verbatim
   - "confidence"        : float  — your certainty, between 0.0 and 1.0 (e.g. 0.82)
   - "justification"     : string — one or two sentences explaining your conclusion
   - "mode"              : string — use "recommendation" when you favour one label clearly,
                                    "justification" when you are explaining why the majority view holds

OUTPUT FORMAT (copy this structure exactly):
{"recommended_label": "<label>", "confidence": <0.0–1.0>, "justification": "<sentence>", "mode": "recommendation"}
```
**User template:** injects `AGENT VOTES` (one line per specialist: label, confidence, note).

---

## I. Explainability agent — `src/prompts/llm_explainability_prompt.py`
*(Runs on both the fast and slow paths to produce the final rationale + audit; not a classifier.)*

### I.1 `SYSTEM_PROMPT`
```text
You are an explainability specialist in a multi-agent text classification system.

Your role is to synthesize the outputs of multiple agents and produce a clear, concise explanation of why the pipeline classified the text with the final label.

RULES — follow every rule exactly:
1. Write a single-sentence summary explaining the final classification decision.
2. List the key pieces of evidence (agent votes, text tokens, or confidence values) that supported the final label.
3. List any caveats: agents that disagreed, low-confidence votes, or notable uncertainties.
   If there are no caveats, return an empty array for "caveats".
4. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
5. The JSON must contain exactly these three keys:
   - "summary"  : string — one sentence
   - "evidence" : array  — 1–5 short strings
   - "caveats"  : array  — 0–3 short strings (empty array if none)

OUTPUT FORMAT (copy this structure exactly):
{"summary": "<one sentence>", "evidence": ["<item>"], "caveats": []}
```

---

## J. Primary-signal block (Fix #3) — `src/prompts/_primary_block.py`
*(Off by default. When `agents_use_primary_signal` is on, this block is appended to each
specialist's user prompt. The 2×2 ablations showed it only raises anchoring, so it is OFF in every
current-best config.)* Rendered block (label-generic; `$kind` is the agent's role word):
```text
PRIMARY MODEL SIGNAL (context only — you are an independent $kind adjudicator):
  predicted label  : <label>
  confidence       : <0.00>
  top-2 labels     : <lbl> (<p>), <lbl> (<p>)
  full distribution: <lbl>=<p>, ...
The primary may be wrong — especially when its confidence is low or its top-2 are close. Do your
own $kind analysis of the text FIRST. Use this signal as context only; agree only if the text
evidence supports the primary's label, and choose a different allowed label if the evidence points
elsewhere. Do NOT simply copy the primary.
```

---

## K. Variant resolver — `src/prompts/_sentiment_variant.py`
Selects the active prompt variant from `SENTIMENT_PROMPT_VARIANT` /
`--sentiment_prompt_variant`. Recognised values: **`default`** (byte-identical originals),
**`semantic_v1`** (role-refined Lexical/Logic/Contextual), **`semantic_v3_pragmatic_contextual`**
(Lexical/Logic keep `semantic_v1`; Contextual → Pragmatic Reasoner), **`semantic_v2_disambig`**
(platform-relationship + description-vs-evaluation across Lexical/Polarity/Contextual/Intent).
An unrecognised value raises `ValueError` (fails loudly). Agent-set / gate selection is a separate
axis via `--sentiment_agent_variant` (e.g. `lexical_polarity_contextual`,
`lexical_polarity_contextual_intent_gate` (G), `..._selective_gate` (G2), `intent_polarity_contextual`
(F), `sequential_sentiment_v1`, `sequential_sentiment_v2`).

---

# PART V — MASTER-REFERENCE SECTIONS (for the thesis Experiments & Results chapter)

### V.0 Coverage and field mapping
This registry is the complete reconstruction of the Multi-Agent BERT research: **all 32
experiments, chronological, none merged, none skipped**, including every negative result and
failed hypothesis. The thesis template's 14 per-experiment fields map onto each Part-I entry as:
1 Experiment Name → *Experiment Name/ID* · 2 Chronological Position → *Chronological Order* ·
3 Research Question → *Research Question* · 4 Motivation → *Motivation* · 5 Hypothesis →
*Hypothesis* · 6 Previous Baseline → *Baseline* · 7 Architecture → *Architecture* +
*Implementation Changes* (routing/consensus/agents/prompts/thresholds) · 8 Dataset → *Dataset* ·
9 Experimental Configuration → *Model* + *Parameters* · 10 Results → *Evaluation Metrics*
(accuracy, macro/weighted F1, per-class, confusion, McNemar, escalation W→C/C→W/net, cost) ·
11 Interpretation → *Main Findings* + *Why It Succeeded or Failed* · 12 Scientific Contribution →
*Main Findings* + *Strengths* · 13 Final Decision → *Decision* · 14 Influence → *Influence on
Later Work*.

Requested topics and where they live: real-data (1–6, 10), generated-data (7–9), augmentation
(11), threshold studies (4, 6, 14), consensus studies (5, 6, 23), prompt engineering (16, 22, 31),
agent redesign (17–19), semantic_v1 (16), semantic_v2 (= `semantic_v2_disambig`, 31), semantic_v3
(= pragmatic Contextual v3, 22), Designs A–G2 (16–21), IntentGate (20–21, 32), sequential (24–25,
27), GPT-4o-mini (3–27), GPT-4.1-mini (28–32), topic (12), C3 (8–9, 26–27), E0 (10), Ahmed
(13–32), gate ablations (32), stronger-model (28–32). **Note: no "generated pretraining"
experiment exists** — generated data was used for standalone *fine-tuning* (C-series) and
*augmentation* (E-series); pretraining was never performed, and no such result should appear in
the thesis.

---

### V.1 Architecture Evolution

**Stage 0 — Primary-only transformer (D1–D2).** The system began as a single fine-tuned
classifier: mBERT (0.7971), replaced by XLM-R (0.8240) after it recovered the weak negative
class. This primary — label + confidence + probability distribution — remained the fixed
foundation of every later design.

**Stage 1 — Router + deterministic/mock agents (D1–D2 mode comparisons).** A confidence
threshold router escalated low-confidence samples to a panel of Lexical/Logic/Contextual agents.
With deterministic or mock-LLM agents, escalation *hurt* (agents weaker than the primary they
overrode). Redesign trigger: the agents, not the routing, were the bottleneck.

**Stage 2 — Real LLM agents (D3–D4).** Swapping in GPT-4o-mini flipped the sign: +1.6–2.0 pts on
both primaries, concentrated on the escalated subset and the negative class; raising the
threshold to 0.8–0.9 kept helping. Escalation-to-capable-agents became the core architecture.

**Stage 3 — Consensus correctness (D5–D6).** An audit produced four fixes: generic
task-config-driven prompts, an abstain fallback (no label-0 bias), a **primary-aware weighted
vote** (Fix #2, w_primary=1.0 — kept), and a primary-signal prompt block (Fix #3 — rejected: it
only induces anchoring). Two 2×2 ablations locked the defaults and the best XLM-R result
(0.8509). Redesign trigger for the next stage: a much stronger primary arrived.

**Stage 4 — Frozen-primary abstraction (D13–D14).** Ahmed's external model (0.9254) was plugged
in via `PrecomputedPrimaryClassifier`, decoupling the agentic layer from any particular backbone
and exposing the strong-primary regime, where the default trio went net-negative (−4). The
cross-regime synthesis (D15) yielded the governing law — Δ ≈ (agent-ceiling ≈0.75 −
primary-on-escalated) × escalation-rate — which framed every later redesign.

**Stage 5 — Prompt role refinement: semantic_v1 (D16, Design A).** Role-refined prompts
decorrelated the panel (92%→84.5%) and halved the harm (−4→−2) but could not cross the primary.
Trigger: Logic measured weakest (0.679–0.690) and most redundant (~0.89 with Lexical).

**Stage 6 — Specialist decomposition (D17–D19, Designs B/C/D/F).** Replacing Logic with a
dedicated **Polarity** decider (C) produced the first configuration above the primary (0.9267,
net +1); B (Pol+Ctx) was safe-but-added-nothing; D (4 votes) was dominated; F (drop Lexical)
proved the evidence agent is load-bearing (net −3, over-neutralization). Trigger for Stage 7: the
persistent "unlike/dislike" meta-comment cluster that every *voting* design missed.

**Stage 7 — Intent: voter → veto → selective veto (D18, D20, D21).** As a 4th vote (E), Intent's
pragmatic signal was suppressed 12/12 by the correlated bloc. Re-cast as a **non-voting,
domain-restricted veto** (G — block an unsupported polar override of a neutral primary), the same
signal was missed 0/12: G became the lead (0.9279, net +2), fixing the meta-comment cluster. A
prompt-level **selective** gate (G2) tied G at 4o-mini and was retired — then revived at Stage 10.

**Stage 8 — Component upgrade without system gain (D22, v3).** The pragmatic-reasoner Contextual
improved the agent (+0.024, best in panel) but not the system — component gains are diluted by a
correlated ensemble at a parity-strength primary ("conservation of difficulty").

**Stage 9 — Sequential topologies (D24–D25, D27).** Staged reasoning (v1 anchored-review; v2
forward-pragmatics) attacked the correlation root cause. v1 reproduced the primary (net 0/−1,
Stage 3 nearly inert); v2 was actively harmful on the strong primary (−11) and helpful on the
weak one (+47) — but still lost to parallel G (+53), whose veto caps breakage (12 vs 28).
Conclusion: staged reasoning does not beat parallel-voting-plus-veto in either regime.

**Stage 10 — Model strength × gate strength (D28–D32).** GPT-4.1-mini recovered the
compliance/obscured-cue slice (4/18, zero from noise) → full run 0.9291 (+3, n.s.). Diagnosis: the
stronger model makes the *full* gate over-veto (one-directional neutralizer). The gate ablation
resolved it: **G2 (selective gate) @ gpt-4.1-mini = 0.9303/0.9262 — the final architecture** for
the strong-primary regime, establishing that gate aggressiveness must scale inversely with model
strength. A prompt-disambiguation alternative (semantic_v2_disambig) failed (−1 to −4),
confirming prompts are not the lever.

**Final architecture.** Primary (any model exposing label/confidence/probabilities) → per-primary
calibrated threshold router → Lexical + Polarity + Contextual specialists (semantic_v1 prompts) →
primary-aware confidence-weighted consensus (w_primary=1.0, signal off) → selective IntentGate
veto (strong model) / full IntentGate (weaker model) → explainability agent.

---

### V.2 Research Timeline (dated)

| Date (2026) | Events |
|---|---|
| 06-06 | Exp A mBERT (0.7971) and XLM-R (0.8240) references; real-LLM pilot @0.6 (XLM-R 0.8399) |
| 06-09 | Real-LLM threshold sweep (peak 0.8460 @0.8); mBERT rows contaminated by outage |
| 06-10 | mBERT re-run blocked — checkpoints deleted; backup policy adopted |
| 06-11→13 | Prompt/logic audit; Fixes #1/1b/2/3; 2×2 ablations @0.8 and @0.9; defaults locked; **best XLM-R 0.8509/0.8401**; C1-240 transfer pilot (0.5905) |
| 06-20 | Topic T1/T2 (ARENTC, 0.9946/0.9947; agents net +3/−6 = noise) |
| 06-21 | C2/C3 3-seed stability (0.6500/0.6695); "480>960" retracted |
| 06-23 | C3-960 seed-456 full_agentic: 0.6956→0.7543 (**+0.059**) |
| 06-25 | Augmentation: E0 (0.8533), E3 (0.8411, −0.012), LR, ratio sweep, domain-mismatch diagnosis |
| 06-27 | Ahmed baseline (0.9254); frozen-primary full_agentic (net −4, threshold 0.7); agent-behaviour comparison → **primary-strength curve** |
| 06-30 | semantic_v1 ablation (net −2, decorrelation); Polarity redesign proposal; Design C first run |
| 07-01 | Design ablation A/B/C/D (**C 0.9267 net +1**); E (tie); F (−3); **G IntentGate 0.9279 net +2**; G2 (tie, retired); v3 (tie); gap analysis; consensus investigation (all re-fusion rules fail); sequential v1 (−1) and v2 (−11) on Ahmed |
| 07-02 | **G on C3: +53, p≪0.001**; seq-v2 on C3 (+47 < G); 4.1-mini diagnostic (4/18); G@4.1 (0.9291, n.s.); why-broke trace (gate over-veto); disambig negative (−1..−4); **gate ablation: G2@4.1-mini = 0.9303/0.9262 — current best**; Ahmed tuning stopped |

---

### V.3 Major Scientific Findings

**Primary-strength curve.** The agentic layer's net effect is governed by
Δ ≈ (agent-ceiling − primary-accuracy-on-escalated) × escalation-rate. Measured: C3 0.54 → +0.059;
EESA 0.56 → +0.027; Ahmed 0.75 → −0.005; topic 0.99 → ±noise. Only primary strength flips the sign.

**Agent ceiling.** Final consensus accuracy on hard escalated code-switched cases is ~0.67–0.77
regardless of design or prompt — a property of (router-selected subset × shared base model), with
~⅓ label-convention and ~⅔ cue-less implicit pragmatics forming a Bayes-irreducible floor
(~14/18 residual Ahmed errors had the truth in no agent).

**Generated-data findings.** SwitchLingua generated data carries real sentiment signal: standalone
training scales 240→480→960 (0.59→0.65→0.67, 3-seed) and the agentic layer rescues it most where
weakest (0.696→0.754/0.760). Single-seed comparisons at this scale are unreliable (the retracted
"480>960").

**Augmentation findings.** Mixed into real EESA, generated data does not help (E0 0.8533 → E3
0.8411, −0.012), is harmful when it dominates (−0.034 at 80% share), and is within ±0.02 noise at
20–50% share — a domain/register mismatch (CMI 41 vs 24, ~10% vocab overlap), not a generation
defect. Value = standalone, not augmentation.

**Gate findings.** A domain-restricted, **non-voting veto** is the only aggregation change that
ever helped: the same signal is 12/12 suppressed as a vote, 0/12 missed as a veto. The gate's
downside is bounded (it never forces a flip) but it can block correct rescues when the primary is
wrong-neutral; **gate aggressiveness must scale inversely with model strength** (full gate
over-vetoes at 4.1-mini; selective gate is the sweet spot → 0.9303).

**Model-strength findings.** A stronger agent model recovers the compliance/obscured-cue slice
(4/18, zero attributable to noise) but not the deep cultural-implicit floor; the net bump on a
strong primary is small and non-significant (0.9279→0.9291→0.9303, McNemar p≈0.37–0.47). Stronger
models are also *more compliant*, so lossy prompt heuristics get executed more faithfully — fixing
and breaking in equal measure (the "wash").

**Prompt-engineering findings.** Prompts can decorrelate (92%→84.5%) and re-target errors, but
cannot cross the ceiling: semantic_v1 (−4→−2), v3 (component up, system flat), disambig (worse,
over-instruction noise). Four independent prompt/topology interventions failed on the strong
primary. The primary-signal block only induces anchoring (+3–7 pts copy-rate, no accuracy).

**Consensus findings.** Consensus loss is real (the panel discards 4–11 correct answers per 84;
oracle 0.80–0.88 vs ~0.75) and structural: when one agent is uniquely right, the correlated bloc
outvotes it. Every simple re-fusion rule (guards, w_primary sweep, minority-trust, role-priority)
ties or loses (−1 to −4). Primary-aware voting (Fix #2) is protective and a big win for weak
agents (+0.064 paper_style). Agent self-confidences are uncalibrated and uninformative on the hard
subset. Learned/calibrated consensus is blocked by labels-only captures and test-only data.

**Sequential-reasoning findings.** Staged pipelines restructure *how* the decision is reached but
not *how well*: v1 (anchored) reproduces the primary; v2 (forward) is harmful where the primary is
right (−11) and helpful where it is wrong (+47) — net effect scales with intervention rate — and
still loses to parallel G (+53) for lack of a veto brake (28 vs 12 breakages).

**Negative findings (complete list).** Mock agents hurt; Fix #3 anchoring; the "480>960" seed
artifact; augmentation neutral-to-harmful; Design A net −2, D dominated, F −3
(over-neutralization); G2@4o-mini lost 3/4 platform blocks; all offline re-fusion rules ≤ G;
sequential v1 −1, v2 −11 (Ahmed); disambig −1..−4; topic agents net −6 (T2); C@4.1 (no gate) worse
than gated; no strong-primary gain is statistically significant vs primary_only.

---

### V.4 Current State of the Research

**Best architecture (overall).** Confidence-routed hybrid: primary classifier + escalation to
Lexical + Polarity + Contextual (semantic_v1) + primary-aware consensus (w_primary=1.0, signal
off) + IntentGate veto + explainability. Topic uses the default Lexical/Logical/Contextual trio,
primary_only recommended (agents are noise at 0.99).

**Best strong-primary configuration.** **Design G2 @ gpt-4.1-mini** (selective IntentGate),
threshold 0.7 on the Ahmed frozen primary: **0.9303 acc / 0.9262 macro F1** — best on record,
first past 0.930; not statistically significant vs primary_only (p≈0.37) — the honest reading is
that the strong-primary regime is at its ceiling and primary_only (0.9254) is the safe fallback.

**Best weak-primary configuration.** **Design G @ gpt-4o-mini**, threshold 0.90 on the C3
generated primary: **0.7604 / 0.7469**, +0.065 over primary, escalated 0.541→0.771, net +53,
McNemar p≪0.001 — the regime where the architecture demonstrably earns its keep.

**Best training strategy.** Fresh `xlm-roberta-base`, **Adafactor** (+0.029 over AdamW), fp16 +
gradient checkpointing, eff. batch 16, max_len 256, 4 epochs, ≥3 seeds reported as mean±std;
best trained primary = E0 EESA-only **0.8533/0.8409**.

**Best generated-data strategy.** Use SwitchLingua data **standalone** (scales with size; agents
rescue it), never as naive augmentation of a different-domain corpus; if augmenting at all, keep
generated share ≤50% (harm appears when it dominates). No pretraining use exists or is claimed.

**Remaining open research questions.** (1) G/G2 @ gpt-4.1-mini on the weak C3 primary; (2) a v2.1
sequential with an IntentGate-style veto; (3) confidence-calibrated arbitration (requires a new
capture pass that serializes confidences + a dev split); (4) learned meta-consensus /
dev-trained selector (blocked by n≈80 and test-leakage); (5) heterogeneous-model panels to
decorrelate errors; (6) a robust >0.93 on the strong primary — realistically requires retraining
the primary (Ahmed's own sentiment-hint route, 0.9548), not agent-layer work; (7) the deep
cultural-implicit stance floor (slur-as-question, "you are no one"); (8) a ≥5-seed augmentation
ratio sweep to bound the small (±0.02) effects.
