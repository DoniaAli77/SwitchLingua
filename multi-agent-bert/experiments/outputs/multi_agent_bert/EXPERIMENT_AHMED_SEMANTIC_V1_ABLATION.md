# B1 — Ahmed Frozen-Primary Full-Agentic with `semantic_v1` Sentiment Prompts

A/B test of the role-refined `semantic_v1` sentiment prompts against the original
prompts, on the Ahmed frozen-primary full_agentic pipeline. **Only the system-prompt
variant changed** (`--sentiment_prompt_variant semantic_v1`); every other setting is
identical to the previous Ahmed run. No training, no generation, no architecture/
consensus/router change. Date: 2026-06-30.

## Setup (identical to the previous Ahmed run, except the prompt variant)
- Frozen primary = `PrecomputedPrimaryClassifier` on Ahmed's aligned predictions
  (`ahmed_eesa_test_predictions_aligned.csv`), EESA test (818).
- active_task = sentiment_classification · pipeline_mode = full_agentic ·
  threshold = 0.7 · Fix-2 primary-aware consensus ON (w_primary = 1.0) ·
  agents_use_primary_signal = false · LLM = GPT-4o-mini (temperature 0.0).
- **Only change:** `--sentiment_prompt_variant semantic_v1`.
- Config verified identical via the run that reproduced the original (same
  `src/config/default.yaml`, threshold, weight). Per-agent outputs captured by a
  deterministic (temp 0) re-run of the 84 escalated samples — its transitions match
  the headline run exactly (CW 14 / WC 12 / escalated 0.7262), validating the capture.

---

## 1. Primary_only (frozen primary) — reproduced
accuracy **0.9254** · macro F1 **0.9207**. Identical to the baseline by construction:
the variant only affects escalated specialist agents, never the frozen primary.

## 2. semantic_v1 full_agentic
accuracy **0.9230** · macro F1 **0.9183** · weighted F1 **0.9228** (0 connection / 0
quota errors in the headline run).

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| positive | 0.941 | 0.964 | 0.952 | 363 |
| negative | 0.937 | 0.898 | 0.917 | 197 |
| neutral | 0.887 | 0.884 | 0.885 | 258 |

## 3–7. Escalation + agent effect (84 escalated)
| metric | value |
|---|---|
| escalated | **84 / 818 (10.3%)** |
| escalated-only accuracy (final) | **0.7262** (61/84) |
| **wrong→correct** | **12** |
| **correct→wrong** | **14** |
| **net change** | **−2** |

(net −2 / 818 = −0.0024 overall, i.e. 0.9254 → 0.9230.)

## 8. Agent agreement — DROPPED vs old 92% (decorrelation achieved)
| | old (default) | **semantic_v1** | Δ |
|---|---|---|---|
| all 3 agents agree | 77/84 = **91.7%** | 71/84 = **84.5%** | **−7.2 pts** |
| pairwise lexical–logic | 0.964 | **0.893** | −0.071 |
| pairwise lexical–contextual | 0.940 | **0.905** | −0.036 |
| pairwise logic–contextual | 0.929 | **0.893** | −0.036 |

The agents now **disagree more often** — the explicit design goal. The three reasoning
modes (lexical evidence / target attribution / whole-message intent) pull apart instead
of voting as one correlated bloc.

## Per-agent accuracy on the 84 escalated (item: did Logic improve? is Contextual best?)
| agent | old | **semantic_v1** | Δ |
|---|---|---|---|
| lexical | 0.7143 | **0.7262** | +0.012 |
| **logic** | 0.6786 | **0.6905** | **+0.012 (improved)** |
| **contextual** | 0.7262 | **0.7500** | **+0.024 (still best)** |
| final consensus | 0.7024 | **0.7262** | +0.024 |
| (Ahmed primary) | 0.7500 | 0.7500 | — |

- **Logic improved** (0.679 → 0.690) but remains the **weakest** agent.
- **Contextual still performs best** (0.750) — now **tied with the Ahmed primary**, and
  clearly ahead of lexical/logic.
- Final consensus rose +0.024, the same lift as contextual — the panel is now led more
  by the (better) contextual reasoning and less dragged by the correlated bloc.

## 9. Break matrix (correct→wrong), old vs semantic_v1
| break (true→final) | old | **semantic_v1** |
|---|---|---|
| neutral→negative | 7 | **5** |
| neutral→positive | 4 | **4** |
| negative→neutral | 2 | **3** |
| positive→neutral | 2 | **2** |
| **total breaks** | **15** | **14** |

The dominant failure (neutral→negative, surface-cue over-reading) **fell 7→5**.

## 10. Did artifact / literalism errors decrease? — YES (partially)
Per-sample old↔new comparison of the 84 escalated:

| transition vs old | count | meaning |
|---|---|---|
| **fixed** (old correct→wrong ⇒ now correct→correct) | **2** | literalism breaks the agents stopped making |
| **new rescue** (old wrong→wrong ⇒ now wrong→correct) | **1** | a case the refined agents now fix |
| newly broken (old correct→correct ⇒ now correct→wrong) | **1** | one regression (over-applied "description≠evaluation") |
| still broken (correct→wrong in both) | **13** | unchanged breaks |
| lost rescue (old wrong→correct ⇒ now wrong→wrong) | **0** | no previously-good fixes were lost |

Net = +2 fixed/rescued − 1 newly-broken = the **net −4 → −2** improvement, plus the +1
W→C. **The two fixes are exactly the targeted literalism patterns**, and the way they
were fixed (agents now disagreeing, target resolved) confirms the mechanism, not luck.

### Examples FIXED by semantic_v1
- **`ahmed-eesa-00362`** (true *neutral*): *"…تذويب الجثة في الحمام و مسلسل Breaking Bad
  لما فشل جيسي…"* — a **plot description** (Breaking Bad: dissolving a body, "Jesse
  failed"). Old: lexical+logic both **negative** (reading *failed/body* as sentiment) →
  final **negative ✗**. New: **all three neutral** → **neutral ✓**. *(description-vs-
  evaluation + plot-words guidance, mainly Logic.)*
- **`ahmed-eesa-00363`** (true *neutral*): *"الناس اللي عامله dislike دول ايه؟"* ("the
  people who hit *dislike* — who are they?"). Old: lexical+logic **negative** (UI word
  *dislike* + third party) → **negative ✗**. New: lexical still flags negative (weak
  cue) but **logic → neutral** (target = *other people*, not the author) and contextual
  neutral → **neutral ✓**. *(sentiment-target attribution + platform-word-as-weak-cue;
  note the agents now disagree — the decorrelation working.)*

### Examples STILL broken
- **`ahmed-eesa-00203`** (true *neutral*): *"هو اللي عامل unlike دة عمله علي اساس اية!!"*
  and **`ahmed-eesa-00245`**: *"الناس الى عاملة unlike دى ازااااى يعنى!!!"* — indignant
  meta-comments about other users' *unlike*. **All three still say negative** in both
  variants. The platform-word guidance did **not** override the author's genuinely
  irritated tone here; these are borderline (arguably mild-negative) and overlap the
  label-convention-disagreement set.
- **`ahmed-eesa-00008`** (*"هوة معتز مسعود دة gay?"*) and **`ahmed-eesa-00097`** (*"…الفيديو
  مفيهوش ولا راجل including Mohamed Ramadan 😂"*): **implicit insult/mockery** that Ahmed
  labels negative but the agents read as neutral — **unchanged**. The contextual sarcasm/
  insult guidance did not catch these hard implicit cases.
- **`ahmed-eesa-00239`** (a *Dodge* car mention read positive): still broken, but
  contextual **flipped to neutral** (partial decorrelation) while lexical/logic held.

### The 1 regression
- **`ahmed-eesa-00193`** (true *negative*): a complaint about an obvious video *"cut"*.
  Old agents negative → correct; new agents all **neutral** (over-applied "this is a
  description, not an evaluation") → **wrong**. This is risk #2 (over-conservatism)
  materialising in exactly one case.

## 11. Cost and calls
- **semantic_v1 headline run:** **336 LLM calls**, 262,404 tokens (prompt 239,254 /
  completion 23,150), **$0.0498** (gpt-4o-mini). Slightly above the old ~$0.043 because
  the `semantic_v1` system prompt is longer (more prompt tokens per call).
- Per-agent capture (analysis instrumentation, deterministic re-run of the 84 escalated)
  spent a comparable amount and hit the org's daily request cap (RPD 10000) mid-run; the
  retry/backoff absorbed it and all 84 completed with 0 `None`.

## 12. Comparison table — old vs semantic_v1
| metric | primary_only | old full_agentic | **semantic_v1** | direction |
|---|---|---|---|---|
| accuracy | **0.9254** | 0.9205 | **0.9230** | ↑ better |
| macro F1 | 0.9207 | 0.9153 | **0.9183** | ↑ better |
| weighted F1 | 0.9254 | 0.9202 | **0.9228** | ↑ better |
| escalated count | — | 84 | 84 | = |
| escalated-only acc | 0.750 | 0.702 | **0.726** | ↑ better |
| wrong→correct | — | 11 | **12** | ↑ better |
| correct→wrong | — | 15 | **14** | ↑ better |
| net change | — | −4 | **−2** | ↑ better |
| neutral→negative breaks | — | 7 | **5** | ↑ better |
| neutral→positive breaks | — | 4 | 4 | = |
| all-3 agent agreement | — | 91.7% | **84.5%** | ↓ (intended) |
| logic acc (escalated) | — | 0.679 | **0.690** | ↑ |
| contextual acc (escalated) | — | 0.726 | **0.750** | ↑ (best) |
| cost / calls | — | ~$0.043 / 336 | **$0.0498 / 336** | — |

---

## Interpretation (the three points)

**1. `semantic_v1` improved the old full_agentic result.** Every transition metric moved
the right way: accuracy **0.9205 → 0.9230**, escalated accuracy **0.702 → 0.726**, net
**−4 → −2**, wrong→correct **11 → 12**, correct→wrong **15 → 14**, neutral→negative breaks
**7 → 5**. The improvement is *mechanistic, not incidental*: agent agreement fell
91.7% → 84.5% (the agents now reason distinctly), the two recovered breaks are exactly
the targeted literalism patterns (plot-description and third-party "dislike"), and they
were fixed by the intended agent (Logic resolving the sentiment target / description-vs-
evaluation).

**2. But `semantic_v1` still does not beat Ahmed primary_only.** primary_only **0.9254**
vs semantic_v1 **0.9230** — the agentic layer is still **net −2** on the escalated subset
and below the frozen primary. The refined agents (consensus 0.726) remain **below Ahmed's
0.750** on these hard escalated cases, so a primary-aware consensus that still lets them
override loses a little ground.

**3. Therefore prompt refinement reduced surface-cue literalism but did not overcome the
primary-strength gap.** `semantic_v1` did what it was designed to do — cut the literalism
breaks and decorrelate the panel — closing **half** the agentic gap (−4 → −2). But the
remaining deficit is **not** a prompt problem: it is the *agent-ceiling* effect. On hard
code-switched cases the LLM agents top out around ~0.73–0.75, which is at/just below
Ahmed's 0.75, so even better-reasoned agents cannot add net value over a primary this
strong. The 13 still-broken cases are dominated by (a) borderline label-convention
disagreements (indignant "unlike" meta-comments) and (b) hard implicit insult/sarcasm —
neither of which a general prompt refinement is expected to fully solve. The path to
net-positive on a strong primary is therefore **not more prompt tuning** but protecting
the primary (non-overriding consensus / no-escalate), consistent with the consensus-
simulation and router-selectability findings.

## Artifacts
- Headline run: `experiment_ahmed_semantic_v1/full_agentic_th07_semantic_v1/` (metrics,
  predictions, `__llm_usage.json`, `run.log`).
- Per-agent capture: `experiment_ahmed_semantic_v1/error_attribution/attribution_table.{csv,json}`
  + `error_attribution_capture.log`; driver `scripts/ahmed_semantic_v1_attribution.py`.
- Old baseline for comparison: `experiment_ahmed_frozen_primary/error_attribution/`.
