# sequential_sentiment_v1 — Minimal 3-Stage Pipeline: Prompts + Controller (Design Only)

**Design only. No code, no runs, no LLM calls.** Exact stage prompts and the deterministic
controller for a minimal sequential sentiment pipeline. General sentiment analysis — no dataset,
platform, author, or benchmark named in any prompt. Final labels: **positive / negative / neutral**.
Every stage returns **JSON only**; every stage carries a **confidence** for later calibration; all
intermediate outputs are persisted. Date: 2026-07-01.

## Architecture (3 LLM stages + 1 deterministic controller)

```
text ──▶ Stage 1: Intent ──▶ Stage 2: Polarity ──▶ Stage 3: Pragmatic ──▶ Stage 4: Controller ──▶ final
              (JSON)            (text + Intent)      (text+Intent+Polarity)     (deterministic)
```

**Why no separate Lexical stage in v1:** lower cost, one fewer serial hop, and Stage 2 (Polarity)
already extracts cue evidence internally (its `evidence` field). A dedicated Lexical extractor is a
**v2 option** only if Stage 2's cue handling proves weak in the stage-level error analysis (§5).

---

## 1. Stage 1 — Intent / opinion-expression detector

**Input:** original text only.
**Model call params (proposed):** temperature 0, `response_format = json_object`, max_tokens ~300.

### System prompt
```
You analyze a short social-media style message and decide ONLY whether the author is expressing
their own evaluative opinion, and about what. You do NOT decide sentiment polarity here.

Determine four things:

1. opinion_expressed — Does the AUTHOR express their own evaluative stance (a like/dislike,
   praise/criticism, approval/disapproval)?
     true    = the author gives their own evaluation.
     false   = no evaluation by the author (a neutral question, a factual/plot description, a
               relayed/quoted opinion, a request or advice, or talk that only MENTIONS or reports
               something without evaluating it).
     unclear = genuinely ambiguous.

2. target — What the opinion (if any) is about: an entity, person, product, topic, or event.
   Use null if there is no clear target or no opinion.

3. speech_act — The primary communicative act:
     "evaluate"  = giving a judgment/opinion
     "describe"  = stating facts, plot, or content without judging
     "ask"       = asking a question
     "advise"    = giving advice, a request, or a call to action
     "quote"     = relaying/quoting someone else's words or opinion
     "other"

4. use_vs_mention — Is the emotional/entity language USED to express the author's stance, or merely
   MENTIONED / referred to?
     "use"           = the author is genuinely expressing evaluation.
     "mention"       = names or refers to something (a title, brand, entity, or another person's
                       view) without the author evaluating it.
     "platform_meta" = talk ABOUT the platform/interface/actions (posting, blocking, following,
                       trending, comments) rather than about a subject the author is evaluating.

Base the decision on what the author is DOING with the message, not on the presence of emotional
words alone. A message can contain strong words yet express no author opinion (e.g. quoting,
describing, or naming something). Handle code-switched / mixed-language and informal text.

Output JSON ONLY, no prose, exactly these keys:
{
  "opinion_expressed": true | false | "unclear",
  "target": "<string or null>",
  "speech_act": "evaluate" | "describe" | "ask" | "advise" | "quote" | "other",
  "use_vs_mention": "use" | "mention" | "platform_meta",
  "confidence": <number 0.0-1.0>,   // your confidence in opinion_expressed
  "evidence": ["<short quoted span>", "..."]  // spans that drove the decision; [] if none
}
```

### User message template
```
Message:
"""
{text}
"""
Return the JSON object only.
```

---

## 2. Stage 2 — Polarity resolver

**Input:** original text + Stage 1 Intent JSON.
**Params:** temperature 0, `json_object`, max_tokens ~350.

### System prompt
```
You assign sentiment polarity to a short message, using the message and a prior INTENT analysis.

Rules:
- If intent.opinion_expressed is false, the author is usually NOT evaluating anything → prefer
  "neutral" with LOWER confidence, UNLESS the text plainly carries the author's own praise or
  insult that the intent step may have missed.
- If intent.opinion_expressed is true or "unclear", decide the polarity of the author's stance:
    "positive" = praise, liking, approval, admiration (explicit or implicit).
    "negative" = criticism, dislike, insult, disapproval (explicit or implicit).
    "neutral"  = no clear evaluative direction, or purely factual/mention/meta content.
- Handle: negation (a positive word under negation can become negative and vice-versa),
  intensifiers and elongation/repetition (strengthen but do not flip), mixed polarity (choose the
  DOMINANT stance; if truly balanced with no dominant side, "neutral" and set "mixed": true),
  explicit praise/insult, and IMPLICIT praise/insult (sarcasm-free implication, admiration, or
  put-downs with no explicit sentiment word).
- Judge the AUTHOR's stance, not the sentiment of a quoted/mentioned/described thing.
- Work with code-switched / mixed-language and informal text.

Do NOT decide sarcasm/irony here — a later step handles that. Give your best literal-plus-implicit
polarity read.

Output JSON ONLY, exactly these keys:
{
  "label": "positive" | "negative" | "neutral",
  "confidence": <number 0.0-1.0>,
  "mixed": true | false,
  "reasoning": "<one or two sentences>",
  "evidence": ["<short quoted cue span>", "..."]   // cues you used; [] if none
}
```

### User message template
```
Message:
"""
{text}
"""
Intent analysis (JSON):
{intent_json}

Return the JSON object only.
```

---

## 3. Stage 3 — Pragmatic verifier

**Input:** original text + Stage 1 Intent JSON + Stage 2 Polarity JSON.
**Params:** temperature 0, `json_object`, max_tokens ~350.

### System prompt
```
You are the final pragmatic check on a sentiment decision. You receive the message, an INTENT
analysis, and a proposed POLARITY. Decide whether to KEEP or REVISE the polarity.

Check specifically:
1. Sarcasm / irony — does the author mean the OPPOSITE of the literal words (mock praise, ironic
   complaint, exaggerated fake enthusiasm)? If so, the true stance is usually the opposite of the
   literal polarity.
2. Implicature — is there an implied stance not stated outright (implicit praise or insult,
   rhetorical questions that carry judgment)?
3. Description vs evaluation — if the text only DESCRIBES, MENTIONS, quotes, or asks without the
   author evaluating, the stance is "neutral" even if emotional words appear.
4. Do NOT over-neutralize: if the author clearly praises or criticizes (explicitly or implicitly),
   KEEP that positive/negative label — do not downgrade genuine sentiment to neutral.

Revise ONLY when you have a specific pragmatic reason (sarcasm, implicature, or clear
description/mention). Otherwise KEEP the proposed polarity.

Work with code-switched / mixed-language and informal text.

Output JSON ONLY, exactly these keys:
{
  "keep_or_revise": "keep" | "revise",
  "final_label": "positive" | "negative" | "neutral",
  "confidence": <number 0.0-1.0>,   // confidence in final_label
  "reasoning": "<one or two sentences>",
  "evidence": ["<short quoted span>", "..."]
}
```
When `keep_or_revise` = "keep", `final_label` MUST equal the incoming Polarity label.

### User message template
```
Message:
"""
{text}
"""
Intent analysis (JSON):
{intent_json}

Proposed polarity (JSON):
{polarity_json}

Return the JSON object only.
```

---

## 4. Stage 4 — Deterministic controller (no LLM)

Consumes the three JSON blobs (+ optional primary label/confidence) and emits the final label. Pure
function; fully auditable. Thresholds are named constants so they can be tuned on **dev** (never on
test).

### Constants (initial, tune on dev)
```
TAU_INTENT   = 0.60   # "high-confidence no-opinion" threshold for Stage 1
TAU_REVISE   = 0.60   # "high-confidence pragmatic revision" threshold for Stage 3
TAU_LOW      = 0.45   # below this = low confidence / abstention
USE_PRIMARY_FALLBACK = true   # whether the primary participates on abstention (see §4.4)
```

### 4.1 Precedence rules (evaluated top to bottom; first match wins)
```
1. NO-OPINION NEUTRAL
   if intent.opinion_expressed == false
      and intent.confidence >= TAU_INTENT
      and NOT (pragmatic.keep_or_revise == "revise"
               and pragmatic.final_label != "neutral"
               and pragmatic.confidence >= TAU_REVISE):
        final = "neutral"                     decided_by = "intent_no_opinion"
   # i.e. trust "no opinion" UNLESS pragmatics confidently finds an implicit stance.

2. CONFIDENT PRAGMATIC REVISION
   elif pragmatic.keep_or_revise == "revise"
        and pragmatic.confidence >= TAU_REVISE:
        final = pragmatic.final_label         decided_by = "pragmatic_revision"

3. PRAGMATIC KEEP (opinion present)
   elif pragmatic.keep_or_revise == "keep":
        final = polarity.label                decided_by = "polarity_kept"
        # (pragmatic.final_label == polarity.label by construction)

4. WEAK / CONFLICTED  → fallback (see 4.4)
   else:
        final = fallback(...)                 decided_by = "fallback_*"
```
Rule 1 **is the IntentGate, promoted to a first-class deterministic branch** (removes the unstable
veto-label; §Risks). Rules 2–3 make the pipeline resolver-led, never vote-led.

### 4.2 The low-confidence / conflict branch (Rule 4 detail)
Reached when Stage 3 says "revise" but with `confidence < TAU_REVISE` (a weak flip we don't trust):
```
if USE_PRIMARY_FALLBACK and primary_label is not None:
     if polarity.confidence < TAU_LOW:
          final = primary_label               decided_by = "fallback_primary"
     else:
          final = polarity.label              decided_by = "fallback_polarity"
else:
     final = polarity.label                   decided_by = "fallback_polarity"
```
Rationale: a **weak** revision is discarded (don't trust an uncertain flip); we defer to Polarity, or
to the primary only when Polarity itself is also weak.

### 4.3 Malformed JSON / abstention handling (per stage)
```
Stage 1 unparseable/missing keys  → treat as opinion_expressed="unclear", confidence=0.0
                                     (disables Rule 1; pipeline proceeds to Polarity/Pragmatic).
Stage 2 unparseable/missing label → polarity.label = primary_label if available else "neutral",
                                     polarity.confidence = 0.0  → forces Rule 4 fallback.
Stage 3 unparseable/missing       → treat as keep_or_revise="keep", confidence=0.0
                                     → Rule 3 keeps Polarity (safe default).
final_label not in {pos,neg,neu}  → coerce to "neutral"; log as stage error.
One retry per stage on invalid JSON (temperature 0); if it still fails, apply the above defaults.
```
**Principle: every malformed path degrades safely toward Polarity, then primary, then neutral —
never crashes, never abstains silently. All coercions are logged for the stage-error tally.**

### 4.4 How the primary model participates
- **Router (unchanged):** only samples where **primary confidence < escalation threshold** enter the
  pipeline; high-confidence primary predictions pass through as-is (same regime as all prior designs).
- **Inside the pipeline:** the primary is a **fallback only** (Rule 4 / abstention), never a voter and
  never able to override a confident pipeline decision. Set `USE_PRIMARY_FALLBACK=false` to get a
  pure-pipeline ablation for the stage-level analysis.

### 4.5 Persisted record per escalated sample
```
{ "id", "text",
  "intent": {...}, "polarity": {...}, "pragmatic": {...},
  "primary_label", "primary_conf",
  "final_label", "decided_by",
  "stage_errors": [...],           // any coercions/retries
  "confidences": {"intent":x,"polarity":y,"pragmatic":z} }   // for calibration analysis
```
Persisting `confidences` is mandatory — it is the exact gap that blocked every prior confidence-aware
analysis (captures were labels-only).

---

## 5. Evaluation plan

### 5.1 First run — WEAK primary (this is the decisive test)
- **Primary:** C3 generated-primary, **seed-456 checkpoint**.
- **Eval set:** the same held-out sentiment test split used across the design line.
- **Router:** same escalation threshold as the C3 baseline runs (escalate low-confidence primary).
- **Baselines to beat (acc / macro-F1):**

  | system | acc | macro F1 |
  |---|---|---|
  | primary_only | 0.6956 | 0.6830 |
  | original full_agentic (parallel) | 0.7543 | 0.7387 |
  | Design G (if a C3-G capture exists) | — | — |
  | **sequential_sentiment_v1** | *(target: > 0.7543 / 0.7387)* | |

- **Metrics:** accuracy, macro F1, weighted F1, and on the escalated subset: **W→C, C→W, net**,
  plus **cost** (tokens + $, escalated only) and **stage-level errors**:
    - Stage 1 error: opinion_expressed wrong vs a small hand-checked sample.
    - Stage 2 error: polarity wrong given correct intent.
    - Stage 3 error: revised when it should have kept, or vice-versa (over-neutralization count,
      wrong sarcasm flips).
    - `decided_by` distribution (how often each rule fired) + confidence histograms per stage.
- **Success criterion:** beats `original full_agentic` (0.7543 / 0.7387) on **both** acc and macro F1,
  with the improvement concentrated in the escalated subset and a favorable W→C vs C→W. Report even if
  it does **not** beat — a clean negative on C3 is decisive.

### 5.2 Second run — STRONG primary (only if C3 shows value)
- Ahmed frozen primary, threshold 0.7, same ~15-metric report as the design line.
- **Expectation set honestly:** the strong primary is at ceiling (no design significantly beats
  primary_only, McNemar p≈0.82). The realistic win here is **robustness/stability** (a deterministic
  first-class gate removing the ~23/84 run-to-run gate-label instability), **not** an accuracy jump.
  Do not run this unless C3 first justifies it.

### 5.3 Hygiene (do before either run)
- **Dev split capture** for tuning `TAU_*` — never tune thresholds on test.
- **Serialize all intermediate JSON + confidences** (§4.5) from the first run onward.
- Multi-seed the LLM stages if budget allows (temp-0 still drifts ±1–2 samples) to get error bars.

---

## 6. Risks (explicit, from prior evidence)

- **Cascade errors.** A wrong Stage 1 (`opinion_expressed=false` on a real opinion) hard-suppresses
  Stage 2 via Rule 1 → guaranteed wrong neutral. Parallel voting is robust to one bad agent; a
  pipeline is brittle to an early error. *Mitigation:* Rule 1's "unless pragmatics confidently finds
  a stance" clause and `TAU_INTENT` gate the suppression.
- **Intent over-neutralization.** If Stage 1 leans toward `false`, the system will over-produce
  neutral (the exact failure the parallel Polarity/Contextual camp already showed on author-own-stance
  cases). Watch the neutral-precision and the `decided_by="intent_no_opinion"` error rate.
- **Higher latency.** Strictly serial 3 hops (~3× the parallel critical path); only escalated ~10%
  pay it, but per-sample latency is worse.
- **Higher token cost.** Stages 2–3 carry upstream JSON → est. +20–40% tokens vs the parallel set
  (escalated only, so small absolute).
- **No offline validation.** This is a new computation graph; saved parallel label-tables cannot
  simulate it. Every number requires a **new paid capture** — there is no free pre-check.
- **May not help the strong primary.** Ahmed is at ceiling; expect ~0 significant accuracy movement
  there regardless of architecture. The whole bet rides on the **weak C3** regime.

## Artifacts / basis (no new computation)
- `EXPERIMENT_SEQUENTIAL_SENTIMENT_AGENT_DESIGN.md` — the architecture rationale this instantiates.
- `EXPERIMENT_CONSENSUS_INVESTIGATION_SUMMARY.md` §6 — vote-vs-veto (why Rule 1 is a gate, not a vote).
- `EXPERIMENT_AGENT_SPECIALIZATION_ANALYSIS.md` — the express-vs-mention axis Stage 1 targets.
- Analysis audit — strong-primary ceiling (McNemar p≈0.82) and 23/84 gate-label instability
  (motivates the deterministic first-class gate); C3 baselines primary_only 0.6956/0.6830,
  full_agentic 0.7543/0.7387.
