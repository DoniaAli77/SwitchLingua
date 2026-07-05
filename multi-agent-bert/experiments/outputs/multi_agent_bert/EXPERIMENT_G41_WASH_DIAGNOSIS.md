# Why gpt-4.1-mini Didn't Improve G on Ahmed — and the (non-dataset-specific) Prompt Fix

**Analysis only — no LLM calls.** Diagnoses why upgrading the agents to gpt-4.1-mini netted
~0 on Ahmed (5 fixed, 5 broken) by examining the *broken* cases, and derives a **general**
(non-EESA) prompt change that targets the actual mechanism. Date: 2026-07-02.

## The finding: it's a WASH on two axes, not a capability gain
Comparing G@4o-mini vs G@4.1-mini on the 84 escalated: **5 fixed, 5 broken, 13 both-wrong.**
The stronger model didn't reduce errors — it **reshuffled** them. And the 5 fixed and 5 broken
are the **same two axes**, pulled opposite ways:

### Axis A — "platform action" (like/dislike) → the prompt says "lean neutral"
- **FIXED** (genuinely meta → true neutral): `00240` "اللى عاملين dislike", `00298`
  "الفتوة=dislike" — 4.1 correctly neutralized.
- **BROKEN** (author's OWN stance *via* the dislike → true negative): `00041` "123 الف dislike
  يارب تكون الرسالة وصلت" (endorsing the dislikes), `00045` "ال عاملين dislike ليه يا ولاد
  المرة!" (attacking the dislikers) — 4.1 **wrongly** neutralized.

**Mechanism:** the prompt rule is *"treat platform words (like, dislike…) as WEAK cues → lean
neutral."* gpt-4.1-mini follows this rule **more consistently** than 4o-mini. But the rule is a
**directional heuristic that is wrong ~half the time**: whether "dislike" is neutral-meta or
the author's own negative stance depends on the author's *relationship to the action*
(reporting it vs endorsing it vs attacking it), not on the word. A better model applying a
lossy rule more faithfully → more correct neutralizations **and** more wrong ones. Net wash.

### Axis B — implicit stance / description-vs-evaluation
- **FIXED**: `00542` misspelled "جميلة" (praise), `00642` "عاش عاش" (bravo), `00193` implicit neg.
- **BROKEN**: `00100`, `00127` implicit insults 4.1 **over-neutralized**; `00362` a *Breaking
  Bad plot description* ("فشل جيسي الفاشل"…) 4.1 **over-read as negative** (it's description → neutral).

Same tension: the prompt pushes "no explicit cue / describes content → lean neutral," so the
stronger model sometimes reads the implicit stance better (fixed) and sometimes over-applies
the neutral lean or over-reads described words (broken).

## Diagnosis (direct answer)
**The bottleneck is NOT model capability — it's that the prompts encode two ambiguous axes as
directional shortcuts** ("platform word → weak/neutral"; "no explicit cue → neutral"). A
stronger model executes those shortcuts more faithfully, which is why it fixes and breaks in
equal measure. **You cannot get past this by swapping models; you have to remove the shortcut
and make the prompt DISAMBIGUATE the axis.** And crucially, the disambiguation is a **general
pragmatic distinction, not an EESA rule** — so it can be added without dataset tailoring.

## Proposed prompt change — general, not dataset-specific

**Do NOT** add Egyptian idioms, emoji tables, or "dislike"-keyword lists (that would be
EESA-specific and overfit). **Instead**, replace the directional heuristic with a
*relationship* question that applies to any platform, any language, any corpus:

**Current (lossy shortcut), in Lexical/Polarity/Intent prompts:**
> "Treat platform/interface words (like, dislike, comment, share…) as WEAK cues unless the
> author clearly states their own opinion." / "if the text only MENTIONS platform actions →
> neutral."

**Proposed (general disambiguation):**
> "When the text refers to a platform action (like, dislike, comment, share, follow, trend…),
> do not treat it as neutral by default. Decide the AUTHOR'S RELATIONSHIP to that action:
> (a) merely **reporting or counting** it → neutral;
> (b) **endorsing / celebrating** it → carry the polarity the author endorses;
> (c) **objecting to it or attacking the people doing it** → negative.
> The platform word itself is not evidence of neutrality; the author's stance toward the
> action is what counts."

And for Axis B (already partly in Contextual's semantic_v1 / v3, generalize it):
> "Distinguish **recounting/describing** content (a plot, events, or others' actions) — which
> is neutral even when it contains negative words — from the author **evaluating** it. Negative
> or positive words *inside described content* are not the author's own stance."

Both are general use-vs-mention / description-vs-evaluation pragmatics — no dataset terms.

## Honest caveat
This targets the **exact wash mechanism** (it converts a coin-flip heuristic into a decision
the author's stance actually determines), so it is the **right** change to try. But the
disambiguation ("is the author endorsing or attacking the dislikes?") is itself a pragmatic
judgment — it will help the **clear** cases and still miss the **genuinely ambiguous** ones,
and it will **not** touch the deep cultural-implicit-insult floor (`00008`, `00182`) that no
prompt fixes. So expect it to **shift the wash toward net-positive**, not to break the ceiling.
It is most likely to pay off on the **weak C3 primary**, where the recoverable slice is larger.

## Recommended test
Implement as a new opt-in prompt variant (e.g. `semantic_v2_disambig`) — Lexical/Polarity/
Intent get the platform-relationship rule, Contextual gets the description-vs-evaluation
generalization — keep it dataset-agnostic, add tests, then run **G@4.1-mini with the new
variant on Ahmed** (does 5-fixed/5-broken become, say, 7/3?) and, if positive, on **C3**.

## Artifacts
- Fixed/broken split: `_G41_broken.txt`, computed from
  `experiment_G_ahmed_gpt41mini/` vs `experiment_ahmed_designG_intent_gate/error_attribution/`.
- Basis: `EXPERIMENT_G_AHMED_GPT41MINI_RESULTS.md`, `EXPERIMENT_G_PROMPT_VS_FAILURE_MAPPING.md`.
