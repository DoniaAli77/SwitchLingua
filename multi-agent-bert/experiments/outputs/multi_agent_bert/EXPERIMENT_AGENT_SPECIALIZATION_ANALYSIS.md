# Agent Specialization — Do Different Agents Own Different Linguistic Phenomena?

**Evidence only — no OpenAI calls, no code/prompt changes, no reruns.** Tests whether Lexical,
Polarity, Contextual (and the IntentGate) are genuine specialists for distinct linguistic
phenomena, or correlated generalists solving the same problem. Uses the saved Design G, v3, and
G2 captures (same agent set: Lexical + Polarity + Contextual + non-voting IntentGate), 84 escalated
samples, each observed across the three runs. **Extremely critical read.** Date: 2026-07-01.

## 0. The result that dominates everything: they almost always AGREE
- **Of 84 escalated samples, only 16 (19%) have any agent disagreement in any of the 3 runs.**
- On the **other 68/84 (81%)** the three agents emit **identical** labels. Including the ~734
  non-escalated cases (agents never run), the panel agrees on **~96% of the full 818**.
- **This is the first and strongest evidence *against* specialization.** True specialists — each
  applying a different lens — would *disagree constantly*. These agents disagree on only 16 edge
  cases out of 84 of the *hardest* inputs. High agreement = high redundancy = mostly the same
  computation, not division of labour. Any "specialization" can therefore live in **at most 16
  samples = 2% of the test set.**

## 1–2. The 16 disagreement samples, who was correct, tagged by phenomenon
Per-agent correctness shown as (correct runs / runs observed). "Winner" = agent(s) most often right.

| id | phenomenon (read) | true | Lexical | Polarity | Contextual | winner |
|---|---|---|---|---|---|---|
| 00046 | platform-meta + sarcastic-negative | neg | **3/3** | 0/3 | 0/3 | Lexical |
| 00363 | platform-meta *question* (mention) | neu | 0/3 | **3/3** | **3/3** | Pol, Ctx |
| 00449 | platform-meta *question* (mention) | neu | 0/3 | **3/3** | **3/3** | Pol, Ctx |
| 00487 | platform + author's own *like* (positive act) | pos | **3/3** | 0/3 | **3/3** | Lex, Ctx |
| 00517 | platform + author positive ("I like with my heart" ❤) | pos | **3/3** | **3/3** | 2/3 | Lex, Pol |
| 00100 | implicit insult / aggressive rhetorical | neg | 0/3 | 1/3 | **3/3** | Contextual |
| 00182 | implicit insult ("you are no one") | neg | 0/3 | 0/3 | **1/1** | Contextual |
| 00113 | implicit "wtf" / description-vs-eval | neg | **3/3** | 0/3 | 1/3 | Lexical |
| 00239 | named-entity/brand mention (":D") | neu | 0/3 | **3/3** | **3/3** | Pol, Ctx |
| 00396 | named-entity mention ("Black widow") | neu | 0/1 | **1/1** | **1/1** | Pol, Ctx |
| 00320 | brand-spotting + emojis (🙂💜🌚) | neu | 0/3 | **3/3** | 0/3 | Polarity |
| 00310 | pun / ambiguous + repeated-char | neu | 1/3 | **3/3** | **3/3** | Pol, Ctx |
| 00330 | speech-act: advice / request | neu | 2/3 | **3/3** | 0/3 | Polarity |
| 00362 | plot / content description | neu | 2/3 | **1/1** | **1/1** | Pol, Ctx |
| 00542 | obscured/misspelled praise ("beautiful") | pos | **3/3** | 0/3 | 1/3 | Lexical |
| 00706 | fan excitement / implicit praise | pos | **3/3** | 0/3 | **3/3** | Lex, Ctx |

## 3–4. Per-phenomenon tally, best/worst agent, and honest significance
| phenomenon | N (samples) | most-correct | most-wrong | statistically meaningful? |
|---|---|---|---|---|
| Explicit / obscured lexical cue (wtf, misspelled praise) | 2 (00113, 00542) | **Lexical** (2/2) | Polarity | **No — anecdotal (N=2)** |
| Platform-meta *question* (mention-vs-use, neutral) | 3 (00363, 00449, +00046-inverse) | **Polarity, Contextual** | Lexical (0/2) | weak (N≈3) |
| Platform + author's own stance (positive act) | 3 (00046, 00487, 00517) | **Lexical** | Polarity | weak (N≈3) |
| Implicit insult / sarcasm (no explicit cue) | 3 (00100, 00182, 00113) | **Contextual** (2/3) | Lexical | weak (N=3, one exception) |
| Named-entity / brand mention → neutral | 3 (00239, 00396, 00320) | **Polarity** (3/3) | Lexical (0/3) | weak (N=3) |
| Speech-act / plot description → neutral | 2 (00330, 00362) | **Polarity** (2/2) | Contextual (1/2) | **No (N=2)** |
| Fan / implicit praise | 1 (00706) | Lexical, Contextual | Polarity | **No — single example** |
| Pun / ambiguous | 1 (00310) | Polarity, Contextual | Lexical | **No — single example** |

**Every category has N ≤ 3. None is statistically meaningful.** A single sample flipping (temp-0
noise, which we know is ±1–2) would change the "winner" in most rows.

## 5. Phenomenon → best-specialist confusion matrix (with the critical caveat)
| phenomenon ↓ | best specialist | but note |
|---|---|---|
| explicit / obscured cue | **Lexical** | over-fires on platform words elsewhere |
| named-entity / mention → neutral | **Polarity** | fails when a subtle stance *is* present |
| platform meta-question | **Polarity / Contextual** | same camp — not a distinct expert |
| platform + author stance | **Lexical** | Polarity over-neutralizes here |
| implicit insult / sarcasm | **Contextual** | fails on brand-spot (00320) and advice (00330) |
| speech-act / description | **Polarity (/Contextual)** | Contextual inconsistent |
| fan / implicit praise | **Lexical / Contextual** | N=1 |

**What the matrix actually shows is not three experts but a two-camp split on ONE axis —
*express vs mention*:**
- **Lexical** = the *explicit-cue* camp: right when a real cue exists (author's own like, "wtf",
  misspelled "beautiful"), wrong when a platform/brand word is merely mentioned.
- **Polarity + Contextual** = the *stance-detection* camp: right when the author expresses **no**
  evaluation (meta-question, mention, description → neutral), wrong when a subtle stance **is**
  present (fan praise, author-like → they over-neutralize).
- **Contextual** adds a weak *implicit/sarcasm* edge on top (00100, 00182), but is **inconsistent**
  — it fails inside its own supposed domain (00320 brand-spotting, 00330 advice), so it is not a
  reliable sarcasm specialist.

So the agents do **not** partition the phenomenon space into clean expert domains. They cluster on
a single pragmatic axis (is an evaluation expressed?), and even there each camp **fails within its
own domain** depending on the sub-case.

## 6. Does this justify adaptive routing instead of static voting? — No.
1. **No agent dominates any phenomenon.** Each "winner" also *fails* inside its supposed domain
   (Contextual on brand-spot/advice; Polarity on genuine subtle positives; Lexical on meta). A
   router needs a dominant expert per class; there isn't one.
2. **N is 1–3 per phenomenon** — far too few to *estimate* a routing policy, let alone validate it.
3. **The routing decision is the hard problem itself.** To route "this is sarcasm → Contextual"
   you must first detect sarcasm — the exact pragmatic judgment the agents are failing. Circular.
4. **Total addressable ≤ 16 samples (2% of test)**, and the consensus-loss analysis showed only
   ~4–11 are even recoverable — a routing mechanism's ceiling is a handful of noise-adjacent samples.
5. **The one robust axis (mention-vs-use) is already captured** — not by a voting specialist but by
   the **IntentGate veto**, which on the neutral-meta cases (00363, 00449, 00239) does the right
   thing more reliably than any single agent, without needing phenomenon detection.

## 7. Did we design true specialists, or are they solving the same problems? — **Mostly the same.**
> **The agents are correlated generalists, not phenomenon-experts.** They emit identical labels on
> 81% of the hardest (escalated) cases and ~96% overall — the signature of redundancy, not
> specialization. On the 16 genuinely-disagreeing edge cases, the pattern is a **weak, noisy,
> two-camp lean on a single express-vs-mention axis** (Lexical = explicit cues; Polarity/Contextual
> = stance/mention detection), with a faint Contextual edge on implicit sarcasm — but **every
> category has N ≤ 3, no agent dominates its domain, and each "specialist" fails inside its own
> supposed lane.** The role prompts induced mild *tendencies*, not true experts.

**Honest bottom line: no strong specialization exists.** What looks like specialization is (a) 84%
correlated agreement + (b) a handful of edge cases where a two-camp directional bias occasionally
separates the agents — indistinguishable from noise at these sample sizes. There is **not enough
evidence to claim per-phenomenon experts**, and therefore **not enough to justify adaptive routing**
over the current consensus + IntentGate veto. If anything, the data says the agents converged to
*the same* pragmatic problem and differ only in which surface bias they apply when it is unresolved.

## Caveats on this evidence
- **Tiny N**: 16 disagreement samples, split into ~8 phenomena → mostly 1–3 each. All phenomenon
  "winners" are anecdotal; none survive a significance test.
- **Phenomenon tags are the analyst's reading** of short code-switched text (all 16 are
  code-switched), not gold pragmatic labels — some are debatable.
- **Temp-0 noise** (±1–2 samples) is on the order of the entire signal here.
- Confidences were not serialized, so "confidence within a phenomenon" could not be examined.

## Artifacts
- Disagreement extraction from `experiment_ahmed_designG_intent_gate/`, `…_v3_pragmatic/`,
  `…_designG2_selective_gate/` `error_attribution/attribution_table.json` (labels + true).
- Related: `EXPERIMENT_CONSENSUS_LOSS_ANALYSIS.md` (lone-dissent precision: Contextual 0.58, others
  <0.5), `EXPERIMENT_ESCALATED_CEILING_ROOT_CAUSE.md` (correlation floor),
  `EXPERIMENT_AGENT_BEHAVIOR_COMPARISON.md` (92% agreement measured earlier).
