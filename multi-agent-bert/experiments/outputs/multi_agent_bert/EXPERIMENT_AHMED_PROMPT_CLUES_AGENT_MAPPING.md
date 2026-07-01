# Ahmed's Prompt/Feature Strategy → Our Agent Architecture (mapping analysis)

Connects the design clues in Ahmed's thesis (*Sentiment Analysis for Arabic-English
Code-Switched Data*, GIU 2025 — the EESA corpus, its sentiment lexicon, 19 handcrafted
word features, and the zero-shot prompt suite) to our multi-agent sentiment decomposition,
using the Design-A/B/C/D/E/F ablation results on the Ahmed frozen primary. **Analysis
only — no runs, no training.** Date: 2026-07-01.

> Note on the frozen primary: Ahmed's reported **ensemble traditional-neural test
> F1 = 92.54%** is exactly our frozen-primary `0.9254`. That primary is the model that
> *learned from* the lexicon + handcrafted features below — which is why its behaviour
> already encodes much of what our agents are trying to re-derive.

---

## 1. Ahmed's clues (from the thesis)
- **Sentiment lexicon** (2,324 pos/neg words & compound phrases from EESA + NileULex) and
  **19 handcrafted binary word features**: positive / negative / neutral, **negation**,
  **intensified**, **named entity**, **compound phrase**, **repeated characters**,
  **emoticons**, "contextually (non-neutral)", token language/case, etc.
- **Annotation guidelines** (used both to label and as a zero-shot prompt): judge from the
  **author's perspective, not the reader's**; **sarcasm → negative**; advice / suggestions
  / questions / ads / hyperlinks → **neutral**. The canonical examples:
  - *"Why are there plenty of dislikes in the comments"* → **neutral** (a question; the
    negative token "dislikes" is *mentioned*, not *expressed*).
  - *"People who are clicking on the dislike button are ignorant"* → **negative** (the
    author *expresses* a stance toward those people).
- **Rule-Based Inference prompt** — an explicit decision sequence: review hints → find
  pos/neg phrases → **"Is the author expressing a genuine sentiment about an entity? →
  continue. Is the author questioning the sentiment expressed by others? → neutral."** →
  both pos+neg ⇒ sarcasm ⇒ negative → only positive ⇒ check ad/promotion/sarcasm.
- **Semantic Translation prompt** — translate the whole code-switched message to English,
  *then* judge sentiment (meaning-level, whole-message reasoning).
- **Result hierarchy:** every strong Ahmed model **depends on the lexicon / sentiment
  hints / handcrafted features** (ensemble 92.54%; fine-tuned GPT-4o-mini **with sentiment
  hints** 95.48%, up from 93.77% without). Removing explicit evidence *never* helped him.

## 2. Mapping to our agents
| Ahmed clue | Our agent | How it maps |
|---|---|---|
| **1. Sentiment hints / lexicon / handcrafted pos-neg-neutral features** | **Lexical** (evidence) + **Polarity** (decision) | Lexical = the *inventory* of explicit evaluative cues and their strength (Ahmed's lexicon lookup + pos/neg/neutral features). Polarity = the *decision* over those cues (Ahmed's "determine polarity from the found phrases"). |
| **2. Annotation-guideline prompting** (author-vs-reader; dislikes-question = neutral; ad/question = neutral) | **Intent** (opinion-expression detector) | This is *exactly* Intent's job: does the **author express their own evaluation**, or is it a question / meta-comment / mention? Ahmed's two dislikes examples are the textbook Intent cases. |
| **3. Semantic-translation / meaning-level prompting** | **Contextual** (whole-message interpreter) | Translate-then-judge = interpret the entire message's communicative intent, incl. sarcasm — Contextual's remit. |
| **4. Negation / intensifier / emoticon / repeated-char features** | **Polarity** (primary) + **Lexical** (report) | Ahmed handles these as features that *flip or amplify* polarity → Polarity owns negation/intensification/mixed-polarity; Lexical reports emojis/repeated-chars as **weak** cues. |
| **5. Strong EESA performance = learned "real cue vs meta/mention"** | the **whole panel**, but especially **Intent + Lexical** | Ahmed's model learned *when a surface cue is genuine sentiment vs platform/meta/content mention*. In our decomposition that knowledge is split: Lexical says "cue present", Intent says "is it the author's expressed evaluation", Contextual says "does the whole message agree". |

**Reading:** Ahmed's own best pipeline is **evidence (lexicon/hints) → intent gate (author
expressing vs questioning) → polarity decision (negation/sarcasm) → whole-message check**.
That is Lexical + Intent + Polarity + Contextual — but with **Intent acting as a gate**,
not as a co-equal vote (see §4).

## 3. Design F (Intent + Polarity + Contextual) answered
Headline F on the Ahmed frozen primary: **acc 0.9218 / macroF1 0.9180 / net −3**, escalated
acc 0.7143, W→C 11, C→W 14, breaks **pos→neu 5, neg→neu 4** (total 14, up from C's 11).

- **Did removing Lexical hurt explicit-cue detection? YES — clearly.** The failure signature
  is **over-neutralization**: polar→neutral breaks jumped (**pos→neu 2→5, neg→neu 3→4**).
  Without an explicit-evidence anchor, three neutral-leaning deciders (Intent's "prefer
  neutral if no clear stance", Polarity's discipline, Contextual's caution) collectively
  flatten genuinely evaluative comments to neutral. This is the exact behaviour Ahmed's
  ablation predicts: he shows removing the lexicon / sentiment-hint features **reduces**
  performance (e.g. GPT-4o-mini 93.77% → 95.48% *with* hints); his models never do without
  explicit evidence.
- **Did Intent reduce platform/meta false sentiment? Partially — on the neutral side.**
  F's **neutral→positive breaks fell to 1** (lowest of any design) and neutral→negative
  stayed low (4) — consistent with Intent doing Ahmed's "dislikes-question = neutral" job.
  But that gain is swamped by the polar→neutral damage, so net is −3.
- **Did Polarity preserve the sentiment-hint benefit? Only partly.** Polarity *decides*
  well but cannot *substitute* for the explicit cue inventory. Ahmed separates "provide the
  sentiment hints" from "decide polarity"; collapsing both into Polarity (no Lexical) loses
  the hint-provision half — the model no longer reliably "knows a positive/negative word is
  present", which Ahmed explicitly names as the reason hints help.
- **Did Contextual still handle implicit/sarcastic cases? Yes, unchanged** — but Contextual
  reasons at the discourse level and does not restore the lost explicit-cue anchor, so it
  cannot compensate for the missing Lexical evidence.
- **Is F closer to Ahmed's logic than C? NO.** Ahmed's entire feature/prompt strategy is
  **evidence-centric** — lexicon, sentiment hints, handcrafted pos/neg features are in every
  strong configuration. Dropping the Lexical/evidence agent moves *away* from Ahmed's design,
  not toward it. F resembles only the *intent-gate* fragment of Ahmed's Rule-Based prompt
  while discarding the *evidence* fragment that Ahmed shows is essential.
- **Is C still better because Lexical provides explicit evidence? YES.** C = Lexical +
  Polarity + Contextual keeps the explicit-evidence anchor (Ahmed's lexicon/hints) **and**
  the polarity decision **and** the whole-message check — the three Ahmed components that
  survive as *votes*. C is net **+1** vs F's **−3**; the difference is precisely the
  explicit-evidence anchor Ahmed's work says you must keep.

## 4. Proposed future architecture (design only — DO NOT implement yet)
**Intent-as-gate** — directly mirrors Ahmed's Rule-Based Inference sequence (intent is a
*branch*, not a *vote*):
1. **Intent gate first:** does the author express an evaluative opinion at all?
   - *No* (question / meta-comment about likes-dislikes-shares / quote / plot description /
     ad) → **prefer neutral / low confidence**, short-circuit.
   - *Yes* → proceed.
2. **Polarity + Contextual decide the polarity** of the expressed evaluation (Polarity for
   negation/intensifier/mixed/sarcasm-as-negative; Contextual for whole-message/implicit),
   **with Lexical supplying the explicit-cue evidence** they weigh.

Why this is the promising shape (per the ablations + Ahmed):
- Design E showed Intent as a *co-equal 4th vote* is a wash (net +1, same as C, higher cost)
  and Design F showed Intent+Polarity+Contextual *without Lexical* over-neutralizes (−3).
  Both say **Intent should not be a voter**. As a **gate**, Intent contributes its
  meta/mention discrimination *without* casting a neutral-leaning vote that flattens polar
  cases — exactly how Ahmed uses it (step 4 of Rule-Based Inference).
- It keeps Lexical as the evidence anchor (Ahmed's lexicon/hints) inside the polarity branch.

Risks to check when/if implemented: a wrong "no-opinion" gate decision hard-neutralizes a
truly polar comment (needs a confidence threshold / fallback to the primary), and it changes
control flow (gate) rather than just the vote set — more than a drop-in.

## 5. Conclusion / recommendation
- **Lexical is genuinely useful and must stay** — removing it (F) causes Ahmed-predicted
  over-neutralization (polar→neutral breaks jump, net −3). This is the clearest evidence in
  the whole ablation that the **explicit-evidence agent is load-bearing**.
- **Best current sentiment decomposition = C: Lexical + Polarity + Contextual**, which is
  the subset of Ahmed's evidence→polarity→whole-message logic that works as independent
  votes. Keep B/E as opt-in curiosities; retire D and F for sentiment.
- **Most promising next idea = Intent-as-gate** (not a vote), matching Ahmed's Rule-Based
  Inference control flow — proposed here, **not implemented**, and **not** to be run before
  the pending C3 generated-primary check.

## Artifacts
- Ablation reports: `EXPERIMENT_SENTIMENT_AGENT_DESIGN_ABLATION.md` (A–D),
  `EXPERIMENT_SENTIMENT_INTENT_AGENT_ABLATION.md` (E),
  `EXPERIMENT_SENTIMENT_DESIGN_F_INTENT_POLARITY_CONTEXTUAL.md` (F).
- Source thesis clues: EESA corpus §4.1, lexicon + 19 features §4.2.1 / Table 4.4,
  annotation guidelines §4.1.2 & Appendix B, prompt suite §4.3.2 / §4.3.3.
