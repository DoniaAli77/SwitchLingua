# Ahmed Escalated — Qualitative Linguistic Error Analysis

Manual/heuristic reading of the 84 escalated samples' actual code-switched text (no
LLM calls, no training, no generation). Classifies *why* the agents broke Ahmed
(correct→wrong) or fixed Ahmed (wrong→correct). Texts read directly (Egyptian-dialect
Arabic + English); interpretation is the analyst's. Date: 2026-06-27.

## 1. correct→wrong (15) — why the agents got it wrong
| id | true | ahmed | agents → final | text (gloss) | reason |
|---|---|---|---|---|---|
| 203 | neutral | neutral | neg | "هو اللي عامل **unlike** دة عمله علي اساس اية!" (on what basis did whoever *unliked* do it?) | **platform artifact "unlike" → negative** (neutral meta-comment about the dislike button) |
| 245 | neutral | neutral | neg | "هى الناس الى عاملة **unlike** دى ازااااى!!!" (how do these people who *unlike*…) | platform artifact "unlike" → negative |
| 258 | neutral | neutral | neg | "مين ال 74 ال عملين **dislike**" (who are the 74 who *disliked*) | platform artifact "dislike" → negative |
| 363 | neutral | neutral | neg | "الناس اللي عامله **dislike** دول ايه؟؟" (who are these people who *disliked*??) | platform artifact "dislike" → negative |
| 239 | neutral | neutral | pos | "العربية الى بيتكلم عنها Dodg **:D**" (the car he means is a Dodge :D) | **emoji ":D" → positive** (neutral factual) |
| 320 | neutral | neutral | pos | "…شعار **bts**… 🙂**💜**🌚" (…notice the BTS logo… 💜) | **emoji/fandom "💜/BTS" → positive** (neutral observation) |
| 307 | neutral | neutral | pos | "مفيش مستحيل **JUST DO IT** … 😄" (nothing's impossible JUST DO IT… 😄) | **English slogan + 😄 → positive** (borderline) |
| 330 | neutral | neutral | pos | "ياريت تزود… احسن مسلسل… وشكرا" (I wish you'd add… the best series… thanks) | polite "best/thanks" → positive (**label ambiguity**) |
| 362 | neutral | neutral | neg | Breaking Bad plot recap: "فشل جيسي **الفاشل**… **تذويب الجثة**" (Jesse the *failure*… dissolving the *corpse*) | **plot/topic words ("failure","corpse") read as sentiment** |
| 396 | neutral | neutral | neg | "Black widow يا **دود**" (Black Widow, hey *Dood/worm*) | **ambiguous slang/nickname** + short, no context |
| 290 | neutral | neutral | neg | "ليه كده يا رمضان؟؟ 😏… عايز تعمل snoop dog؟؟ 🙄" (why like this Ramadan?? you wanted to act like Snoop Dog?? 🙄) | mild teasing/criticism → negative (**label ambiguity**, defensible) |
| 008 | negative | negative | neutral | "هوة معتز مسعود دة **gay** ?" (is Moataz Masoud *gay*?) | **implicit slur/insult missed** → agents under-called to neutral |
| 097 | negative | negative | neutral | "الفيديو مفيهوش ولا راجل including Mohamed Ramadan **😂**" (the video has no real man, incl. M. Ramadan 😂) | **sarcastic insult missed** (😂 misleads) → neutral |
| 642 | positive | positive | neutral | "**عاش عاش** اسمك ايه في لعبة Free Fire" (*bravo bravo*, what's your name in Free Fire) | **Arabic praise cue "عاش عاش" missed** → neutral |
| 635 | positive | positive | neutral | "دا هيبقا مسلسل out of season" (this'll be an out-of-season series) | subtly-positive idiom missed → neutral (**label ambiguity**) |

## 2. wrong→correct (11) — why the agents helped
| id | true | ahmed → final | text (gloss) | reason |
|---|---|---|---|---|
| 542 | positive | neutral→**pos** | "تول عومراك **جاميلا** XD XD" (…*beautiful* XD) | caught **Arabic positive word** + laughter Ahmed missed |
| 715 | positive | negative→**pos** | "عمرو وكنده **cuteness overload**" | caught **English positive phrase** (Ahmed wrongly negative) |
| 706 | positive | neutral→**pos** | BTS fan comment "…🙃**💜💜💜💜**" | caught **positive emoji/fandom (💜)** Ahmed read as neutral |
| 815 | positive | neutral→**pos** | "…ال new man بحياتك… **😂♥️**" (playful) | caught **playful positive tone + ♥️😂** |
| 021 | negative | neutral→**neg** | "…ممثلة تعمل دور **الممحونة**" (…the *vulgar/horny* role) | caught **Arabic vulgar/insult word** Ahmed missed |
| 045 | negative | neutral→**neg** | "ال عاملين dislike ليه **ياولاد المره**!" (…you *sons-of-[insult]*) | caught **Arabic insult** (here "dislike" *is* negative — real insult attached) |
| 043 | negative | neutral→**neg** | "…بيكسبوا فلوس **من وراء الغلابه😒**" (profit *off the poor* 😒) | caught **critical phrase + negative emoji 😒** |
| 127 | negative | neutral→**neg** | "بتقول I don't think بعدين تقول في بطنه بطيخه!" (mocking) | caught **mocking/sarcasm** in longer text |
| 081 | negative | positive→**neg** | long film review: "vfx **رخيص**… اداء شبه **منعدم**… بدون مستوى" (cheap vfx… nonexistent acting) | caught **long-form critical content** (Ahmed misled by "الفيلم الحلو") |
| 198 | neutral | positive→**neu** | "اول مرة اسمعها بعد ما سمعتها من فرقة big" (first time hearing it…) | **corrected Ahmed's over-call** back to neutral |
| 270 | neutral | negative→**neu** | "الاغنية فيها تقليد من i am one" (the song imitates 'I am one') | **corrected over-call** (Ahmed read "تقليد/imitation" as negative) |

## 3–5. Summary

### 1. Top failure reasons (correct→wrong)
1. **Platform-artifact over-reading: "unlike"/"dislike" → negative (4/15).** Neutral
   meta-comments *about the dislike button/count* ("who disliked this??") are read as
   negative sentiment. This is the single biggest, most systematic failure.
2. **Emoji / English-slogan over-weighting → positive (3/15):** `:D`, `💜`, `😄`,
   `JUST DO IT` flip neutral observations to positive.
3. **Plot/topic/named-entity words confused with sentiment (2/15):** Breaking Bad's
   "failure/corpse", "Black Widow" — dark *content* vocabulary read as negative
   *sentiment*.
4. **Implicit insult / sarcasm / praise *missed* → under-called to neutral (4/15):**
   slur-as-question, sarcastic 😂 insult, "عاش عاش/bravo" — agents default to neutral
   when sentiment is implicit rather than lexical.
5. **Label ambiguity (3/15):** several "neutral" gold labels are genuinely borderline
   (mild teasing, polite "thanks/best") where the agents' polar read is defensible.

### 2. Top success reasons (wrong→correct)
1. **Caught an explicit sentiment cue Ahmed missed (9/11):** Arabic positive
   ("جميلة"), Arabic insult/vulgar ("الممحونة","ياولاد المره"), English positive
   ("cuteness overload"), positive emoji/fandom (💜♥️😂), critical phrase + 😒, and
   sarcasm/mockery in longer text.
2. **Corrected Ahmed's over-call back to neutral (2/11).**

### 3. Misunderstanding vs. EESA label convention
**Mostly genuine agent misunderstanding (~11/15), with a real label-convention/
ambiguity component (~3–4/15).** The artifact, emoji, and plot-word failures are
clear agent errors. But the recurring **"unlike/dislike → neutral" convention is
EESA-specific** (a meta-comment about dislikes is neutral, not negative) — the agents
simply don't know it, whereas Ahmed (trained on EESA) does. A few neutral labels are
also genuinely borderline.

### 4. Are the agents too polarity-sensitive (esp. neutral→negative)? — YES
Confirmed and *explained*: neutral→negative is driven specifically by **the literal
"unlike/dislike" keyword and dark-content vocabulary**, and neutral→positive by
**emojis/English slogans**. The agents react to **surface lexical/emoji cues** and do
not judge whether the comment as a whole is sentiment-bearing — so any
negative-flavoured token (or any emoji) tips a neutral comment.

### 5. The weak component — a mix, ranked
1. **Prompt design (primary).** The agents are **surface-cue literalists**: emojis,
   English words, and platform keywords are over-weighted; implicit sarcasm/insult is
   missed. A prompt that asks "does the comment *express* an opinion, or just mention/
   react with an emoji?" and that names platform artifacts would fix most breaks.
2. **Noisy social-media / platform artifacts (EESA-specific).** "unlike/dislike/clip/
   lyrics" meta-comments are a recurring EESA pattern the generic agents misread.
3. **Dialect & implicit-sentiment handling.** Missed Egyptian-dialect sarcasm/insult
   and implicit praise ("عاش عاش").
4. **Dataset label ambiguity (minor).** A handful of borderline neutrals.
5. **Consensus override = amplifier, not root.** It lets the (correlated, surface-
   driven) agents win — but the *linguistic* root is #1–#3.

## Representative examples (spanning categories)
| id | true | ahmed | final | text (gloss) | verdict |
|---|---|---|---|---|---|
| 258 | neutral | neutral | **neg** | "مين ال 74 ال عملين dislike" (who are the 74 who disliked) | **FAIL** — "dislike" artifact over-read |
| 320 | neutral | neutral | **pos** | "…شعار bts… 💜🌚" | **FAIL** — 💜/fandom emoji over-read |
| 362 | neutral | neutral | **neg** | Breaking Bad: "الفاشل… تذويب الجثة" | **FAIL** — plot words as sentiment |
| 008 | negative | negative | **neutral** | "هوة معتز مسعود دة gay ?" | **FAIL** — implicit slur missed |
| 290 | neutral | neutral | **neg** | "ليه كده يا رمضان؟؟ 🙄" | **FAIL/ambiguous** — mild criticism, label borderline |
| 715 | positive | negative | **pos** | "عمرو وكنده cuteness overload" | **SUCCESS** — English positive phrase caught |
| 021 | negative | neutral | **neg** | "…دور الممحونة" | **SUCCESS** — Arabic vulgar/insult caught |
| 043 | negative | neutral | **neg** | "…من وراء الغلابه😒" | **SUCCESS** — critical phrase + 😒 caught |
| 081 | negative | positive | **neg** | long critical film review ("vfx رخيص…") | **SUCCESS** — long-form critique caught |
| 270 | neutral | negative | **neu** | "فيها تقليد من i am one" | **SUCCESS** — over-call corrected to neutral |

## 6. correct→wrong: is GPT *linguistically reasonable* (convention disagreement) or *genuinely wrong*?
Per-case verdict — **A** = GPT's read is defensible, the EESA gold label is the
debatable/convention call; **B** = genuine GPT error (EESA clearly right); **C** =
text too ambiguous to judge.

| id | true → GPT | text (gloss) | verdict | why |
|---|---|---|---|---|
| 290 | neutral → **neg** | "ليه كده يا رمضان؟؟ 🙄… تعمل snoop dog؟؟" (mocking M. Ramadan, eye-roll) | **A** | clearly mild criticism/mockery → GPT's *negative* is the natural read; EESA "neutral" is the odd label |
| 307 | neutral → **pos** | "مفيش مستحيل JUST DO IT… 😄" (nothing's impossible…) | **A** | genuinely hopeful/motivational → *positive* is reasonable; EESA reserves "positive" |
| 320 | neutral → **pos** | "…شعار bts… 💜🌚" (fan pointing out BTS logo) | **A** | fan affect + 💜 → *positive* defensible; EESA treats "pointing-out" as neutral |
| 330 | neutral → **pos** | "ياريت تزود… احسن مسلسل… وشكرا" (nice suggestion + thanks) | **A** | appreciative "best/thanks" → mildly *positive* defensible |
| 635 | positive → **neutral** | "دا هيبقا مسلسل out of season" | **A** | the phrase carries no overt positive cue → GPT's *neutral* is reasonable; EESA "positive" is non-obvious |
| 239 | neutral → **pos** | "…Dodge :D" (identifying a car :D) | **A/B** | only a friendly `:D` on neutral content — *weakly* defensible positive |
| 008 | negative → **neutral** | "هوة معتز مسعود دة gay ?" | **B** | a veiled slur/insinuation → EESA "negative" justified; GPT missed the pragmatic intent |
| 097 | negative → **neutral** | "…مفيهوش ولا راجل… Mohamed Ramadan 😂" (no *real man*) | **B** | explicit insult → clearly negative; GPT under-read (😂 misled) |
| 203 | neutral → **neg** | "اللي عامل unlike… علي اساس اية!" | **B** | commenter is *defending* the video against dislikers → GPT's negative is the **wrong direction**, not a convention gap |
| 245 | neutral → **neg** | "الناس… unlike… ازااااى!!!" | **B** | same — pro-video, GPT wrong direction (artifact) |
| 258 | neutral → **neg** | "مين ال 74… dislike" | **B** | neutral meta-question; GPT triggered by "dislike" keyword |
| 363 | neutral → **neg** | "الناس… dislike دول ايه؟؟" | **B** | same artifact misread |
| 362 | neutral → **neg** | Breaking Bad recap ("الفاشل… الجثة") | **B** | dark *plot* vocabulary ≠ commenter sentiment; genuine confusion |
| 642 | positive → **neutral** | "عاش عاش… Free Fire" (*bravo bravo*) | **B** | "عاش عاش" is an unambiguous Arabic cheer; GPT missed it |
| 396 | neutral → **neg** | "Black widow يا دود" | **C** | too short/ambiguous ("يا دود" = nickname or mild insult) to adjudicate |

**Tally: A (GPT reasonable / EESA-convention debatable) ≈ 5–6 · B (genuine GPT error)
≈ 8 · C (ambiguous) = 1.**

### What this means
- **~5–6 of the 15 "agent breaks" are not really agent failures** — GPT's reading is
  linguistically defensible and the **EESA gold label is the debatable one.** These
  cluster as **neutral→positive over-calls** on *mildly* positive comments (motivational,
  fan-affect, polite-thanks): **EESA labels mildly-positive content as neutral**, i.e. a
  stricter "positive" threshold than general sentiment intuition. One (290) is a
  neutral→negative where the comment is genuinely mild criticism.
- **~8 are genuine GPT errors**, and they are systematic: the **"unlike/dislike" wrong-
  direction misread (4)** (the commenter is *defending* the video, GPT sees the keyword
  and says negative), **plot/content words as sentiment (1)**, and **missed implicit
  insult/praise (3)**. These are real and prompt-fixable.
- So the **effective** agent error rate on the escalated breaks is **~8/15, not 15/15** —
  roughly a third of the "harm" is annotation-convention disagreement, not model error.
  This matters for interpreting the −4 net: part of it is GPT being *reasonably*
  different from EESA, not *wrong*.

## Bottom line
The agents and Ahmed fail/succeed by the **same mechanism**: the agents are **explicit-
cue detectors**. They *help* when a real sentiment word/emoji/insult is present that
Ahmed missed, and *hurt* when an explicit cue is present but the comment is actually a
**neutral meta-reaction** (emoji on a fact, "dislike" about the dislike button, dark
plot vocabulary). Ahmed already encodes EESA's "neutral meta-comment" convention, so
the agents' literal reading specifically erodes those. **The fix is linguistic, not
architectural: prompt the agents to (a) ignore platform artifacts and bare emojis as
sentiment, (b) require an actual opinion to be *expressed*, and (c) handle Egyptian-
dialect sarcasm/insult — i.e. reduce surface-cue literalism — rather than changing the
consensus rule.**
