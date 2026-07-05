"""Prompt templates for the ``sequential_sentiment_v1`` staged pipeline.

This module holds the three **stage** prompts for the experimental sequential
sentiment architecture (opt-in; the default parallel pipeline never imports it):

* **Stage 1 — Intent / opinion-expression detector.** Decides whether the author
  expresses their own evaluative opinion, its target, the speech act, and
  use-vs-mention. It does **not** output a sentiment label.
* **Stage 2 — Polarity resolver.** Given the text and the Stage-1 Intent JSON,
  assigns a sentiment label with cue evidence handled internally (no separate
  Lexical stage in v1).
* **Stage 3 — Pragmatic verifier.** Given the text, Intent JSON, and Polarity
  JSON, checks sarcasm / implicature / description-vs-evaluation and decides
  whether to KEEP or REVISE the polarity.

Design constraints (kept verbatim from the approved design report):

* General sentiment-reasoning guidance only — no dataset, corpus, platform,
  author, register, or benchmark is named.
* Every stage returns **JSON only** (no markdown fences, no prose).
* Every stage returns a ``confidence`` for later calibration analysis.
* Final labels are the shared ``positive / negative / neutral`` set; the allowed
  label list is injected verbatim into the Stage-2 / Stage-3 user prompts so the
  module stays label-generic like the rest of the codebase.
"""

from __future__ import annotations

import json
from string import Template
from typing import Dict, List

# ===========================================================================
# Stage 1 — Intent / opinion-expression detector
# ===========================================================================

INTENT_SYSTEM_PROMPT = """\
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
{"opinion_expressed": true, "target": "<string or null>", "speech_act": "evaluate", "use_vs_mention": "use", "confidence": 0.0, "evidence": ["<span>"]}\
""".strip()


_INTENT_USER_TEMPLATE = Template("""\
Message:
\"\"\"
$text
\"\"\"
Return the JSON object only.\
""")


# ===========================================================================
# Stage 2 — Polarity resolver
# ===========================================================================

POLARITY_SYSTEM_PROMPT = """\
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
{"label": "<label>", "confidence": 0.0, "mixed": false, "reasoning": "<one sentence>", "evidence": ["<span>"]}\
""".strip()


_POLARITY_USER_TEMPLATE = Template("""\
TASK: $task_name (polarity resolution)

ALLOWED LABELS (choose exactly one - no other value is valid):
$labels_csv

LABEL DESCRIPTIONS:
$labels_block

Message:
\"\"\"
$text
\"\"\"

Intent analysis (JSON):
$intent_json

Return the JSON object only. "label" must be exactly one of: $labels_csv\
""")


# ===========================================================================
# Stage 3 — Pragmatic verifier
# ===========================================================================

PRAGMATIC_SYSTEM_PROMPT = """\
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
{"keep_or_revise": "keep", "final_label": "<label>", "confidence": 0.0, "reasoning": "<one sentence>", "evidence": ["<span>"]}\
""".strip()


_PRAGMATIC_USER_TEMPLATE = Template("""\
TASK: $task_name (pragmatic verification)

ALLOWED LABELS (choose exactly one - no other value is valid):
$labels_csv

LABEL DESCRIPTIONS:
$labels_block

Message:
\"\"\"
$text
\"\"\"

Intent analysis (JSON):
$intent_json

Proposed polarity (JSON):
$polarity_json

Return the JSON object only. "final_label" must be exactly one of: $labels_csv\
""")


# ===========================================================================
# User-prompt builders
# ===========================================================================

def _labels_block(labels: List[str], label_descriptions: Dict[str, str]) -> str:
    return "\n".join(
        f"  {lbl} - {label_descriptions.get(lbl, '(no description)')}"
        for lbl in labels
    )


def build_intent_user_prompt(text: str) -> str:
    """Render the Stage-1 (intent) user prompt for one message."""
    return _INTENT_USER_TEMPLATE.substitute(text=text)


def build_polarity_user_prompt(
    task_name: str,
    labels: List[str],
    label_descriptions: Dict[str, str],
    text: str,
    intent: Dict,
) -> str:
    """Render the Stage-2 (polarity) user prompt, embedding the Stage-1 JSON."""
    return _POLARITY_USER_TEMPLATE.substitute(
        task_name=task_name,
        labels_csv=", ".join(labels),
        labels_block=_labels_block(labels, label_descriptions),
        text=text,
        intent_json=json.dumps(intent, ensure_ascii=False),
    )


def build_pragmatic_user_prompt(
    task_name: str,
    labels: List[str],
    label_descriptions: Dict[str, str],
    text: str,
    intent: Dict,
    polarity: Dict,
) -> str:
    """Render the Stage-3 (pragmatic) user prompt, embedding Stage-1 and Stage-2 JSON."""
    return _PRAGMATIC_USER_TEMPLATE.substitute(
        task_name=task_name,
        labels_csv=", ".join(labels),
        labels_block=_labels_block(labels, label_descriptions),
        text=text,
        intent_json=json.dumps(intent, ensure_ascii=False),
        polarity_json=json.dumps(polarity, ensure_ascii=False),
    )
