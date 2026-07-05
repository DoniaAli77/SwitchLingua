"""Prompt templates for ``sequential_sentiment_v2`` (forward-pragmatics pipeline).

Opt-in; the default pipeline never imports this. v2 fixes v1's confirmation-anchored
review stage by moving pragmatics **upstream as structured features** and deciding
polarity **once**, feature-aware, with no prior label shown to the final stage:

* **Stage 1 — Intent** (lean opinion-existence gate; no sentiment label).
* **Stage 2 — Pragmatic feature extractor** (structured features; **no** sentiment label).
* **Stage 3 — Polarity resolver** (decides positive/negative/neutral once, using the text
  + Intent + Pragmatic features; is NOT shown a prior polarity to ratify).

General sentiment-reasoning guidance only — no dataset/author/benchmark named. JSON-only.
Every stage carries a ``confidence`` for calibration. Allowed labels are injected verbatim
into the Stage-3 user prompt (label-generic, as in v1).
"""

from __future__ import annotations

import json
from string import Template
from typing import Dict, List

# ===========================================================================
# Stage 1 — Intent (lean opinion-existence detector)
# ===========================================================================

INTENT_V2_SYSTEM_PROMPT = """\
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
{"opinion_expressed": true, "target": "<string or null>", "confidence": 0.0, "evidence": ["<span>"]}\
""".strip()

_INTENT_V2_USER_TEMPLATE = Template("""\
Message:
\"\"\"
$text
\"\"\"
Return the JSON object only.\
""")


# ===========================================================================
# Stage 2 — Pragmatic feature extractor (NO sentiment label)
# ===========================================================================

PRAGMATIC_FEATURES_SYSTEM_PROMPT = """\
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
{"speech_act": "evaluate", "target": "<string or null>", "target_attribution": "author", "use_vs_mention": "use", "platform_meta": false, "description_vs_evaluation": "evaluation", "sarcasm_or_irony": false, "implicit_stance": "none", "stance_strength": "moderate", "confidence": 0.0, "evidence": ["<span>"]}\
""".strip()

_PRAGMATIC_FEATURES_USER_TEMPLATE = Template("""\
Message:
\"\"\"
$text
\"\"\"

Intent analysis (JSON):
$intent_json

Return the JSON object only (features, NO sentiment label).\
""")


# ===========================================================================
# Stage 3 — Polarity resolver (decides once, feature-aware, sees no prior label)
# ===========================================================================

POLARITY_RESOLVER_SYSTEM_PROMPT = """\
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
{"label": "<label>", "confidence": 0.0, "used_features": ["sarcasm_or_irony"], "reasoning": "<one sentence>", "evidence": ["<span>"]}\
""".strip()

_POLARITY_RESOLVER_USER_TEMPLATE = Template("""\
TASK: $task_name (feature-aware polarity resolution)

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

Pragmatic features (JSON):
$pragmatic_json

Decide the AUTHOR's sentiment now, using the features above. Return the JSON object only.
"label" must be exactly one of: $labels_csv\
""")


# ===========================================================================
# User-prompt builders
# ===========================================================================

def _labels_block(labels: List[str], label_descriptions: Dict[str, str]) -> str:
    return "\n".join(
        f"  {lbl} - {label_descriptions.get(lbl, '(no description)')}"
        for lbl in labels
    )


def build_intent_v2_user_prompt(text: str) -> str:
    return _INTENT_V2_USER_TEMPLATE.substitute(text=text)


def build_pragmatic_features_user_prompt(text: str, intent: Dict) -> str:
    return _PRAGMATIC_FEATURES_USER_TEMPLATE.substitute(
        text=text,
        intent_json=json.dumps(intent, ensure_ascii=False),
    )


def build_polarity_resolver_user_prompt(
    task_name: str,
    labels: List[str],
    label_descriptions: Dict[str, str],
    text: str,
    intent: Dict,
    pragmatic: Dict,
) -> str:
    return _POLARITY_RESOLVER_USER_TEMPLATE.substitute(
        task_name=task_name,
        labels_csv=", ".join(labels),
        labels_block=_labels_block(labels, label_descriptions),
        text=text,
        intent_json=json.dumps(intent, ensure_ascii=False),
        pragmatic_json=json.dumps(pragmatic, ensure_ascii=False),
    )
