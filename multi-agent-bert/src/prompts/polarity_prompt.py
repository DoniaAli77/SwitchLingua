"""Prompt templates for the PolarityAgent (experimental sentiment specialist).

Design goals
------------
* **Decision-framed** — unlike a cue lister, this agent must *decide* whether the
  author expresses an evaluative attitude and, if so, its polarity. It does not
  merely enumerate sentiment words (that is the Lexical agent's job), nor does it
  perform full pragmatic/social interpretation (that is the Contextual agent's job).
* **Label-locked** — the allowed label list is embedded verbatim.
* **Schema-first** — the exact JSON contract appears in both the system and user
  prompt, identical to the other classifier agents.

This prompt is **general sentiment-reasoning guidance** — not tied to any dataset,
corpus, register, or language pair.

Response contract
-----------------
PolarityAgent expects exactly this JSON (no markdown fences, no extra keys):

.. code-block:: json

    {
        "label":      "<one of the allowed labels>",
        "confidence": <float 0.0–1.0>,
        "reasoning":  "<one sentence>",
        "evidence":   ["<token or short phrase>", ...]
    }
"""

from __future__ import annotations

from string import Template

from src.prompts._primary_block import render_primary_block

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
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
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}\
""".strip()


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

_USER_TEMPLATE = Template("""\
TASK: $task_name (polarity decision)

ALLOWED LABELS (choose exactly one — no other value is valid):
$labels_csv

LABEL DESCRIPTIONS:
$labels_block

TEXT TO CLASSIFY:
$text
$primary_block
Decide whether the author expresses an evaluative attitude and, if so, its polarity \
(in any language present, e.g. Arabic and English). Distinguish expressing a polarity \
from merely mentioning sentiment-related words or artifacts.
Respond with JSON only. "label" must be exactly one of: $labels_csv\
""")


# ---------------------------------------------------------------------------
# Experimental variant: semantic_v2_disambig
# ---------------------------------------------------------------------------
# Refines step 3 (mention -> neutral) into a general RELATIONSHIP disambiguation
# for platform actions, plus a description-vs-evaluation rule. General pragmatics
# only — no dataset-specific terms.

_DISAMBIG_ADDENDUM = """\
PLATFORM-ACTION & DESCRIPTION DISAMBIGUATION (sentiment):
- A platform action (like, dislike, unlike, comment, share, follow, trend, view) is NOT
  neutral by default. Decide the author's RELATIONSHIP to it: merely reporting or counting
  it → neutral; endorsing or celebrating it → carry the endorsed polarity; objecting to it,
  or attacking the people doing it → negative.
- A sentiment word merely MENTIONED, quoted, or attributed to other people is not the
  author expressing it.
- Positive or negative words appearing INSIDE described content (a plot, events, a report or
  quote of others) are not the author's own stance → prefer neutral unless the author
  themselves evaluates.\
""".strip()

_OUTPUT_FORMAT_MARKER = "OUTPUT FORMAT (copy this structure exactly):"
SYSTEM_PROMPT_DISAMBIG = SYSTEM_PROMPT.replace(
    _OUTPUT_FORMAT_MARKER,
    f"{_DISAMBIG_ADDENDUM}\n\n{_OUTPUT_FORMAT_MARKER}",
    1,
)


def get_system_prompt(variant: str | None = None) -> str:
    """Return the polarity system prompt for the active sentiment variant.

    Single role prompt by default; ``semantic_v2_disambig`` adds the platform-action /
    description disambiguation guidance.
    """
    from src.prompts._sentiment_variant import active_variant

    if active_variant(variant) == "semantic_v2_disambig":
        return SYSTEM_PROMPT_DISAMBIG
    return SYSTEM_PROMPT


def build_user_prompt(
    task_name: str,
    labels: list[str],
    label_descriptions: dict[str, str],
    text: str,
    primary_signal: dict | None = None,
) -> str:
    """Render the user prompt for a single polarity decision request.

    ``primary_signal`` (optional) renders the primary-signal context block; when
    ``None`` the block is empty and the prompt is unchanged.
    """
    labels_csv = ", ".join(labels)
    labels_block = "\n".join(
        f"  {lbl} — {label_descriptions.get(lbl, '(no description)')}"
        for lbl in labels
    )
    block = render_primary_block(primary_signal, "polarity")
    primary_block = f"\n{block}\n" if block else ""
    return _USER_TEMPLATE.substitute(
        task_name=task_name,
        labels_csv=labels_csv,
        labels_block=labels_block,
        text=text,
        primary_block=primary_block,
    )
