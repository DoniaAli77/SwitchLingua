"""Prompt templates for the ContextualAgent.

Keeping templates here (separate from agent logic) means they can be
reviewed, versioned, and swapped without touching agent code.

Design goals
------------
* **Label-locked** – the allowed label list is embedded verbatim so the model
  cannot invent a label not in the list.
* **Schema-first** – the exact JSON schema appears in both the system prompt
  (as a contract) and the user prompt (as a reminder), eliminating ambiguity.
* **Short and strict** – no conversational filler; every sentence is either a
  constraint or a definition.
* **Easy to parse** – ``ContextualAgent`` expects plain JSON with exactly four
  keys: ``label``, ``confidence``, ``reasoning``, ``evidence``.

Response contract
-----------------
The agent expects the LLM to return **exactly** this JSON schema (no
markdown fences, no extra keys at the top level):

.. code-block:: json

    {
        "label":      "<one of the allowed labels>",
        "confidence": <float 0.0–1.0>,
        "reasoning":  "<one sentence>",
        "evidence":   ["<token or short phrase>"]
    }

Any deviation causes ``ContextualAgent`` to raise ``ContextualParseError``.
"""

from __future__ import annotations

from string import Template

from src.prompts._primary_block import render_primary_block

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
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
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}\
""".strip()


# ---------------------------------------------------------------------------
# Experimental variant: semantic_v1
# ---------------------------------------------------------------------------
# Role-preserving refinement. The Contextual Agent still classifies, but is
# directed to its intended role: interpret the WHOLE message — its communicative
# intent — detect implicit sarcasm/mockery/praise, and prioritise the overall
# message over isolated surface cues (emojis, emotional words). General
# sentiment-reasoning guidance only — not tied to any dataset. Enabled via
# SENTIMENT_PROMPT_VARIANT=semantic_v1.

_SEMANTIC_V1_ADDENDUM = """\
WHOLE-MESSAGE INTERPRETATION GUIDANCE (sentiment) — judge overall intent:
- Interpret the overall communicative intent of the ENTIRE message.
- Decide whether the text is an opinion, a meta-comment, a joke, a quote, a
  plot / content description, or a platform interaction.
- Do NOT overrule a neutral reading just because emotional words or emojis appear.
- Use context to detect implicit sarcasm, mockery, praise, or insult.
- If surface cues conflict with the overall message, PRIORITIZE the overall message.
- If the author's stance is genuinely unclear, prefer neutral or lower confidence.\
""".strip()

#: System prompt for the ``semantic_v1`` variant — original prompt with the
#: whole-message guidance inserted before the OUTPUT FORMAT block. The default
#: ``SYSTEM_PROMPT`` above is left byte-for-byte unchanged.
_OUTPUT_FORMAT_MARKER = "OUTPUT FORMAT (copy this structure exactly):"
SYSTEM_PROMPT_SEMANTIC_V1 = SYSTEM_PROMPT.replace(
    _OUTPUT_FORMAT_MARKER,
    f"{_SEMANTIC_V1_ADDENDUM}\n\n{_OUTPUT_FORMAT_MARKER}",
    1,
)


# ---------------------------------------------------------------------------
# Experimental variant: semantic_v3_pragmatic_contextual (Pragmatic Reasoner)
# ---------------------------------------------------------------------------
# Contextual-only upgrade. The Contextual Agent becomes an explicit PRAGMATIC
# REASONER: it resolves the message's pragmatic structure (speech act, target
# attribution, mention-vs-use, implicature/sarcasm, description-vs-evaluation)
# BEFORE assigning sentiment. Targets the remaining pragmatic errors while keeping
# the JSON schema and the role identical. General sentiment-reasoning guidance
# only — not tied to any dataset. Lexical and Polarity are unaffected by this
# variant (they keep their semantic_v1 behaviour). Enabled via
# SENTIMENT_PROMPT_VARIANT=semantic_v3_pragmatic_contextual.

_PRAGMATIC_ADDENDUM = """\
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
calibrate confidence to how clearly the pragmatic structure supports the decision.\
""".strip()

#: System prompt for the ``semantic_v3_pragmatic_contextual`` variant — the original
#: contextual prompt with the pragmatic-reasoning block inserted before OUTPUT FORMAT.
#: (Built from the default base, not stacked on semantic_v1, so it is a self-contained
#: pragmatic Contextual.) The default ``SYSTEM_PROMPT`` is left byte-for-byte unchanged.
SYSTEM_PROMPT_PRAGMATIC = SYSTEM_PROMPT.replace(
    _OUTPUT_FORMAT_MARKER,
    f"{_PRAGMATIC_ADDENDUM}\n\n{_OUTPUT_FORMAT_MARKER}",
    1,
)


# ---------------------------------------------------------------------------
# Experimental variant: semantic_v2_disambig
# ---------------------------------------------------------------------------
# Whole-message interpretation + two explicit disambiguations: the author's
# RELATIONSHIP to a platform action, and DESCRIPTION vs EVALUATION. General
# pragmatics only — no dataset-specific terms.

_DISAMBIG_ADDENDUM = """\
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
- If the author's stance is genuinely unclear, prefer neutral or lower confidence.\
""".strip()

SYSTEM_PROMPT_DISAMBIG = SYSTEM_PROMPT.replace(
    _OUTPUT_FORMAT_MARKER,
    f"{_DISAMBIG_ADDENDUM}\n\n{_OUTPUT_FORMAT_MARKER}",
    1,
)


def get_system_prompt(variant: str | None = None) -> str:
    """Return the contextual system prompt for the active sentiment variant.

    Defaults to the original ``SYSTEM_PROMPT``. ``semantic_v1`` → whole-message
    guidance; ``semantic_v3_pragmatic_contextual`` → the Pragmatic Reasoner prompt;
    ``semantic_v2_disambig`` → whole-message + platform-relationship / description-vs-eval.
    """
    from src.prompts._sentiment_variant import active_variant

    v = active_variant(variant)
    if v == "semantic_v3_pragmatic_contextual":
        return SYSTEM_PROMPT_PRAGMATIC
    if v == "semantic_v2_disambig":
        return SYSTEM_PROMPT_DISAMBIG
    if v == "semantic_v1":
        return SYSTEM_PROMPT_SEMANTIC_V1
    return SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------
# Placeholders (Python string.Template syntax):
#   $task_name     — task identifier, e.g. "topic_classification"
#   $labels_csv    — comma-separated allowed labels, e.g. "tech, sports, finance"
#   $labels_block  — one line per label: "  label_name — description"
#   $text          — the raw input text to classify
# ---------------------------------------------------------------------------

_USER_TEMPLATE = Template("""\
TASK: $task_name

ALLOWED LABELS (you must pick exactly one of these — no other value is valid):
$labels_csv

LABEL DESCRIPTIONS:
$labels_block
$prior_context_block

TEXT:
$text
$primary_block
Respond with JSON only. Use the schema from the system prompt.
"label" must be one of: $labels_csv\
""")


def build_user_prompt(
    task_name: str,
    labels: list[str],
    label_descriptions: dict[str, str],
    text: str,
    prior_agent_summaries: list[str] | None = None,
    primary_signal: dict | None = None,
) -> str:
    """Render the user prompt for a single classification request.

    Parameters
    ----------
    task_name:
        Descriptive task identifier, e.g. ``"topic_classification"``.
    labels:
        Ordered list of allowed output labels.  Shown both as a CSV list
        (for quick scanning) and as a description block (for context).
    label_descriptions:
        Human-readable description per label.  Labels absent from this dict
        are shown as ``"(no description)"``.
    text:
        The raw input text to classify.  Not wrapped in extra quotes so
        Arabic and mixed-script text renders cleanly.
    prior_agent_summaries:
        Optional compact summaries from upstream agents (primary/lexical/logic).
        When provided, they are appended under a dedicated context block.

    Returns
    -------
    str
        Fully rendered prompt string ready to concatenate with
        ``SYSTEM_PROMPT`` and send to ``LLMClient.generate``.
    """
    labels_csv = ", ".join(labels)
    desc_lines = [
        f"  {lbl} — {label_descriptions.get(lbl, '(no description)')}"
        for lbl in labels
    ]
    prior_context_block = ""
    if prior_agent_summaries:
        summary_lines = "\n".join(f"  - {item}" for item in prior_agent_summaries)
        prior_context_block = (
            "\nPRIOR AGENT SUMMARIES (context only; use as weak hints):\n"
            f"{summary_lines}\n"
        )

    block = render_primary_block(primary_signal, "contextual")
    primary_block = f"\n{block}\n" if block else ""

    return _USER_TEMPLATE.substitute(
        task_name=task_name,
        labels_csv=labels_csv,
        labels_block="\n".join(desc_lines),
        prior_context_block=prior_context_block,
        text=text,
        primary_block=primary_block,
    )
