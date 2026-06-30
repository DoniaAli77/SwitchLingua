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


def get_system_prompt(variant: str | None = None) -> str:
    """Return the contextual system prompt for the active sentiment variant.

    Defaults to the original ``SYSTEM_PROMPT`` unless ``semantic_v1`` is active.
    """
    from src.prompts._sentiment_variant import active_variant

    return (
        SYSTEM_PROMPT_SEMANTIC_V1
        if active_variant(variant) == "semantic_v1"
        else SYSTEM_PROMPT
    )


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
