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

Respond with JSON only. Use the schema from the system prompt.
"label" must be one of: $labels_csv\
""")


def build_user_prompt(
    task_name: str,
    labels: list[str],
    label_descriptions: dict[str, str],
    text: str,
    prior_agent_summaries: list[str] | None = None,
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

    return _USER_TEMPLATE.substitute(
        task_name=task_name,
        labels_csv=labels_csv,
        labels_block="\n".join(desc_lines),
        prior_context_block=prior_context_block,
        text=text,
    )
