"""Prompt templates for LLMLogicAgent.

Design goals
------------
* **Role-framed** — the model is asked to act as a logical reasoning specialist
  that applies rule-based and structural inference rather than surface lexical
  matching.
* **Label-locked** — the allowed label list is embedded verbatim.
* **Schema-first** — exact JSON contract in both system and user prompt.
* **Pattern-aware** — the model is directed to reason about relational patterns,
  co-occurrence structures, and discourse-level cues.

Response contract
-----------------
LLMLogicAgent expects exactly this JSON (no markdown fences, no extra keys):

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

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a logical reasoning specialist in a multi-agent text classification system.

Your role is to choose the most likely classification label for the ACTIVE TASK
by applying RULE-BASED AND STRUCTURAL REASONING:
- Identify relational patterns between concepts (e.g. entity-action-object structures)
- Detect co-occurrence of task-relevant concept pairs (in any language present)
- Apply discourse-level cues: enumeration, cause-effect, negation, and contrast
- Reason about which allowed label best fits the text for the active task

RULES — follow every rule exactly:
1. Choose EXACTLY ONE label from the allowed list provided. Do NOT invent labels.
2. Base the decision on logical inference — patterns and relationships, not just
   surface words — matched against the LABEL DESCRIPTIONS for the active task.
3. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
4. The JSON must contain exactly these four keys:
   - "label"      : string — must be one of the allowed labels, copied verbatim
   - "confidence" : float  — your certainty, between 0.0 and 1.0 (e.g. 0.78)
   - "reasoning"  : string — one sentence describing the logical pattern you identified
   - "evidence"   : array  — 1–5 short phrases or concept pairs from the text

OUTPUT FORMAT (copy this structure exactly):
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}\
""".strip()


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

_USER_TEMPLATE = Template("""\
TASK: $task_name (logical/rule-based reasoning)

ALLOWED LABELS (choose exactly one — no other value is valid):
$labels_csv

LABEL DESCRIPTIONS:
$labels_block

TEXT TO CLASSIFY:
$text

Apply logical and rule-based reasoning: identify relational patterns and structural cues \
in both Arabic and English that point to one domain.
Respond with JSON only. "label" must be exactly one of: $labels_csv\
""")


def build_user_prompt(
    task_name: str,
    labels: list[str],
    label_descriptions: dict[str, str],
    text: str,
) -> str:
    """Render the user prompt for a single logic classification request."""
    labels_csv = ", ".join(labels)
    labels_block = "\n".join(
        f"  {lbl} — {label_descriptions.get(lbl, '(no description)')}"
        for lbl in labels
    )
    return _USER_TEMPLATE.substitute(
        task_name=task_name,
        labels_csv=labels_csv,
        labels_block=labels_block,
        text=text,
    )
