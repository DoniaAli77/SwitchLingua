"""Prompt templates for LLMLexicalAgent.

Design goals
------------
* **Role-framed** — the model is asked to act as a lexical specialist that
  focuses on surface-level vocabulary cues, not deep semantic reasoning.
* **Label-locked** — the allowed label list is embedded verbatim.
* **Schema-first** — exact JSON contract appears in both the system and user
  prompt.
* **Evidence-grounded** — the model must cite concrete tokens from the text.

Response contract
-----------------
LLMLexicalAgent expects exactly this JSON (no markdown fences, no extra keys):

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
You are a lexical analysis specialist in a multi-agent text classification system.

Your role is to choose the most likely classification label for the ACTIVE TASK,
based on VOCABULARY CUES ONLY:
- Surface-level words, terms, and phrases that appear explicitly in the text
- Task-relevant terminology and characteristic expressions (in any language
  present in the text, e.g. Arabic and English)
- Named entities and salient tokens

RULES — follow every rule exactly:
1. Choose EXACTLY ONE label from the allowed list provided. Do NOT invent labels.
2. Base the decision on explicit lexical evidence — words and phrases visible in
   the text — matched against the LABEL DESCRIPTIONS for the active task.
3. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
4. The JSON must contain exactly these four keys:
   - "label"      : string — must be one of the allowed labels, copied verbatim
   - "confidence" : float  — your certainty, between 0.0 and 1.0 (e.g. 0.82)
   - "reasoning"  : string — one sentence citing the key vocabulary you found
   - "evidence"   : array  — 1–5 tokens or short phrases from the text that support the label

OUTPUT FORMAT (copy this structure exactly):
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}\
""".strip()


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

_USER_TEMPLATE = Template("""\
TASK: $task_name (lexical analysis)

ALLOWED LABELS (choose exactly one — no other value is valid):
$labels_csv

LABEL DESCRIPTIONS:
$labels_block

TEXT TO CLASSIFY:
$text
$primary_block
Perform lexical analysis: identify task-relevant vocabulary (in any language \
present, e.g. Arabic and English).
Respond with JSON only. "label" must be exactly one of: $labels_csv\
""")


def build_user_prompt(
    task_name: str,
    labels: list[str],
    label_descriptions: dict[str, str],
    text: str,
    primary_signal: dict | None = None,
) -> str:
    """Render the user prompt for a single lexical classification request.

    ``primary_signal`` (optional) renders the primary-signal context block; when
    ``None`` the block is empty and the prompt is unchanged.
    """
    labels_csv = ", ".join(labels)
    labels_block = "\n".join(
        f"  {lbl} — {label_descriptions.get(lbl, '(no description)')}"
        for lbl in labels
    )
    block = render_primary_block(primary_signal, "lexical")
    primary_block = f"\n{block}\n" if block else ""
    return _USER_TEMPLATE.substitute(
        task_name=task_name,
        labels_csv=labels_csv,
        labels_block=labels_block,
        text=text,
        primary_block=primary_block,
    )
