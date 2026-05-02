"""Prompt templates for the DeliberationAgent.

Keeping templates here (separate from agent logic) allows them to be
reviewed, versioned, and swapped without touching agent code.

Design goals
------------
* **Vote-aware** — the prompt embeds every available specialist vote so the
  LLM can see where agents agree and where they conflict.
* **Label-locked** — the allowed label list is injected verbatim so the model
  cannot invent a label not in the task config.
* **Schema-first** — the exact JSON schema is stated in both the system prompt
  and the user prompt so there is no ambiguity about what to return.
* **Weak-signal framing** — the prompt explicitly asks the model to recommend
  OR justify, lowering the bar when agents strongly agree.

Response contract
-----------------
The agent expects the LLM to return **exactly** this JSON schema (no markdown
fences, no extra keys at the top level):

.. code-block:: json

    {
        "recommended_label": "<one of the allowed labels>",
        "confidence":        <float 0.0–1.0>,
        "justification":     "<one or two sentences>",
        "mode":              "recommendation" | "justification"
    }

Any deviation causes ``DeliberationAgent`` to raise ``DeliberationParseError``
and write nothing to ``state.deliberation_output``.
"""

from __future__ import annotations

from string import Template
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a deliberation engine reviewing the outputs of multiple specialist text classification agents.

RULES — follow every rule exactly:
1. Read the agent votes below and determine whether they agree or conflict.
2. Choose EXACTLY ONE label from the allowed list as your recommendation.
   Do NOT invent, abbreviate, or paraphrase a label.
3. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
4. The JSON must contain exactly these four keys:
   - "recommended_label" : string — must be one of the allowed labels, copied verbatim
   - "confidence"        : float  — your certainty, between 0.0 and 1.0 (e.g. 0.82)
   - "justification"     : string — one or two sentences explaining your conclusion
   - "mode"              : string — use "recommendation" when you favour one label clearly,
                                    "justification" when you are explaining why the majority view holds

OUTPUT FORMAT (copy this structure exactly):
{"recommended_label": "<label>", "confidence": <0.0–1.0>, "justification": "<sentence>", "mode": "recommendation"}\
""".strip()


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------
# Placeholders (Python string.Template syntax):
#   $task_name    — task identifier, e.g. "topic_classification"
#   $labels_csv   — comma-separated allowed labels
#   $labels_block — one line per label: "  label_name — description"
#   $votes_block  — one line per agent vote
# ---------------------------------------------------------------------------

_USER_TEMPLATE = Template("""\
TASK: $task_name

ALLOWED LABELS (you must pick exactly one as recommended_label — no other value is valid):
$labels_csv

LABEL DESCRIPTIONS:
$labels_block

AGENT VOTES:
$votes_block

Review the votes above and produce a deliberation output.\
""")


def build_user_prompt(
    task_name: str,
    labels: List[str],
    label_descriptions: Dict[str, str],
    agent_votes: List[Tuple[str, Optional[str], Optional[float], str]],
) -> str:
    """Render the deliberation user prompt.

    Parameters
    ----------
    task_name:
        Task identifier embedded verbatim in the prompt.
    labels:
        Ordered list of all allowed labels.
    label_descriptions:
        Mapping of ``label -> description`` for the prompt body.
    agent_votes:
        List of ``(agent_name, label, confidence, notes)`` tuples, one per
        specialist agent that produced output.  ``label`` and ``confidence``
        may be ``None`` when an agent did not produce a usable classification.

    Returns
    -------
    str
        Fully rendered user prompt string.
    """
    labels_csv = ", ".join(labels)
    labels_block = "\n".join(
        f"  {lbl} — {label_descriptions.get(lbl, 'No description.')}"
        for lbl in labels
    )

    if agent_votes:
        vote_lines: List[str] = []
        for agent_name, label, confidence, notes in agent_votes:
            conf_str = f"{confidence:.3f}" if confidence is not None else "n/a"
            label_str = label if label is not None else "n/a"
            line = f"  {agent_name}: label='{label_str}', confidence={conf_str}"
            if notes:
                line += f", note='{notes}'"
            vote_lines.append(line)
        votes_block = "\n".join(vote_lines)
    else:
        votes_block = "  (no agent votes available)"

    return _USER_TEMPLATE.substitute(
        task_name=task_name,
        labels_csv=labels_csv,
        labels_block=labels_block,
        votes_block=votes_block,
    )
