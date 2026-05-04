"""Prompt templates for LLMExplainabilityAgent.

Design goals
------------
* **Synthesis-focused** — the model is given all available specialist agent
  outputs and asked to produce a concise, readable explanation of the
  pipeline's final decision.
* **Schema-first** — exact JSON contract in both system and user prompt.
* **Evidence-grounded** — the model must cite specific agent votes or text
  tokens as evidence, and flag any disagreements as caveats.

Response contract
-----------------
LLMExplainabilityAgent expects exactly this JSON (no markdown fences, no extra
keys at the top level):

.. code-block:: json

    {
        "summary":  "<one sentence explaining the final decision>",
        "evidence": ["<agent vote or text token that supported the decision>", ...],
        "caveats":  ["<any dissenting vote or uncertainty>", ...]
    }
"""

from __future__ import annotations

from string import Template

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an explainability specialist in a multi-agent text classification system.

Your role is to synthesize the outputs of multiple agents and produce a clear, \
concise explanation of why the pipeline classified the text with the final label.

RULES — follow every rule exactly:
1. Write a single-sentence summary explaining the final classification decision.
2. List the key pieces of evidence (agent votes, text tokens, or confidence values) \
that supported the final label.
3. List any caveats: agents that disagreed, low-confidence votes, or notable uncertainties.
   If there are no caveats, return an empty array for "caveats".
4. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
5. The JSON must contain exactly these three keys:
   - "summary"  : string — one sentence
   - "evidence" : array  — 1–5 short strings
   - "caveats"  : array  — 0–3 short strings (empty array if none)

OUTPUT FORMAT (copy this structure exactly):
{"summary": "<one sentence>", "evidence": ["<item>"], "caveats": []}\
""".strip()


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

_USER_TEMPLATE = Template("""\
CLASSIFICATION TASK: $task_name
FINAL LABEL: $final_label
FINAL CONFIDENCE: $final_confidence

AGENT OUTPUTS:
$agent_block

TEXT THAT WAS CLASSIFIED:
$text

Synthesize the above into a concise explanation of the final classification decision.
Respond with JSON only using the schema from the system prompt.\
""")


def build_user_prompt(
    task_name: str,
    final_label: str,
    final_confidence: str,
    agent_summaries: list[str],
    text: str,
) -> str:
    """Render the user prompt for a single explainability request.

    Parameters
    ----------
    task_name:
        Descriptive task identifier, e.g. ``"topic_classification"``.
    final_label:
        The label chosen by the pipeline's final output.
    final_confidence:
        Formatted confidence string, e.g. ``"87.3%"``.
    agent_summaries:
        One string per agent that ran, each summarising its vote and confidence.
        An empty list is acceptable (e.g. primary-only mode).
    text:
        The raw input text that was classified.

    Returns
    -------
    str
        Fully rendered prompt string ready to concatenate with ``SYSTEM_PROMPT``
        and send to ``LLMClient.generate``.
    """
    if agent_summaries:
        agent_block = "\n".join(f"  - {s}" for s in agent_summaries)
    else:
        agent_block = "  (no specialist agents ran; primary model only)"

    return _USER_TEMPLATE.substitute(
        task_name=task_name,
        final_label=final_label,
        final_confidence=final_confidence,
        agent_block=agent_block,
        text=text,
    )
