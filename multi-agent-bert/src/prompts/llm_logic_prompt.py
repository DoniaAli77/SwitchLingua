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

from src.prompts._primary_block import render_primary_block

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
# Experimental variant: semantic_v1
# ---------------------------------------------------------------------------
# Role-preserving refinement. The Logic Agent still reasons from structure, but
# is directed to do the structural work it is best placed for: resolve the
# sentiment TARGET first (the author vs other people), separate description from
# evaluation, and avoid reading emotionally-loaded content as the author's own
# negativity. General sentiment-reasoning guidance only — not tied to any
# dataset. Enabled via SENTIMENT_PROMPT_VARIANT=semantic_v1.

_SEMANTIC_V1_ADDENDUM = """\
STRUCTURE & TARGET GUIDANCE (sentiment) — resolve structure before polarity:
- FIRST identify the sentiment TARGET: what or who is the author evaluating?
- Distinguish the AUTHOR'S OWN sentiment from discussion of other people's
  actions, reactions, or opinions.
- Do NOT classify the text as negative merely because it MENTIONS negative words,
  dislike counts, plot events, death, failure, or other emotionally loaded content.
- Decide whether the text EXPRESSES an evaluation, or merely DESCRIBES / MENTIONS
  something.
- Handle negation, contrast, sarcasm, rhetorical questions, and implicit insults
  or praise, including polarity flips in their scope.
- If the text discusses platform behavior or other users without the author's own
  clear evaluation, prefer neutral or low confidence.\
""".strip()

#: System prompt for the ``semantic_v1`` variant — original prompt with the
#: structure/target guidance inserted before the OUTPUT FORMAT block. The default
#: ``SYSTEM_PROMPT`` above is left byte-for-byte unchanged.
_OUTPUT_FORMAT_MARKER = "OUTPUT FORMAT (copy this structure exactly):"
SYSTEM_PROMPT_SEMANTIC_V1 = SYSTEM_PROMPT.replace(
    _OUTPUT_FORMAT_MARKER,
    f"{_SEMANTIC_V1_ADDENDUM}\n\n{_OUTPUT_FORMAT_MARKER}",
    1,
)


def get_system_prompt(variant: str | None = None) -> str:
    """Return the logic system prompt for the active sentiment variant.

    Defaults to the original ``SYSTEM_PROMPT`` unless ``semantic_v1`` is active.
    """
    from src.prompts._sentiment_variant import active_variant

    # semantic_v3_pragmatic_contextual is a Contextual-only upgrade: Logic keeps its
    # semantic_v1 behaviour under it (Design G uses Polarity in the logic slot, so this
    # only matters for a default-trio run of v3).
    return (
        SYSTEM_PROMPT_SEMANTIC_V1
        if active_variant(variant) in ("semantic_v1", "semantic_v3_pragmatic_contextual")
        else SYSTEM_PROMPT
    )


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
$primary_block
Apply logical and rule-based reasoning: identify relational patterns and structural cues \
(in any language present, e.g. Arabic and English) that point to one allowed label.
Respond with JSON only. "label" must be exactly one of: $labels_csv\
""")


def build_user_prompt(
    task_name: str,
    labels: list[str],
    label_descriptions: dict[str, str],
    text: str,
    primary_signal: dict | None = None,
) -> str:
    """Render the user prompt for a single logic classification request.

    ``primary_signal`` (optional) renders the primary-signal context block; when
    ``None`` the block is empty and the prompt is unchanged.
    """
    labels_csv = ", ".join(labels)
    labels_block = "\n".join(
        f"  {lbl} — {label_descriptions.get(lbl, '(no description)')}"
        for lbl in labels
    )
    block = render_primary_block(primary_signal, "logical")
    primary_block = f"\n{block}\n" if block else ""
    return _USER_TEMPLATE.substitute(
        task_name=task_name,
        labels_csv=labels_csv,
        labels_block=labels_block,
        text=text,
        primary_block=primary_block,
    )
