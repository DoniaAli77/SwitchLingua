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
# Experimental variant: semantic_v1
# ---------------------------------------------------------------------------
# Role-preserving refinement. The Lexical Agent still reasons from explicit
# vocabulary cues, but is guided to weigh those cues more carefully: to separate
# a sentiment word being *mentioned* from the author *expressing* it, to treat
# interface/emoji/punctuation artifacts as weak cues, and to lower confidence
# when evidence is weak. General sentiment-reasoning guidance only — not tied to
# any dataset. Enabled via SENTIMENT_PROMPT_VARIANT=semantic_v1.

_SEMANTIC_V1_ADDENDUM = """\
LEXICAL EVIDENCE GUIDANCE (sentiment) — weigh vocabulary cues carefully:
- Identify the explicit positive, negative, and neutral lexical cues actually present.
- Do NOT assign strong sentiment from isolated words alone — a single token is weak,
  defeasible evidence, not a decision on its own.
- Distinguish a sentiment word being MENTIONED or REFERENCED from the AUTHOR
  EXPRESSING that sentiment (e.g. naming a feeling is not the same as feeling it).
- Treat platform / interface words (like, dislike, unlike, comment, share, clip,
  lyrics, video, button, subscribe) as WEAK cues unless the author clearly states
  their own opinion.
- Treat emojis, slogans, and repeated punctuation as weak SUPPORTING cues only,
  never decisive evidence by themselves.
- If the lexical evidence is weak, conflicting, or only artifact-based, return
  LOWER confidence.
- Your job is to report the lexical evidence and its strength — not to resolve the
  full pragmatic meaning (target attribution and overall intent are other agents' roles).\
""".strip()

#: System prompt for the ``semantic_v1`` variant — the original prompt with the
#: lexical-evidence guidance inserted immediately before the OUTPUT FORMAT block
#: (so the JSON contract remains the final instruction). The default
#: ``SYSTEM_PROMPT`` above is left byte-for-byte unchanged.
_OUTPUT_FORMAT_MARKER = "OUTPUT FORMAT (copy this structure exactly):"
SYSTEM_PROMPT_SEMANTIC_V1 = SYSTEM_PROMPT.replace(
    _OUTPUT_FORMAT_MARKER,
    f"{_SEMANTIC_V1_ADDENDUM}\n\n{_OUTPUT_FORMAT_MARKER}",
    1,
)


def get_system_prompt(variant: str | None = None) -> str:
    """Return the lexical system prompt for the active sentiment variant.

    Defaults to the original ``SYSTEM_PROMPT`` unless ``semantic_v1`` is active.
    """
    from src.prompts._sentiment_variant import active_variant

    # semantic_v3_pragmatic_contextual is a Contextual-only upgrade: Lexical keeps its
    # semantic_v1 behaviour under it (unchanged from Design G's semantic_v1 run).
    return (
        SYSTEM_PROMPT_SEMANTIC_V1
        if active_variant(variant) in ("semantic_v1", "semantic_v3_pragmatic_contextual")
        else SYSTEM_PROMPT
    )


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
