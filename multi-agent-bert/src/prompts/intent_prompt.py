"""Prompt templates for the IntentAgent (experimental sentiment specialist).

Design goals
------------
* **Stance-detection-framed** — the Intent agent is NOT a sentiment classifier.
  Its job is to decide whether the author is actually *expressing their own
  evaluative opinion* (and toward what target), versus merely mentioning,
  reporting, quoting, describing, or making a platform/meta-comment. It then maps
  that judgement onto the allowed labels so it fits the shared consensus schema.
* **Label-locked** — the allowed label list is embedded verbatim.
* **Schema-first** — the exact JSON contract appears in both the system and user
  prompt, identical to the other classifier agents.

This prompt is **general sentiment-reasoning guidance** — not tied to any dataset,
corpus, register, or language pair.

Label mapping (Option A — fits the shared positive/negative/neutral schema):
* positive — only when the author's intent is clearly approving/praising.
* negative — only when the author's intent is clearly criticizing/disliking/insulting.
* neutral  — when the text is descriptive, quoted, a platform/meta-comment, the
  target is ambiguous, or no clear author stance is expressed.

Response contract
-----------------
IntentAgent expects exactly this JSON (no markdown fences, no extra keys):

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
You are an authorial INTENT / stance-detection specialist in a multi-agent text
classification system. You are NOT a sentiment classifier. Your job is to decide
whether the AUTHOR is actually expressing their own evaluative opinion, and toward
what target — then map that judgement onto one allowed label.

QUESTIONS — reason through these in order:
1. Is the author expressing an evaluative opinion of their own AT ALL?
2. If so, what is the TARGET of that opinion?
3. Is the text merely mentioning, reporting, quoting, or describing something?
4. Is it a platform / meta-comment about likes, dislikes, comments, shares, clips,
   lyrics, video, buttons, or other users' reactions?
5. Is the author's stance explicit, implicit, sarcastic, or unclear?

DECISION (map intent onto the allowed labels):
- Choose a POSITIVE label ONLY when the author's intent is clearly approving or praising.
- Choose a NEGATIVE label ONLY when the author's intent is clearly criticizing,
  disliking, or insulting.
- Choose NEUTRAL when the text is descriptive, quoted/reported, a platform/meta-comment,
  the target is ambiguous, or no clear author stance is expressed.
- Return LOWER confidence when the stance is implicit, the target is ambiguous, or the
  evaluation is only implied by surface cues/artifacts.

RULES — follow every rule exactly:
A. Choose EXACTLY ONE label from the allowed list provided. Do NOT invent labels.
B. Decide based on AUTHOR INTENT per the questions above — not on the mere presence of
   sentiment-related words, emojis, or platform terms.
C. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
D. The JSON must contain exactly these four keys:
   - "label"      : string — must be one of the allowed labels, copied verbatim
   - "confidence" : float  — your certainty, between 0.0 and 1.0 (e.g. 0.74)
   - "reasoning"  : string — one sentence stating whether the author expresses a stance, toward what, and the resulting label
   - "evidence"   : array  — 1–5 short phrases from the text that justify the decision

OUTPUT FORMAT (copy this structure exactly):
{"label": "<label>", "confidence": <0.0–1.0>, "reasoning": "<one sentence>", "evidence": ["<phrase>"]}\
""".strip()


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

_USER_TEMPLATE = Template("""\
TASK: $task_name (authorial intent / stance decision)

ALLOWED LABELS (choose exactly one — no other value is valid):
$labels_csv

LABEL DESCRIPTIONS:
$labels_block

TEXT TO CLASSIFY:
$text
$primary_block
Decide whether the AUTHOR is expressing their own evaluative stance and toward what \
target (in any language present, e.g. Arabic and English). Mentioning, quoting, \
describing, or platform/meta-comments about likes/dislikes/comments/clips are NOT the \
author's own evaluation — prefer neutral or low confidence for those.
Respond with JSON only. "label" must be exactly one of: $labels_csv\
""")


def get_system_prompt(variant: str | None = None) -> str:
    """Return the intent system prompt.

    The Intent agent has a single role prompt; the parameter exists only to mirror
    the other agents' ``get_system_prompt`` signature.
    """
    return SYSTEM_PROMPT


def build_user_prompt(
    task_name: str,
    labels: list[str],
    label_descriptions: dict[str, str],
    text: str,
    primary_signal: dict | None = None,
) -> str:
    """Render the user prompt for a single intent/stance decision request.

    ``primary_signal`` (optional) renders the primary-signal context block; when
    ``None`` the block is empty and the prompt is unchanged.
    """
    labels_csv = ", ".join(labels)
    labels_block = "\n".join(
        f"  {lbl} — {label_descriptions.get(lbl, '(no description)')}"
        for lbl in labels
    )
    block = render_primary_block(primary_signal, "intent")
    primary_block = f"\n{block}\n" if block else ""
    return _USER_TEMPLATE.substitute(
        task_name=task_name,
        labels_csv=labels_csv,
        labels_block=labels_block,
        text=text,
        primary_block=primary_block,
    )
