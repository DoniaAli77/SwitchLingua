"""Prompt templates for LLMNERAgent (sequence labeling / NER).

The NER counterpart of ``llm_lexical_prompt`` etc.: instead of one label per
sentence, the model returns **one BIO tag per token**. Designed for
Arabic-English code-switched text, where the model must recognise entities
written in **either** script (Arabic-script names have no capital-letter cue).

Dynamic entity types
--------------------
Which entities the model looks for is driven entirely by ``task_config.labels``
(and optional ``task_config.label_descriptions``) — exactly like the topic
classifier's categories. The entity types are derived from the BIO labels at
prompt-build time, so adding e.g. ``B-MISC``/``I-MISC`` to the label set is a
config change with no code change.

Response contract
-----------------
LLMNERAgent expects exactly this JSON (no markdown fences, no extra keys):

.. code-block:: json

    {
        "tags":      ["O", "B-PER", "I-PER", ...],   // exactly one per token, in order
        "reasoning": "<one short sentence>"
    }

``tags`` must have the same length as the input token list and use only the
allowed BIO tags.
"""

from __future__ import annotations

from string import Template
from typing import Dict, List, Optional

# Fallback descriptions for the common CoNLL entity types (used only when the
# task config does not supply its own label_descriptions).
_DEFAULT_TYPE_DESC: Dict[str, str] = {
    "PER": "a person's name (first, last, or full)",
    "PERS": "a person's name (first, last, or full)",
    "PERSON": "a person's name (first, last, or full)",
    "ORG": "an organization, company, institution, or team",
    "ORGANISATION": "an organization, company, institution, or team",
    "ORGANIZATION": "an organization, company, institution, or team",
    "LOC": "a location, city, country, or place",
    "LOCATION": "a location, city, country, or place",
    "MISC": "a named entity that is not a person, organization, or location",
    "GPE": "a geopolitical entity (country, city, state)",
    "DATE": "a date or time expression",
}


def entity_types_from_labels(labels: List[str]) -> List[str]:
    """Derive the ordered unique entity types from a BIO label set.

    ``["O","B-PER","I-PER","B-ORG",...]`` → ``["PER", "ORG", ...]`` (``O`` dropped).
    """
    types: List[str] = []
    for lbl in labels:
        if lbl == "O":
            continue
        # Handle both BIO labels ("B-PER" -> "PER") and bare type labels
        # ("PERS" -> "PERS"), so type-level label sets work too.
        etype = lbl.split("-", 1)[1] if "-" in lbl else lbl
        if etype not in types:
            types.append(etype)
    return types


def _type_descriptions(
    types: List[str],
    label_descriptions: Optional[Dict[str, str]],
) -> str:
    """Render a '  TYPE — description' block, preferring config descriptions."""
    label_descriptions = label_descriptions or {}
    lines = []
    for t in types:
        # Prefer an explicit description for the type or its B-/I- tag; else default.
        desc = (
            label_descriptions.get(t)
            or label_descriptions.get(f"B-{t}")
            or _DEFAULT_TYPE_DESC.get(t.upper())
            or "a named entity of this type"
        )
        lines.append(f"  {t} — {desc}")
    return "\n".join(lines)


_SYSTEM_TEMPLATE = Template("""\
You are a Named Entity Recognition (NER) specialist in a multi-agent system for
CODE-SWITCHED Arabic-English text.

Your job: assign ONE tag to EACH token, using the BIO scheme, for the entity
types defined by the ACTIVE TASK (listed below). The text mixes Arabic and
English, sometimes in the same sentence. Entities may be written in EITHER
script:
- Arabic-script names (e.g. أحمد, القاهرة, جامعة القاهرة) are entities too —
  Arabic has no capital letters, so judge by meaning and context, not casing.
- English/Latin names (e.g. Google, United Nations, London) as well.

ENTITY TYPES for this task:
$types_block

BIO scheme:
- "B-XXX" marks the FIRST token of an entity of type XXX.
- "I-XXX" marks a CONTINUATION token of the same entity (multi-word entities).
- "O" marks a token that is not part of any entity.

RULES — follow every rule exactly:
1. Output ONE tag per input token, in the SAME ORDER, using ONLY the allowed tags.
2. Use only the entity types listed above. Do NOT invent new types.
3. Multi-word entities: first token B-, following tokens I- (same type).
4. Do NOT merge, split, add, or drop tokens. The number of tags MUST equal the
   number of input tokens.
5. Respond with ONLY a JSON object. No markdown fences, no prose, no extra keys.
6. The JSON must contain exactly these two keys:
   - "tags"      : array of strings — one allowed tag per token, in order
   - "reasoning" : string — one short sentence naming the entities you found

OUTPUT FORMAT (copy this structure exactly):
{"tags": ["O", "B-PER", ...], "reasoning": "<one sentence>"}\
""")


def get_system_prompt(
    labels: Optional[List[str]] = None,
    label_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Return the NER system prompt, with entity types derived from *labels*.

    When *labels* is None a generic PER/ORG/LOC description is used (backwards
    compatible), but callers should pass the task labels so the entity types
    are fully task-driven.
    """
    types = entity_types_from_labels(labels) if labels else ["PER", "ORG", "LOC"]
    return _SYSTEM_TEMPLATE.substitute(
        types_block=_type_descriptions(types, label_descriptions)
    ).strip()


_USER_TEMPLATE = Template("""\
TASK: $task_name (named entity recognition, BIO tagging)

ALLOWED TAGS (use only these, copied verbatim):
$tags_csv

SENTENCE:
$text

TOKENS (index: token) — return exactly $n tags, one per token, in this order:
$numbered_tokens

Respond with JSON only: an object with "tags" (an array of exactly $n tags from
the allowed list, in order) and a one-sentence "reasoning".\
""")


def build_user_prompt(
    task_name: str,
    labels: List[str],
    tokens: List[str],
    text: str,
) -> str:
    """Render the user prompt for one NER tagging request."""
    tags_csv = ", ".join(labels)
    numbered = "\n".join(f"  {i}: {tok}" for i, tok in enumerate(tokens))
    return _USER_TEMPLATE.substitute(
        task_name=task_name,
        tags_csv=tags_csv,
        text=text,
        numbered_tokens=numbered,
        n=len(tokens),
    )
