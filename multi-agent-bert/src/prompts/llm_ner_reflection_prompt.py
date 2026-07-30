"""Prompt templates for LLMNERReflectionAgent.

Reflection (a.k.a. review-and-correct) prompt for NER. Instead of tagging from
scratch, the LLM is shown the PRIMARY model's draft tags and asked to *correct
only the mistakes* — keeping correct tags, and focusing on what the primary is
weak at (Arabic-script entities, multi-word spans, and entity types the frozen
primary cannot predict). This is the pattern shown to make agents help rather
than hurt in recent multi-agent NER work (KDR-Agent reflective analysis;
CROSSAGENTIE debate). Entity types remain dynamic from ``task_config.labels``.

Response contract (same as the base LLM NER agent):

.. code-block:: json

    {"tags": ["O", "B-PER", ...], "reasoning": "<what you corrected and why>"}
"""

from __future__ import annotations

from string import Template
from typing import Dict, List, Optional

from src.prompts.llm_ner_prompt import _type_descriptions, entity_types_from_labels

_SYSTEM_TEMPLATE = Template("""\
You are a Named Entity Recognition (NER) REVIEWER in a multi-agent system for
CODE-SWITCHED Arabic-English text.

A primary model has already produced DRAFT BIO tags. Your job is NOT to tag from
scratch — it is to REVIEW the draft and CORRECT ONLY THE MISTAKES, then return
the full corrected tag list.

The primary model is usually right, but it is weak at:
- Arabic-script entities (أحمد, القاهرة, جامعة القاهرة) — Arabic has no capital
  letters, so the model often misses these; add them.
- Multi-word entities (mark continuation tokens I-XXX correctly).
- Entity types it was never trained on (see the task's ENTITY TYPES below) —
  the primary can only guess a limited set, so fill in the types it cannot do.

ENTITY TYPES for this task:
$types_block

REVIEWER RULES — follow exactly:
1. KEEP every draft tag that is already correct. Change ONLY genuine errors.
2. Return ONE tag per token, in order, using ONLY the allowed tags.
3. BIO scheme: first token of an entity B-XXX, continuation tokens I-XXX, other O.
4. Do NOT add, drop, merge, or split tokens. Tag count MUST equal token count.
5. Respond with ONLY a JSON object, no markdown, exactly these keys:
   - "tags"      : array of corrected tags, one per token, in order
   - "reasoning" : one short sentence naming what you corrected (or "no changes")

OUTPUT FORMAT (copy exactly):
{"tags": ["O", "B-PER", ...], "reasoning": "<what you corrected>"}\
""")


def get_reflection_system_prompt(
    labels: Optional[List[str]] = None,
    label_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Return the reflection system prompt with task-driven entity types."""
    types = entity_types_from_labels(labels) if labels else ["PER", "ORG", "LOC"]
    return _SYSTEM_TEMPLATE.substitute(
        types_block=_type_descriptions(types, label_descriptions)
    ).strip()


_USER_TEMPLATE = Template("""\
TASK: $task_name (named entity recognition — REVIEW the draft)

ALLOWED TAGS (use only these, copied verbatim):
$tags_csv

SENTENCE:
$text

DRAFT (index: token -> primary model's tag). Correct only the wrong ones:
$draft_block

Return JSON only: "tags" = the full corrected list of exactly $n tags (one per
token, in order), and a one-sentence "reasoning".\
""")


def build_reflection_user_prompt(
    task_name: str,
    labels: List[str],
    tokens: List[str],
    text: str,
    draft_tags: List[str],
) -> str:
    """Render the reflection user prompt, showing the primary's draft per token."""
    tags_csv = ", ".join(labels)
    draft_block = "\n".join(
        f"  {i}: {tok} -> {draft_tags[i] if i < len(draft_tags) else 'O'}"
        for i, tok in enumerate(tokens)
    )
    return _USER_TEMPLATE.substitute(
        task_name=task_name,
        tags_csv=tags_csv,
        text=text,
        draft_block=draft_block,
        n=len(tokens),
    )
