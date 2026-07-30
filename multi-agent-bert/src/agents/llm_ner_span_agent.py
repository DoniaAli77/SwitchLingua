"""llm_ner_span_agent.py — span-extraction NER agent with DETERMINISTIC alignment.

Fixes the alignment-drift failure mode: instead of asking the LLM to emit one
tag per token in order (which drifts on long/Arabic sentences), we ask it to
LIST the entities as ``{text, type}`` — the LLM's strength — and then a pure,
deterministic function maps each entity's text back onto the token positions.
The LLM never counts tokens, so positional drift is impossible.

Contract from the LLM:

.. code-block:: json

    {"entities": [{"text": "الجنوب اللبناني", "type": "LOC"},
                  {"text": "Lebanon", "type": "LOC"}],
     "reasoning": "<one sentence>"}

``align_entities_to_tokens`` then produces one type-level tag per token.
Sequence-labeling only; classification untouched.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.agents.llm_ner_agent import (
    _LLM_CONFIDENCE,
    _SLOT_ATTR,
    LLMNERParseError,
    _extract_json,
    coerce_to_valid,
)
from src.llm.base_client import LLMClient
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    SequenceLabelingOutput,
    TokenTag,
)

_SKIP_NOTE = "LLMNERSpanAgent: task_type is not sequence_labeling — skipped."


def _norm(tok: str) -> str:
    """Normalise a token for matching: strip edge punctuation, lower-case."""
    return tok.strip(" \t\"'`.,;:!?()[]{}«»،؛؟").lower()


def _find_span(norm_tokens: List[str], ent_tokens: List[str], claimed: List[bool]) -> Optional[int]:
    """Return the first start index where *ent_tokens* matches consecutively and
    none of those positions are already claimed; else None."""
    k = len(ent_tokens)
    if k == 0:
        return None
    for i in range(len(norm_tokens) - k + 1):
        if any(claimed[i + j] for j in range(k)):
            continue
        if all(norm_tokens[i + j] == ent_tokens[j] for j in range(k)):
            return i
    return None


def align_entities_to_tokens(
    tokens: List[str],
    entities: List[Dict],
    valid: set,
) -> List[str]:
    """Deterministically map LLM entity spans onto per-token TYPE tags.

    Parameters
    ----------
    tokens:
        The canonical token list.
    entities:
        List of ``{"text": str, "type": str}`` from the LLM.
    valid:
        Allowed tag set (type-level, e.g. ``{"O","PERS","LOC","ORG","MISC"}``).

    Returns
    -------
    list[str]
        One tag per token (default ``"O"``). Entities whose text cannot be
        located in *tokens* are dropped (safe — cannot misplace them). Longer
        entities are matched first; each token is claimed by at most one entity.
    """
    tags = ["O"] * len(tokens)
    claimed = [False] * len(tokens)
    norm_tokens = [_norm(t) for t in tokens]

    # Longest entities first so multi-word spans win over their sub-words.
    def _elen(e):
        return len(str(e.get("text", "")).split())
    for ent in sorted(entities, key=_elen, reverse=True):
        text = str(ent.get("text", "")).strip()
        etype = coerce_to_valid(str(ent.get("type", "")), valid)
        if not text or etype == "O":
            continue
        ent_tokens = [_norm(x) for x in text.split() if _norm(x)]
        pos = _find_span(norm_tokens, ent_tokens, claimed)
        if pos is None:
            continue
        for j in range(len(ent_tokens)):
            tags[pos + j] = etype
            claimed[pos + j] = True
    return tags


def spans_from_tags(tokens: List[str], tags: List[str]) -> List[Dict]:
    """Convert per-token IO/BIO tags into ``{text, type}`` entity spans.

    Consecutive tokens of the same entity type are merged into one span (IO
    scheme). Used to turn gold train sentences into few-shot examples.
    """
    spans: List[Dict] = []
    cur_type, buf = None, []

    def _flush():
        if buf:
            spans.append({"text": " ".join(buf), "type": cur_type})

    for tok, tag in zip(tokens, tags):
        ty = tag.split("-", 1)[1] if "-" in tag else (None if tag == "O" else tag)
        if ty is None:
            _flush(); buf.clear(); cur_type = None
        elif ty == cur_type:
            buf.append(tok)
        else:
            _flush(); buf.clear(); cur_type = ty; buf.append(tok)
    _flush()
    return spans


def build_span_prompt(task_name: str, labels: List[str], tokens: List[str],
                      text: str, descriptions: Optional[Dict[str, str]] = None,
                      examples: Optional[List] = None) -> str:
    ent_types = [l for l in labels if l != "O"]
    desc = descriptions or {}
    type_lines = "\n".join(
        f"  {t} — {desc.get(t, 'a named entity of this type')}" for t in ent_types)

    ex_block = ""
    if examples:
        parts = []
        for ex_text, ex_ents in examples:
            parts.append(f'Sentence: {ex_text}\n'
                         f'{{"entities": {json.dumps(ex_ents, ensure_ascii=False)}}}')
        ex_block = ("EXAMPLES — how THIS dataset annotates entities. Follow these "
                    "conventions exactly (note what IS and is NOT an entity):\n\n"
                    + "\n\n".join(parts) + "\n\n")

    return (
        "You are a Named Entity Recognition specialist for code-switched "
        "Arabic-English text. LIST the named entities in the sentence. Do NOT "
        "tag every word — only real named entities.\n\n"
        f"ENTITY TYPES:\n{type_lines}\n\n"
        + ex_block +
        f"NOW TAG THIS SENTENCE: {text}\n\n"
        "TOKENS (copy entity text EXACTLY as these tokens appear, space-separated):\n"
        + " ".join(tokens) +
        "\n\nRULES:\n"
        "1. Copy each entity's text using the EXACT tokens above (same words, "
        "same order). Multi-word entities: include all their tokens.\n"
        "2. Arabic-script names are entities too.\n"
        "3. Use only the entity types listed above, following the EXAMPLES' conventions.\n"
        "4. Respond with JSON ONLY, exactly:\n"
        '{"entities": [{"text": "<exact tokens>", "type": "<TYPE>"}], '
        '"reasoning": "<one sentence>"}'
    )


class LLMNERSpanAgent(BaseAgent[PipelineState]):
    """LLM span-extraction tagger with deterministic token alignment."""

    def __init__(self, llm_client: LLMClient, output_slot: str = "contextual",
                 descriptions: Optional[Dict[str, str]] = None,
                 examples: Optional[List] = None, name=None, logger=None):
        super().__init__(name=name or "LLMNERSpanAgent", logger=logger)
        if output_slot not in _SLOT_ATTR:
            raise ValueError(f"output_slot must be one of {sorted(_SLOT_ATTR)}.")
        self.llm_client = llm_client
        self.output_slot = output_slot
        self.descriptions = descriptions
        # Optional few-shot examples: list of (sentence_text, [{text,type}, ...]).
        self.examples = examples

    def run(self, state: PipelineState) -> PipelineState:
        if state.task_config.task_type != "sequence_labeling":
            state.append_history(component=self.name, summary=_SKIP_NOTE)
            return state
        task = state.task_config
        tokens: List[str] = state.extras.get("tokens") or state.input_text.split()
        prompt = build_span_prompt(task.task_name, task.labels, tokens,
                                   state.input_text, self.descriptions
                                   or getattr(task, "label_descriptions", None),
                                   examples=self.examples)
        raw = self.llm_client.generate(prompt)
        try:
            entities, reasoning = self._parse(raw)
            note = ""
        except LLMNERParseError as exc:
            entities, reasoning, note = [], "", f"span parse fail ({exc}); all-O"
        tags = align_entities_to_tokens(tokens, entities, set(task.labels))

        seq = SequenceLabelingOutput(
            tags=[TokenTag(token=t, tag=g, confidence=_LLM_CONFIDENCE)
                  for t, g in zip(tokens, tags)],
            notes=note or reasoning,
            features={"raw_llm_response": raw, "n_entities": len(entities)})
        setattr(state, _SLOT_ATTR[self.output_slot], AgentOutput(
            agent_name=self.name, model_output=ModelOutput(),
            sequence_output=seq, notes=note))
        state.append_history(
            component=self.name,
            summary=f"Span NER ({self.output_slot}): {len(entities)} entity(ies). {reasoning}".strip(),
            outputs={"n_entities": len(entities), "fallback": bool(note)})
        return state

    @staticmethod
    def _parse(raw: str):
        try:
            data = json.loads(_extract_json(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMNERParseError(f"JSON decode failed: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("entities"), list):
            raise LLMNERParseError("Missing or non-list 'entities'.")
        ents = [e for e in data["entities"] if isinstance(e, dict)]
        return ents, str(data.get("reasoning", ""))
