"""NER Lexical Agent — gazetteer-based token tagger for sequence-labeling tasks.

Only runs when ``state.task_config.task_type == "sequence_labeling"``.

Tokenisation
------------
Uses ``state.extras["tokens"]`` when present (pre-tokenised dataset tokens).
Falls back to simple whitespace split when the key is absent.

Tagging logic
-------------
For each token the agent looks up a ``gazetteer`` dict:
``{entity_type: [surface_form, …]}``, e.g.
``{"PER": ["Ahmed", "Sara"], "ORG": ["Google", "OpenAI"]}``.

Look-up is case-insensitive.  When a surface form matches, the token
receives a ``B-<TYPE>`` tag.  Consecutive tokens that belong to the same
entity type following a ``B-`` continue as ``I-<TYPE>`` tokens.

All unmatched tokens receive the ``O`` tag.

Output
------
Writes a :class:`~src.state.schema.SequenceLabelingOutput` to
``state.lexical_output.sequence_output``.  Never sets ``model_output.label``.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    SequenceLabelingOutput,
    TokenTag,
)

_SKIP_NOTE = "NERLexicalAgent: task_type is not sequence_labeling — skipped."


def _token_entity_type(
    token: str,
    gazetteer: Dict[str, List[str]],
) -> Optional[str]:
    """Return the entity type for *token* via case-insensitive gazetteer lookup.

    Returns ``None`` when no match is found.
    """
    lower = token.lower()
    for etype, names in gazetteer.items():
        for name in names:
            if name.lower() == lower:
                return etype
    return None


def _tag_tokens(
    tokens: List[str],
    gazetteer: Dict[str, List[str]],
    labels: List[str],
) -> List[TokenTag]:
    """Assign one BIO tag per token using the gazetteer.

    Consecutive tokens of the same entity type are labelled ``I-<TYPE>``
    after the first ``B-<TYPE>``.  Unknown tokens get ``O``.
    """
    valid_tags = set(labels)
    result: List[TokenTag] = []
    prev_type: Optional[str] = None

    for token in tokens:
        etype = _token_entity_type(token, gazetteer)
        if etype is None:
            tag = "O"
            prev_type = None
            conf = 1.0
        elif etype == prev_type:
            tag = f"I-{etype}"
            conf = 0.85
        else:
            tag = f"B-{etype}"
            prev_type = etype
            conf = 0.9

        # Gracefully fall back to O if the computed tag is not in the label set.
        if tag not in valid_tags:
            tag = "O"
            conf = 0.3

        result.append(TokenTag(token=token, tag=tag, confidence=conf))

    return result


class NERLexicalAgent(BaseAgent[PipelineState]):
    """Gazetteer-based BIO tagger for sequence-labeling tasks.

    Parameters
    ----------
    gazetteer:
        Mapping of ``{entity_type: [surface_form, …]}``.  E.g.
        ``{"PER": ["Ahmed", "Sara"], "LOC": ["Paris", "Cairo"]}``.
        Entity types must match the label schema (labels like ``B-PER``
        are expected, so entity types are upper-cased prefixes).
        Pass an empty dict for an all-``O`` tagger.
    name:
        Optional display name for logging and history.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        gazetteer: Optional[Dict[str, List[str]]] = None,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "NERLexicalAgent", logger=logger)
        self.gazetteer: Dict[str, List[str]] = gazetteer or {}

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def validate_before(self, state: PipelineState) -> None:
        if not state.task_config.labels:
            raise ValueError(f"{self.name}: state.task_config.labels is empty.")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Tag tokens; write result to ``state.lexical_output.sequence_output``."""

        if state.task_config.task_type != "sequence_labeling":
            self.logger.debug(_SKIP_NOTE)
            state.append_history(
                component=self.name,
                summary=_SKIP_NOTE,
            )
            return state

        tokens: List[str] = state.extras.get("tokens") or state.input_text.split()

        tagged = _tag_tokens(tokens, self.gazetteer, state.task_config.labels)

        seq_out = SequenceLabelingOutput(
            tags=tagged,
            notes=f"{self.name}: tagged {len(tagged)} token(s) via gazetteer.",
            features={"gazetteer_size": sum(len(v) for v in self.gazetteer.values())},
        )

        state.lexical_output = AgentOutput(
            agent_name=self.name,
            model_output=ModelOutput(),   # intentionally empty for NER
            sequence_output=seq_out,
            notes=seq_out.notes,
        )

        tag_summary = [f"{tt.token}:{tt.tag}" for tt in tagged]
        state.append_history(
            component=self.name,
            summary=f"Gazetteer tagging complete: {len(tagged)} token(s).",
            outputs={"tags": tag_summary},
        )

        self.logger.debug(
            "%s: tagged %d token(s). Summary: %s",
            self.name,
            len(tagged),
            tag_summary,
        )
        return state
