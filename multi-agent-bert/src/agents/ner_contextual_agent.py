"""NER Contextual Agent — heuristic contextual BIO tagger for sequence-labeling.

Only runs when ``state.task_config.task_type == "sequence_labeling"``.

Tokenisation
------------
Uses ``state.extras["tokens"]`` when present.
Falls back to simple whitespace split otherwise.

Tagging logic
-------------
Uses lightweight contextual heuristics without an LLM backend, making it
safe and deterministic for testing:

1. **Capitalisation heuristic** — tokens that start with an uppercase letter
   (non-Arabic) or are in the ``known_entities`` dict are tagged as named
   entities.

2. **Context window** — if the previous token was tagged as ``B-PER`` /
   ``I-PER``, the current token (when capitalised) continues the span as
   ``I-PER``.  Likewise for ORG and LOC.

3. **known_entities** — an optional ``{surface_form: entity_type}`` dict for
   exact-match overrides (case-insensitive).  Takes precedence over heuristics.

4. **Arabic tokens** — Arabic-script tokens are never auto-capitalised.
   They only receive a non-O tag if present in ``known_entities``.

Output
------
Writes a :class:`~src.state.schema.SequenceLabelingOutput` to
``state.contextual_output.sequence_output``.  Never sets ``model_output.label``.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    SequenceLabelingOutput,
    TokenTag,
)

_SKIP_NOTE = "NERContextualAgent: task_type is not sequence_labeling — skipped."

# Entity types in priority order for capitalisation heuristic.
_DEFAULT_CAP_TYPE = "PER"

# Arabic Unicode block range.
_ARABIC_START = 0x0600
_ARABIC_END = 0x06FF


def _is_arabic(token: str) -> bool:
    """Return True if the majority of characters in *token* are Arabic."""
    arabic = sum(
        1 for ch in token
        if _ARABIC_START <= ord(ch) <= _ARABIC_END
    )
    return arabic > len(token) / 2


def _tag_tokens(
    tokens: List[str],
    known_entities: Dict[str, str],
    labels: List[str],
) -> List[TokenTag]:
    """Assign one BIO tag per token using contextual heuristics.

    Parameters
    ----------
    tokens:
        Ordered list of token strings.
    known_entities:
        ``{surface_form_lower: entity_type}`` override map.
    labels:
        Full label set from ``task_config.labels``.
    """
    valid_tags = set(labels)
    result: List[TokenTag] = []
    prev_type: Optional[str] = None

    for token in tokens:
        lower = token.lower()
        etype: Optional[str] = None
        conf = 0.0

        # 1. Exact-match override (highest priority).
        if lower in known_entities:
            etype = known_entities[lower]
            conf = 0.92

        # 2. Capitalisation heuristic for Latin-script tokens.
        elif not _is_arabic(token) and token and token[0].isupper():
            etype = _DEFAULT_CAP_TYPE
            conf = 0.65

        if etype is None:
            tag = "O"
            prev_type = None
            conf = 1.0
        elif etype == prev_type:
            tag = f"I-{etype}"
        else:
            tag = f"B-{etype}"
            prev_type = etype

        if tag not in valid_tags:
            tag = "O"
            conf = 0.3
            prev_type = None

        result.append(TokenTag(token=token, tag=tag, confidence=conf))

    return result


class NERContextualAgent(BaseAgent[PipelineState]):
    """Heuristic contextual BIO tagger for sequence-labeling tasks.

    Parameters
    ----------
    known_entities:
        Mapping of ``{surface_form: entity_type}``.  Look-up is
        case-insensitive.  E.g.
        ``{"google": "ORG", "paris": "LOC", "ahmed": "PER"}``.
    name:
        Optional display name for logging and history.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        known_entities: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "NERContextualAgent", logger=logger)
        # Normalise keys to lower-case for case-insensitive lookup.
        self.known_entities: Dict[str, str] = {
            k.lower(): v for k, v in (known_entities or {}).items()
        }

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
        """Tag tokens; write result to ``state.contextual_output.sequence_output``."""

        if state.task_config.task_type != "sequence_labeling":
            self.logger.debug(_SKIP_NOTE)
            state.append_history(
                component=self.name,
                summary=_SKIP_NOTE,
            )
            return state

        tokens: List[str] = state.extras.get("tokens") or state.input_text.split()

        tagged = _tag_tokens(tokens, self.known_entities, state.task_config.labels)

        seq_out = SequenceLabelingOutput(
            tags=tagged,
            notes=f"{self.name}: tagged {len(tagged)} token(s) via contextual heuristics.",
            features={"known_entity_count": len(self.known_entities)},
        )

        state.contextual_output = AgentOutput(
            agent_name=self.name,
            model_output=ModelOutput(),   # intentionally empty for NER
            sequence_output=seq_out,
            notes=seq_out.notes,
        )

        tag_summary = [f"{tt.token}:{tt.tag}" for tt in tagged]
        state.append_history(
            component=self.name,
            summary=f"Contextual heuristic tagging complete: {len(tagged)} token(s).",
            outputs={"tags": tag_summary},
        )

        self.logger.debug(
            "%s: tagged %d token(s). Summary: %s",
            self.name,
            len(tagged),
            tag_summary,
        )
        return state
