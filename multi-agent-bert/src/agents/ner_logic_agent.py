"""NER Logic Agent — regex-rule BIO tagger for sequence-labeling tasks.

Only runs when ``state.task_config.task_type == "sequence_labeling"``.

Tokenisation
------------
Uses ``state.extras["tokens"]`` when present.
Falls back to simple whitespace split otherwise.

Tagging logic
-------------
Accepts a ``rule_map``:
``{entity_type: [regex_pattern, …]}``, e.g.
``{"PER": [r"\\b(Dr|Mr|Ms)\\.?\\s+\\w+"]}``.

Each token is tested against every compiled pattern.  The first entity type
whose pattern produces a full-token match (``re.fullmatch``) or a whole-word
match (``re.search`` with word boundaries for ASCII, plain substring for
non-ASCII) wins and the token gets ``B-<TYPE>`` or ``I-<TYPE>`` depending on
whether the previous token shared the same entity type.

All unmatched tokens receive ``O``.

Output
------
Writes a :class:`~src.state.schema.SequenceLabelingOutput` to
``state.logic_output.sequence_output``.  Never sets ``model_output.label``.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from src.agents.base_agent import BaseAgent
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    SequenceLabelingOutput,
    TokenTag,
)

_SKIP_NOTE = "NERLogicAgent: task_type is not sequence_labeling — skipped."


def _token_matches_pattern(token: str, pattern: re.Pattern[str]) -> bool:
    """Return True if *token* matches *pattern*.

    Full-match is tried first.  For ASCII tokens a whole-word search inside
    the token is used as a fallback; for non-ASCII the pattern is searched as
    a substring.
    """
    if pattern.fullmatch(token):
        return True
    if token.isascii():
        return bool(re.search(r"(?i)\b" + pattern.pattern + r"\b", token, re.UNICODE))
    return bool(pattern.search(token))


def _tag_tokens(
    tokens: List[str],
    compiled_rules: List[Tuple[str, re.Pattern[str]]],
    labels: List[str],
) -> List[TokenTag]:
    """Assign one BIO tag per token using compiled regex rules.

    Parameters
    ----------
    tokens:
        Ordered list of token strings.
    compiled_rules:
        List of ``(entity_type, compiled_pattern)`` pairs in priority order
        (first match wins).
    labels:
        Full label set from ``task_config.labels`` — used for validity check.
    """
    valid_tags = set(labels)
    result: List[TokenTag] = []
    prev_type: Optional[str] = None

    for token in tokens:
        matched_type: Optional[str] = None
        for etype, pattern in compiled_rules:
            if _token_matches_pattern(token, pattern):
                matched_type = etype
                break

        if matched_type is None:
            tag = "O"
            prev_type = None
            conf = 1.0
        elif matched_type == prev_type:
            tag = f"I-{matched_type}"
            conf = 0.8
        else:
            tag = f"B-{matched_type}"
            prev_type = matched_type
            conf = 0.85

        if tag not in valid_tags:
            tag = "O"
            conf = 0.3

        result.append(TokenTag(token=token, tag=tag, confidence=conf))

    return result


class NERLogicAgent(BaseAgent[PipelineState]):
    """Regex-rule BIO tagger for sequence-labeling tasks.

    Parameters
    ----------
    rule_map:
        Mapping of ``{entity_type: [regex_pattern_str, …]}``.  Patterns are
        compiled at construction time with ``re.IGNORECASE | re.UNICODE``.
        Bad patterns are skipped with a warning.
    name:
        Optional display name for logging and history.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        rule_map: Optional[Dict[str, List[str]]] = None,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "NERLogicAgent", logger=logger)
        self._compiled: List[Tuple[str, re.Pattern[str]]] = []
        for etype, patterns in (rule_map or {}).items():
            for pat in patterns:
                try:
                    self._compiled.append(
                        (etype, re.compile(pat, re.IGNORECASE | re.UNICODE))
                    )
                except re.error as exc:
                    logging.getLogger(self.name).warning(
                        "Skipping invalid regex '%s' for entity type '%s': %s",
                        pat, etype, exc,
                    )

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
        """Tag tokens; write result to ``state.logic_output.sequence_output``."""

        if state.task_config.task_type != "sequence_labeling":
            self.logger.debug(_SKIP_NOTE)
            state.append_history(
                component=self.name,
                summary=_SKIP_NOTE,
            )
            return state

        tokens: List[str] = state.extras.get("tokens") or state.input_text.split()

        tagged = _tag_tokens(tokens, self._compiled, state.task_config.labels)

        seq_out = SequenceLabelingOutput(
            tags=tagged,
            notes=f"{self.name}: tagged {len(tagged)} token(s) via regex rules.",
            features={"rule_count": len(self._compiled)},
        )

        state.logic_output = AgentOutput(
            agent_name=self.name,
            model_output=ModelOutput(),   # intentionally empty for NER
            sequence_output=seq_out,
            notes=seq_out.notes,
        )

        tag_summary = [f"{tt.token}:{tt.tag}" for tt in tagged]
        state.append_history(
            component=self.name,
            summary=f"Regex-rule tagging complete: {len(tagged)} token(s).",
            outputs={"tags": tag_summary},
        )

        self.logger.debug(
            "%s: tagged %d token(s). Summary: %s",
            self.name,
            len(tagged),
            tag_summary,
        )
        return state
