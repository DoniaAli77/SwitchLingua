"""NER Consensus Agent — token-level voting for sequence-labeling tasks.

Only runs when ``state.task_config.task_type == "sequence_labeling"``.

Algorithm
---------
For each token position ``i``, the agent collects the tag predicted by each
specialist slot (lexical, contextual, logic) that has a
:class:`~src.state.schema.SequenceLabelingOutput` available.

    score[tag] += weight[slot] * token_tag.confidence

The tag with the highest accumulated score wins.  Ties are broken
deterministically by the position of the tag in ``task_config.labels``
(earlier index wins), matching the classification ConsensusAgent's behaviour.

If no slot produced a usable output for a position, the token gets tag ``"O"``
at confidence ``0.0``.

Token list resolution
---------------------
1. ``state.extras["tokens"]``  — pre-tokenised list (highest priority).
2. Tags list of the first available ``sequence_output`` — token strings are
   taken from :attr:`~src.state.schema.TokenTag.token` in order.
3. ``state.input_text.split()`` — whitespace-split fallback.

Slots with a different token count are still included: only positions that
exist in that slot's output contribute a vote.  The canonical token count
comes from the resolved token list (step 1/2/3 above).

State writes
------------
- ``state.final_output.payload["sequence_output"]``
    List of ``{"token": str, "tag": str, "confidence": float}`` dicts —
    one per token — for easy JSON serialisation.
- ``state.final_output.payload["token_count"]`` — int.
- ``state.consensus_output``
    :class:`~src.state.schema.ConsensusOutput` with ``label=None``,
    ``confidence=None``, ``rationale="sequence_labeling_completed"``, and
    ``votes={"token_count": "<N>", "summary": "<tag_distribution>"}`` for
    lightweight traceability.
- ``state.consensus_output.votes`` is a ``Dict[str, str]`` (schema
    constraint), so numeric values are stored as strings.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.state.schema import (
    AgentOutput,
    ConsensusOutput,
    FinalOutput,
    PipelineState,
    SequenceLabelingOutput,
    TokenTag,
)

_SKIP_NOTE = "NERConsensusAgent: task_type is not sequence_labeling — skipped."

# Default per-slot weights (same slot names as classification ConsensusAgent).
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "lexical": 1.0,
    "contextual": 1.0,
    "logic": 1.0,
}


def _extract_seq_output(
    agent_output: Optional[AgentOutput],
) -> Optional[SequenceLabelingOutput]:
    """Return the sequence_output from an AgentOutput, or None."""
    if agent_output is None:
        return None
    return agent_output.sequence_output


def _resolve_tokens(state: PipelineState) -> List[str]:
    """Return the canonical token list for the current sample.

    Priority: extras["tokens"] → first available sequence_output tokens →
    whitespace split.
    """
    tokens = state.extras.get("tokens")
    if tokens:
        return list(tokens)

    for slot_output in (state.lexical_output, state.logic_output, state.contextual_output):
        seq = _extract_seq_output(slot_output)
        if seq and seq.tags:
            return [tt.token for tt in seq.tags]

    return state.input_text.split()


def _vote_token(
    position: int,
    slots: Dict[str, Optional[SequenceLabelingOutput]],
    weights: Dict[str, float],
    labels: List[str],
) -> TokenTag:
    """Elect the winning tag for one token position via weighted voting.

    Parameters
    ----------
    position:
        0-based token index.
    slots:
        Mapping of slot name → ``SequenceLabelingOutput | None``.
    weights:
        Per-slot weights (clamped to ≥ 0).
    labels:
        Full ordered tag label set from ``task_config.labels``.
    """
    valid_tags = set(labels)
    scores: Dict[str, float] = {lbl: 0.0 for lbl in labels}
    active_weight_sum = 0.0

    for slot_name, seq_out in slots.items():
        weight = weights.get(slot_name, 0.0)
        if weight <= 0.0 or seq_out is None:
            continue
        if position >= len(seq_out.tags):
            continue  # slot produced fewer tokens — skip this position
        tt = seq_out.tags[position]
        if tt.tag not in valid_tags:
            continue
        conf = max(0.0, min(1.0, tt.confidence))
        scores[tt.tag] += weight * conf
        active_weight_sum += weight

    if active_weight_sum == 0.0:
        return TokenTag(token="?", tag="O", confidence=0.0)

    # Tie-break: highest score wins; equal scores → earliest in labels list.
    winner = max(labels, key=lambda t: (scores[t], -labels.index(t)))
    final_conf = round(scores[winner] / active_weight_sum, 6)
    return TokenTag(token="?", tag=winner, confidence=final_conf)


class NERConsensusAgent(BaseAgent[PipelineState]):
    """Token-level weighted-voting consensus for sequence-labeling tasks.

    Parameters
    ----------
    weights:
        Mapping of ``{slot: weight}`` where *slot* is ``"lexical"``,
        ``"contextual"``, or ``"logic"``.  Missing slots default to ``1.0``.
        Negative weights are clamped to ``0.0``.
    name:
        Optional display name for logging and history.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "NERConsensusAgent", logger=logger)
        raw: Dict[str, float] = {**_DEFAULT_WEIGHTS, **(weights or {})}
        self.weights: Dict[str, float] = {
            slot: max(0.0, raw.get(slot, _DEFAULT_WEIGHTS.get(slot, 1.0)))
            for slot in ("lexical", "contextual", "logic")
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
        """Aggregate token-level votes; write consensus + final output."""

        if state.task_config.task_type != "sequence_labeling":
            self.logger.debug(_SKIP_NOTE)
            state.append_history(component=self.name, summary=_SKIP_NOTE)
            return state

        labels: List[str] = state.task_config.labels
        tokens: List[str] = _resolve_tokens(state)

        slots: Dict[str, Optional[SequenceLabelingOutput]] = {
            "lexical":     _extract_seq_output(state.lexical_output),
            "contextual":  _extract_seq_output(state.contextual_output),
            "logic":       _extract_seq_output(state.logic_output),
        }

        tagged: List[TokenTag] = []
        for i, token in enumerate(tokens):
            voted = _vote_token(i, slots, self.weights, labels)
            # Restore the canonical token string (vote helper uses "?").
            tagged.append(TokenTag(token=token, tag=voted.tag, confidence=voted.confidence))

        # --- Serialisable payload for final_output -----------------------
        seq_payload = [
            {"token": tt.token, "tag": tt.tag, "confidence": tt.confidence}
            for tt in tagged
        ]

        if state.final_output is None:
            state.final_output = FinalOutput()
        state.final_output.payload["sequence_output"] = seq_payload
        state.final_output.payload["token_count"] = len(tagged)

        # --- Lightweight consensus_output summary ------------------------
        tag_counts = Counter(tt.tag for tt in tagged)
        distribution = ", ".join(
            f"{tag}:{count}" for tag, count in sorted(tag_counts.items())
        )
        state.consensus_output = ConsensusOutput(
            label=None,
            confidence=None,
            votes={
                "token_count": str(len(tagged)),
                "tag_distribution": distribution,
            },
            rationale="sequence_labeling_completed",
        )

        tag_summary = [f"{tt.token}:{tt.tag}" for tt in tagged]
        state.append_history(
            component=self.name,
            summary=f"NER consensus complete: {len(tagged)} token(s). {distribution}",
            outputs={
                "token_count": len(tagged),
                "tag_distribution": dict(tag_counts),
                "tags": tag_summary,
            },
        )

        self.logger.debug(
            "%s: %d token(s) tagged. Distribution: %s",
            self.name, len(tagged), distribution,
        )
        return state
