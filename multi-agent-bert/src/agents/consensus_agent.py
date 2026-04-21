"""Consensus agent for a stateful multi-agent text classification pipeline.

Aggregates the outputs of lexical, contextual, and logic specialist agents
using configurable per-agent weights.  The final label is the one with the
highest weighted score; the final confidence is that score normalised by the
sum of all active weights.

Algorithm
---------
For every specialist slot (lexical, contextual, logic) that produced a valid
output with a non-None label and confidence:

    score[label] += weight[agent] * agent_confidence

The label with the highest score wins.  If two or more labels tie, the one
that appears first in ``task_config.labels`` is chosen (deterministic
tie-breaking without randomness).

If *no* agent produced a usable output the agent writes a low-confidence
fallback to ``state.consensus_output`` and ``state.final_output`` rather than
raising, so the pipeline keeps flowing.

State writes
------------
- ``state.consensus_output`` — :class:`~src.state.schema.ConsensusOutput`
  with ``label``, ``confidence``, ``votes`` (raw scores by label), and
  ``rationale`` (vote breakdown by agent).
- ``state.final_output``    — :class:`~src.state.schema.FinalOutput`
  mirroring the consensus label and confidence.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from src.agents.base_agent import BaseAgent
from src.state.schema import (
    AgentOutput,
    ConsensusOutput,
    FinalOutput,
    ModelOutput,
    PipelineState,
    TaskConfig,
)

# Sentinel note written when no agent produced usable output.
_NO_VOTE_NOTE = "ConsensusAgent: no agent produced a usable output; fallback applied."

# Default per-agent weights used when the caller passes nothing.
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "lexical": 1.0,
    "contextual": 1.0,
    "logic": 1.0,
}


def _extract_vote(
    agent_output: Optional[AgentOutput],
) -> Tuple[Optional[str], Optional[float]]:
    """Return ``(label, confidence)`` from an AgentOutput, or ``(None, None)``.

    Returns ``(None, None)`` when the output is absent, its label is None, or
    its confidence is None / out of range.
    """
    if agent_output is None:
        return None, None
    mo: ModelOutput = agent_output.model_output
    if mo.label is None or mo.confidence is None:
        return None, None
    if not (0.0 <= mo.confidence <= 1.0):
        return None, None
    return mo.label, mo.confidence


class ConsensusAgent(BaseAgent[PipelineState]):
    """Weighted-voting consensus over specialist agent outputs.

    Parameters
    ----------
    weights:
        Mapping of ``{agent_slot: weight}`` where *agent_slot* is one of
        ``"lexical"``, ``"contextual"``, or ``"logic"``.  Missing slots
        default to ``1.0``.  Weights of ``0.0`` effectively silence an agent.
        Negative weights are silently clamped to ``0.0``.
    name:
        Optional display name used for logging.
    logger:
        Optional pre-configured logger.

    Examples
    --------
    Equal weights (default behaviour)::

        agent = ConsensusAgent()

    Trust contextual twice as much::

        agent = ConsensusAgent(weights={"lexical": 1.0, "contextual": 2.0, "logic": 1.0})

    Silence lexical::

        agent = ConsensusAgent(weights={"lexical": 0.0, "contextual": 1.0, "logic": 1.0})
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "ConsensusAgent", logger=logger)
        raw: Dict[str, float] = {**_DEFAULT_WEIGHTS, **(weights or {})}
        # Clamp negatives; keep only the three known slots.
        self.weights: Dict[str, float] = {
            slot: max(0.0, raw.get(slot, 1.0))
            for slot in ("lexical", "contextual", "logic")
        }

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def validate_before(self, state: PipelineState) -> None:
        if not state.task_config.labels:
            raise ValueError("ConsensusAgent: state.task_config.labels is empty.")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Aggregate agent votes and write consensus + final output to state."""

        task: TaskConfig = state.task_config
        labels: List[str] = task.labels

        agent_slots = {
            "lexical": state.lexical_output,
            "contextual": state.contextual_output,
            "logic": state.logic_output,
        }

        # score[label] accumulates weighted confidence values.
        scores: Dict[str, float] = {lbl: 0.0 for lbl in labels}
        # vote_details[agent_slot] = human-readable breakdown string
        vote_details: Dict[str, str] = {}
        active_weight_sum: float = 0.0

        for slot, output in agent_slots.items():
            weight = self.weights[slot]
            label, confidence = _extract_vote(output)

            if label is None or weight == 0.0:
                vote_details[slot] = "no vote"
                continue

            # Skip out-of-vocabulary labels with a warning.
            if not task.is_allowed_label(label):
                self.logger.warning(
                    "%s: agent '%s' returned an out-of-vocabulary label '%s'; skipping.",
                    self.name, slot, label,
                )
                vote_details[slot] = f"invalid label '{label}' — skipped"
                continue

            scores[label] += weight * confidence  # type: ignore[operator]
            active_weight_sum += weight
            vote_details[slot] = (
                f"{label} (weight={weight:.2f}, conf={confidence:.4f}, "
                f"contribution={weight * confidence:.4f})"  # type: ignore[operator]
            )

        # --- No usable votes → fallback -------------------------------------
        if active_weight_sum == 0.0:
            self.logger.warning("%s: no usable agent votes; applying fallback.", self.name)
            fallback_label = labels[0]
            fallback_conf = round(1.0 / len(labels), 6)
            state.consensus_output = ConsensusOutput(
                label=fallback_label,
                confidence=fallback_conf,
                votes={lbl: 0.0 for lbl in labels},
                rationale=_NO_VOTE_NOTE,
            )
            state.final_output = FinalOutput(
                label=fallback_label,
                confidence=fallback_conf,
            )
            state.append_history(
                component=self.name,
                summary=f"No usable votes — fallback to '{fallback_label}' (uniform confidence).",
                outputs={
                    "label": fallback_label,
                    "confidence": fallback_conf,
                    "votes": {lbl: 0.0 for lbl in labels},
                    "fallback": True,
                },
            )
            return state

        # --- Determine winner -----------------------------------------------
        # Break ties deterministically: highest score first; among ties,
        # the label that appears earliest in task_config.labels wins.
        best_label = max(labels, key=lambda lbl: (scores[lbl], -labels.index(lbl)))
        final_confidence = round(scores[best_label] / active_weight_sum, 6)

        rationale = "; ".join(
            f"{slot}={detail}" for slot, detail in vote_details.items()
        )

        state.consensus_output = ConsensusOutput(
            label=best_label,
            confidence=final_confidence,
            votes={lbl: round(scores[lbl], 6) for lbl in labels},
            rationale=rationale,
        )
        state.final_output = FinalOutput(
            label=best_label,
            confidence=final_confidence,
        )

        self.logger.debug(
            "%s: winner='%s' conf=%.4f scores=%s",
            self.name, best_label, final_confidence, scores,
        )
        state.append_history(
            component=self.name,
            summary=(
                f"Consensus: '{best_label}' "
                f"(confidence={final_confidence:.3f}). {rationale}"
            ),
            outputs={
                "label": best_label,
                "confidence": final_confidence,
                "votes": {lbl: round(scores[lbl], 6) for lbl in labels},
                "vote_details": dict(vote_details),
                "fallback": False,
            },
        )
        return state

