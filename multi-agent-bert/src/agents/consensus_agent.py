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
    DeliberationOutput,
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
    "deliberation": 0.0,  # Off by default; set > 0 to include deliberation vote.
    # Primary-aware consensus (audit C1): the primary model participates as a
    # weighted vote/prior on the escalation path. Confidence-scaled, so a
    # near-threshold-confident primary anchors more than a barely-confident one.
    # Set 0.0 to reproduce the legacy agents-only behaviour (ablation).
    "primary": 1.0,
}


def _select_winner(
    tied: List[str],
    primary_label: Optional[str],
    vote_counts: Dict[str, int],
    max_contribution: Dict[str, float],
) -> str:
    """Pick the winner among score-tied labels without using config label order.

    Precedence: (1) the primary's label if it is tied; (2) the label with more
    voting agents; (3) the label with the highest single agent contribution;
    (4) deterministic alphabetical (NOT ``task_config.labels`` order).
    ``tied`` must be non-empty.
    """
    if len(tied) == 1:
        return tied[0]
    # 1. Conservative anchor: prefer the primary's own label.
    if primary_label is not None and primary_label in tied:
        return primary_label
    # 2. Most voting agents.
    best_count = max(vote_counts.get(lbl, 0) for lbl in tied)
    cands = [lbl for lbl in tied if vote_counts.get(lbl, 0) == best_count]
    if len(cands) == 1:
        return cands[0]
    # 3. Highest single agent contribution.
    best_contrib = max(max_contribution.get(lbl, 0.0) for lbl in cands)
    cands = [
        lbl for lbl in cands
        if abs(max_contribution.get(lbl, 0.0) - best_contrib) <= 1e-9
    ]
    if len(cands) == 1:
        return cands[0]
    # 4. Deterministic, label-order-free last resort.
    return sorted(cands)[0]


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

    Include deliberation at weight 1.5::

        agent = ConsensusAgent(weights={"deliberation": 1.5})
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "ConsensusAgent", logger=logger)
        raw: Dict[str, float] = {**_DEFAULT_WEIGHTS, **(weights or {})}
        # Clamp negatives; keep all known slots (incl. the primary prior).
        self.weights: Dict[str, float] = {
            slot: max(0.0, raw.get(slot, _DEFAULT_WEIGHTS.get(slot, 0.0)))
            for slot in ("lexical", "contextual", "logic", "deliberation", "primary")
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
        # Per-label agent tally + best single contribution, for non-positional
        # tie-breaking (primary is handled separately as an anchor).
        vote_counts: Dict[str, int] = {lbl: 0 for lbl in labels}
        max_contribution: Dict[str, float] = {lbl: 0.0 for lbl in labels}
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

            contribution = weight * confidence  # type: ignore[operator]
            scores[label] += contribution
            active_weight_sum += weight
            vote_counts[label] += 1
            max_contribution[label] = max(max_contribution[label], contribution)
            vote_details[slot] = (
                f"{label} (weight={weight:.2f}, conf={confidence:.4f}, "
                f"contribution={contribution:.4f})"
            )

        # --- Optional deliberation vote ------------------------------------
        delib_weight = self.weights["deliberation"]
        if delib_weight > 0.0 and state.deliberation_output is not None:
            d_label = state.deliberation_output.recommended_label
            d_conf = state.deliberation_output.confidence
            if (
                d_label is not None
                and d_conf is not None
                and 0.0 <= d_conf <= 1.0
                and task.is_allowed_label(d_label)
            ):
                d_contribution = delib_weight * d_conf  # type: ignore[operator]
                scores[d_label] += d_contribution
                active_weight_sum += delib_weight
                vote_counts[d_label] += 1
                max_contribution[d_label] = max(max_contribution[d_label], d_contribution)
                vote_details["deliberation"] = (
                    f"{d_label} (weight={delib_weight:.2f}, conf={d_conf:.4f}, "
                    f"contribution={d_contribution:.4f})"
                )
            else:
                self.logger.warning(
                    "%s: deliberation output is unusable (label=%r, conf=%r); skipping.",
                    self.name, d_label, d_conf,
                )
                vote_details["deliberation"] = "no vote"

        # --- All agents abstained → defer to the primary (never labels[0]) ---
        if active_weight_sum == 0.0:
            primary = state.primary_model_output
            if primary.label is not None and task.is_allowed_label(primary.label):
                fb_label: Optional[str] = primary.label
                fb_conf: Optional[float] = primary.confidence
                source = "primary_fallback"
                self.logger.warning(
                    "%s: all agents abstained; deferring to primary '%s'.",
                    self.name, fb_label,
                )
            else:
                # No usable primary either — return a no-decision (label=None),
                # NEVER labels[0].
                fb_label = None
                fb_conf = None
                source = "no_decision"
                self.logger.warning(
                    "%s: all agents abstained and no usable primary; no decision.",
                    self.name,
                )
            rationale = f"{_NO_VOTE_NOTE} fallback_source={source}"
            state.consensus_output = ConsensusOutput(
                label=fb_label,
                confidence=fb_conf,
                votes={lbl: 0.0 for lbl in labels},
                rationale=rationale,
            )
            state.final_output = FinalOutput(
                label=fb_label,
                confidence=fb_conf,
                payload={"source": source},
            )
            state.append_history(
                component=self.name,
                summary=f"All agents abstained — {source} (label={fb_label}).",
                outputs={
                    "label": fb_label,
                    "confidence": fb_conf,
                    "votes": {lbl: 0.0 for lbl in labels},
                    "abstain_fallback": source,
                },
            )
            return state

        # --- Primary-aware vote (audit C1) ----------------------------------
        # The primary participates as a confidence-scaled weighted prior, so the
        # agents override it only when they collectively out-vote it. Reached
        # only when >=1 agent voted (the all-abstain case is handled above and
        # already defers to the primary).
        w_primary = self.weights["primary"]
        primary = state.primary_model_output
        primary_label: Optional[str] = None
        if (
            w_primary > 0.0
            and primary.label is not None
            and primary.confidence is not None
            and task.is_allowed_label(primary.label)
        ):
            primary_label = primary.label
            p_contribution = w_primary * primary.confidence
            scores[primary_label] += p_contribution
            active_weight_sum += w_primary
            vote_details["primary"] = (
                f"{primary_label} (weight={w_primary:.2f}, conf={primary.confidence:.4f}, "
                f"contribution={p_contribution:.4f})"
            )

        # --- Determine winner (non-positional tie-break) --------------------
        max_score = max(scores[lbl] for lbl in labels)
        tied = sorted(lbl for lbl in labels if abs(scores[lbl] - max_score) <= 1e-9)
        best_label = _select_winner(tied, primary_label, vote_counts, max_contribution)
        final_confidence = (
            round(scores[best_label] / active_weight_sum, 6) if active_weight_sum else 0.0
        )

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

