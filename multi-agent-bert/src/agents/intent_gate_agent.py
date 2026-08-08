"""Standalone IntentGate guard — decoupled from the ConsensusAgent.

Design G/G2 use a *non-voting* IntentGate: an Intent agent writes ``state.polarity_output``
with weight 0 (it never votes), and this guard runs **after** consensus. If the consensus
OVERRODE the primary and the gate SIDES WITH the primary, the override is blocked (reverted
to the primary). It never forces a flip toward the gate's own opinion and never fires when
consensus already agrees with the primary.

Separation of concerns
----------------------
Previously this logic lived inside :class:`~src.agents.consensus_agent.ConsensusAgent`.
It is now a standalone post-consensus pipeline stage so the ConsensusAgent is pure
weighted-voting and the guard is independently testable, swappable, and off by default
(the orchestrator only runs it when an ``intent_gate_agent`` is supplied — sentiment G/G2).
Behaviour is identical to the previous embedded guard.

The guard is **task-generic**: it keys only on ``gate_label == primary_label`` agreement,
never on a hardcoded label name.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.agents.base_agent import BaseAgent
from src.state.schema import PipelineState


class IntentGateAgent(BaseAgent[PipelineState]):
    """Post-consensus non-voting veto: block an unsupported override of the primary.

    Parameters
    ----------
    gate_slot:
        State attribute (without the ``_output`` suffix) holding the gate agent's
        :class:`~src.state.schema.AgentOutput`. Defaults to ``"polarity"`` (i.e.
        ``state.polarity_output``), matching how Design G/G2 inject the IntentGate.
    lazy_agent:
        Optional gate agent (an ``IntentAgent``) invoked **on demand**, only once an
        override of the primary has been detected. When supplied, the gate's LLM call
        is deferred to this post-consensus stage instead of running as a pre-consensus
        specialist (Design G2-lazy). ``None`` (default) → the gate output is expected
        to be already present in ``gate_slot`` (Designs G/G2). The gate's judgement is
        identical either way: its prompt sees only the raw text and label set, never
        the consensus or primary prediction — so deferring the call changes cost and
        call-ordering, not the decision.
    """

    def __init__(
        self,
        gate_slot: str = "polarity",
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        lazy_agent: Optional[Any] = None,
    ) -> None:
        super().__init__(name=name or "IntentGateAgent", logger=logger)
        self._gate_slot = gate_slot
        self._lazy_agent = lazy_agent

    def run(self, state: PipelineState) -> PipelineState:
        """Revert consensus→primary iff consensus overrode the primary and the gate agrees."""
        consensus = state.consensus_output
        if consensus is None or consensus.label is None:
            return state  # nothing to guard

        primary = state.primary_model_output
        primary_label = primary.label if primary is not None else None
        best_label = consensus.label

        # Guard only an *override* of a usable primary; no-op otherwise.
        if primary_label is None or best_label == primary_label:
            return state

        # Lazy mode: an override exists, so the gate's judgement is now needed.
        # Invoke the gate agent here (its only LLM call) instead of pre-consensus.
        gate_out = getattr(state, f"{self._gate_slot}_output", None)
        if gate_out is None and self._lazy_agent is not None:
            self.logger.debug(
                "%s: override detected ('%s'->'%s'); invoking gate agent on demand.",
                self.name, primary_label, best_label,
            )
            state = self._lazy_agent.run(state)
            gate_out = getattr(state, f"{self._gate_slot}_output", None)

        gate_label = (
            gate_out.model_output.label
            if gate_out is not None and gate_out.model_output is not None
            else None
        )
        # Block only when the gate SIDES WITH the primary (never forces a different flip).
        if gate_label is None or gate_label != primary_label:
            return state

        # --- BLOCK: revert the override to the primary ---
        pre_gate_label = best_label
        votes = consensus.votes or {}
        conf = consensus.confidence
        # Recompute confidence as scores[primary]/active_weight_sum, reconstructing the
        # weight sum from the stored votes (scores) and the consensus confidence.
        v_best = votes.get(best_label)
        v_primary = votes.get(primary_label, 0.0)
        if conf and v_best:
            active_weight_sum = v_best / conf
            new_conf = round(v_primary / active_weight_sum, 6) if active_weight_sum else conf
        else:
            new_conf = conf

        consensus.label = primary_label
        consensus.confidence = new_conf
        consensus.rationale = (
            (consensus.rationale or "")
            + f"; intent_gate=BLOCKED override '{pre_gate_label}'->'{primary_label}'"
        )
        if state.final_output is not None:
            state.final_output.label = primary_label
            state.final_output.confidence = new_conf

        self.logger.debug(
            "%s: BLOCKED override '%s'->'%s' (gate=%s == primary)",
            self.name, pre_gate_label, primary_label, gate_label,
        )
        state.append_history(
            component=self.name,
            summary=(
                f"IntentGate blocked override '{pre_gate_label}'->'{primary_label}' "
                f"(gate sided with primary)."
            ),
            outputs={
                "label": primary_label,
                "confidence": new_conf,
                "intent_gate_blocked": True,
                "pre_gate_label": pre_gate_label,
            },
        )
        return state
