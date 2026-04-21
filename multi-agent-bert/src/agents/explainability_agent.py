"""Explainability agent for a stateful multi-agent text classification pipeline.

Generates a concise, deterministic, human-readable explanation of how the
pipeline arrived at its final label.  The explanation is template-based —
no LLM call is made.

Two code paths
--------------
**Accepted primary prediction** (``routing_info.decision == "accept_primary"``)
    The primary model's confidence was above the threshold; specialist agents
    were not consulted.  The explanation says so and reports the primary label
    and confidence.

**Escalated prediction** (any other routing decision, or no routing info)
    Specialist agents were run.  The explanation lists which agents supported
    the winning consensus label, what evidence/notes they each contributed,
    and flags any agents that disagreed.

State writes
------------
``state.explanation_output`` — :class:`~src.state.schema.ExplanationOutput`
    * ``summary``  — one-sentence overall verdict.
    * ``evidence`` — one entry per supporting agent (or primary model when accepted).
    * ``caveats``  — one entry per disagreeing agent, or empty for full agreement.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from src.agents.base_agent import BaseAgent
from src.state.schema import (
    AgentOutput,
    ConsensusOutput,
    ExplanationOutput,
    ModelOutput,
    PipelineState,
    RoutingInfo,
)

# Routing decision value written by the router when confidence is above threshold.
_ACCEPT_DECISION = "accept_primary"


def _agent_vote(output: Optional[AgentOutput]) -> Tuple[Optional[str], Optional[float], str]:
    """Return ``(label, confidence, notes)`` from an AgentOutput.

    Returns ``(None, None, "")`` when the output is absent or has no label.
    ``notes`` is taken from ``AgentOutput.notes``; ``evidence`` list items from
    ``ModelOutput.probabilities`` are not surfaced here — only the top label.
    """
    if output is None:
        return None, None, ""
    mo: ModelOutput = output.model_output
    return mo.label, mo.confidence, output.notes


def _fmt_conf(confidence: Optional[float]) -> str:
    """Format confidence as a percentage string, or '?' when absent."""
    if confidence is None:
        return "?"
    return f"{confidence * 100:.1f}%"


def _build_accepted_explanation(
    primary: ModelOutput,
    routing: Optional[RoutingInfo],
) -> ExplanationOutput:
    """Explanation for the fast-path: primary model accepted without escalation."""
    label = primary.label or "unknown"
    conf_str = _fmt_conf(primary.confidence)
    threshold_str = (
        f" (threshold: {routing.threshold * 100:.1f}%)" if routing is not None else ""
    )

    summary = (
        f"Primary model predicted '{label}' with {conf_str} confidence"
        f"{threshold_str}. No specialist agents were consulted."
    )
    evidence = [
        f"Primary model: label='{label}', confidence={conf_str}"
    ]
    if primary.raw_text:
        evidence.append(f"Primary model raw output: {primary.raw_text}")

    return ExplanationOutput(summary=summary, evidence=evidence, caveats=[])


def _build_escalated_explanation(
    consensus: Optional[ConsensusOutput],
    lexical: Optional[AgentOutput],
    contextual: Optional[AgentOutput],
    logic: Optional[AgentOutput],
    primary: ModelOutput,
) -> ExplanationOutput:
    """Explanation for the escalated path: specialist agents were consulted."""

    winning_label: Optional[str] = None
    final_conf: Optional[float] = None

    if consensus is not None:
        winning_label = consensus.label
        final_conf = consensus.confidence
    elif primary.label is not None:
        # Consensus was skipped — fall back to primary.
        winning_label = primary.label
        final_conf = primary.confidence

    label_str = winning_label or "unknown"
    conf_str = _fmt_conf(final_conf)

    summary = (
        f"Specialist agents were consulted. "
        f"Consensus reached: '{label_str}' (confidence: {conf_str})."
    )

    evidence: List[str] = []
    caveats: List[str] = []

    agent_slots = [
        ("Lexical", lexical),
        ("Contextual", contextual),
        ("Logic", logic),
    ]

    for agent_name, output in agent_slots:
        label, conf, notes = _agent_vote(output)
        if label is None:
            continue
        conf_s = _fmt_conf(conf)
        if label == winning_label:
            entry = f"{agent_name}: supported '{label}' ({conf_s})"
            if notes and notes != "No keywords matched; uniform fallback applied.":
                entry += f" — {notes}"
            evidence.append(entry)
        else:
            caveat = f"{agent_name}: voted for '{label}' ({conf_s})"
            if notes:
                caveat += f" — {notes}"
            caveats.append(caveat)

    # Add primary model as background context.
    if primary.label:
        evidence.append(
            f"Primary model: label='{primary.label}', "
            f"confidence={_fmt_conf(primary.confidence)}"
        )

    if not evidence:
        evidence.append("No agent produced a usable supporting vote.")

    return ExplanationOutput(summary=summary, evidence=evidence, caveats=caveats)


class ExplainabilityAgent(BaseAgent[PipelineState]):
    """Template-based explanation generator for the multi-agent pipeline.

    Reads all available pipeline outputs and produces a structured
    :class:`~src.state.schema.ExplanationOutput` in ``state.explanation_output``.
    No configuration is required; the agent adapts its template based on
    whether the prediction was accepted or escalated.

    Parameters
    ----------
    name:
        Optional display name for logging.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "ExplainabilityAgent", logger=logger)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Generate and write an explanation to ``state.explanation_output``."""

        routing: Optional[RoutingInfo] = state.routing_info
        primary: ModelOutput = state.primary_model_output

        # Determine path: accepted (fast) vs escalated (specialist agents ran).
        is_accepted = (
            routing is not None
            and routing.decision == _ACCEPT_DECISION
        )

        if is_accepted:
            explanation = _build_accepted_explanation(primary, routing)
            self.logger.debug(
                "%s: primary prediction accepted — no escalation.", self.name
            )
        else:
            explanation = _build_escalated_explanation(
                consensus=state.consensus_output,
                lexical=state.lexical_output,
                contextual=state.contextual_output,
                logic=state.logic_output,
                primary=primary,
            )
            n_support = len(explanation.evidence)
            n_dissent = len(explanation.caveats)
            self.logger.debug(
                "%s: escalated — %d supporting, %d dissenting.",
                self.name, n_support, n_dissent,
            )

        state.explanation_output = explanation
        state.append_history(
            component=self.name,
            summary=explanation.summary,
            outputs={
                "path": "accepted" if is_accepted else "escalated",
                "evidence": list(explanation.evidence),
                "caveats": list(explanation.caveats),
            },
        )
        return state

