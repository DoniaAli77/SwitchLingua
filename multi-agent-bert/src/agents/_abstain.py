"""Shared abstain / no-vote helper for classification specialist agents.

Replaces the old ``labels[0]`` fallback (which silently voted for the first
configured label, creating a first-label / positive bias) with a task-generic
**abstain**: an :class:`~src.state.schema.AgentOutput` whose label is ``None``.

``ConsensusAgent._extract_vote`` already returns ``(None, None)`` for a ``None``
label, so an abstaining agent is naturally excluded from the weighted vote — no
label names are involved anywhere, so this is fully task-generic.
"""

from __future__ import annotations

from src.state.schema import AgentOutput, ModelOutput, PipelineState

# Stable key used in AgentOutput.features and assertable in tests.
ABSTAIN_FLAG = "abstained"


def abstain_output(agent_name: str, state: PipelineState, reason: str) -> AgentOutput:
    """Return an abstaining (no-vote) :class:`AgentOutput`.

    The result carries ``label=None``, ``confidence=None`` and empty
    ``probabilities`` so it is excluded by consensus and passes
    ``ModelOutput.validate_labels`` (which skips validation when label is None
    and has no probability keys to check).

    Parameters
    ----------
    agent_name:
        Name of the abstaining agent (for ``AgentOutput.agent_name``).
    state:
        Current pipeline state (used only for ``input_text``).
    reason:
        Human-readable reason for abstaining (e.g. "parse failure",
        "no keyword match"); stored in ``notes`` and ``features``.
    """
    return AgentOutput(
        agent_name=agent_name,
        model_output=ModelOutput(
            label=None,
            confidence=None,
            probabilities={},
            raw_text=state.input_text,
        ),
        notes=reason,
        features={ABSTAIN_FLAG: True, "abstain_reason": reason},
    )
