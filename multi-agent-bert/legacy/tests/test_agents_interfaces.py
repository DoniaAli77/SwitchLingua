"""Interface tests for all agent components."""

import pytest

from agents import (
    ConsensusAgent,
    ContextualAgent,
    ExplainabilityAgent,
    LexicalAgent,
    LogicAgent,
    PrimaryClassifier,
    Router,
)
from models.state import InputState, PipelineState


@pytest.mark.parametrize(
    "agent",
    [
        PrimaryClassifier(),
        Router(),
        LexicalAgent(),
        ContextualAgent(),
        LogicAgent(),
        ConsensusAgent(),
        ExplainabilityAgent(),
    ],
)
def test_agent_run_is_declared_but_not_implemented(agent: object) -> None:
    """Each agent should expose run(state) and raise NotImplementedError for now."""

    state = PipelineState(input=InputState(input_text="stub text"))
    with pytest.raises(NotImplementedError):
        getattr(agent, "run")(state)
