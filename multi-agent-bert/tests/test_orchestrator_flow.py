"""Orchestrator flow tests for stateful component sequencing."""

from dataclasses import dataclass, field
from typing import List

from models.state import PipelineState
from pipeline.orchestrator import PipelineOrchestrator


@dataclass(slots=True)
class RecorderAgent:
    """Test double that records call order and returns state unchanged."""

    name: str
    calls: List[str] = field(default_factory=list)

    def run(self, state: PipelineState) -> PipelineState:
        self.calls.append(self.name)
        return state


def test_orchestrator_calls_components_in_expected_order() -> None:
    """Orchestrator should invoke all components in starter sequence."""

    calls: List[str] = []
    orchestrator = PipelineOrchestrator(
        primary_classifier=RecorderAgent("primary", calls),
        router=RecorderAgent("router", calls),
        lexical_agent=RecorderAgent("lexical", calls),
        contextual_agent=RecorderAgent("contextual", calls),
        logic_agent=RecorderAgent("logic", calls),
        consensus_agent=RecorderAgent("consensus", calls),
        explainability_agent=RecorderAgent("explainability", calls),
    )

    final_state = orchestrator.run(text="hello world", request_id="r1", language="en")

    assert final_state.input.input_text == "hello world"
    assert final_state.request.request_id == "r1"
    assert calls == [
        "primary",
        "router",
        "lexical",
        "contextual",
        "logic",
        "consensus",
        "explainability",
    ]
