"""Orchestrator flow tests for mode-aware component sequencing."""

from dataclasses import dataclass
from typing import List

from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.router import Router
from src.state.schema import (
    ModelOutput,
    PipelineState,
    StateMetadata,
    TaskConfig,
)


@dataclass(slots=True)
class _RecorderAgent:
    name: str
    calls: List[str]

    def run(self, state: PipelineState) -> PipelineState:
        self.calls.append(self.name)
        return state


@dataclass(slots=True)
class _PrimaryRecorder:
    calls: List[str]
    confidence: float
    label: str = "positive"

    def run(self, state: PipelineState) -> PipelineState:
        self.calls.append("primary")
        state.primary_model_output = ModelOutput(
            label=self.label,
            confidence=self.confidence,
            probabilities={
                "positive": self.confidence,
                "negative": 1.0 - self.confidence,
            },
            raw_text=state.input_text,
        )
        return state


@dataclass(slots=True)
class _RouterRecorder:
    calls: List[str]

    def run(self, state: PipelineState) -> PipelineState:
        self.calls.append("router")
        return Router().run(state)


def _make_state(
    pipeline_mode: str,
    *,
    threshold: float = 0.5,
    enable_deliberation: bool = False,
) -> PipelineState:
    return PipelineState(
        metadata=StateMetadata(sample_id="s1"),
        input_text="hello",
        task_config=TaskConfig(
            task_name="sentiment",
            labels=["positive", "negative"],
            label_descriptions={"positive": "pos", "negative": "neg"},
            threshold=threshold,
            enable_deliberation=enable_deliberation,
            pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
        ),
    )


def _build_orchestrator(calls: List[str], primary_confidence: float) -> PipelineOrchestrator:
    return PipelineOrchestrator(
        primary_classifier=_PrimaryRecorder(calls=calls, confidence=primary_confidence),
        router=_RouterRecorder(calls=calls),
        lexical_agent=_RecorderAgent("lexical", calls),
        contextual_agent=_RecorderAgent("contextual", calls),
        logic_agent=_RecorderAgent("logic", calls),
        consensus_agent=_RecorderAgent("consensus", calls),
        explainability_agent=_RecorderAgent("explainability", calls),
        deliberation_agent=_RecorderAgent("deliberation", calls),
    )


def _has_mode_history(state: PipelineState, expected_mode: str) -> bool:
    for event in state.history:
        if event.component != "orchestrator":
            continue
        if event.outputs.get("pipeline_mode") == expected_mode:
            return True
    return False


def test_primary_only_skips_router_and_specialists() -> None:
    calls: List[str] = []
    orch = _build_orchestrator(calls=calls, primary_confidence=0.15)
    state = _make_state("primary_only", threshold=0.9, enable_deliberation=True)

    out = orch.run(state)

    assert calls == ["primary"]
    assert out.routing_info is None
    assert out.final_output is not None
    assert out.final_output.label == "positive"
    assert out.explanation_output is not None
    assert "Primary-only mode" in out.explanation_output.summary
    assert _has_mode_history(out, "primary_only")


def test_paper_style_escalation_runs_lexical_and_logic_only() -> None:
    calls: List[str] = []
    orch = _build_orchestrator(calls=calls, primary_confidence=0.20)
    state = _make_state("paper_style", threshold=0.9, enable_deliberation=True)

    out = orch.run(state)

    assert calls == [
        "primary",
        "router",
        "lexical",
        "logic",
        "consensus",
        "explainability",
    ]
    assert "contextual" not in calls
    assert "deliberation" not in calls
    assert out.routing_info is not None
    assert out.routing_info.decision == "escalate"
    assert _has_mode_history(out, "paper_style")


def test_full_agentic_escalation_keeps_existing_behavior() -> None:
    calls: List[str] = []
    orch = _build_orchestrator(calls=calls, primary_confidence=0.20)
    state = _make_state("full_agentic", threshold=0.9, enable_deliberation=True)

    out = orch.run(state)

    assert calls == [
        "primary",
        "router",
        "lexical",
        "logic",
        "contextual",
        "deliberation",
        "consensus",
        "explainability",
    ]
    assert out.routing_info is not None
    assert out.routing_info.decision == "escalate"
    assert _has_mode_history(out, "full_agentic")
