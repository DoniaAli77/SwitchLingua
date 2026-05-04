"""Smoke tests for the three pipeline modes.

These tests verify pipeline *integrity* — that each mode produces the
expected set of outputs and leaves the state in a consistent shape.
Accuracy is not checked here.

Labels used: business, education, sports, tech (topic classification).

A high routing threshold (0.99) ensures the mock primary classifier's
heuristic confidence always falls below it, forcing escalation in both
paper_style and full_agentic modes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_pipeline import build_orchestrator
from src.state.schema import PipelineState, StateMetadata, TaskConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_LABELS = ["business", "education", "sports", "tech"]
_DESCS = {
    "business": "companies markets products corporate strategy",
    "education": "learning schools universities courses exams",
    "sports": "match team player score goals coach",
    "tech": "software apps AI machine learning cloud",
}
_TEXT = "The new app uses machine learning to predict sports scores."


def _make_state(pipeline_mode: str, *, enable_deliberation: bool = False) -> PipelineState:
    return PipelineState(
        metadata=StateMetadata(sample_id=f"smoke_{pipeline_mode}"),
        input_text=_TEXT,
        task_config=TaskConfig(
            task_name="topic",
            labels=_LABELS,
            label_descriptions=_DESCS,
            threshold=0.99,          # high threshold → heuristic primary always escalates
            pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
            enable_deliberation=enable_deliberation,
        ),
    )


def _build(pipeline_mode: str, *, enable_deliberation: bool = False) -> PipelineState:
    """Build a fully-wired orchestrator, run it, and return the result state."""
    task_config = TaskConfig(
        task_name="topic",
        labels=_LABELS,
        label_descriptions=_DESCS,
        threshold=0.99,
        pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
        enable_deliberation=enable_deliberation,
    )
    orch = build_orchestrator(
        task_config=task_config,
        threshold=0.99,
        enable_deliberation=enable_deliberation,
    )
    state = _make_state(pipeline_mode, enable_deliberation=enable_deliberation)
    return orch.run(state)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _history_components(state: PipelineState) -> list[str]:
    return [e.component for e in state.history]


# ===========================================================================
# primary_only
# ===========================================================================

class TestPrimaryOnly:

    @pytest.fixture(scope="class")
    def result(self) -> PipelineState:
        return _build("primary_only")

    def test_final_output_not_none(self, result: PipelineState) -> None:
        assert result.final_output is not None

    def test_final_label_in_labels(self, result: PipelineState) -> None:
        assert result.final_output.label in _LABELS

    def test_routing_info_is_none(self, result: PipelineState) -> None:
        """Router is skipped in primary_only mode."""
        assert result.routing_info is None

    def test_lexical_output_is_none(self, result: PipelineState) -> None:
        assert result.lexical_output is None

    def test_contextual_output_is_none(self, result: PipelineState) -> None:
        assert result.contextual_output is None

    def test_logic_output_is_none(self, result: PipelineState) -> None:
        assert result.logic_output is None

    def test_consensus_output_is_none(self, result: PipelineState) -> None:
        assert result.consensus_output is None

    def test_no_pipeline_error(self, result: PipelineState) -> None:
        assert "pipeline_error" not in result.extras


# ===========================================================================
# paper_style
# ===========================================================================

class TestPaperStyle:

    @pytest.fixture(scope="class")
    def result(self) -> PipelineState:
        return _build("paper_style")

    def test_routing_decision_is_escalate(self, result: PipelineState) -> None:
        assert result.routing_info is not None
        assert result.routing_info.decision == "escalate"

    def test_lexical_output_not_none(self, result: PipelineState) -> None:
        assert result.lexical_output is not None

    def test_logic_output_not_none(self, result: PipelineState) -> None:
        assert result.logic_output is not None

    def test_contextual_output_not_none(self, result: PipelineState) -> None:
        assert result.contextual_output is not None

    def test_consensus_output_not_none(self, result: PipelineState) -> None:
        assert result.consensus_output is not None

    def test_final_output_not_none(self, result: PipelineState) -> None:
        assert result.final_output is not None

    def test_final_label_in_labels(self, result: PipelineState) -> None:
        assert result.final_output.label in _LABELS

    def test_no_deliberation_output(self, result: PipelineState) -> None:
        """paper_style never runs deliberation."""
        assert result.deliberation_output is None

    def test_history_contains_transformer_contextual_agent(self, result: PipelineState) -> None:
        """paper_style should use TransformerContextualAgent as the contextual agent."""
        components = _history_components(result)
        assert "TransformerContextualAgent" in components

    def test_no_pipeline_error(self, result: PipelineState) -> None:
        assert "pipeline_error" not in result.extras


# ===========================================================================
# full_agentic — without deliberation
# ===========================================================================

class TestFullAgenticNoDeliberation:

    @pytest.fixture(scope="class")
    def result(self) -> PipelineState:
        return _build("full_agentic", enable_deliberation=False)

    def test_routing_decision_is_escalate(self, result: PipelineState) -> None:
        assert result.routing_info is not None
        assert result.routing_info.decision == "escalate"

    def test_lexical_output_not_none(self, result: PipelineState) -> None:
        assert result.lexical_output is not None

    def test_logic_output_not_none(self, result: PipelineState) -> None:
        assert result.logic_output is not None

    def test_contextual_output_not_none(self, result: PipelineState) -> None:
        assert result.contextual_output is not None

    def test_consensus_output_not_none(self, result: PipelineState) -> None:
        assert result.consensus_output is not None

    def test_final_output_not_none(self, result: PipelineState) -> None:
        assert result.final_output is not None

    def test_final_label_in_labels(self, result: PipelineState) -> None:
        assert result.final_output.label in _LABELS

    def test_deliberation_output_is_none(self, result: PipelineState) -> None:
        """enable_deliberation=False → deliberation stage skipped."""
        assert result.deliberation_output is None

    def test_no_pipeline_error(self, result: PipelineState) -> None:
        assert "pipeline_error" not in result.extras


# ===========================================================================
# full_agentic — with deliberation
# ===========================================================================

class TestFullAgenticWithDeliberation:

    @pytest.fixture(scope="class")
    def result(self) -> PipelineState:
        return _build("full_agentic", enable_deliberation=True)

    def test_routing_decision_is_escalate(self, result: PipelineState) -> None:
        assert result.routing_info is not None
        assert result.routing_info.decision == "escalate"

    def test_lexical_output_not_none(self, result: PipelineState) -> None:
        assert result.lexical_output is not None

    def test_logic_output_not_none(self, result: PipelineState) -> None:
        assert result.logic_output is not None

    def test_contextual_output_not_none(self, result: PipelineState) -> None:
        assert result.contextual_output is not None

    def test_consensus_output_not_none(self, result: PipelineState) -> None:
        assert result.consensus_output is not None

    def test_final_output_not_none(self, result: PipelineState) -> None:
        assert result.final_output is not None

    def test_final_label_in_labels(self, result: PipelineState) -> None:
        assert result.final_output.label in _LABELS

    def test_deliberation_output_not_none(self, result: PipelineState) -> None:
        """enable_deliberation=True → deliberation_output should be written."""
        assert result.deliberation_output is not None

    def test_no_pipeline_error(self, result: PipelineState) -> None:
        assert "pipeline_error" not in result.extras
