"""Tests for dataclass state and model contracts."""

from models import ClassificationLabel, ClassificationResult, PipelineState
from models.state import InputState


def test_pipeline_state_minimal_init() -> None:
    """PipelineState should initialize nested state containers."""

    state = PipelineState(input=InputState(input_text="hello"))
    assert state.input.input_text == "hello"
    assert state.execution.primary_result is None
    assert state.diagnostics.errors == []


def test_classification_result_defaults() -> None:
    """ClassificationResult should expose stable default values."""

    result = ClassificationResult()
    assert result.label == ClassificationLabel.UNKNOWN
    assert result.confidence == 0.0
    assert result.rationale == ""
