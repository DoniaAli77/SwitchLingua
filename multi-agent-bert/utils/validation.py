"""Validation helpers for pipeline inputs and outputs."""

from __future__ import annotations

from models.state import PipelineState


def validate_input_text(text: str) -> None:
    """Validate raw input text before pipeline execution."""

    raise NotImplementedError("Input validation logic is not implemented yet.")


def validate_state(state: PipelineState) -> None:
    """Validate state consistency after component execution."""

    raise NotImplementedError("State validation logic is not implemented yet.")
