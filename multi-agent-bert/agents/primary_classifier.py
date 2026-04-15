"""Primary classifier component stub."""

from __future__ import annotations

from models.state import PipelineState


class PrimaryClassifier:
    """Creates the first-pass classification result from raw input text."""

    def run(self, state: PipelineState) -> PipelineState:
        """Update pipeline state with primary classification output."""

        raise NotImplementedError("Primary classification logic is not implemented yet.")
