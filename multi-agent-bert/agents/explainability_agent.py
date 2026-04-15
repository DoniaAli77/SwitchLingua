"""Explainability agent stub."""

from __future__ import annotations

from models.state import PipelineState


class ExplainabilityAgent:
    """Builds structured explanations for the final classification decision."""

    def run(self, state: PipelineState) -> PipelineState:
        """Attach explanation payload to pipeline state."""

        raise NotImplementedError("Explainability logic is not implemented yet.")
