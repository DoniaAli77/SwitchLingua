"""Contextual specialist agent stub."""

from __future__ import annotations

from models.state import PipelineState


class ContextualAgent:
    """Analyzes pragmatic and situational context for label refinement."""

    def run(self, state: PipelineState) -> PipelineState:
        """Attach contextual analysis output to pipeline state."""

        raise NotImplementedError("Contextual analysis logic is not implemented yet.")
