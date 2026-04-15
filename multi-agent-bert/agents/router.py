"""Router component stub."""

from __future__ import annotations

from models.state import PipelineState


class Router:
    """Determines which specialist agents should process the current state."""

    def run(self, state: PipelineState) -> PipelineState:
        """Update pipeline state with routing decision metadata."""

        raise NotImplementedError("Routing logic is not implemented yet.")
