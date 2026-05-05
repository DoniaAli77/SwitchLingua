"""Consensus agent stub."""

from __future__ import annotations

from models.state import PipelineState


class ConsensusAgent:
    """Combines specialist outputs into a unified final decision."""

    def run(self, state: PipelineState) -> PipelineState:
        """Attach consensus result to pipeline state."""

        raise NotImplementedError("Consensus logic is not implemented yet.")
