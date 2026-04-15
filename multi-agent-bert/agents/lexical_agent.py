"""Lexical specialist agent stub."""

from __future__ import annotations

from models.state import PipelineState


class LexicalAgent:
    """Analyzes lexical surface signals such as token-level cues."""

    def run(self, state: PipelineState) -> PipelineState:
        """Attach lexical analysis output to pipeline state."""

        raise NotImplementedError("Lexical analysis logic is not implemented yet.")
