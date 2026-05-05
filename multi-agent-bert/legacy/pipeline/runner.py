"""Convenience pipeline runner entrypoint."""

from __future__ import annotations

from typing import Optional

from models.state import PipelineState
from pipeline.orchestrator import PipelineOrchestrator


def run_pipeline(
    text: str,
    request_id: Optional[str] = None,
    language: Optional[str] = None,
    orchestrator: Optional[PipelineOrchestrator] = None,
) -> PipelineState:
    """Execute the pipeline using default orchestrator if none is provided."""

    active_orchestrator = orchestrator or PipelineOrchestrator.default()
    return active_orchestrator.run(text=text, request_id=request_id, language=language)
