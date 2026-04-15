"""Dataclass state container shared across all pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from models.contracts import (
    AgentOutput,
    ClassificationResult,
    ConsensusResult,
    ExplanationResult,
    RoutingDecision,
)


@dataclass(slots=True)
class RequestState:
    """Request-level metadata and runtime context."""

    request_id: Optional[str] = None
    language: Optional[str] = None
    sample_id: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass(slots=True)
class InputState:
    """Raw inputs received by the pipeline."""

    input_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskState:
    """Task configuration for generic classification use cases."""

    task_name: str = ""
    labels: List[str] = field(default_factory=list)
    label_descriptions: Dict[str, str] = field(default_factory=dict)
    threshold: float = 0.0


@dataclass(slots=True)
class AgentState:
    """Specialist-agent outputs."""

    lexical_output: Optional[AgentOutput] = None
    contextual_output: Optional[AgentOutput] = None
    logic_output: Optional[AgentOutput] = None


@dataclass(slots=True)
class ExecutionState:
    """Execution outputs produced during pipeline progression."""

    primary_result: Optional[ClassificationResult] = None
    routing_decision: Optional[RoutingDecision] = None
    consensus_result: Optional[ConsensusResult] = None
    explanation_result: Optional[ExplanationResult] = None


@dataclass(slots=True)
class DiagnosticsState:
    """Warnings and errors emitted while processing."""

    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass(slots=True)
class FinalState:
    """Final output fields returned to callers."""

    final_label: Optional[str] = None
    final_confidence: Optional[float] = None


@dataclass(slots=True)
class PipelineState:
    """Top-level shared state passed across all pipeline components."""

    request: RequestState = field(default_factory=RequestState)
    input: InputState = field(default_factory=lambda: InputState(input_text=""))
    task: TaskState = field(default_factory=TaskState)
    agents: AgentState = field(default_factory=AgentState)
    execution: ExecutionState = field(default_factory=ExecutionState)
    diagnostics: DiagnosticsState = field(default_factory=DiagnosticsState)
    final: FinalState = field(default_factory=FinalState)
