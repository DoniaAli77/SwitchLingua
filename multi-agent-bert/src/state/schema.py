"""Shared state schema for a generic multi-agent text-classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class StateMetadata:
    """Metadata for one sample flowing through the pipeline."""

    sample_id: str
    timestamp: Optional[datetime] = None


@dataclass(slots=True)
class TaskConfig:
    """Task-level configuration shared across all components.

    This schema stays generic for tasks like sentiment, topic, and NER.
    """

    task_name: str
    labels: List[str] = field(default_factory=list)
    label_descriptions: Dict[str, str] = field(default_factory=dict)

    def is_allowed_label(self, label: str) -> bool:
        """Return True if label is part of the configured label space."""

        return label in self.labels

    def validate_label(self, label: str, field_name: str = "label") -> None:
        """Validate a label against configured labels and raise on mismatch."""

        if not self.labels:
            raise ValueError("TaskConfig.labels cannot be empty.")
        if label not in self.labels:
            allowed = ", ".join(self.labels)
            raise ValueError(f"Invalid {field_name}: '{label}'. Allowed labels: [{allowed}]")


@dataclass(slots=True)
class ModelOutput:
    """Generic model output payload usable by primary and specialist agents."""

    label: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""

    def validate_labels(self, task_config: TaskConfig, field_name: str = "label") -> None:
        """Validate single label and probability-label keys against allowed labels."""

        if self.label is not None:
            task_config.validate_label(self.label, field_name=field_name)

        for prob_label in self.probabilities:
            task_config.validate_label(prob_label, field_name=f"{field_name}.probabilities_key")


@dataclass(slots=True)
class AgentOutput:
    """Output wrapper for specialist agents."""

    agent_name: str
    model_output: ModelOutput = field(default_factory=ModelOutput)
    notes: str = ""
    features: Dict[str, Any] = field(default_factory=dict)

    def validate_labels(self, task_config: TaskConfig) -> None:
        """Validate labels used inside the enclosed model output."""

        self.model_output.validate_labels(task_config, field_name=f"{self.agent_name}.label")


@dataclass(slots=True)
class RoutingInfo:
    """Routing configuration and decision state."""

    threshold: float
    decision: str


@dataclass(slots=True)
class ConsensusOutput:
    """Merged decision across specialist agents."""

    label: Optional[str] = None
    confidence: Optional[float] = None
    votes: Dict[str, str] = field(default_factory=dict)
    rationale: str = ""

    def validate_labels(self, task_config: TaskConfig) -> None:
        """Validate consensus label against allowed labels when available."""

        if self.label is not None:
            task_config.validate_label(self.label, field_name="consensus.label")


@dataclass(slots=True)
class ExplanationOutput:
    """Structured explainability payload for traceability and debugging."""

    summary: str = ""
    evidence: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


@dataclass(slots=True)
class FinalOutput:
    """Final user-facing output of the pipeline."""

    label: Optional[str] = None
    confidence: Optional[float] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def validate_labels(self, task_config: TaskConfig) -> None:
        """Validate final label against allowed labels when available."""

        if self.label is not None:
            task_config.validate_label(self.label, field_name="final.label")


@dataclass(slots=True)
class PipelineState:
    """Shared mutable state passed through all pipeline components."""

    metadata: StateMetadata
    input_text: str
    task_config: TaskConfig

    primary_model_output: ModelOutput = field(default_factory=ModelOutput)
    routing_info: Optional[RoutingInfo] = None

    lexical_output: Optional[AgentOutput] = None
    contextual_output: Optional[AgentOutput] = None
    logic_output: Optional[AgentOutput] = None

    consensus_output: Optional[ConsensusOutput] = None
    explanation_output: Optional[ExplanationOutput] = None
    final_output: Optional[FinalOutput] = None

    extras: Dict[str, Any] = field(default_factory=dict)

    def validate_labels(self) -> None:
        """Validate all labels present in state against task-config labels."""

        self.primary_model_output.validate_labels(self.task_config, field_name="primary_model_output.label")

        if self.lexical_output is not None:
            self.lexical_output.validate_labels(self.task_config)
        if self.contextual_output is not None:
            self.contextual_output.validate_labels(self.task_config)
        if self.logic_output is not None:
            self.logic_output.validate_labels(self.task_config)

        if self.consensus_output is not None:
            self.consensus_output.validate_labels(self.task_config)
        if self.final_output is not None:
            self.final_output.validate_labels(self.task_config)


def utc_now() -> datetime:
    """Return timezone-aware UTC datetime for state metadata."""

    return datetime.now(timezone.utc)
