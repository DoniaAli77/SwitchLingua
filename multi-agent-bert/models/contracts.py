"""Typed contracts used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class TaskType(str, Enum):
    """Describes how the task produces output labels."""

    CLASSIFICATION = "classification"       # one label per input
    SEQUENCE_LABELING = "sequence_labeling"  # one label per token (e.g. NER)


class ClassificationLabel(str, Enum):
    """Allowed top-level labels for classification outputs."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class RouteTarget(str, Enum):
    """Possible routing destinations for specialist analysis."""

    LEXICAL = "lexical"
    CONTEXTUAL = "contextual"
    LOGIC = "logic"
    ALL = "all"


@dataclass(slots=True)
class TokenTag:
    """Label assigned to a single token in sequence-labeling tasks."""

    token: str
    tag: str
    confidence: float = 0.0


@dataclass(slots=True)
class SequenceLabelingResult:
    """Result payload for sequence-labeling tasks such as NER.

    ``tags`` contains one :class:`TokenTag` per input token in order.
    ``label`` is omitted because there is no single document-level label;
    downstream consumers should iterate ``tags`` directly.
    """

    tags: List[TokenTag] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClassificationResult:
    """Result payload produced by classifier-like components."""

    label: ClassificationLabel = ClassificationLabel.UNKNOWN
    confidence: float = 0.0
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RoutingDecision:
    """Router decision describing where the state should flow."""

    targets: List[RouteTarget] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class AgentOutput:
    """Generic output wrapper for specialist agents.

    ``result`` holds either a :class:`ClassificationResult` (single-label
    tasks) or a :class:`SequenceLabelingResult` (token-level tasks).
    Check ``isinstance(output.result, SequenceLabelingResult)`` to branch.
    """

    agent_name: str
    result: Optional[Union[ClassificationResult, SequenceLabelingResult]] = None
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConsensusResult:
    """Combined decision created from multiple specialist outputs."""

    final: ClassificationResult = field(default_factory=ClassificationResult)
    disagreements: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ExplanationResult:
    """Structured explainability payload for downstream consumers."""

    summary: str = ""
    evidence: List[str] = field(default_factory=list)
    confidence_notes: str = ""
