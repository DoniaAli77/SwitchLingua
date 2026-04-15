"""Typed contracts used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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
    """Generic output wrapper for specialist agents."""

    agent_name: str
    result: Optional[ClassificationResult] = None
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
