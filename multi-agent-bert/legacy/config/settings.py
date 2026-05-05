"""Typed configuration access for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(slots=True)
class PipelineConfig:
    """Top-level pipeline settings loaded from YAML."""

    name: str = "stateful_multi_agent_classifier"
    version: str = "0.1.0"


@dataclass(slots=True)
class ClassificationConfig:
    """Classification label settings."""

    labels: List[str] = field(default_factory=list)


@dataclass(slots=True)
class RoutingConfig:
    """Keyword groups used by router decision heuristics."""

    lexical_keywords: List[str] = field(default_factory=list)
    contextual_keywords: List[str] = field(default_factory=list)
    logic_keywords: List[str] = field(default_factory=list)


@dataclass(slots=True)
class AppConfig:
    """Complete application configuration object."""

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)


def load_config(config_path: str | Path) -> AppConfig:
    """Load YAML configuration and return typed config dataclasses.

    Business mapping logic is intentionally not implemented yet.
    """

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        _raw: Dict[str, Any] = yaml.safe_load(file) or {}

    raise NotImplementedError("Map raw YAML into AppConfig dataclasses.")
