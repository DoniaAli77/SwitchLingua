"""Generic abstract base agent for stateful multi-agent pipelines."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

StateT = TypeVar("StateT")


class BaseAgent(ABC, Generic[StateT]):
    """Reusable abstract agent with logging and validation hooks."""

    def __init__(self, name: str | None = None, logger: logging.Logger | None = None) -> None:
        self.name = name or self.__class__.__name__
        self.logger = logger or logging.getLogger(self.name)

    @abstractmethod
    def run(self, state: StateT) -> StateT:
        """Execute agent logic and return updated state."""

    def execute(self, state: StateT) -> StateT:
        """Template execution wrapper with validation hooks and logging."""

        self.logger.debug("Starting agent execution: %s", self.name)
        self.validate_before(state)
        updated_state = self.run(state)
        self.validate_after(updated_state)
        self.logger.debug("Finished agent execution: %s", self.name)
        return updated_state

    def validate_before(self, state: StateT) -> None:
        """Optional hook to validate state before run. Override as needed."""

    def validate_after(self, state: StateT) -> None:
        """Optional hook to validate state after run. Override as needed."""
