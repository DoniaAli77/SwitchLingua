"""Abstract LLM client interface for the multi-agent pipeline.

Any real backend (OpenAI, Anthropic, local Ollama, etc.) implements
``LLMClient`` and is injected into ``ContextualAgent`` at construction time.
The pipeline itself never imports a concrete backend, keeping all agents
testable without network access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Minimal interface every LLM backend must satisfy."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send *prompt* to the model and return the raw text response.

        Parameters
        ----------
        prompt:
            The fully rendered prompt string to send.

        Returns
        -------
        str
            Raw model response.  The caller is responsible for parsing it.

        Raises
        ------
        LLMClientError
            On any non-recoverable backend error (network, auth, quota …).
        """


class LLMClientError(RuntimeError):
    """Raised by ``LLMClient.generate`` on non-recoverable backend errors."""
