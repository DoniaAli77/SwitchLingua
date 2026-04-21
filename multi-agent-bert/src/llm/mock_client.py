"""Mock LLM client for testing the ContextualAgent without a real backend.

Three modes:

``fixed``
    Always returns the same hard-coded JSON string.  Useful for happy-path
    tests where you want full control of the response.

``label_echo``
    Inspects the prompt for the first occurrence of any label in the
    ``allowed_labels`` list passed at construction and returns a valid JSON
    response for that label.  Mimics a "dumb but correct" model that echoes
    whatever label it sees first in the prompt.

``raise_on_call``
    Raises ``LLMClientError`` on every call.  Used to test error-handling
    paths in the agent.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Literal, Optional

from src.llm.base_client import LLMClient, LLMClientError

MockMode = Literal["fixed", "label_echo", "raise_on_call"]

# Sentinel used in notes when the mock fell back to the first allowed label.
LABEL_ECHO_FALLBACK_NOTE = "mock_label_echo: no label found in prompt; used first allowed label"


class MockLLMClient(LLMClient):
    """Deterministic LLM stand-in for pipeline tests.

    Parameters
    ----------
    mode:
        Operating mode (see module docstring).
    fixed_response:
        Raw string returned verbatim in ``"fixed"`` mode.
    allowed_labels:
        Label list used by ``"label_echo"`` mode to detect which label the
        prompt references.  Must be supplied when mode is ``"label_echo"``.
    fixed_confidence:
        Confidence value embedded in ``"label_echo"`` responses (default 0.85).
    call_log:
        If a list is passed, every prompt sent to ``generate`` is appended to
        it, enabling call-count assertions in tests.
    """

    def __init__(
        self,
        mode: MockMode = "fixed",
        fixed_response: Optional[str] = None,
        allowed_labels: Optional[List[str]] = None,
        fixed_confidence: float = 0.85,
        call_log: Optional[List[str]] = None,
    ) -> None:
        self.mode = mode
        self.fixed_response = fixed_response or json.dumps(
            {
                "label": "tech",
                "confidence": 0.85,
                "reasoning": "Mock reasoning.",
                "evidence": ["mock evidence"],
            }
        )
        self.allowed_labels: List[str] = allowed_labels or []
        self.fixed_confidence = fixed_confidence
        self.call_log: List[str] = call_log if call_log is not None else []

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        self.call_log.append(prompt)

        if self.mode == "raise_on_call":
            raise LLMClientError("MockLLMClient: raise_on_call mode is active.")

        if self.mode == "fixed":
            return self.fixed_response

        # mode == "label_echo"
        detected = self._detect_label(prompt)
        note = "" if detected else LABEL_ECHO_FALLBACK_NOTE
        label = detected or (self.allowed_labels[0] if self.allowed_labels else "unknown")
        return json.dumps(
            {
                "label": label,
                "confidence": self.fixed_confidence,
                "reasoning": f"Mock: detected label '{label}' in prompt context.",
                "evidence": [f"prompt_contains:{label}"],
                "mock_note": note,
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_label(self, prompt: str) -> Optional[str]:
        """Return the first allowed label found in *prompt*, or None."""
        for label in self.allowed_labels:
            if re.search(r"\b" + re.escape(label) + r"\b", prompt, re.IGNORECASE):
                return label
        return None
