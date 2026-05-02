"""Deliberation agent for the stateful multi-agent text classification pipeline.

Reviews the outputs of lexical, contextual, and logic specialist agents and
proposes either a refined label recommendation or a justification for why one
label should be preferred over the others.

The agent is designed to run **after** all three specialist agents and
**before** the ConsensusAgent.  Its output (``state.deliberation_output``)
can optionally be included as an extra weighted input to the consensus vote.

Usage
-----
Wire the agent into the orchestrator and enable the feature via::

    task_config.enable_deliberation = True

The orchestrator is responsible for the gate — ``DeliberationAgent.run``
always executes when called.  The flag check belongs in the orchestrator so
the agent remains unit-testable without any config coupling.

Error taxonomy
--------------
``DeliberationParseError``
    The LLM returned something that is not valid JSON, is missing a required
    key, or contains an out-of-vocabulary label.  The agent catches this,
    logs a warning, and leaves ``state.deliberation_output`` as ``None``
    rather than crashing the pipeline.

``LLMClientError``
    A non-recoverable backend error (network, auth, quota).  The agent
    re-raises this so the orchestrator can decide how to handle it.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base_agent import BaseAgent
from src.llm.base_client import LLMClient, LLMClientError  # noqa: F401 (re-exported for callers)
from src.prompts.deliberation_prompt import SYSTEM_PROMPT, build_user_prompt
from src.state.schema import DeliberationOutput, PipelineState

# Keys required in every valid LLM deliberation response.
_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"recommended_label", "confidence", "justification", "mode"}
)

# Valid deliberation modes.
_VALID_MODES: frozenset[str] = frozenset({"recommendation", "justification"})

_PARSE_FAIL_NOTE = (
    "DeliberationAgent: response parse failed; no deliberation output stored."
)

# Max characters allowed in per-agent notes before truncation.
_MAX_NOTE_CHARS = 120


class DeliberationParseError(ValueError):
    """Raised when the LLM response cannot be parsed into the expected schema."""


class DeliberationAgent(BaseAgent[PipelineState]):
    """LLM-backed deliberation agent that reviews all specialist outputs.

    Reads the outputs of the lexical, contextual, and logic agents from state,
    builds a structured prompt summarising their votes, calls the LLM, and
    writes a :class:`~src.state.schema.DeliberationOutput` to
    ``state.deliberation_output``.

    On a parse/validation error the agent writes nothing to state (leaving
    ``state.deliberation_output`` as ``None``) and appends a history event
    describing the failure.

    Parameters
    ----------
    llm_client:
        Any :class:`~src.llm.base_client.LLMClient` implementation.
        Pass ``MockLLMClient`` in tests.
    name:
        Optional agent name for logging and history events.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "DeliberationAgent", logger=logger)
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def validate_before(self, state: PipelineState) -> None:
        if not state.task_config.labels:
            raise ValueError("DeliberationAgent: state.task_config.labels is empty.")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Build deliberation prompt, call LLM, parse result, write to state."""

        task = state.task_config
        votes = self._collect_votes(state)

        prompt = build_user_prompt(
            task_name=task.task_name,
            labels=task.labels,
            label_descriptions=task.label_descriptions,
            agent_votes=votes,
        )

        self.logger.debug(
            "%s: sending deliberation prompt (%d chars)", self.name, len(prompt)
        )

        raw_response = self.llm_client.generate(f"{SYSTEM_PROMPT}\n\n{prompt}")

        try:
            parsed = self._parse_response(raw_response, task.labels)
        except DeliberationParseError as exc:
            self.logger.warning("%s: parse error — %s", self.name, exc)
            state.append_history(
                component=self.name,
                summary=f"Parse error — no deliberation output stored. {exc}",
                outputs={"error": str(exc), "fallback": True},
            )
            return state

        state.deliberation_output = DeliberationOutput(
            recommended_label=parsed["recommended_label"],
            confidence=parsed["confidence"],
            justification=parsed["justification"],
            mode=parsed["mode"],
        )

        self.logger.debug(
            "%s: recommended_label=%s confidence=%.4f mode=%s",
            self.name,
            parsed["recommended_label"],
            parsed["confidence"],
            parsed["mode"],
        )
        state.append_history(
            component=self.name,
            summary=(
                f"Deliberation complete. "
                f"recommended_label='{parsed['recommended_label']}' "
                f"confidence={parsed['confidence']:.3f} mode='{parsed['mode']}'"
            ),
            outputs={
                "recommended_label": parsed["recommended_label"],
                "confidence": parsed["confidence"],
                "justification": parsed["justification"],
                "mode": parsed["mode"],
                "fallback": False,
            },
        )
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_votes(
        self, state: PipelineState
    ) -> List[Tuple[str, Optional[str], Optional[float], str]]:
        """Collect ``(agent_name, label, confidence, notes)`` from agent outputs.

        Notes are sanitized — whitespace is normalized and long strings are
        truncated to avoid injecting excessive text into the prompt.
        """
        votes: List[Tuple[str, Optional[str], Optional[float], str]] = []

        for slot_name, output in [
            ("lexical", state.lexical_output),
            ("contextual", state.contextual_output),
            ("logic", state.logic_output),
        ]:
            if output is None:
                continue
            mo = output.model_output
            notes = re.sub(r"\s+", " ", output.notes or "").strip()
            if len(notes) > _MAX_NOTE_CHARS:
                notes = notes[:_MAX_NOTE_CHARS].rstrip() + "..."
            votes.append((slot_name, mo.label, mo.confidence, notes))

        return votes

    def _parse_response(
        self, raw: str, allowed_labels: List[str]
    ) -> Dict[str, Any]:
        """Parse and validate the LLM deliberation JSON response.

        Raises ``DeliberationParseError`` on any structural problem so the
        caller can decide how to handle the failure without crashing.
        """
        raw = raw.strip()

        # Strip optional markdown code fences (```json ... ``` or ``` ... ```)
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            data: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DeliberationParseError(f"Invalid JSON: {exc}") from exc

        missing = _REQUIRED_KEYS - data.keys()
        if missing:
            raise DeliberationParseError(f"Missing required keys: {sorted(missing)}")

        label = data["recommended_label"]
        if label not in allowed_labels:
            raise DeliberationParseError(
                f"recommended_label '{label}' is not in allowed labels: {allowed_labels}"
            )

        confidence = data["confidence"]
        if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
            raise DeliberationParseError(
                f"confidence must be a float in [0.0, 1.0]; got {confidence!r}"
            )

        mode = data["mode"]
        if mode not in _VALID_MODES:
            raise DeliberationParseError(
                f"mode must be one of {sorted(_VALID_MODES)}; got {mode!r}"
            )

        return {
            "recommended_label": label,
            "confidence": float(confidence),
            "justification": str(data["justification"]),
            "mode": mode,
        }
