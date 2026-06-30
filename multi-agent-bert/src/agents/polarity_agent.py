"""LLM-based sentiment POLARITY specialist agent (experimental variant).

Part of the ``lexical_polarity_contextual`` sentiment agent variant: it replaces
the :class:`~src.agents.llm_logic_agent.LLMLogicAgent` in the full_agentic
specialist trio with a polarity *decider*. To keep the router and consensus
untouched, it writes to the **same** ``state.logic_output`` slot the logic agent
uses, so it is a drop-in substitution at the orchestrator's logic stage.

Unlike the Logic agent (structural/relational reasoning) this agent answers a
single question — "Is the author expressing an evaluative attitude? If yes, what
polarity?" — distinguishing expressed polarity from mere mention of sentiment
words or artifacts. It does not enumerate lexical cues (Lexical's job) and does
not perform full pragmatic interpretation (Contextual's job).

The agent:

1. Builds a polarity-decision prompt.
2. Calls ``LLMClient.generate(prompt)``.
3. Parses the JSON response against the same strict schema as the other agents.
4. Validates the returned label against ``task_config.labels``.
5. Writes an ``AgentOutput`` to ``state.logic_output``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.agents._abstain import abstain_output
from src.prompts._primary_block import build_primary_signal
from src.agents.base_agent import BaseAgent
from src.llm.base_client import LLMClient, LLMClientError  # noqa: F401
from src.prompts.polarity_prompt import build_user_prompt, get_system_prompt
from src.state.schema import AgentOutput, ModelOutput, PipelineState

_REQUIRED_KEYS: frozenset[str] = frozenset({"label", "confidence", "reasoning", "evidence"})

_PARSE_FAIL_NOTE = "PolarityAgent: response parse failed; abstain fallback applied."
_INVALID_LABEL_NOTE = "PolarityAgent: LLM returned an invalid label; abstain fallback applied."


class PolarityParseError(ValueError):
    """Raised when the LLM response cannot be parsed into the expected schema."""


def _extract_json(raw: str) -> str:
    """Strip markdown fences and surrounding whitespace from a raw LLM response."""
    stripped = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        return fence.group(1)
    return stripped


class PolarityAgent(BaseAgent[PipelineState]):
    """LLM-backed sentiment polarity decider for the full_agentic pipeline mode.

    Drop-in replacement for :class:`LLMLogicAgent` in the
    ``lexical_polarity_contextual`` variant; writes ``state.logic_output``.

    Parameters
    ----------
    llm_client:
        Any ``LLMClient`` implementation.  Pass ``MockLLMClient`` in tests.
    name:
        Optional agent name for logging and ``AgentOutput.agent_name``.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "PolarityAgent", logger=logger)
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def validate_before(self, state: PipelineState) -> None:
        if not state.input_text or not state.input_text.strip():
            raise ValueError("PolarityAgent: state.input_text is empty or blank.")
        if not state.task_config.labels:
            raise ValueError("PolarityAgent: state.task_config.labels is empty.")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Call LLM, parse response, write result to state.logic_output."""
        task = state.task_config

        primary_signal = (
            build_primary_signal(state.primary_model_output)
            if task.agents_use_primary_signal
            else None
        )
        prompt = build_user_prompt(
            task_name=task.task_name,
            labels=task.labels,
            label_descriptions=task.label_descriptions,
            text=state.input_text,
            primary_signal=primary_signal,
        )

        self.logger.debug("%s: sending prompt (%d chars)", self.name, len(prompt))

        raw_response = self.llm_client.generate(f"{get_system_prompt()}\n\n{prompt}")

        try:
            parsed = self._parse_response(raw_response)
        except PolarityParseError as exc:
            self.logger.warning("%s: parse error — %s", self.name, exc)
            state.logic_output = self._fallback_output(state, _PARSE_FAIL_NOTE, raw_response)
            state.append_history(
                component=self.name,
                summary=f"Parse error — falling back. {exc}",
                outputs={"error": str(exc), "fallback": True},
            )
            return state

        label: str = parsed["label"]
        confidence: float = parsed["confidence"]
        reasoning: str = parsed["reasoning"]
        evidence: List[str] = parsed["evidence"]

        if not task.is_allowed_label(label):
            self.logger.warning(
                "%s: LLM returned invalid label '%s'; falling back.", self.name, label
            )
            state.logic_output = self._fallback_output(
                state, _INVALID_LABEL_NOTE, raw_response
            )
            state.append_history(
                component=self.name,
                summary=f"Invalid label '{label}' returned by LLM — falling back.",
                outputs={"invalid_label": label, "fallback": True},
            )
            return state

        probabilities = self._build_probabilities(label, confidence, task.labels)

        state.logic_output = AgentOutput(
            agent_name=self.name,
            model_output=ModelOutput(
                label=label,
                confidence=confidence,
                probabilities=probabilities,
                raw_text=state.input_text,
            ),
            notes=reasoning,
            features={"evidence": evidence, "raw_llm_response": raw_response},
        )

        self.logger.debug("%s: label=%s confidence=%.4f", self.name, label, confidence)
        state.append_history(
            component=self.name,
            summary=f"Polarity label '{label}' (confidence={confidence:.3f}). {reasoning}",
            outputs={
                "label": label,
                "confidence": confidence,
                "probabilities": dict(probabilities),
                "evidence": evidence,
                "fallback": False,
            },
        )
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse and validate the LLM JSON response.

        Raises
        ------
        PolarityParseError
            When the response is not valid JSON or is missing required keys.
        """
        try:
            data = json.loads(_extract_json(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            raise PolarityParseError(f"JSON decode failed: {exc}") from exc

        if not isinstance(data, dict):
            raise PolarityParseError(f"Expected a JSON object, got {type(data).__name__}")

        missing = _REQUIRED_KEYS - data.keys()
        if missing:
            raise PolarityParseError(f"Missing required keys: {sorted(missing)}")

        if not isinstance(data["confidence"], (int, float)):
            raise PolarityParseError(
                f"'confidence' must be a number, got {type(data['confidence']).__name__}"
            )
        if not isinstance(data["evidence"], list):
            raise PolarityParseError(
                f"'evidence' must be a list, got {type(data['evidence']).__name__}"
            )
        return data

    def _fallback_output(
        self, state: PipelineState, note: str, raw_response: str
    ) -> AgentOutput:
        # Abstain (no vote) instead of defaulting to labels[0]; consensus excludes
        # None-label outputs. Attach the raw response for debugging.
        out = abstain_output(self.name, state, note)
        out.features["raw_llm_response"] = raw_response
        return out

    @staticmethod
    def _build_probabilities(
        label: str, confidence: float, labels: List[str]
    ) -> Dict[str, float]:
        """Assign *confidence* to *label* and distribute the rest uniformly."""
        n = len(labels)
        if n == 0:
            return {}
        remainder = (1.0 - confidence) / (n - 1) if n > 1 else 0.0
        return {lbl: confidence if lbl == label else remainder for lbl in labels}
