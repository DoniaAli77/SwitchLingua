"""Contextual agent for stateful multi-agent text classification pipelines.

Delegates classification to an LLM via a pluggable ``LLMClient``.  The agent:

1. Builds a structured prompt from the task config and input text.
2. Calls ``LLMClient.generate(prompt)``.
3. Parses the JSON response against a strict schema.
4. Validates the returned label against ``task_config.labels``.
5. Writes an ``AgentOutput`` to ``state.contextual_output``.

Error taxonomy
--------------
``ContextualParseError``
    The LLM returned something that is not valid JSON or is missing a
    required key.  The agent catches this and writes a low-confidence
    fallback output rather than crashing the pipeline.

``LLMClientError``
    A non-recoverable backend error (network, auth, quota).  The agent
    re-raises this so the orchestrator can decide how to handle it.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.llm.base_client import LLMClient, LLMClientError  # noqa: F401 (re-exported for callers)
from src.prompts.contextual_prompt import SYSTEM_PROMPT, build_user_prompt
from src.state.schema import AgentOutput, ModelOutput, PipelineState

# Keys that must be present in a valid LLM response.
_REQUIRED_KEYS: frozenset[str] = frozenset({"label", "confidence", "reasoning", "evidence"})

_PARSE_FAIL_NOTE = "ContextualAgent: response parse failed; uniform fallback applied."
_INVALID_LABEL_NOTE = "ContextualAgent: LLM returned an invalid label; uniform fallback applied."
_MAX_SUMMARY_CHARS = 160


class ContextualParseError(ValueError):
    """Raised when the LLM response cannot be parsed into the expected schema."""


class ContextualAgent(BaseAgent[PipelineState]):
    """LLM-backed contextual classifier with a pluggable backend.

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
        super().__init__(name=name or "ContextualAgent", logger=logger)
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def validate_before(self, state: PipelineState) -> None:
        if not state.input_text or not state.input_text.strip():
            raise ValueError("ContextualAgent: state.input_text is empty or blank.")
        if not state.task_config.labels:
            raise ValueError("ContextualAgent: state.task_config.labels is empty.")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Call LLM, parse response, write result to state.contextual_output."""

        task = state.task_config
        prior_summaries: List[str] = []
        if task.contextual_use_prior_outputs:
            prior_summaries = self._build_prior_summaries(state)

        prompt = build_user_prompt(
            task_name=task.task_name,
            labels=task.labels,
            label_descriptions=task.label_descriptions,
            text=state.input_text,
            prior_agent_summaries=prior_summaries,
        )

        self.logger.debug("%s: sending prompt (%d chars)", self.name, len(prompt))

        raw_response = self.llm_client.generate(
            f"{SYSTEM_PROMPT}\n\n{prompt}"
        )

        try:
            parsed = self._parse_response(raw_response)
        except ContextualParseError as exc:
            self.logger.warning("%s: parse error — %s", self.name, exc)
            state.contextual_output = self._fallback_output(
                state, _PARSE_FAIL_NOTE, raw_response
            )
            state.append_history(
                component=self.name,
                summary=f"Parse error — falling back. {exc}",
                outputs={
                    "error": str(exc),
                    "fallback": True,
                    "used_prior_outputs": task.contextual_use_prior_outputs,
                },
            )
            return state

        label: str = parsed["label"]
        confidence: float = parsed["confidence"]
        reasoning: str = parsed["reasoning"]
        evidence: List[str] = parsed["evidence"]

        # Validate returned label against task config.
        if not task.is_allowed_label(label):
            self.logger.warning(
                "%s: LLM returned invalid label '%s'; falling back.", self.name, label
            )
            state.contextual_output = self._fallback_output(
                state, _INVALID_LABEL_NOTE, raw_response
            )
            state.append_history(
                component=self.name,
                summary=f"Invalid label '{label}' returned by LLM — falling back.",
                outputs={
                    "invalid_label": label,
                    "fallback": True,
                    "used_prior_outputs": task.contextual_use_prior_outputs,
                },
            )
            return state

        # Build uniform probabilities with the chosen label given its stated confidence.
        probabilities = self._build_probabilities(label, confidence, task.labels)

        model_output = ModelOutput(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            raw_text=state.input_text,
        )

        state.contextual_output = AgentOutput(
            agent_name=self.name,
            model_output=model_output,
            notes=reasoning,
            features={
                "evidence": evidence,
                "raw_llm_response": raw_response,
            },
        )

        self.logger.debug(
            "%s: label=%s confidence=%.4f", self.name, label, confidence
        )
        state.append_history(
            component=self.name,
            summary=f"Label '{label}' (confidence={confidence:.3f}). {reasoning}",
            outputs={
                "label": label,
                "confidence": confidence,
                "probabilities": dict(probabilities),
                "evidence": evidence,
                "fallback": False,
                "used_prior_outputs": task.contextual_use_prior_outputs,
            },
        )
        return state

    def _build_prior_summaries(self, state: PipelineState) -> List[str]:
        """Build safe, compact upstream summaries for contextual prompting.

        This intentionally excludes raw input text and full LLM responses to
        reduce leakage risk. Only label/confidence/short notes are passed.
        """
        summaries: List[str] = []

        primary = state.primary_model_output
        if primary.label is not None and primary.confidence is not None:
            summaries.append(
                f"Primary model -> label='{primary.label}', confidence={primary.confidence:.3f}"
            )

        if state.lexical_output is not None:
            lex_model = state.lexical_output.model_output
            if lex_model.label is not None and lex_model.confidence is not None:
                lex_note = self._sanitize_summary_text(state.lexical_output.notes)
                summaries.append(
                    "Lexical agent -> "
                    f"label='{lex_model.label}', confidence={lex_model.confidence:.3f}, "
                    f"note='{lex_note}'"
                )

        if state.logic_output is not None:
            logic_model = state.logic_output.model_output
            if logic_model.label is not None and logic_model.confidence is not None:
                logic_note = self._sanitize_summary_text(state.logic_output.notes)
                summaries.append(
                    "Logic agent -> "
                    f"label='{logic_model.label}', confidence={logic_model.confidence:.3f}, "
                    f"note='{logic_note}'"
                )

        return summaries

    @staticmethod
    def _sanitize_summary_text(text: str) -> str:
        """Normalize and truncate free-text notes before including in prompts."""
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if len(cleaned) <= _MAX_SUMMARY_CHARS:
            return cleaned
        return cleaned[:_MAX_SUMMARY_CHARS].rstrip() + "..."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse and validate the LLM JSON response.

        Raises ``ContextualParseError`` on any structural problem.
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
            raise ContextualParseError(f"Invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ContextualParseError("Response is not a JSON object.")

        missing = _REQUIRED_KEYS - data.keys()
        if missing:
            raise ContextualParseError(f"Missing keys in response: {sorted(missing)}")

        if not isinstance(data["label"], str):
            raise ContextualParseError("'label' must be a string.")

        try:
            data["confidence"] = float(data["confidence"])
        except (TypeError, ValueError) as exc:
            raise ContextualParseError(f"'confidence' is not numeric: {exc}") from exc

        if not (0.0 <= data["confidence"] <= 1.0):
            raise ContextualParseError(
                f"'confidence' out of range [0,1]: {data['confidence']}"
            )

        if not isinstance(data["reasoning"], str):
            raise ContextualParseError("'reasoning' must be a string.")

        if not isinstance(data["evidence"], list):
            raise ContextualParseError("'evidence' must be a list.")

        return data

    @staticmethod
    def _build_probabilities(
        chosen_label: str,
        confidence: float,
        labels: List[str],
    ) -> Dict[str, float]:
        """Assign *confidence* to *chosen_label* and split the rest uniformly."""
        remaining = max(0.0, 1.0 - confidence)
        other_labels = [lbl for lbl in labels if lbl != chosen_label]
        per_other = round(remaining / len(other_labels), 6) if other_labels else 0.0
        probs = {lbl: per_other for lbl in labels}
        probs[chosen_label] = round(confidence, 6)
        return probs

    def _fallback_output(
        self,
        state: PipelineState,
        note: str,
        raw_response: str,
    ) -> AgentOutput:
        """Return a uniform-confidence fallback ``AgentOutput``."""
        labels = state.task_config.labels
        uniform = round(1.0 / len(labels), 6)
        return AgentOutput(
            agent_name=self.name,
            model_output=ModelOutput(
                label=labels[0],
                confidence=uniform,
                probabilities={lbl: uniform for lbl in labels},
                raw_text=state.input_text,
            ),
            notes=note,
            features={"raw_llm_response": raw_response},
        )
