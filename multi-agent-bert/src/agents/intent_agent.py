"""LLM-based authorial INTENT / stance specialist agent (experimental variant).

Part of the ``lexical_intent_polarity_contextual`` sentiment agent variant (Design
E). The Intent agent is NOT a sentiment classifier: it decides whether the AUTHOR
is actually expressing their own evaluative opinion (and toward what target), and
maps that judgement onto the allowed labels so it fits the shared consensus schema.

In Design E it runs as the 4th specialist, writing ``state.polarity_output`` (the
generic 4th-specialist slot the orchestrator/consensus already support), so no
consensus/architecture change is needed beyond the existing four-slot wiring.

The agent:

1. Builds an intent/stance-decision prompt.
2. Calls ``LLMClient.generate(prompt)``.
3. Parses the JSON response against the same strict schema as the other agents.
4. Validates the returned label against ``task_config.labels``.
5. Writes an ``AgentOutput`` to its configured slot (default ``logic_output``;
   Design E injects it at ``polarity_output``).
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
from src.prompts.intent_prompt import build_user_prompt, get_system_prompt
from src.state.schema import AgentOutput, ModelOutput, PipelineState

_REQUIRED_KEYS: frozenset[str] = frozenset({"label", "confidence", "reasoning", "evidence"})

_PARSE_FAIL_NOTE = "IntentAgent: response parse failed; abstain fallback applied."
_INVALID_LABEL_NOTE = "IntentAgent: LLM returned an invalid label; abstain fallback applied."


class IntentParseError(ValueError):
    """Raised when the LLM response cannot be parsed into the expected schema."""


def _extract_json(raw: str) -> str:
    """Strip markdown fences and surrounding whitespace from a raw LLM response."""
    stripped = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        return fence.group(1)
    return stripped


class IntentAgent(BaseAgent[PipelineState]):
    """LLM-backed authorial-intent decider for the full_agentic pipeline mode.

    Parameters
    ----------
    llm_client:
        Any ``LLMClient`` implementation.  Pass ``MockLLMClient`` in tests.
    name:
        Optional agent name for logging and ``AgentOutput.agent_name``.
    logger:
        Optional pre-configured logger.
    output_attr:
        State slot to write to. Defaults to ``logic_output``; Design E uses
        ``polarity_output`` so Intent coexists with the other three agents.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        output_attr: str = "logic_output",
        system_variant: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "IntentAgent", logger=logger)
        self.llm_client = llm_client
        self._output_attr = output_attr
        # Which intent system prompt to use. None → default intent prompt;
        # "selective" → the Design-G2 selective-gate prompt.
        self._system_variant = system_variant

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def validate_before(self, state: PipelineState) -> None:
        if not state.input_text or not state.input_text.strip():
            raise ValueError("IntentAgent: state.input_text is empty or blank.")
        if not state.task_config.labels:
            raise ValueError("IntentAgent: state.task_config.labels is empty.")

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Call LLM, parse response, write result to the configured slot."""
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

        raw_response = self.llm_client.generate(f"{get_system_prompt(self._system_variant)}\n\n{prompt}")

        try:
            parsed = self._parse_response(raw_response)
        except IntentParseError as exc:
            self.logger.warning("%s: parse error — %s", self.name, exc)
            setattr(state, self._output_attr, self._fallback_output(state, _PARSE_FAIL_NOTE, raw_response))
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
            setattr(
                state, self._output_attr,
                self._fallback_output(state, _INVALID_LABEL_NOTE, raw_response),
            )
            state.append_history(
                component=self.name,
                summary=f"Invalid label '{label}' returned by LLM — falling back.",
                outputs={"invalid_label": label, "fallback": True},
            )
            return state

        probabilities = self._build_probabilities(label, confidence, task.labels)

        setattr(state, self._output_attr, AgentOutput(
            agent_name=self.name,
            model_output=ModelOutput(
                label=label,
                confidence=confidence,
                probabilities=probabilities,
                raw_text=state.input_text,
            ),
            notes=reasoning,
            features={"evidence": evidence, "raw_llm_response": raw_response},
        ))

        self.logger.debug("%s: label=%s confidence=%.4f", self.name, label, confidence)
        state.append_history(
            component=self.name,
            summary=f"Intent label '{label}' (confidence={confidence:.3f}). {reasoning}",
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
        """Parse and validate the LLM JSON response."""
        try:
            data = json.loads(_extract_json(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            raise IntentParseError(f"JSON decode failed: {exc}") from exc

        if not isinstance(data, dict):
            raise IntentParseError(f"Expected a JSON object, got {type(data).__name__}")

        missing = _REQUIRED_KEYS - data.keys()
        if missing:
            raise IntentParseError(f"Missing required keys: {sorted(missing)}")

        if not isinstance(data["confidence"], (int, float)):
            raise IntentParseError(
                f"'confidence' must be a number, got {type(data['confidence']).__name__}"
            )
        if not isinstance(data["evidence"], list):
            raise IntentParseError(
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
