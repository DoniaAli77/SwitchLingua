"""LLM-based explainability agent for full_agentic pipeline mode.

Generates a concise, human-readable explanation of the pipeline's final
classification decision by asking an LLM to synthesize all available agent
outputs.  Unlike the template-based
:class:`~src.agents.explainability_agent.ExplainabilityAgent`, this agent
can produce nuanced, narrative-style explanations that reflect the full
multi-agent reasoning chain.

The agent:

1. Collects summaries from all available pipeline outputs
   (primary, lexical, logic, contextual, deliberation, consensus).
2. Builds a structured prompt using
   :mod:`src.prompts.llm_explainability_prompt`.
3. Calls ``LLMClient.generate(prompt)``.
4. Parses the JSON response (keys: ``summary``, ``evidence``, ``caveats``).
5. Writes an :class:`~src.state.schema.ExplanationOutput` to
   ``state.explanation_output``.
6. Falls back to a short deterministic explanation on parse failure.

Error taxonomy
--------------
``LLMExplainabilityParseError``
    The LLM returned something that is not valid JSON or is missing a
    required key.  A deterministic fallback explanation is written instead.

``LLMClientError``
    A non-recoverable backend error (network, auth, quota).  Re-raised so
    the orchestrator can decide how to handle it.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.llm.base_client import LLMClient, LLMClientError  # noqa: F401
from src.prompts.llm_explainability_prompt import SYSTEM_PROMPT, build_user_prompt
from src.state.schema import (
    AgentOutput,
    ConsensusOutput,
    ExplanationOutput,
    ModelOutput,
    PipelineState,
)

_REQUIRED_KEYS: frozenset[str] = frozenset({"summary", "evidence", "caveats"})

_PARSE_FAIL_NOTE = (
    "LLMExplainabilityAgent: response parse failed; deterministic fallback applied."
)


class LLMExplainabilityParseError(ValueError):
    """Raised when the LLM response cannot be parsed into the expected schema."""


def _extract_json(raw: str) -> str:
    """Strip markdown fences and surrounding whitespace."""
    stripped = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    return fence.group(1) if fence else stripped


def _fmt_conf(confidence: Optional[float]) -> str:
    if confidence is None:
        return "?"
    return f"{confidence * 100:.1f}%"


def _collect_agent_summaries(state: PipelineState) -> List[str]:
    """Build one-line summaries for every agent output present in state."""
    summaries: List[str] = []

    primary: ModelOutput = state.primary_model_output
    if primary.label is not None:
        summaries.append(
            f"Primary model: label='{primary.label}', "
            f"confidence={_fmt_conf(primary.confidence)}"
        )

    for slot_name, output in (
        ("Lexical", state.lexical_output),
        ("Logic", state.logic_output),
        ("Contextual", state.contextual_output),
    ):
        output: Optional[AgentOutput]
        if output is None:
            continue
        mo = output.model_output
        if mo.label is None:
            continue
        note = f" — {output.notes}" if output.notes else ""
        summaries.append(
            f"{slot_name}: label='{mo.label}', "
            f"confidence={_fmt_conf(mo.confidence)}{note}"
        )

    if state.deliberation_output is not None:
        delb = state.deliberation_output
        if delb.recommended_label:
            summaries.append(
                f"Deliberation: recommended='{delb.recommended_label}', "
                f"confidence={_fmt_conf(delb.confidence)} — {delb.justification}"
            )

    consensus: Optional[ConsensusOutput] = state.consensus_output
    if consensus is not None and consensus.label is not None:
        summaries.append(
            f"Consensus: label='{consensus.label}', "
            f"confidence={_fmt_conf(consensus.confidence)} — {consensus.rationale}"
        )

    return summaries


def _deterministic_fallback(state: PipelineState, note: str) -> ExplanationOutput:
    """Build a short, deterministic explanation without the LLM."""
    final = state.final_output
    if final is not None and final.label is not None:
        summary = (
            f"The pipeline classified the text as '{final.label}' "
            f"(confidence: {_fmt_conf(final.confidence)}). "
            "Detailed LLM explanation unavailable."
        )
        evidence = [f"Final output: label='{final.label}', confidence={_fmt_conf(final.confidence)}"]
    else:
        summary = "Pipeline explanation unavailable (no final output recorded)."
        evidence = []

    return ExplanationOutput(summary=summary, evidence=evidence, caveats=[note])


class LLMExplainabilityAgent(BaseAgent[PipelineState]):
    """LLM-backed explanation generator for the full_agentic pipeline mode.

    Parameters
    ----------
    llm_client:
        Any ``LLMClient`` implementation.  Pass ``MockLLMClient`` in tests.
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
        super().__init__(name=name or "LLMExplainabilityAgent", logger=logger)
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Generate an LLM explanation and write it to state.explanation_output."""
        task = state.task_config

        final = state.final_output
        final_label = final.label if (final and final.label) else "unknown"
        final_conf = _fmt_conf(final.confidence if final else None)

        agent_summaries = _collect_agent_summaries(state)

        prompt = build_user_prompt(
            task_name=task.task_name,
            final_label=final_label,
            final_confidence=final_conf,
            agent_summaries=agent_summaries,
            text=state.input_text,
        )

        self.logger.debug("%s: sending prompt (%d chars)", self.name, len(prompt))

        raw_response = self.llm_client.generate(f"{SYSTEM_PROMPT}\n\n{prompt}")

        try:
            parsed = self._parse_response(raw_response)
        except LLMExplainabilityParseError as exc:
            self.logger.warning("%s: parse error — %s", self.name, exc)
            state.explanation_output = _deterministic_fallback(state, _PARSE_FAIL_NOTE)
            state.append_history(
                component=self.name,
                summary=f"Parse error — deterministic fallback. {exc}",
                outputs={"error": str(exc), "fallback": True},
            )
            return state

        explanation = ExplanationOutput(
            summary=parsed["summary"],
            evidence=parsed["evidence"],
            caveats=parsed["caveats"],
        )

        state.explanation_output = explanation
        self.logger.debug("%s: explanation written (%d evidence items)", self.name, len(explanation.evidence))
        state.append_history(
            component=self.name,
            summary=explanation.summary,
            outputs={
                "evidence": list(explanation.evidence),
                "caveats": list(explanation.caveats),
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
        LLMExplainabilityParseError
            When the response is not valid JSON or is missing required keys.
        """
        try:
            data = json.loads(_extract_json(raw))
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMExplainabilityParseError(f"JSON decode failed: {exc}") from exc

        if not isinstance(data, dict):
            raise LLMExplainabilityParseError(
                f"Expected a JSON object, got {type(data).__name__}"
            )

        missing = _REQUIRED_KEYS - data.keys()
        if missing:
            raise LLMExplainabilityParseError(f"Missing required keys: {sorted(missing)}")

        if not isinstance(data["evidence"], list):
            raise LLMExplainabilityParseError(
                f"'evidence' must be a list, got {type(data['evidence']).__name__}"
            )
        if not isinstance(data["caveats"], list):
            raise LLMExplainabilityParseError(
                f"'caveats' must be a list, got {type(data['caveats']).__name__}"
            )
        return data
