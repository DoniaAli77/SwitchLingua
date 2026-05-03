"""Pipeline orchestrator for the stateful multi-agent text classification system.

Execution flow
--------------
1. Primary classifier  — writes ``state.primary_model_output``
2. Router              — writes ``state.routing_info`` (and ``state.final_output``
                         when decision is ``"accept_primary"``)
3a. If decision == ``"accept_primary"`` (fast path):
       ExplainabilityAgent — writes a short accepted-prediction explanation
3b. If decision == ``"escalate"`` (slow path, ``paper_style``):
       LexicalAgent        — writes ``state.lexical_output``
       LogicAgent          — writes ``state.logic_output``
       ContextualAgent     — writes ``state.contextual_output``
       ConsensusAgent      — writes ``state.consensus_output`` + ``state.final_output``
       ExplainabilityAgent — writes a full escalated explanation
3c. If decision == ``"escalate"`` (slow path, ``full_agentic``):
       Same stages as 3b, **plus**:
       DeliberationAgent   — (optional) writes ``state.deliberation_output``
                             when ``task_config.enable_deliberation`` is True
                             and the orchestrator was constructed with a
                             ``deliberation_agent`` instance.  Runs between
                             ContextualAgent and ConsensusAgent.

Mode summary
------------
* ``primary_only``  — primary classifier only; router and all specialist agents skipped.
* ``paper_style``   — primary → router → (if escalated) lexical + logic + contextual
                      → consensus → explainability.  No deliberation.
* ``full_agentic``  — same as paper_style, plus optional DeliberationAgent before consensus.

Error handling
--------------
Any unhandled exception inside a stage is caught, logged, and stored in
``state.extras["pipeline_error"]`` as a dict with ``stage`` and ``message``
keys.  The state at the point of failure is returned immediately.  All
upstream results are preserved.

Usage
-----
Build the orchestrator with all components wired up, then call
``orchestrator.run(state)`` from any entry point.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, Optional

from src.agents.consensus_agent import ConsensusAgent
from src.agents.contextual_agent import ContextualAgent
from src.agents.deliberation_agent import DeliberationAgent
from src.agents.explainability_agent import ExplainabilityAgent
from src.agents.lexical_agent import LexicalAgent
from src.agents.logic_agent import LogicAgent
from src.models.mock_primary_classifier import MockPrimaryClassifier
from src.pipeline.router import Router
from src.state.schema import ExplanationOutput, FinalOutput, PipelineMode, PipelineState

_ESCALATE = "escalate"
_PRIMARY_ONLY: PipelineMode = "primary_only"
_PAPER_STYLE: PipelineMode = "paper_style"
_FULL_AGENTIC: PipelineMode = "full_agentic"
_ALLOWED_MODES = {_PRIMARY_ONLY, _PAPER_STYLE, _FULL_AGENTIC}

log = logging.getLogger(__name__)


def _run_stage(
    stage_name: str,
    fn,
    state: PipelineState,
) -> tuple[PipelineState, bool]:
    """Run a single pipeline stage, returning ``(state, ok)``.

    On exception the error details are attached to ``state.extras`` and
    ``ok=False`` is returned so the orchestrator can stop the chain.
    """
    log.info("Stage: %s — starting", stage_name)
    try:
        state = fn(state)
        log.info("Stage: %s — done", stage_name)
        return state, True
    except Exception as exc:  # noqa: BLE001
        log.error("Stage: %s — FAILED: %s", stage_name, exc)
        state.extras["pipeline_error"] = {
            "stage": stage_name,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        state.append_history(
            component=stage_name,
            summary=f"Stage failed: {exc}",
            outputs={
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        return state, False


class PipelineOrchestrator:
    """Wires all pipeline components together and executes them in order.

    Parameters
    ----------
    primary_classifier::
        Any object with a ``run(state) -> state`` method.  Use
        :class:`~src.models.mock_primary_classifier.MockPrimaryClassifier`
        in tests or demos and swap in a real model for production.
    router:
        A :class:`~src.pipeline.router.Router` instance.
    lexical_agent:
        Required for the escalation path.
    contextual_agent:
        LLM-backed contextual agent used in ``full_agentic`` mode (and in
        ``paper_style`` mode when *paper_contextual_agent* is ``None``).
    logic_agent:
        Required for the escalation path.
    consensus_agent:
        Required for the escalation path.
    explainability_agent:
        Used on both paths (short explanation on fast path, full on escalation).
    deliberation_agent:
        Optional.  When supplied **and** ``state.task_config.enable_deliberation``
        is ``True``, the deliberation stage runs between the contextual stage and
        the consensus stage.  Pass ``None`` (default) to keep deliberation off
        regardless of the config flag.
    paper_contextual_agent:
        Optional non-LLM contextual agent (e.g.
        :class:`~src.agents.transformer_contextual_agent.TransformerContextualAgent`)
        used **only** in ``paper_style`` mode.  When ``None`` (default), the
        orchestrator falls back to *contextual_agent* for backward compatibility.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        primary_classifier: MockPrimaryClassifier,
        router: Router,
        lexical_agent: LexicalAgent,
        contextual_agent: ContextualAgent,
        logic_agent: LogicAgent,
        consensus_agent: ConsensusAgent,
        explainability_agent: ExplainabilityAgent,
        deliberation_agent: Optional[DeliberationAgent] = None,
        paper_contextual_agent: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._primary = primary_classifier
        self._router = router
        self._lexical = lexical_agent
        self._contextual = contextual_agent
        self._logic = logic_agent
        self._consensus = consensus_agent
        self._explain = explainability_agent
        self._deliberation = deliberation_agent
        self._paper_contextual = paper_contextual_agent
        self.logger = logger or log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Execute the full pipeline and return the final state.

        Returns the state after whichever stage last succeeded.  Check
        ``state.extras.get("pipeline_error")`` to detect failures.
        """
        pipeline_mode = self._resolve_pipeline_mode(state)
        self.logger.info(
            "Pipeline started — sample_id=%s task=%s mode=%s",
            state.metadata.sample_id,
            state.task_config.task_name,
            pipeline_mode,
        )
        state.append_history(
            component="orchestrator",
            summary="Pipeline started.",
            outputs={
                "sample_id": state.metadata.sample_id,
                "task": state.task_config.task_name,
                "input_length": len(state.input_text),
                "pipeline_mode": pipeline_mode,
            },
        )

        # Stage 1: Primary classifier
        state, ok = _run_stage("primary_classifier", self._primary.run, state)
        if not ok:
            return state

        if pipeline_mode == _PRIMARY_ONLY:
            primary = state.primary_model_output
            state.final_output = FinalOutput(
                label=primary.label,
                confidence=primary.confidence,
                payload={
                    "source": "primary_model",
                    "probabilities": dict(primary.probabilities),
                    "raw_text": primary.raw_text,
                    "pipeline_mode": pipeline_mode,
                },
            )
            state.explanation_output = ExplanationOutput(
                summary="Primary-only mode used. Router and specialist agents were skipped.",
                evidence=[
                    (
                        "Final output came directly from the primary model "
                        f"(label='{primary.label}', confidence={primary.confidence})."
                    )
                ],
                caveats=[],
            )
            state.append_history(
                component="orchestrator",
                summary="Primary-only mode finalized output from primary classifier.",
                outputs={
                    "pipeline_mode": pipeline_mode,
                    "router_ran": False,
                    "specialist_agents_ran": False,
                },
            )
        else:
            # Stage 2: Router
            state, ok = _run_stage("router", self._router.run, state)
            if not ok:
                return state

            decision = state.routing_info.decision if state.routing_info else _ESCALATE

            if decision != _ESCALATE:
                # Fast path ─────────────────────────────────────────────────
                self.logger.info(
                    "Decision: %s — skipping specialist agents", decision
                )
                state, _ = _run_stage("explainability_agent", self._explain.run, state)
            else:
                # Escalation path ────────────────────────────────────────────
                self.logger.info(
                    "Decision: escalate — running specialist agents (mode=%s)",
                    pipeline_mode,
                )

                # In paper_style mode, use the non-LLM paper_contextual_agent
                # when one is provided; otherwise fall back to contextual_agent
                # for full backward compatibility.
                contextual_for_mode = (
                    self._paper_contextual
                    if (pipeline_mode == _PAPER_STYLE and self._paper_contextual is not None)
                    else self._contextual
                )

                stages = [
                    ("lexical_agent", self._lexical),
                    ("logic_agent", self._logic),
                    ("contextual_agent", contextual_for_mode),
                ]

                for stage_name, agent in stages:
                    state, ok = _run_stage(stage_name, agent.run, state)
                    if not ok:
                        return state

                # Optional deliberation stage (runs before consensus) in full-agentic only.
                if (
                    pipeline_mode == _FULL_AGENTIC
                    and state.task_config.enable_deliberation
                    and self._deliberation is not None
                ):
                    state, ok = _run_stage(
                        "deliberation_agent", self._deliberation.run, state
                    )
                    if not ok:
                        return state

                state, ok = _run_stage("consensus_agent", self._consensus.run, state)
                if not ok:
                    return state

                state, _ = _run_stage("explainability_agent", self._explain.run, state)

        self.logger.info(
            "Pipeline finished — label=%s confidence=%s",
            state.final_output.label if state.final_output else None,
            f"{state.final_output.confidence:.3f}"
            if state.final_output and state.final_output.confidence is not None
            else None,
        )
        state.append_history(
            component="orchestrator",
            summary=(
                f"Pipeline finished. "
                f"label='{state.final_output.label if state.final_output else None}' "
                f"confidence="
                + (
                    f"{state.final_output.confidence:.3f}"
                    if state.final_output and state.final_output.confidence is not None
                    else "n/a"
                )
            ),
            outputs={
                "label": state.final_output.label if state.final_output else None,
                "confidence": state.final_output.confidence if state.final_output else None,
                "routing_decision": (
                    state.routing_info.decision if state.routing_info else None
                ),
                "pipeline_mode": pipeline_mode,
                "history_length": len(state.history),
            },
        )
        return state

    @staticmethod
    def _resolve_pipeline_mode(state: PipelineState) -> PipelineMode:
        mode = getattr(state.task_config, "pipeline_mode", _FULL_AGENTIC)
        if mode not in _ALLOWED_MODES:
            return _FULL_AGENTIC
        return mode
