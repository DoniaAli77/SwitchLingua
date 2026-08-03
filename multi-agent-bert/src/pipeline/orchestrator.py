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
* ``paper_style``   — primary → router → (if escalated) lexical + logic + paper_contextual
                      → consensus → explainability.  No deliberation.
                      Uses the non-LLM specialist agents to stay close to the reference
                      BERT multi-agent framework.
* ``full_agentic``  — primary → router → (if escalated) llm_lexical + llm_logic + contextual
                      → (optional) deliberation → consensus → explainability.
                      Uses LLM-backed specialist agents when provided; otherwise falls
                      back to the keyword/regex agents for backward compatibility.

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
from src.agents.llm_explainability_agent import LLMExplainabilityAgent
from src.agents.llm_lexical_agent import LLMLexicalAgent
from src.agents.llm_logic_agent import LLMLogicAgent
from src.agents.logic_agent import LogicAgent
from src.agents.ner_consensus_agent import NERConsensusAgent
from src.agents.ner_contextual_agent import NERContextualAgent
from src.agents.ner_lexical_agent import NERLexicalAgent
from src.agents.ner_logic_agent import NERLogicAgent
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
    llm_lexical_agent:
        Optional LLM-backed lexical agent used **only** in ``full_agentic`` mode.
        When ``None`` (default), the orchestrator falls back to *lexical_agent*
        for backward compatibility.
    llm_logic_agent:
        Optional LLM-backed logic agent used **only** in ``full_agentic`` mode.
        When ``None`` (default), the orchestrator falls back to *logic_agent*
        for backward compatibility.
    llm_explainability_agent:
        Optional LLM-backed explainability agent used **only** in ``full_agentic``
        mode.  When ``None`` (default), the orchestrator falls back to
        *explainability_agent* for backward compatibility.
    ner_lexical_agent:
        NER-specific lexical (gazetteer) agent.  Used on the NER path in all
        modes except ``primary_only``.  When ``None`` a default
        :class:`~src.agents.ner_lexical_agent.NERLexicalAgent` (empty gazetteer)
        is created automatically.
    ner_logic_agent:
        NER-specific logic (regex-rule) agent.  Same defaulting behaviour as
        *ner_lexical_agent*.
    ner_contextual_agent:
        NER-specific contextual (heuristic) agent.  Same defaulting behaviour.
    ner_consensus_agent:
        NER consensus agent.  Same defaulting behaviour.
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
        llm_lexical_agent: Optional[LLMLexicalAgent] = None,
        llm_logic_agent: Optional[LLMLogicAgent] = None,
        llm_explainability_agent: Optional[LLMExplainabilityAgent] = None,
        polarity_agent: Optional[Any] = None,
        intent_gate_agent: Optional[Any] = None,
        sequential_stages: Optional[list] = None,
        sequential_controller: Optional[Any] = None,
        ner_lexical_agent: Optional[NERLexicalAgent] = None,
        ner_logic_agent: Optional[NERLogicAgent] = None,
        ner_contextual_agent: Optional[NERContextualAgent] = None,
        ner_consensus_agent: Optional[NERConsensusAgent] = None,
        ner_primary: Optional[Any] = None,
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
        self._llm_lexical = llm_lexical_agent
        self._llm_logic = llm_logic_agent
        self._llm_explain = llm_explainability_agent
        # Optional 4th specialist (four-agent sentiment variant D); writes
        # state.polarity_output. None for every other configuration.
        self._polarity = polarity_agent
        # Optional non-voting IntentGate guard (Design G/G2), run as a SEPARATE
        # stage right after consensus. None for every other configuration → the
        # consensus decision stands unchanged (the guard is fully decoupled).
        self._intent_gate = intent_gate_agent
        # Optional sequential-agentic escalation (sequential_sentiment_v1):
        # an ordered list of (stage_name, agent) run in series, followed by a
        # deterministic controller that writes final_output. Both None for every
        # other configuration → the parallel trio + consensus path is unchanged.
        self._sequential_stages = sequential_stages
        self._sequential_controller = sequential_controller
        # NER-path agents (lazy defaults created here so callers don't need to
        # supply them when not running sequence-labeling tasks).
        self._ner_lexical: NERLexicalAgent = ner_lexical_agent or NERLexicalAgent()
        self._ner_logic: NERLogicAgent = ner_logic_agent or NERLogicAgent()
        self._ner_contextual: NERContextualAgent = ner_contextual_agent or NERContextualAgent()
        self._ner_consensus: NERConsensusAgent = ner_consensus_agent or NERConsensusAgent()
        # Optional real NER primary model (e.g. TransformerNERTagger). Defaults to
        # None → NOT auto-created, so the NER path carries no torch dependency
        # unless a caller explicitly wires a model in. When present it runs in the
        # primary position of the NER path (before the heuristic specialists),
        # mirroring the classification path's primary → specialists → consensus.
        self._ner_primary = ner_primary
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

        # Dispatch to the appropriate path based on task_type.
        if state.task_config.task_type == "sequence_labeling":
            return self._run_ner_path(state, pipeline_mode)

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
                # Fast path and paper_style always use template explainability.
                state, _ = _run_stage("explainability_agent", self._explain.run, state)
            else:
                # Escalation path ────────────────────────────────────────────
                self.logger.info(
                    "Decision: escalate — running specialist agents (mode=%s)",
                    pipeline_mode,
                )

                # Sequential-agentic escalation (sequential_sentiment_v1): opt-in.
                # When a controller is injected and we are in full_agentic mode, run
                # the staged reasoning pipeline (intent → polarity → pragmatic →
                # deterministic controller) instead of the parallel trio + consensus.
                # Default (controller None) leaves the parallel path untouched.
                if (
                    pipeline_mode == _FULL_AGENTIC
                    and self._sequential_controller is not None
                ):
                    for stage_name, agent in (self._sequential_stages or []):
                        state, ok = _run_stage(stage_name, agent.run, state)
                        if not ok:
                            return state
                    state, ok = _run_stage(
                        "sequential_controller", self._sequential_controller.run, state
                    )
                    if not ok:
                        return state
                    explain_for_mode = (
                        self._llm_explain
                        if self._llm_explain is not None
                        else self._explain
                    )
                    state, _ = _run_stage(
                        "explainability_agent", explain_for_mode.run, state
                    )
                    return self._finalize(state, pipeline_mode)

                # Resolve per-mode specialist agents.
                #
                # paper_style: use the non-LLM baseline agents to stay close to
                #   the reference BERT multi-agent framework.
                # full_agentic: use LLM-backed specialist agents when provided;
                #   fall back to keyword/regex agents for backward compatibility.
                if pipeline_mode == _PAPER_STYLE:
                    lexical_for_mode = self._lexical
                    logic_for_mode = self._logic
                    contextual_for_mode = (
                        self._paper_contextual
                        if self._paper_contextual is not None
                        else self._contextual
                    )
                else:  # full_agentic
                    lexical_for_mode = (
                        self._llm_lexical
                        if self._llm_lexical is not None
                        else self._lexical
                    )
                    logic_for_mode = (
                        self._llm_logic
                        if self._llm_logic is not None
                        else self._logic
                    )
                    contextual_for_mode = self._contextual

                stages = [
                    ("lexical_agent", lexical_for_mode),
                    ("logic_agent", logic_for_mode),
                    ("contextual_agent", contextual_for_mode),
                ]
                # Optional custom execution order for the three specialist stages
                # (sequential-chain ablation). Reorders WHICH agent runs when;
                # each agent still writes its own slot, so consensus is unchanged.
                # Unknown/omitted names keep their default relative position.
                custom_order = getattr(state.task_config, "agent_stage_order", None)
                if custom_order:
                    rank = {name: i for i, name in enumerate(custom_order)}
                    stages.sort(key=lambda s: (rank.get(s[0], len(rank) + 1)))
                # Four-agent sentiment variant D: run an extra Polarity stage
                # (writes state.polarity_output) alongside the standard trio.
                # Injected only for that variant; None otherwise → trio unchanged.
                # Always appended AFTER the (possibly reordered) trio so the gate
                # LLM stays last, independent of the voter order.
                if pipeline_mode == _FULL_AGENTIC and self._polarity is not None:
                    stages.append(("polarity_agent", self._polarity))

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

                # Optional non-voting IntentGate guard (Design G/G2), decoupled
                # from consensus and run as its own post-consensus stage. No-op
                # unless an intent_gate_agent was supplied.
                if self._intent_gate is not None:
                    state, ok = _run_stage("intent_gate", self._intent_gate.run, state)
                    if not ok:
                        return state

                # In full_agentic mode, use the LLM explainability agent when
                # provided; otherwise fall back to the template agent for both
                # paper_style and full_agentic backward compatibility.
                explain_for_mode = (
                    self._llm_explain
                    if (
                        pipeline_mode == _FULL_AGENTIC
                        and self._llm_explain is not None
                    )
                    else self._explain
                )
                state, _ = _run_stage("explainability_agent", explain_for_mode.run, state)

        return self._finalize(state, pipeline_mode)

    def _finalize(self, state: PipelineState, pipeline_mode: PipelineMode) -> PipelineState:
        """Shared pipeline-completion logging + history; returns the state."""
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

    # ------------------------------------------------------------------
    # NER path
    # ------------------------------------------------------------------

    def _run_ner_path(
        self,
        state: PipelineState,
        pipeline_mode: PipelineMode,
    ) -> PipelineState:
        """Execute the sequence-labeling (NER) pipeline and return final state.

        Modes
        -----
        primary_only
            Runs only the primary classifier if it can produce a
            ``sequence_output`` (i.e. it is NER-aware).  In the common case
            (classification-only primary model), the primary stage is skipped
            and an empty FinalOutput is written so downstream consumers always
            have a well-formed state.

        paper_style / full_agentic
            NERLexicalAgent → NERLogicAgent → NERContextualAgent →
            NERConsensusAgent → ExplainabilityAgent.

            Both modes run the same deterministic NER agents.  The distinction
            is preserved so future LLM-backed NER agents can be slotted into
            ``full_agentic`` without breaking this interface.
        """
        self.logger.info(
            "NER path — mode=%s task=%s",
            pipeline_mode,
            state.task_config.task_name,
        )

        if pipeline_mode == _PRIMARY_ONLY:
            # When a real NER primary model is wired in, run it alone and finalize
            # its per-token output directly — the NER analogue of classification
            # primary_only (primary model, no specialist agents).
            if self._ner_primary is not None:
                state, ok = _run_stage("ner_primary_model", self._ner_primary.run, state)
                if not ok:
                    return state
                self._finalize_ner_primary_only(state, pipeline_mode)
                return state

            # No NER primary available: the classification primary is
            # classification-only, so write a minimal stub final_output so callers
            # never get None.
            state, ok = _run_stage("primary_classifier", self._primary.run, state)
            if not ok:
                return state
            if state.final_output is None:
                state.final_output = FinalOutput(
                    payload={
                        "source": "primary_model_ner_stub",
                        "pipeline_mode": pipeline_mode,
                        "note": (
                            "primary_only mode on NER task: "
                            "no sequence output from primary model."
                        ),
                    }
                )
            state.append_history(
                component="orchestrator",
                summary="NER primary-only mode: primary stage ran; specialist agents skipped.",
                outputs={"pipeline_mode": pipeline_mode, "specialist_agents_ran": False},
            )
            return state

        # paper_style and full_agentic mirror the classification path:
        #   NER primary model → router → (accept primary | escalate → agents → consensus)
        # The router only engages when a real NER primary is wired in; with only
        # heuristic agents (no primary) the path escalates unconditionally, exactly
        # as before, so existing heuristic-only behaviour is unchanged.
        if self._ner_primary is not None:
            state, ok = _run_stage("ner_primary_model", self._ner_primary.run, state)
            if not ok:
                return state

            decision = self._ner_route(state)
            if decision != _ESCALATE:
                # Fast path: primary is confident on every token — accept it and
                # skip the specialist agents (NER analogue of accept_primary).
                self.logger.info("NER decision: %s — skipping specialist agents", decision)
                self._finalize_ner_primary_only(state, pipeline_mode)
                state.final_output.payload["source"] = "ner_primary_accepted"
            else:
                self.logger.info("NER decision: escalate — running specialist agents")
                state, ok = self._run_ner_specialists(state)
                if not ok:
                    return state
        else:
            # No primary wired: heuristic specialists + consensus, as before.
            state, ok = self._run_ner_specialists(state)
            if not ok:
                return state

        # Explainability — use LLM explainability in full_agentic when present.
        explain_for_mode = (
            self._llm_explain
            if (pipeline_mode == _FULL_AGENTIC and self._llm_explain is not None)
            else self._explain
        )
        state, _ = _run_stage("explainability_agent", explain_for_mode.run, state)

        self.logger.info(
            "NER pipeline finished — token_count=%s",
            state.final_output.payload.get("token_count") if state.final_output else None,
        )
        state.append_history(
            component="orchestrator",
            summary="NER pipeline finished.",
            outputs={
                "pipeline_mode": pipeline_mode,
                "token_count": (
                    state.final_output.payload.get("token_count")
                    if state.final_output else None
                ),
                "history_length": len(state.history),
            },
        )
        return state

    def _run_ner_specialists(self, state: PipelineState):
        """Run the four heuristic NER specialist agents in order. Returns (state, ok)."""
        ner_stages = [
            ("ner_lexical_agent",    self._ner_lexical),
            ("ner_logic_agent",      self._ner_logic),
            ("ner_contextual_agent", self._ner_contextual),
            ("ner_consensus_agent",  self._ner_consensus),
        ]
        for stage_name, agent in ner_stages:
            state, ok = _run_stage(stage_name, agent.run, state)
            if not ok:
                return state, False
        return state, True

    def _ner_route(self, state: PipelineState) -> str:
        """Decide accept-primary vs escalate for NER using a min-confidence gate.

        The NER analogue of :class:`~src.pipeline.router.Router`: because NER
        confidence is per-token, the sentence is accepted only when the *least*
        confident token clears ``task_config.threshold`` — i.e. the primary is
        confident about **every** token.  Any uncertain token escalates the whole
        sentence to the specialist agents.  Records the decision on
        ``state.routing_info`` (same shape the classification router uses).
        """
        from src.state.schema import RoutingInfo

        threshold = state.task_config.threshold
        seq = state.ner_model_output.sequence_output if state.ner_model_output else None
        confidences = [tt.confidence for tt in seq.tags] if (seq and seq.tags) else []
        min_conf = min(confidences) if confidences else 0.0

        decision = "accept_primary" if (confidences and min_conf >= threshold) else _ESCALATE
        state.routing_info = RoutingInfo(threshold=threshold, decision=decision)
        state.append_history(
            component="ner_router",
            summary=(
                f"NER decision: {decision} "
                f"(min token confidence={min_conf:.3f} vs threshold={threshold:.3f})"
            ),
            outputs={
                "decision": decision,
                "threshold": threshold,
                "min_token_confidence": round(min_conf, 6),
                "token_count": len(confidences),
            },
        )
        return decision

    @staticmethod
    def _finalize_ner_primary_only(state: PipelineState, pipeline_mode: PipelineMode) -> None:
        """Serialise the NER primary's per-token output into ``final_output``.

        Mirrors the payload shape written by
        :class:`~src.agents.ner_consensus_agent.NERConsensusAgent` so downstream
        consumers (e.g. :class:`~src.evaluation.ner_evaluator.NEREvaluator`) read
        primary-only and full-pipeline NER runs identically.
        """
        seq = state.ner_model_output.sequence_output if state.ner_model_output else None
        tags = list(seq.tags) if seq else []
        seq_payload = [
            {"token": tt.token, "tag": tt.tag, "confidence": tt.confidence}
            for tt in tags
        ]
        if state.final_output is None:
            state.final_output = FinalOutput()
        state.final_output.payload["sequence_output"] = seq_payload
        state.final_output.payload["token_count"] = len(tags)
        state.final_output.payload["source"] = "ner_primary_model"
        state.final_output.payload["pipeline_mode"] = pipeline_mode
        state.append_history(
            component="orchestrator",
            summary=f"NER primary-only mode: primary model tagged {len(tags)} token(s).",
            outputs={"pipeline_mode": pipeline_mode, "token_count": len(tags),
                     "specialist_agents_ran": False},
        )

    @staticmethod
    def _resolve_pipeline_mode(state: PipelineState) -> PipelineMode:
        mode = getattr(state.task_config, "pipeline_mode", _FULL_AGENTIC)
        if mode not in _ALLOWED_MODES:
            return _FULL_AGENTIC
        return mode
