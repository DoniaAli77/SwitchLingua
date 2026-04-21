"""Unit tests for src.agents.explainability_agent.ExplainabilityAgent."""

from __future__ import annotations

import pytest

from src.agents.explainability_agent import (
    ExplainabilityAgent,
    _ACCEPT_DECISION,
    _build_accepted_explanation,
    _build_escalated_explanation,
    _fmt_conf,
    _agent_vote,
)
from src.state.schema import (
    AgentOutput,
    ConsensusOutput,
    ModelOutput,
    PipelineState,
    RoutingInfo,
    StateMetadata,
    TaskConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABELS = ["positive", "negative", "neutral"]


def make_state(
    primary_label: str | None = None,
    primary_conf: float | None = None,
    routing_decision: str | None = None,
    routing_threshold: float = 0.6,
    lexical_label: str | None = None,
    lexical_conf: float | None = None,
    lexical_notes: str = "",
    contextual_label: str | None = None,
    contextual_conf: float | None = None,
    contextual_notes: str = "",
    logic_label: str | None = None,
    logic_conf: float | None = None,
    logic_notes: str = "",
    consensus_label: str | None = None,
    consensus_conf: float | None = None,
) -> PipelineState:
    state = PipelineState(
        metadata=StateMetadata(sample_id="t-01"),
        input_text="some input text",
        task_config=TaskConfig(task_name="sentiment", labels=list(LABELS)),
        primary_model_output=ModelOutput(
            label=primary_label, confidence=primary_conf
        ),
    )
    if routing_decision is not None:
        state.routing_info = RoutingInfo(
            threshold=routing_threshold, decision=routing_decision
        )
    if lexical_label is not None:
        state.lexical_output = AgentOutput(
            agent_name="LexicalAgent",
            model_output=ModelOutput(label=lexical_label, confidence=lexical_conf),
            notes=lexical_notes,
        )
    if contextual_label is not None:
        state.contextual_output = AgentOutput(
            agent_name="ContextualAgent",
            model_output=ModelOutput(label=contextual_label, confidence=contextual_conf),
            notes=contextual_notes,
        )
    if logic_label is not None:
        state.logic_output = AgentOutput(
            agent_name="LogicAgent",
            model_output=ModelOutput(label=logic_label, confidence=logic_conf),
            notes=logic_notes,
        )
    if consensus_label is not None:
        state.consensus_output = ConsensusOutput(
            label=consensus_label, confidence=consensus_conf
        )
    return state


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

class TestFmtConf:
    def test_none_returns_question_mark(self):
        assert _fmt_conf(None) == "?"

    def test_zero(self):
        assert _fmt_conf(0.0) == "0.0%"

    def test_one(self):
        assert _fmt_conf(1.0) == "100.0%"

    def test_rounding(self):
        assert _fmt_conf(0.876) == "87.6%"


class TestAgentVote:
    def test_none_output(self):
        assert _agent_vote(None) == (None, None, "")

    def test_no_label(self):
        ao = AgentOutput("a", model_output=ModelOutput(label=None, confidence=0.9))
        label, _, notes = _agent_vote(ao)
        assert label is None
        assert notes == ""

    def test_returns_label_conf_notes(self):
        ao = AgentOutput(
            "a",
            model_output=ModelOutput(label="positive", confidence=0.8),
            notes="keyword hit",
        )
        assert _agent_vote(ao) == ("positive", 0.8, "keyword hit")


# ---------------------------------------------------------------------------
# Scenario 1: Accepted primary prediction (no escalation)
# ---------------------------------------------------------------------------

class TestAcceptedPrediction:
    def _run(self, routing_threshold: float = 0.6) -> PipelineState:
        state = make_state(
            primary_label="positive",
            primary_conf=0.92,
            routing_decision=_ACCEPT_DECISION,
            routing_threshold=routing_threshold,
        )
        return ExplainabilityAgent().run(state)

    def test_explanation_output_written(self):
        state = self._run()
        assert state.explanation_output is not None

    def test_summary_mentions_primary_label(self):
        state = self._run()
        assert "positive" in state.explanation_output.summary

    def test_summary_mentions_confidence(self):
        state = self._run()
        assert "92.0%" in state.explanation_output.summary

    def test_summary_says_no_specialists(self):
        state = self._run()
        assert "No specialist agents were consulted" in state.explanation_output.summary

    def test_evidence_has_primary_model_entry(self):
        state = self._run()
        assert any("Primary model" in e for e in state.explanation_output.evidence)

    def test_no_caveats_on_accepted_path(self):
        state = self._run()
        assert state.explanation_output.caveats == []

    def test_threshold_appears_in_summary(self):
        state = self._run(routing_threshold=0.75)
        assert "75.0%" in state.explanation_output.summary

    def test_accepted_with_no_routing_info(self):
        # routing_info=None → escalated path; just confirm it does not crash
        state = make_state(primary_label="neutral", primary_conf=0.7)
        state = ExplainabilityAgent().run(state)
        assert state.explanation_output is not None

    def test_accepted_raw_text_added_to_evidence(self):
        state = PipelineState(
            metadata=StateMetadata(sample_id="x"),
            input_text="hi",
            task_config=TaskConfig(task_name="t", labels=LABELS),
            primary_model_output=ModelOutput(
                label="positive", confidence=0.8, raw_text="raw_out"
            ),
            routing_info=RoutingInfo(threshold=0.6, decision=_ACCEPT_DECISION),
        )
        state = ExplainabilityAgent().run(state)
        assert any("raw_out" in e for e in state.explanation_output.evidence)


# ---------------------------------------------------------------------------
# Scenario 2: Escalated with full agreement
# ---------------------------------------------------------------------------

class TestEscalatedFullAgreement:
    def _run(self) -> PipelineState:
        state = make_state(
            primary_label="negative",
            primary_conf=0.45,
            routing_decision="escalate",
            lexical_label="negative", lexical_conf=0.7,
            contextual_label="negative", contextual_conf=0.85,
            logic_label="negative", logic_conf=0.75,
            consensus_label="negative", consensus_conf=0.77,
        )
        return ExplainabilityAgent().run(state)

    def test_explanation_output_written(self):
        assert self._run().explanation_output is not None

    def test_summary_mentions_consensus_label(self):
        assert "negative" in self._run().explanation_output.summary

    def test_summary_mentions_consensus_confidence(self):
        assert "77.0%" in self._run().explanation_output.summary

    def test_summary_mentions_specialist_agents(self):
        assert "Specialist agents" in self._run().explanation_output.summary

    def test_evidence_contains_all_three_agents(self):
        evidence = self._run().explanation_output.evidence
        names = " ".join(evidence)
        assert "Lexical" in names
        assert "Contextual" in names
        assert "Logic" in names

    def test_no_caveats_on_full_agreement(self):
        assert self._run().explanation_output.caveats == []

    def test_evidence_contains_primary_model(self):
        evidence = self._run().explanation_output.evidence
        assert any("Primary model" in e for e in evidence)


# ---------------------------------------------------------------------------
# Scenario 3: Escalated with disagreement
# ---------------------------------------------------------------------------

class TestEscalatedDisagreement:
    def _run(self) -> PipelineState:
        # lexical + contextual → "positive"; logic → "negative"
        state = make_state(
            primary_label="positive",
            primary_conf=0.55,
            routing_decision="escalate",
            lexical_label="positive", lexical_conf=0.7,
            contextual_label="positive", contextual_conf=0.9,
            logic_label="negative", logic_conf=0.8,
            consensus_label="positive", consensus_conf=0.8,
        )
        return ExplainabilityAgent().run(state)

    def test_explanation_output_written(self):
        assert self._run().explanation_output is not None

    def test_summary_reflects_consensus_winner(self):
        assert "positive" in self._run().explanation_output.summary

    def test_supporting_agents_in_evidence(self):
        evidence = " ".join(self._run().explanation_output.evidence)
        assert "Lexical" in evidence
        assert "Contextual" in evidence

    def test_dissenting_agent_in_caveats(self):
        caveats = self._run().explanation_output.caveats
        assert len(caveats) == 1
        assert "Logic" in caveats[0]
        assert "negative" in caveats[0]

    def test_caveats_include_dissenting_label(self):
        caveat = self._run().explanation_output.caveats[0]
        assert "negative" in caveat

    def test_evidence_does_not_include_dissenting_agent(self):
        evidence = " ".join(self._run().explanation_output.evidence)
        # Logic dissented — must not appear in supporting evidence
        assert "Logic" not in evidence


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_consensus_falls_back_to_primary(self):
        state = make_state(
            primary_label="neutral", primary_conf=0.5,
            routing_decision="escalate",
        )
        state = ExplainabilityAgent().run(state)
        assert "neutral" in state.explanation_output.summary

    def test_no_agents_no_consensus_produces_fallback_evidence(self):
        state = make_state(routing_decision="escalate")
        state = ExplainabilityAgent().run(state)
        assert any("No agent" in e for e in state.explanation_output.evidence)

    def test_agent_notes_appear_in_evidence(self):
        state = make_state(
            routing_decision="escalate",
            contextual_label="positive", contextual_conf=0.8,
            contextual_notes="strong lexical cue",
            consensus_label="positive", consensus_conf=0.8,
        )
        state = ExplainabilityAgent().run(state)
        evidence = " ".join(state.explanation_output.evidence)
        assert "strong lexical cue" in evidence

    def test_agent_notes_appear_in_caveats(self):
        state = make_state(
            routing_decision="escalate",
            lexical_label="positive", lexical_conf=0.9,
            logic_label="negative", logic_conf=0.7, logic_notes="contradiction found",
            consensus_label="positive", consensus_conf=0.9,
        )
        state = ExplainabilityAgent().run(state)
        caveats = " ".join(state.explanation_output.caveats)
        assert "contradiction found" in caveats

    def test_execute_wrapper_also_writes(self):
        state = make_state(
            primary_label="positive", primary_conf=0.95,
            routing_decision=_ACCEPT_DECISION,
        )
        state = ExplainabilityAgent().execute(state)
        assert state.explanation_output is not None

    def test_other_state_fields_untouched(self):
        state = make_state(
            primary_label="positive", primary_conf=0.9,
            routing_decision=_ACCEPT_DECISION,
        )
        original_text = state.input_text
        state = ExplainabilityAgent().run(state)
        assert state.input_text == original_text

    def test_only_lexical_present(self):
        state = make_state(
            routing_decision="escalate",
            lexical_label="neutral", lexical_conf=0.6,
            consensus_label="neutral", consensus_conf=0.6,
        )
        state = ExplainabilityAgent().run(state)
        evidence = " ".join(state.explanation_output.evidence)
        assert "Lexical" in evidence

    def test_only_logic_present_as_dissenter(self):
        state = make_state(
            routing_decision="escalate",
            logic_label="negative", logic_conf=0.7,
            consensus_label="positive", consensus_conf=0.5,
        )
        state = ExplainabilityAgent().run(state)
        caveats = state.explanation_output.caveats
        assert any("Logic" in c for c in caveats)
