"""Unit tests for src.agents.consensus_agent.ConsensusAgent."""

from __future__ import annotations

import pytest

from src.agents.consensus_agent import ConsensusAgent, _NO_VOTE_NOTE, _extract_vote
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    StateMetadata,
    TaskConfig,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

LABELS = ["positive", "negative", "neutral"]


def make_state(
    labels: list[str] | None = None,
    lexical_label: str | None = None,
    lexical_conf: float | None = None,
    contextual_label: str | None = None,
    contextual_conf: float | None = None,
    logic_label: str | None = None,
    logic_conf: float | None = None,
) -> PipelineState:
    """Construct a PipelineState with optional pre-filled agent outputs."""
    if labels is None:
        labels = list(LABELS)
    state = PipelineState(
        metadata=StateMetadata(sample_id="t-01"),
        input_text="some text",
        task_config=TaskConfig(task_name="sentiment", labels=labels),
    )
    if lexical_label is not None:
        state.lexical_output = AgentOutput(
            agent_name="LexicalAgent",
            model_output=ModelOutput(label=lexical_label, confidence=lexical_conf),
        )
    if contextual_label is not None:
        state.contextual_output = AgentOutput(
            agent_name="ContextualAgent",
            model_output=ModelOutput(label=contextual_label, confidence=contextual_conf),
        )
    if logic_label is not None:
        state.logic_output = AgentOutput(
            agent_name="LogicAgent",
            model_output=ModelOutput(label=logic_label, confidence=logic_conf),
        )
    return state


# ---------------------------------------------------------------------------
# Helper function: _extract_vote
# ---------------------------------------------------------------------------

class TestExtractVote:
    def test_returns_none_for_missing_output(self):
        assert _extract_vote(None) == (None, None)

    def test_returns_none_when_label_is_none(self):
        ao = AgentOutput("a", model_output=ModelOutput(label=None, confidence=0.9))
        assert _extract_vote(ao) == (None, None)

    def test_returns_none_when_confidence_is_none(self):
        ao = AgentOutput("a", model_output=ModelOutput(label="positive", confidence=None))
        assert _extract_vote(ao) == (None, None)

    def test_returns_none_when_confidence_out_of_range(self):
        ao = AgentOutput("a", model_output=ModelOutput(label="positive", confidence=1.5))
        assert _extract_vote(ao) == (None, None)

    def test_returns_label_and_confidence(self):
        ao = AgentOutput("a", model_output=ModelOutput(label="negative", confidence=0.75))
        assert _extract_vote(ao) == ("negative", 0.75)

    def test_confidence_boundary_zero(self):
        ao = AgentOutput("a", model_output=ModelOutput(label="neutral", confidence=0.0))
        assert _extract_vote(ao) == ("neutral", 0.0)

    def test_confidence_boundary_one(self):
        ao = AgentOutput("a", model_output=ModelOutput(label="neutral", confidence=1.0))
        assert _extract_vote(ao) == ("neutral", 1.0)


# ---------------------------------------------------------------------------
# ConsensusAgent construction
# ---------------------------------------------------------------------------

class TestConsensusAgentInit:
    def test_default_weights_are_equal(self):
        agent = ConsensusAgent()
        assert agent.weights == {
            "lexical": 1.0,
            "contextual": 1.0,
            "logic": 1.0,
            "polarity": 0.0,
            "deliberation": 0.0,
            "primary": 1.0,
        }

    def test_custom_weights_merged(self):
        agent = ConsensusAgent(weights={"contextual": 2.0})
        assert agent.weights["contextual"] == 2.0
        assert agent.weights["lexical"] == 1.0   # default kept
        assert agent.weights["logic"] == 1.0

    def test_negative_weight_clamped_to_zero(self):
        agent = ConsensusAgent(weights={"lexical": -0.5})
        assert agent.weights["lexical"] == 0.0

    def test_default_name(self):
        assert ConsensusAgent().name == "ConsensusAgent"

    def test_custom_name(self):
        assert ConsensusAgent(name="Merger").name == "Merger"


# ---------------------------------------------------------------------------
# validate_before
# ---------------------------------------------------------------------------

class TestValidateBefore:
    def test_raises_when_labels_empty(self):
        state = PipelineState(
            metadata=StateMetadata(sample_id="x"),
            input_text="hi",
            task_config=TaskConfig(task_name="t", labels=[]),
        )
        with pytest.raises(ValueError, match="labels is empty"):
            ConsensusAgent().validate_before(state)

    def test_passes_with_labels(self):
        state = make_state()
        ConsensusAgent().validate_before(state)  # should not raise


# ---------------------------------------------------------------------------
# Full agreement scenarios
# ---------------------------------------------------------------------------

class TestFullAgreement:
    def test_all_agents_agree_positive(self):
        state = make_state(
            lexical_label="positive", lexical_conf=0.8,
            contextual_label="positive", contextual_conf=0.9,
            logic_label="positive", logic_conf=0.7,
        )
        state = ConsensusAgent().run(state)
        assert state.consensus_output.label == "positive"
        assert state.final_output.label == "positive"

    def test_confidence_is_normalised(self):
        # All three agree on "positive" with conf=1.0 and equal weights.
        # score["positive"] = 1*1 + 1*1 + 1*1 = 3; active_weight_sum = 3.
        # normalised = 3/3 = 1.0
        state = make_state(
            lexical_label="positive", lexical_conf=1.0,
            contextual_label="positive", contextual_conf=1.0,
            logic_label="positive", logic_conf=1.0,
        )
        state = ConsensusAgent().run(state)
        assert state.consensus_output.confidence == pytest.approx(1.0)
        assert state.final_output.confidence == pytest.approx(1.0)

    def test_votes_dict_has_all_labels(self):
        state = make_state(
            lexical_label="negative", lexical_conf=0.6,
            contextual_label="negative", contextual_conf=0.8,
            logic_label="negative", logic_conf=0.7,
        )
        state = ConsensusAgent().run(state)
        assert set(state.consensus_output.votes.keys()) == set(LABELS)

    def test_votes_dict_winning_label_nonzero(self):
        state = make_state(
            lexical_label="neutral", lexical_conf=0.5,
            contextual_label="neutral", contextual_conf=0.5,
            logic_label="neutral", logic_conf=0.5,
        )
        state = ConsensusAgent().run(state)
        assert state.consensus_output.votes["neutral"] > 0.0

    def test_rationale_contains_all_slots(self):
        state = make_state(
            lexical_label="positive", lexical_conf=0.7,
            contextual_label="positive", contextual_conf=0.8,
            logic_label="positive", logic_conf=0.6,
        )
        state = ConsensusAgent().run(state)
        rationale = state.consensus_output.rationale
        assert "lexical" in rationale
        assert "contextual" in rationale
        assert "logic" in rationale


# ---------------------------------------------------------------------------
# Disagreement scenarios
# ---------------------------------------------------------------------------

class TestDisagreement:
    def test_majority_vote_two_vs_one(self):
        # lexical + contextual → "positive"; logic → "negative"
        state = make_state(
            lexical_label="positive", lexical_conf=0.8,
            contextual_label="positive", contextual_conf=0.9,
            logic_label="negative", logic_conf=0.95,
        )
        state = ConsensusAgent().run(state)
        # positive score = 1*0.8 + 1*0.9 = 1.7 > negative = 1*0.95 = 0.95
        assert state.consensus_output.label == "positive"

    def test_high_weight_agent_wins_minority(self):
        # contextual weight=5 → its single vote for "negative" should dominate
        state = make_state(
            lexical_label="positive", lexical_conf=0.9,
            contextual_label="negative", contextual_conf=0.8,
            logic_label="positive", logic_conf=0.85,
        )
        agent = ConsensusAgent(weights={"lexical": 1.0, "contextual": 5.0, "logic": 1.0})
        state = agent.run(state)
        # negative = 5*0.8 = 4.0; positive = 1*0.9 + 1*0.85 = 1.75
        assert state.consensus_output.label == "negative"

    def test_silenced_agent_not_counted(self):
        # lexical silenced → only contextual + logic count
        state = make_state(
            lexical_label="negative", lexical_conf=1.0,   # silenced
            contextual_label="positive", contextual_conf=0.9,
            logic_label="positive", logic_conf=0.8,
        )
        agent = ConsensusAgent(weights={"lexical": 0.0, "contextual": 1.0, "logic": 1.0})
        state = agent.run(state)
        assert state.consensus_output.label == "positive"

    def test_tie_is_not_broken_by_label_order(self):
        # positive and negative tie on score; with no primary set, the winner is
        # the deterministic alphabetical label, NOT labels[0] ('positive').
        state = make_state(
            lexical_label="positive", lexical_conf=1.0,
            contextual_label="negative", contextual_conf=1.0,
        )
        agent = ConsensusAgent(weights={"lexical": 1.0, "contextual": 1.0, "logic": 1.0})
        state = agent.run(state)
        assert state.consensus_output.label != "positive"   # no positional bias
        assert state.consensus_output.label == "negative"   # sorted-name fallback

    def test_confidence_reflects_best_score(self):
        # positive score = 1*0.6 = 0.6; negative score = 1*0.4 = 0.4
        # active_weight_sum = 2 (lexical + contextual only)
        # final_confidence = 0.6 / 2 = 0.3
        state = make_state(
            lexical_label="positive", lexical_conf=0.6,
            contextual_label="negative", contextual_conf=0.4,
        )
        state = ConsensusAgent().run(state)
        assert state.consensus_output.confidence == pytest.approx(0.3)

    def test_three_way_split_is_non_positional(self):
        # Each label gets one equal vote → tie on score. With no primary set, the
        # winner is the deterministic alphabetical label, NOT labels[0].
        state = make_state(
            lexical_label="positive", lexical_conf=1.0,
            contextual_label="negative", contextual_conf=1.0,
            logic_label="neutral", logic_conf=1.0,
        )
        state = ConsensusAgent().run(state)
        assert state.consensus_output.label != "positive"  # not labels[0]
        assert state.consensus_output.label == "negative"  # deterministic alpha


# ---------------------------------------------------------------------------
# Missing / partial agent outputs
# ---------------------------------------------------------------------------

class TestMissingOutputs:
    def test_only_one_agent_present(self):
        state = make_state(contextual_label="neutral", contextual_conf=0.7)
        state = ConsensusAgent().run(state)
        assert state.consensus_output.label == "neutral"
        assert state.final_output.label == "neutral"

    def test_all_abstain_no_primary_returns_none_not_first_label(self):
        # No agent votes AND no usable primary → no-decision (label None), NEVER labels[0].
        state = make_state()
        state = ConsensusAgent().run(state)
        assert state.consensus_output is not None
        assert state.final_output is not None
        assert state.consensus_output.label is None      # not LABELS[0]
        assert state.final_output.label is None
        assert _NO_VOTE_NOTE in state.consensus_output.rationale
        assert "no_decision" in state.consensus_output.rationale

    def test_all_abstain_no_primary_confidence_is_none(self):
        state = make_state()
        state = ConsensusAgent().run(state)
        assert state.consensus_output.confidence is None
        assert state.final_output.confidence is None

    def test_all_abstain_falls_back_to_primary(self):
        # No agent votes but a usable primary → defer to primary, NEVER labels[0].
        state = make_state()
        state.primary_model_output = ModelOutput(label="negative", confidence=0.71)
        state = ConsensusAgent().run(state)
        assert state.final_output.label == "negative"    # primary, not LABELS[0] ('positive')
        assert state.consensus_output.label == "negative"
        assert state.final_output.confidence == pytest.approx(0.71)
        assert "primary_fallback" in state.consensus_output.rationale

    def test_all_agents_have_none_confidence(self):
        state = make_state(
            lexical_label="positive", lexical_conf=None,
            contextual_label="negative", contextual_conf=None,
        )
        state = ConsensusAgent().run(state)
        # _extract_vote returns (None, None) for None confidence → fallback
        assert _NO_VOTE_NOTE in state.consensus_output.rationale

    def test_missing_lexical_and_logic(self):
        state = make_state(contextual_label="negative", contextual_conf=0.85)
        state = ConsensusAgent().run(state)
        assert state.consensus_output.label == "negative"
        assert "no vote" in state.consensus_output.rationale

    def test_partial_outputs_rationale_marks_no_vote(self):
        state = make_state(logic_label="positive", logic_conf=0.6)
        state = ConsensusAgent().run(state)
        rationale = state.consensus_output.rationale
        assert "lexical=no vote" in rationale
        assert "contextual=no vote" in rationale


# ---------------------------------------------------------------------------
# Out-of-vocabulary label handling
# ---------------------------------------------------------------------------

class TestOovLabels:
    def test_oov_label_skipped_not_counted(self):
        # lexical returns a label not in task_config.labels
        state = make_state(
            lexical_label="UNKNOWN_LABEL", lexical_conf=1.0,
            contextual_label="positive", contextual_conf=0.9,
        )
        state = ConsensusAgent().run(state)
        assert state.consensus_output.label == "positive"

    def test_oov_label_appears_in_rationale(self):
        state = make_state(
            lexical_label="bogus", lexical_conf=1.0,
            contextual_label="neutral", contextual_conf=0.8,
        )
        state = ConsensusAgent().run(state)
        assert "invalid label" in state.consensus_output.rationale

    def test_all_oov_returns_fallback(self):
        state = make_state(
            lexical_label="bogus1", lexical_conf=1.0,
            contextual_label="bogus2", contextual_conf=1.0,
            logic_label="bogus3", logic_conf=1.0,
        )
        state = ConsensusAgent().run(state)
        assert _NO_VOTE_NOTE in state.consensus_output.rationale


# ---------------------------------------------------------------------------
# State writes
# ---------------------------------------------------------------------------

class TestStateWrites:
    def test_consensus_output_written(self):
        state = make_state(lexical_label="positive", lexical_conf=0.8)
        assert state.consensus_output is None
        state = ConsensusAgent().run(state)
        assert state.consensus_output is not None

    def test_final_output_written(self):
        state = make_state(contextual_label="negative", contextual_conf=0.7)
        assert state.final_output is None
        state = ConsensusAgent().run(state)
        assert state.final_output is not None

    def test_consensus_and_final_labels_match(self):
        state = make_state(logic_label="neutral", logic_conf=0.6)
        state = ConsensusAgent().run(state)
        assert state.consensus_output.label == state.final_output.label

    def test_consensus_and_final_confidence_match(self):
        state = make_state(lexical_label="positive", lexical_conf=0.75)
        state = ConsensusAgent().run(state)
        assert state.consensus_output.confidence == pytest.approx(
            state.final_output.confidence
        )

    def test_existing_outputs_not_modified(self):
        """Other state fields must not be altered by ConsensusAgent."""
        state = make_state(
            lexical_label="positive", lexical_conf=0.8,
            contextual_label="positive", contextual_conf=0.9,
        )
        original_text = state.input_text
        state = ConsensusAgent().run(state)
        assert state.input_text == original_text
        assert state.lexical_output is not None   # untouched
        assert state.contextual_output is not None

    def test_execute_wrapper_also_writes(self):
        """ConsensusAgent.execute() must go through the same path as run()."""
        state = make_state(contextual_label="neutral", contextual_conf=0.65)
        state = ConsensusAgent().execute(state)
        assert state.consensus_output is not None
        assert state.final_output is not None
