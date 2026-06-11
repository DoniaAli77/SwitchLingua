"""Tests for primary-aware consensus (audit C1, Fix #2).

The primary participates as a confidence-scaled weighted vote; agents override it
only when they collectively out-vote it; tie-breaking never uses config label
order. All offline, no OpenAI.
"""

from __future__ import annotations

import pytest

from src.agents._abstain import abstain_output
from src.agents.consensus_agent import ConsensusAgent
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    StateMetadata,
    TaskConfig,
)

LABELS = ["positive", "negative", "neutral"]


def _state(labels=None, primary=None, primary_conf=None):
    st = PipelineState(
        metadata=StateMetadata(sample_id="t"),
        input_text="text",
        task_config=TaskConfig(task_name="sentiment", labels=labels or list(LABELS)),
    )
    if primary is not None:
        st.primary_model_output = ModelOutput(label=primary, confidence=primary_conf)
    return st


def _vote(name, label, conf):
    return AgentOutput(agent_name=name, model_output=ModelOutput(label=label, confidence=conf))


# ---------------------------------------------------------------------------
# Primary participates as a vote
# ---------------------------------------------------------------------------

def test_primary_counted_when_agents_agree():
    st = _state(primary="negative", primary_conf=0.5)
    st.contextual_output = _vote("ContextualAgent", "negative", 0.8)
    st = ConsensusAgent().run(st)
    assert st.final_output.label == "negative"
    assert "primary" in st.consensus_output.rationale


def test_default_includes_primary_weight_one():
    assert ConsensusAgent().weights["primary"] == 1.0


# ---------------------------------------------------------------------------
# Conservative override
# ---------------------------------------------------------------------------

def test_single_agent_does_not_flip_confident_primary():
    # primary 'negative' conf 0.9 → contribution 1.0*0.9 = 0.90
    # one agent 'positive' conf 0.8 → 0.80  → primary wins (not flipped).
    st = _state(primary="negative", primary_conf=0.9)
    st.lexical_output = _vote("LexicalAgent", "positive", 0.8)
    st = ConsensusAgent().run(st)
    assert st.final_output.label == "negative"


def test_two_agents_flip_primary():
    # primary 'negative' conf 0.5 → 0.50 ; two agents 'positive' 0.8 each → 1.60
    st = _state(primary="negative", primary_conf=0.5)
    st.lexical_output = _vote("LexicalAgent", "positive", 0.8)
    st.contextual_output = _vote("ContextualAgent", "positive", 0.8)
    st = ConsensusAgent().run(st)
    assert st.final_output.label == "positive"


def test_low_conf_primary_easier_to_override_than_high_conf():
    # Same single agent 'positive' 0.8 (=0.80).
    # Low-conf primary 'negative' 0.4 → 0.40 → agent wins.
    low = _state(primary="negative", primary_conf=0.4)
    low.lexical_output = _vote("LexicalAgent", "positive", 0.8)
    assert ConsensusAgent().run(low).final_output.label == "positive"
    # High-conf primary 'negative' 0.95 → 0.95 → primary holds.
    high = _state(primary="negative", primary_conf=0.95)
    high.lexical_output = _vote("LexicalAgent", "positive", 0.8)
    assert ConsensusAgent().run(high).final_output.label == "negative"


def test_w_primary_zero_reproduces_agents_only():
    # With w_primary=0 the primary is not a vote → single agent decides.
    st = _state(primary="negative", primary_conf=0.99)
    st.lexical_output = _vote("LexicalAgent", "positive", 0.6)
    st = ConsensusAgent(weights={"primary": 0.0}).run(st)
    assert st.final_output.label == "positive"   # agent-only, primary ignored


# ---------------------------------------------------------------------------
# Tie-breaking (non-positional)
# ---------------------------------------------------------------------------

def test_tie_prefers_primary_label_not_first_label():
    # lexical 'neutral' 1.0 vs logic 'negative' 1.0 tie; primary 'neutral' 0.0
    # contribution keeps the tie but primary anchors the winner to 'neutral',
    # which is NOT labels[0] ('positive').
    st = _state(primary="neutral", primary_conf=0.0)
    st.lexical_output = _vote("LexicalAgent", "neutral", 1.0)
    st.logic_output = _vote("LogicAgent", "negative", 1.0)
    st = ConsensusAgent().run(st)
    assert st.final_output.label == "neutral"     # primary-anchored tie, not 'positive'


def test_tie_without_primary_in_tie_uses_agent_count():
    # 'negative' has 2 agents @0.5 (=1.0 total); 'positive' has 1 agent @1.0 (=1.0).
    # Scores tie at 1.0; primary 'neutral' is not in the tie → most-agents wins → negative.
    st = _state(primary="neutral", primary_conf=0.0)
    st.lexical_output = _vote("LexicalAgent", "negative", 0.5)
    st.logic_output = _vote("LogicAgent", "negative", 0.5)
    st.contextual_output = _vote("ContextualAgent", "positive", 1.0)
    st = ConsensusAgent().run(st)
    assert st.final_output.label == "negative"    # more voting agents, not labels[0]


# ---------------------------------------------------------------------------
# Abstain interplay (Fix #1 preserved) + task-generic
# ---------------------------------------------------------------------------

def test_all_abstain_still_defers_to_primary():
    st = _state(primary="neutral", primary_conf=0.6)
    st.lexical_output = abstain_output("LexicalAgent", st, "x")
    st.contextual_output = abstain_output("ContextualAgent", st, "x")
    st.logic_output = abstain_output("LogicAgent", st, "x")
    st = ConsensusAgent().run(st)
    assert st.final_output.label == "neutral"
    assert "primary_fallback" in st.consensus_output.rationale


def test_task_generic_arbitrary_labels():
    labels = ["a", "b", "c", "d"]
    st = _state(labels=labels, primary="c", primary_conf=0.9)
    st.lexical_output = _vote("LexicalAgent", "a", 0.7)   # one agent disagrees
    st = ConsensusAgent().run(st)
    assert st.final_output.label == "c"   # confident primary holds, no label hardcoding
