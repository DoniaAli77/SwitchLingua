"""Tests for the no-vote / abstain fallback (audit C2).

Verifies that every former labels[0] fallback site now abstains (label=None),
consensus ignores abstentions, all-abstain defers to the primary, and an
all-abstain-with-no-primary case never returns labels[0]. All offline.
"""

from __future__ import annotations

import json

import pytest

from src.agents._abstain import ABSTAIN_FLAG, abstain_output
from src.agents.consensus_agent import ConsensusAgent
from src.agents.contextual_agent import ContextualAgent
from src.agents.lexical_agent import LexicalAgent
from src.agents.logic_agent import LogicAgent
from src.agents.llm_lexical_agent import LLMLexicalAgent
from src.agents.llm_logic_agent import LLMLogicAgent
from src.llm.base_client import LLMClient
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    StateMetadata,
    TaskConfig,
)

LABELS = ["positive", "negative", "neutral"]


def _state(labels=None, text="some code-switched text") -> PipelineState:
    return PipelineState(
        metadata=StateMetadata(sample_id="t-01"),
        input_text=text,
        task_config=TaskConfig(task_name="sentiment", labels=labels or list(LABELS)),
    )


class _FixedClient(LLMClient):
    def __init__(self, response: str) -> None:
        self._r = response

    def generate(self, prompt: str) -> str:  # noqa: D401
        return self._r


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def test_abstain_output_shape():
    out = abstain_output("AnyAgent", _state(), "because reasons")
    assert out.model_output.label is None
    assert out.model_output.confidence is None
    assert out.model_output.probabilities == {}
    assert out.features[ABSTAIN_FLAG] is True
    assert out.features["abstain_reason"] == "because reasons"
    # Abstain output passes label validation (label is None → skipped).
    out.model_output.validate_labels(_state().task_config)


# ---------------------------------------------------------------------------
# LLM agents: parse failure / invalid label → abstain (not labels[0])
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent_cls,attr", [
    (LLMLexicalAgent, "lexical_output"),
    (LLMLogicAgent, "logic_output"),
])
def test_llm_parse_failure_abstains(agent_cls, attr):
    state = agent_cls(llm_client=_FixedClient("not json")).run(_state())
    out = getattr(state, attr)
    assert out.model_output.label is None        # NOT labels[0]
    assert out.features.get(ABSTAIN_FLAG) is True


@pytest.mark.parametrize("agent_cls,attr", [
    (LLMLexicalAgent, "lexical_output"),
    (LLMLogicAgent, "logic_output"),
])
def test_llm_invalid_label_abstains(agent_cls, attr):
    bad = json.dumps({"label": "NOT_A_LABEL", "confidence": 0.9,
                      "reasoning": "x", "evidence": []})
    state = agent_cls(llm_client=_FixedClient(bad)).run(_state())
    out = getattr(state, attr)
    assert out.model_output.label is None
    assert out.features.get(ABSTAIN_FLAG) is True


def test_contextual_parse_failure_abstains():
    state = ContextualAgent(_FixedClient("garbage")).run(_state())
    out = state.contextual_output
    assert out.model_output.label is None
    assert out.features.get(ABSTAIN_FLAG) is True


# ---------------------------------------------------------------------------
# Non-LLM agents: no keyword / rule hit → abstain
# ---------------------------------------------------------------------------

def test_lexical_no_keyword_hit_abstains():
    state = LexicalAgent(keyword_map={"positive": ["great"]}).run(_state(text="zzz qqq"))
    out = state.lexical_output
    assert out.model_output.label is None        # NOT labels[0]
    assert out.features.get(ABSTAIN_FLAG) is True


def test_logic_no_rule_hit_abstains():
    state = LogicAgent(rule_map={"positive": [r"\bgreat\b"]}).run(_state(text="zzz qqq"))
    out = state.logic_output
    assert out.model_output.label is None
    assert out.features.get(ABSTAIN_FLAG) is True


# ---------------------------------------------------------------------------
# Consensus: ignore abstentions; all-abstain → primary; no primary → None
# ---------------------------------------------------------------------------

def _abstain(name) -> AgentOutput:
    return abstain_output(name, _state(), "abstain")


def _vote(name, label, conf) -> AgentOutput:
    return AgentOutput(agent_name=name, model_output=ModelOutput(label=label, confidence=conf))


def test_consensus_ignores_abstentions():
    state = _state()
    state.lexical_output = _abstain("LexicalAgent")
    state.contextual_output = _vote("ContextualAgent", "negative", 0.8)
    state.logic_output = _abstain("LogicAgent")
    state = ConsensusAgent().run(state)
    assert state.final_output.label == "negative"   # the only real vote wins
    assert "no vote" in state.consensus_output.rationale


def test_all_abstain_falls_back_to_primary():
    state = _state()
    state.lexical_output = _abstain("LexicalAgent")
    state.contextual_output = _abstain("ContextualAgent")
    state.logic_output = _abstain("LogicAgent")
    state.primary_model_output = ModelOutput(label="neutral", confidence=0.66)
    state = ConsensusAgent().run(state)
    assert state.final_output.label == "neutral"    # primary, not labels[0]
    assert state.final_output.confidence == pytest.approx(0.66)
    assert "primary_fallback" in state.consensus_output.rationale


def test_all_abstain_no_primary_never_returns_first_label():
    state = _state()  # no primary set → empty ModelOutput(label=None)
    state.lexical_output = _abstain("LexicalAgent")
    state.contextual_output = _abstain("ContextualAgent")
    state.logic_output = _abstain("LogicAgent")
    state = ConsensusAgent().run(state)
    assert state.final_output.label is None          # NOT labels[0] ('positive')
    assert state.consensus_output.label is None
    assert "no_decision" in state.consensus_output.rationale
