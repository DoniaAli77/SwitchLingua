"""Tests for the experimental ``lexical_polarity_contextual`` sentiment agent variant.

Covers:
- default build_orchestrator uses Lexical + Logic + Contextual;
- the experimental variant uses Lexical + Polarity + Contextual;
- the Polarity prompt names no dataset/benchmark and preserves the JSON contract;
- a bad variant value raises clearly;
- env-var gating;
- the PolarityAgent writes a valid AgentOutput to state.logic_output.

No real LLM calls — a tiny offline stub client is used.
"""
from __future__ import annotations

import json
import os

import pytest

from evaluate_pipeline import build_orchestrator
from src.agents._sentiment_agent_variant import active_agent_variant
from src.agents.contextual_agent import ContextualAgent
from src.agents.llm_lexical_agent import LLMLexicalAgent
from src.agents.llm_logic_agent import LLMLogicAgent
from src.agents.polarity_agent import PolarityAgent
from src.prompts import polarity_prompt
from src.state.schema import ModelOutput, PipelineState, StateMetadata, TaskConfig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _StubClient:
    """Returns a fixed JSON response; records the last prompt it received."""

    def __init__(self, label: str = "negative", confidence: float = 0.81) -> None:
        self._response = json.dumps(
            {
                "label": label,
                "confidence": confidence,
                "reasoning": "stub",
                "evidence": ["stub"],
            }
        )
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response


def _task_config() -> TaskConfig:
    labels = ["positive", "negative", "neutral"]
    return TaskConfig(
        task_name="sentiment_classification",
        labels=labels,
        label_descriptions={l: f"{l} sentiment" for l in labels},
    )


@pytest.fixture(autouse=True)
def _clear_env():
    """Ensure no leaked env var between tests; restore afterwards."""
    saved = os.environ.pop("SENTIMENT_AGENT_VARIANT", None)
    yield
    os.environ.pop("SENTIMENT_AGENT_VARIANT", None)
    if saved is not None:
        os.environ["SENTIMENT_AGENT_VARIANT"] = saved


# --------------------------------------------------------------------------- #
# Agent-list wiring
# --------------------------------------------------------------------------- #

def test_default_uses_lexical_logic_contextual():
    tc = _task_config()
    orch = build_orchestrator(tc, threshold=0.7, enable_deliberation=False)
    assert isinstance(orch._llm_lexical, LLMLexicalAgent)
    assert isinstance(orch._llm_logic, LLMLogicAgent)
    assert not isinstance(orch._llm_logic, PolarityAgent)
    assert isinstance(orch._contextual, ContextualAgent)


def test_variant_uses_lexical_polarity_contextual():
    tc = _task_config()
    orch = build_orchestrator(
        tc, threshold=0.7, enable_deliberation=False,
        sentiment_agent_variant="lexical_polarity_contextual",
    )
    # Logic slot is now the Polarity agent; lexical and contextual unchanged.
    assert isinstance(orch._llm_logic, PolarityAgent)
    assert isinstance(orch._llm_lexical, LLMLexicalAgent)
    assert isinstance(orch._contextual, ContextualAgent)


def test_variant_via_env_var():
    tc = _task_config()
    os.environ["SENTIMENT_AGENT_VARIANT"] = "lexical_polarity_contextual"
    orch = build_orchestrator(tc, threshold=0.7, enable_deliberation=False)
    assert isinstance(orch._llm_logic, PolarityAgent)


def test_explicit_default_overrides_env():
    tc = _task_config()
    os.environ["SENTIMENT_AGENT_VARIANT"] = "lexical_polarity_contextual"
    orch = build_orchestrator(
        tc, threshold=0.7, enable_deliberation=False,
        sentiment_agent_variant="default",
    )
    assert isinstance(orch._llm_logic, LLMLogicAgent)
    assert not isinstance(orch._llm_logic, PolarityAgent)


# --------------------------------------------------------------------------- #
# Variant resolver
# --------------------------------------------------------------------------- #

def test_active_agent_variant_default():
    assert active_agent_variant() == "default"
    assert active_agent_variant("default") == "default"


def test_active_agent_variant_experimental():
    assert active_agent_variant("lexical_polarity_contextual") == "lexical_polarity_contextual"


def test_bad_variant_raises():
    with pytest.raises(ValueError):
        active_agent_variant("bogus")
    os.environ["SENTIMENT_AGENT_VARIANT"] = "nonsense"
    with pytest.raises(ValueError):
        active_agent_variant()


# --------------------------------------------------------------------------- #
# Polarity prompt contract
# --------------------------------------------------------------------------- #

_FORBIDDEN = ["eesa", "arensa", "ahmed", "twitter", "arsentd", "tweet"]


def test_polarity_prompt_no_dataset_names():
    text = polarity_prompt.SYSTEM_PROMPT.lower()
    for term in _FORBIDDEN:
        assert term not in text, f"Polarity prompt must not mention '{term}'"
    # User template also clean
    rendered = polarity_prompt.build_user_prompt(
        "sentiment_classification",
        ["positive", "negative", "neutral"],
        {"positive": "p", "negative": "n", "neutral": "x"},
        "some text",
    ).lower()
    for term in _FORBIDDEN:
        assert term not in rendered


def test_polarity_prompt_preserves_json_contract():
    sp = polarity_prompt.SYSTEM_PROMPT
    assert "OUTPUT FORMAT (copy this structure exactly):" in sp
    for key in ('"label"', '"confidence"', '"reasoning"', '"evidence"'):
        assert key in sp
    # JSON contract is the final block of the system prompt
    tail = sp.split("OUTPUT FORMAT (copy this structure exactly):", 1)[1]
    assert '"label"' in tail and '"confidence"' in tail
    # get_system_prompt returns the same single prompt regardless of arg
    assert polarity_prompt.get_system_prompt() == sp
    assert polarity_prompt.get_system_prompt("semantic_v1") == sp


def test_polarity_user_prompt_has_labels_and_text():
    up = polarity_prompt.build_user_prompt(
        "sentiment_classification",
        ["positive", "negative", "neutral"],
        {"positive": "p", "negative": "n", "neutral": "x"},
        "I really did not like it",
    )
    assert "I really did not like it" in up
    assert "positive, negative, neutral" in up


# --------------------------------------------------------------------------- #
# PolarityAgent behaviour
# --------------------------------------------------------------------------- #

def test_polarity_agent_writes_logic_output():
    tc = _task_config()
    agent = PolarityAgent(llm_client=_StubClient(label="negative", confidence=0.77))
    state = PipelineState(
        metadata=StateMetadata(sample_id="t1"),
        input_text="ما عجبني الفيديو",
        task_config=tc,
    )
    state.primary_model_output = ModelOutput(
        label="neutral", confidence=0.6, probabilities={}, raw_text=""
    )
    out_state = agent.run(state)
    assert out_state.logic_output is not None
    assert out_state.logic_output.model_output.label == "negative"
    assert out_state.logic_output.agent_name == "PolarityAgent"
    assert abs(out_state.logic_output.model_output.confidence - 0.77) < 1e-9


def test_polarity_agent_invalid_label_abstains():
    tc = _task_config()
    agent = PolarityAgent(llm_client=_StubClient(label="not_a_label"))
    state = PipelineState(
        metadata=StateMetadata(sample_id="t2"),
        input_text="نص",
        task_config=tc,
    )
    out_state = agent.run(state)
    # Abstain → None label (never labels[0]); consensus excludes None votes.
    assert out_state.logic_output is not None
    assert out_state.logic_output.model_output.label is None
