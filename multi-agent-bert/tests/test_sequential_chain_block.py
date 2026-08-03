"""Tests for the opt-in sequential-chain ablation (same parallel agents, run in a
fixed order, each later agent seeing the earlier specialists' conclusions).

Verifies:
- default (sequential_chain=False) → prompts are byte-identical (no chain block);
- sequential_chain=True → Polarity sees Lexical; Contextual sees Lexical + Logic;
- the block never surfaces the current agent's own slot or the primary model;
- build_agent_chain_block returns "" when there is nothing to show.

No real LLM calls — an offline stub client records the prompt it receives.
"""
from __future__ import annotations

import json

from src.agents.contextual_agent import ContextualAgent
from src.agents.llm_lexical_agent import LLMLexicalAgent
from src.agents.polarity_agent import PolarityAgent
from src.prompts._agent_chain_block import build_agent_chain_block
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    StateMetadata,
    TaskConfig,
)

_CHAIN_HEADER = "PRIOR AGENTS' ANALYSES"


class _StubClient:
    def __init__(self, label: str = "negative", confidence: float = 0.81) -> None:
        self._response = json.dumps(
            {"label": label, "confidence": confidence, "reasoning": "stub", "evidence": ["x"]}
        )
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response


def _state(sequential_chain: bool) -> PipelineState:
    labels = ["positive", "negative", "neutral"]
    tc = TaskConfig(
        task_name="sentiment_classification",
        labels=labels,
        label_descriptions={l: f"{l} sentiment" for l in labels},
        sequential_chain=sequential_chain,
    )
    return PipelineState(
        metadata=StateMetadata(sample_id="t-01"),
        input_text="some code-switched text",
        task_config=tc,
    )


def _agent_out(label: str, conf: float, note: str) -> AgentOutput:
    return AgentOutput(
        agent_name="x",
        model_output=ModelOutput(label=label, confidence=conf),
        notes=note,
    )


# --------------------------------------------------------------------------- #
# build_agent_chain_block unit
# --------------------------------------------------------------------------- #

def test_chain_block_empty_when_no_prior():
    st = _state(sequential_chain=True)
    assert build_agent_chain_block(st, exclude_slot="logic_output", analysis_kind="polarity") == ""


def test_chain_block_lists_prior_and_excludes_self_and_primary():
    st = _state(sequential_chain=True)
    st.primary_model_output = ModelOutput(label="positive", confidence=0.99)
    st.lexical_output = _agent_out("negative", 0.80, "saw dislike")
    st.logic_output = _agent_out("neutral", 0.70, "a question")
    block = build_agent_chain_block(st, exclude_slot="contextual_output", analysis_kind="contextual")
    assert "Lexical agent -> label='negative'" in block
    assert "Polarity agent -> label='neutral'" in block
    # primary must NOT leak into the agent chain block
    assert "positive" not in block
    # excluded slot (self) absent
    assert "Contextual agent" not in block


# --------------------------------------------------------------------------- #
# Default path is unchanged (byte-identical)
# --------------------------------------------------------------------------- #

def test_polarity_default_has_no_chain_block():
    st = _state(sequential_chain=False)
    st.lexical_output = _agent_out("negative", 0.80, "saw dislike")
    client = _StubClient()
    PolarityAgent(llm_client=client).run(st)
    assert _CHAIN_HEADER not in (client.last_prompt or "")


def test_contextual_default_has_no_chain_block():
    st = _state(sequential_chain=False)
    st.lexical_output = _agent_out("negative", 0.80, "cue")
    st.logic_output = _agent_out("neutral", 0.70, "meta")
    client = _StubClient()
    ContextualAgent(llm_client=client).run(st)
    assert _CHAIN_HEADER not in (client.last_prompt or "")


# --------------------------------------------------------------------------- #
# Sequential path injects the chain
# --------------------------------------------------------------------------- #

def test_polarity_sequential_sees_lexical():
    st = _state(sequential_chain=True)
    st.lexical_output = _agent_out("negative", 0.80, "saw the word dislike")
    client = _StubClient()
    PolarityAgent(llm_client=client).run(st)
    prompt = client.last_prompt or ""
    assert _CHAIN_HEADER in prompt
    assert "Lexical agent -> label='negative'" in prompt


def test_contextual_sequential_sees_lexical_and_logic():
    st = _state(sequential_chain=True)
    st.lexical_output = _agent_out("negative", 0.80, "cue")
    st.logic_output = _agent_out("neutral", 0.70, "meta question")
    client = _StubClient()
    ContextualAgent(llm_client=client).run(st)
    prompt = client.last_prompt or ""
    assert _CHAIN_HEADER in prompt
    assert "Lexical agent -> label='negative'" in prompt
    assert "Polarity agent -> label='neutral'" in prompt


def test_lexical_first_sees_nothing_even_in_sequential():
    # Lexical runs first → no prior specialists → block absent even with flag on.
    st = _state(sequential_chain=True)
    client = _StubClient()
    LLMLexicalAgent(llm_client=client).run(st)
    assert _CHAIN_HEADER not in (client.last_prompt or "")
