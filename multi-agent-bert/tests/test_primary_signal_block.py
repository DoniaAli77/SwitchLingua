"""Tests for the optional primary-signal prompt block (Fix #3).

All offline, no OpenAI. Proves: block absent when disabled, present when enabled,
top-2 sorted by probability (not label order), missing primary/probabilities do
not crash, and the JSON output schema is unchanged.
"""

from __future__ import annotations

import json

from src.prompts._primary_block import build_primary_signal, render_primary_block
from src.prompts import llm_lexical_prompt, llm_logic_prompt, contextual_prompt
from src.agents.llm_lexical_agent import LLMLexicalAgent
from src.agents.llm_logic_agent import LLMLogicAgent
from src.agents.contextual_agent import ContextualAgent
from src.llm.base_client import LLMClient
from src.state.schema import (
    ModelOutput,
    PipelineState,
    StateMetadata,
    TaskConfig,
)

LABELS = ["positive", "negative", "neutral"]
SIGNAL_MARKER = "PRIMARY MODEL SIGNAL"


def _task(use_signal: bool, labels=None) -> TaskConfig:
    return TaskConfig(
        task_name="sentiment",
        labels=labels or list(LABELS),
        label_descriptions={lbl: lbl for lbl in (labels or LABELS)},
        agents_use_primary_signal=use_signal,
    )


def _state(use_signal: bool, primary="negative", conf=0.43, probs=None, labels=None):
    st = PipelineState(
        metadata=StateMetadata(sample_id="t"),
        input_text="great شغل but bad جدا",
        task_config=_task(use_signal, labels),
    )
    if primary is not None:
        st.primary_model_output = ModelOutput(
            label=primary, confidence=conf,
            probabilities=probs if probs is not None
            else {"positive": 0.18, "negative": 0.43, "neutral": 0.39},
        )
    return st


class _RecordingClient(LLMClient):
    """Returns a fixed valid JSON and records the prompts it was sent."""

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps({"label": "negative", "confidence": 0.7,
                           "reasoning": "r", "evidence": ["x"]})


# ---------------------------------------------------------------------------
# render_primary_block / build_primary_signal
# ---------------------------------------------------------------------------

def test_build_signal_top2_sorted_by_probability_not_label_order():
    mo = ModelOutput(label="neutral", confidence=0.4,
                     probabilities={"positive": 0.1, "negative": 0.35, "neutral": 0.55})
    sig = build_primary_signal(mo)
    # Highest prob first regardless of label order in the dict/labels list.
    assert [lbl for lbl, _ in sig["top2"]] == ["neutral", "negative"]


def test_render_block_contains_signal_and_anti_anchoring():
    sig = build_primary_signal(ModelOutput(
        label="negative", confidence=0.43,
        probabilities={"positive": 0.18, "negative": 0.43, "neutral": 0.39}))
    block = render_primary_block(sig, "lexical")
    assert SIGNAL_MARKER in block
    assert "negative" in block and "0.43" in block
    assert "top-2 labels" in block and "full distribution" in block
    assert "Do NOT simply copy the primary" in block
    assert "independent lexical adjudicator" in block


def test_render_block_none_signal_is_empty():
    assert render_primary_block(None, "logical") == ""
    assert render_primary_block({"label": None}, "logical") == ""


def test_render_block_missing_probabilities_no_crash():
    sig = build_primary_signal(ModelOutput(label="positive", confidence=0.6, probabilities={}))
    block = render_primary_block(sig, "contextual")
    assert SIGNAL_MARKER in block
    assert "positive" in block
    assert "probability distribution unavailable" in block
    assert "top-2 labels" not in block


def test_build_signal_none_when_no_label():
    assert build_primary_signal(ModelOutput(label=None)) is None
    assert build_primary_signal(None) is None


# ---------------------------------------------------------------------------
# build_user_prompt: block absent by default, present with a signal
# ---------------------------------------------------------------------------

def test_prompts_omit_block_without_signal():
    for mod in (llm_lexical_prompt, llm_logic_prompt, contextual_prompt):
        p = mod.build_user_prompt("sentiment", LABELS, {l: l for l in LABELS}, "txt")
        assert SIGNAL_MARKER not in p


def test_prompts_include_block_with_signal():
    sig = build_primary_signal(ModelOutput(
        label="negative", confidence=0.43,
        probabilities={"positive": 0.18, "negative": 0.43, "neutral": 0.39}))
    for mod in (llm_lexical_prompt, llm_logic_prompt, contextual_prompt):
        p = mod.build_user_prompt("sentiment", LABELS, {l: l for l in LABELS}, "txt",
                                  primary_signal=sig)
        assert SIGNAL_MARKER in p
        assert "negative" in p


# ---------------------------------------------------------------------------
# Agent flag gating (the key behavioral guarantee)
# ---------------------------------------------------------------------------

def test_agents_omit_block_when_flag_off():
    for agent_cls in (LLMLexicalAgent, LLMLogicAgent, ContextualAgent):
        client = _RecordingClient()
        agent_cls(llm_client=client).run(_state(use_signal=False))
        assert SIGNAL_MARKER not in client.prompts[0]


def test_agents_include_block_when_flag_on():
    for agent_cls in (LLMLexicalAgent, LLMLogicAgent, ContextualAgent):
        client = _RecordingClient()
        agent_cls(llm_client=client).run(_state(use_signal=True))
        assert SIGNAL_MARKER in client.prompts[0]
        assert "negative" in client.prompts[0]   # the primary label is shown


def test_flag_on_but_no_primary_label_no_block_no_crash():
    for agent_cls in (LLMLexicalAgent, LLMLogicAgent, ContextualAgent):
        client = _RecordingClient()
        st = _state(use_signal=True, primary=None)  # primary defaults to label=None
        agent_cls(llm_client=client).run(st)
        assert SIGNAL_MARKER not in client.prompts[0]


def test_json_schema_unchanged_with_block_on():
    # With the block enabled the agent still parses the normal schema and writes
    # a valid label (no schema change).
    client = _RecordingClient()
    st = _state(use_signal=True)
    out = LLMLexicalAgent(llm_client=client).run(st).lexical_output
    assert out.model_output.label == "negative"
    assert out.model_output.confidence == 0.7


def test_task_generic_arbitrary_labels():
    labels = ["a", "b", "c", "d"]
    client = _RecordingClient()
    st = _state(use_signal=True, primary="c", conf=0.5,
                probs={"a": 0.1, "b": 0.2, "c": 0.5, "d": 0.2}, labels=labels)
    # Recording client returns 'negative' which is invalid here → agent abstains,
    # but the prompt must still carry the block with the arbitrary primary label.
    LLMLogicAgent(llm_client=client).run(st)
    assert SIGNAL_MARKER in client.prompts[0]
    assert "c (0.50)" in client.prompts[0]
