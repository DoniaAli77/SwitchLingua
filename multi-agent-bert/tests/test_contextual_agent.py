"""Unit tests for ContextualAgent, MockLLMClient, and contextual_prompt."""

from __future__ import annotations

import json

import pytest

from src.agents.contextual_agent import (
    ContextualAgent,
    ContextualParseError,
    _INVALID_LABEL_NOTE,
    _PARSE_FAIL_NOTE,
)
from src.llm.base_client import LLMClientError
from src.llm.mock_client import LABEL_ECHO_FALLBACK_NOTE, MockLLMClient
from src.prompts.contextual_prompt import SYSTEM_PROMPT, build_user_prompt
from src.state.schema import AgentOutput, ModelOutput, PipelineState, StateMetadata, TaskConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABELS = ["tech", "sports", "finance"]
DESCRIPTIONS = {
    "tech": "Technology and software.",
    "sports": "Competitive sports and athletics.",
    "finance": "Financial markets and banking.",
}


def make_state(text: str, labels: list[str] | None = None) -> PipelineState:
    return PipelineState(
        metadata=StateMetadata(sample_id="ctx-001"),
        input_text=text,
        task_config=TaskConfig(
            task_name="topic_classification",
            labels=labels if labels is not None else LABELS,
            label_descriptions=DESCRIPTIONS,
        ),
    )


def valid_response(label: str = "tech", confidence: float = 0.9) -> str:
    return json.dumps({
        "label": label,
        "confidence": confidence,
        "reasoning": "The text discusses software.",
        "evidence": ["software", "API"],
    })


# ---------------------------------------------------------------------------
# Happy-path: fixed mock
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_output_written_to_contextual_output(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response("tech", 0.9))
        agent = ContextualAgent(client)
        state = agent.run(make_state("The new API is live"))
        assert state.contextual_output is not None

    def test_correct_label_and_confidence(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response("sports", 0.75))
        agent = ContextualAgent(client)
        state = agent.run(make_state("The match ended 2-1"))
        out = state.contextual_output.model_output
        assert out.label == "sports"
        assert out.confidence == pytest.approx(0.75)

    def test_probabilities_assigned_to_chosen_label(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response("finance", 0.8))
        agent = ContextualAgent(client)
        state = agent.run(make_state("IPO valuation looks strong"))
        probs = state.contextual_output.model_output.probabilities
        assert probs["finance"] == pytest.approx(0.8)
        # remaining 0.2 split evenly across 2 other labels
        assert probs["tech"] == pytest.approx(0.1)
        assert probs["sports"] == pytest.approx(0.1)

    def test_probabilities_sum_to_one(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response("tech", 0.6))
        agent = ContextualAgent(client)
        state = agent.run(make_state("cloud computing trends"))
        probs = state.contextual_output.model_output.probabilities
        assert sum(probs.values()) == pytest.approx(1.0, rel=1e-5)

    def test_reasoning_stored_in_notes(self):
        resp = json.dumps({
            "label": "tech",
            "confidence": 0.9,
            "reasoning": "Talks about software engineering.",
            "evidence": ["software"],
        })
        client = MockLLMClient(mode="fixed", fixed_response=resp)
        agent = ContextualAgent(client)
        state = agent.run(make_state("software"))
        assert state.contextual_output.notes == "Talks about software engineering."

    def test_evidence_stored_in_features(self):
        resp = json.dumps({
            "label": "tech",
            "confidence": 0.9,
            "reasoning": "r",
            "evidence": ["API", "cloud"],
        })
        client = MockLLMClient(mode="fixed", fixed_response=resp)
        agent = ContextualAgent(client)
        state = agent.run(make_state("API and cloud"))
        assert state.contextual_output.features["evidence"] == ["API", "cloud"]

    def test_raw_llm_response_stored_in_features(self):
        resp = valid_response("tech", 0.9)
        client = MockLLMClient(mode="fixed", fixed_response=resp)
        agent = ContextualAgent(client)
        state = agent.run(make_state("AI software"))
        assert state.contextual_output.features["raw_llm_response"] == resp

    def test_agent_name_in_output(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response())
        agent = ContextualAgent(client, name="my_contextual")
        state = agent.run(make_state("AI"))
        assert state.contextual_output.agent_name == "my_contextual"

    def test_raw_text_preserved(self):
        text = "The league match was exciting"
        client = MockLLMClient(mode="fixed", fixed_response=valid_response("sports", 0.8))
        agent = ContextualAgent(client)
        state = agent.run(make_state(text))
        assert state.contextual_output.model_output.raw_text == text


# ---------------------------------------------------------------------------
# label_echo mock mode
# ---------------------------------------------------------------------------

class TestLabelEchoMode:
    def test_echo_detects_label_from_prompt(self):
        # Use allowed_labels=["finance"] so the mock searches only for "finance".
        # "finance" appears in both the input text and the rendered prompt,
        # so it will always be detected regardless of label ordering.
        client = MockLLMClient(
            mode="label_echo",
            allowed_labels=["finance"],
            fixed_confidence=0.85,
        )
        agent = ContextualAgent(client)
        # Only "finance" is in task labels so the validated label must be finance.
        state = agent.run(make_state("The finance report was released today", ["finance"]))
        assert state.contextual_output.model_output.label == "finance"

    def test_echo_fallback_to_first_label_when_none_found(self):
        client = MockLLMClient(
            mode="label_echo",
            allowed_labels=LABELS,
        )
        agent = ContextualAgent(client)
        state = agent.run(make_state("today is a sunny day"))
        # No label keyword in text → falls back to first allowed label
        assert state.contextual_output.model_output.label == "tech"


# ---------------------------------------------------------------------------
# LLM call log
# ---------------------------------------------------------------------------

class TestCallLog:
    def test_prompt_logged_on_generate(self):
        log: list[str] = []
        client = MockLLMClient(
            mode="fixed", fixed_response=valid_response(), call_log=log
        )
        agent = ContextualAgent(client)
        agent.run(make_state("AI software"))
        assert len(log) == 1
        assert "topic_classification" in log[0]

    def test_system_prompt_prepended(self):
        log: list[str] = []
        client = MockLLMClient(
            mode="fixed", fixed_response=valid_response(), call_log=log
        )
        agent = ContextualAgent(client)
        agent.run(make_state("AI"))
        assert log[0].startswith(SYSTEM_PROMPT[:30])

    def test_prior_agent_summaries_included_when_enabled(self):
        log: list[str] = []
        client = MockLLMClient(mode="fixed", fixed_response=valid_response(), call_log=log)
        agent = ContextualAgent(client)
        state = make_state("AI software")
        state.task_config.contextual_use_prior_outputs = True
        state.primary_model = ModelOutput(label="tech", confidence=0.88)
        state.lexical_output = AgentOutput(
            agent_name="LexicalAgent",
            model_output=ModelOutput(label="tech", confidence=0.75),
            notes="Matched keywords in text.",
        )
        state.logic_output = AgentOutput(
            agent_name="LogicAgent",
            model_output=ModelOutput(label="sports", confidence=0.55),
            notes="One weak rule fired.",
        )

        agent.run(state)

        assert "PRIOR AGENT SUMMARIES" in log[0]
        assert "Primary model -> label='tech', confidence=0.880" in log[0]
        assert "Lexical agent -> label='tech', confidence=0.750" in log[0]
        assert "Logic agent -> label='sports', confidence=0.550" in log[0]

    def test_prior_agent_summaries_omitted_when_disabled(self):
        log: list[str] = []
        client = MockLLMClient(mode="fixed", fixed_response=valid_response(), call_log=log)
        agent = ContextualAgent(client)
        state = make_state("AI software")
        state.task_config.contextual_use_prior_outputs = False
        state.primary_model = ModelOutput(label="tech", confidence=0.88)
        state.lexical_output = AgentOutput(
            agent_name="LexicalAgent",
            model_output=ModelOutput(label="tech", confidence=0.75),
            notes="Matched keywords in text.",
        )
        state.logic_output = AgentOutput(
            agent_name="LogicAgent",
            model_output=ModelOutput(label="sports", confidence=0.55),
            notes="One weak rule fired.",
        )

        agent.run(state)

        assert "PRIOR AGENT SUMMARIES" not in log[0]


# ---------------------------------------------------------------------------
# Parse error fallback
# ---------------------------------------------------------------------------

class TestParseErrorFallback:
    def test_non_json_triggers_fallback(self):
        client = MockLLMClient(mode="fixed", fixed_response="This is plain text.")
        agent = ContextualAgent(client)
        state = agent.run(make_state("AI software"))
        assert state.contextual_output.notes == _PARSE_FAIL_NOTE

    def test_missing_key_triggers_fallback(self):
        resp = json.dumps({"label": "tech", "confidence": 0.9})  # missing reasoning + evidence
        client = MockLLMClient(mode="fixed", fixed_response=resp)
        agent = ContextualAgent(client)
        state = agent.run(make_state("AI software"))
        assert state.contextual_output.notes == _PARSE_FAIL_NOTE

    def test_invalid_confidence_type_triggers_fallback(self):
        resp = json.dumps({
            "label": "tech", "confidence": "high",
            "reasoning": "r", "evidence": [],
        })
        client = MockLLMClient(mode="fixed", fixed_response=resp)
        agent = ContextualAgent(client)
        state = agent.run(make_state("AI"))
        assert state.contextual_output.notes == _PARSE_FAIL_NOTE

    def test_confidence_out_of_range_triggers_fallback(self):
        resp = json.dumps({
            "label": "tech", "confidence": 1.5,
            "reasoning": "r", "evidence": [],
        })
        client = MockLLMClient(mode="fixed", fixed_response=resp)
        agent = ContextualAgent(client)
        state = agent.run(make_state("AI"))
        assert state.contextual_output.notes == _PARSE_FAIL_NOTE

    def test_markdown_fenced_json_parsed_correctly(self):
        raw = "```json\n" + valid_response("tech", 0.9) + "\n```"
        client = MockLLMClient(mode="fixed", fixed_response=raw)
        agent = ContextualAgent(client)
        state = agent.run(make_state("AI software"))
        assert state.contextual_output.model_output.label == "tech"

    def test_fallback_uses_uniform_distribution(self):
        client = MockLLMClient(mode="fixed", fixed_response="not json")
        agent = ContextualAgent(client)
        state = agent.run(make_state("AI", labels=["tech", "sports", "finance"]))
        probs = state.contextual_output.model_output.probabilities
        for prob in probs.values():
            assert prob == pytest.approx(1 / 3, rel=1e-4)


# ---------------------------------------------------------------------------
# Invalid label in response
# ---------------------------------------------------------------------------

class TestInvalidLabelFallback:
    def test_unlisted_label_triggers_fallback(self):
        resp = json.dumps({
            "label": "music",  # not in LABELS
            "confidence": 0.9,
            "reasoning": "r",
            "evidence": [],
        })
        client = MockLLMClient(mode="fixed", fixed_response=resp)
        agent = ContextualAgent(client)
        state = agent.run(make_state("I love music"))
        assert state.contextual_output.notes == _INVALID_LABEL_NOTE

    def test_fallback_label_is_first_in_list(self):
        resp = json.dumps({
            "label": "xyz",
            "confidence": 0.9,
            "reasoning": "r",
            "evidence": [],
        })
        client = MockLLMClient(mode="fixed", fixed_response=resp)
        agent = ContextualAgent(client)
        labels = ["finance", "tech", "sports"]
        state = agent.run(make_state("text", labels))
        assert state.contextual_output.model_output.label == "finance"


# ---------------------------------------------------------------------------
# LLMClientError propagation
# ---------------------------------------------------------------------------

class TestLLMClientError:
    def test_llm_error_propagates(self):
        client = MockLLMClient(mode="raise_on_call")
        agent = ContextualAgent(client)
        with pytest.raises(LLMClientError):
            agent.run(make_state("AI software"))


# ---------------------------------------------------------------------------
# Validation hooks
# ---------------------------------------------------------------------------

class TestValidation:
    def test_raises_on_blank_input(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response())
        agent = ContextualAgent(client)
        with pytest.raises(ValueError, match="input_text"):
            agent.execute(make_state("   "))

    def test_raises_on_empty_labels(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response())
        agent = ContextualAgent(client)
        state = make_state("AI", labels=[])
        with pytest.raises(ValueError, match="labels"):
            agent.execute(state)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

class TestPromptTemplate:
    def test_prompt_contains_task_name(self):
        prompt = build_user_prompt("topic_clf", LABELS, DESCRIPTIONS, "hello")
        assert "topic_clf" in prompt

    def test_prompt_contains_all_labels(self):
        prompt = build_user_prompt("task", LABELS, DESCRIPTIONS, "text")
        for label in LABELS:
            assert label in prompt

    def test_prompt_contains_descriptions(self):
        prompt = build_user_prompt("task", LABELS, DESCRIPTIONS, "text")
        assert "Technology and software." in prompt

    def test_prompt_contains_input_text(self):
        text = "This is a unique test sentence."
        prompt = build_user_prompt("task", LABELS, DESCRIPTIONS, text)
        assert text in prompt

    def test_missing_description_shows_placeholder(self):
        prompt = build_user_prompt("task", ["new_label"], {}, "text")
        assert "(no description)" in prompt

    def test_prompt_contains_prior_context_block_when_given(self):
        prompt = build_user_prompt(
            "task",
            LABELS,
            DESCRIPTIONS,
            "text",
            prior_agent_summaries=["Primary model -> label='tech', confidence=0.900"],
        )
        assert "PRIOR AGENT SUMMARIES" in prompt
        assert "Primary model -> label='tech', confidence=0.900" in prompt
