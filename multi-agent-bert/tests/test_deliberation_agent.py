"""Unit tests for DeliberationAgent, deliberation_prompt, and consensus integration."""

from __future__ import annotations

import json

import pytest

from src.agents.deliberation_agent import (
    DeliberationAgent,
    DeliberationParseError,
    _PARSE_FAIL_NOTE,
)
from src.llm.base_client import LLMClientError
from src.llm.mock_client import MockLLMClient
from src.prompts.deliberation_prompt import build_user_prompt
from src.state.schema import (
    AgentOutput,
    DeliberationOutput,
    ModelOutput,
    PipelineState,
    StateMetadata,
    TaskConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABELS = ["tech", "sports", "finance"]
DESCRIPTIONS = {
    "tech": "Technology and software.",
    "sports": "Competitive sports.",
    "finance": "Financial markets.",
}


def make_state(
    text: str = "test input text",
    labels: list[str] | None = None,
    lexical_label: str | None = None,
    contextual_label: str | None = None,
    logic_label: str | None = None,
) -> PipelineState:
    """Construct a minimal PipelineState with optional pre-filled agent outputs."""
    state = PipelineState(
        metadata=StateMetadata(sample_id="delib-001"),
        input_text=text,
        task_config=TaskConfig(
            task_name="topic_classification",
            labels=labels if labels is not None else list(LABELS),
            label_descriptions=DESCRIPTIONS,
        ),
    )
    if lexical_label is not None:
        state.lexical_output = AgentOutput(
            agent_name="LexicalAgent",
            model_output=ModelOutput(label=lexical_label, confidence=0.70),
            notes="lexical note",
        )
    if contextual_label is not None:
        state.contextual_output = AgentOutput(
            agent_name="ContextualAgent",
            model_output=ModelOutput(label=contextual_label, confidence=0.80),
            notes="contextual note",
        )
    if logic_label is not None:
        state.logic_output = AgentOutput(
            agent_name="LogicAgent",
            model_output=ModelOutput(label=logic_label, confidence=0.75),
            notes="logic note",
        )
    return state


def valid_response(
    label: str = "tech",
    confidence: float = 0.82,
    mode: str = "recommendation",
) -> str:
    """Return a valid JSON string that DeliberationAgent can parse."""
    return json.dumps(
        {
            "recommended_label": label,
            "confidence": confidence,
            "justification": "Two of three agents agree on this label.",
            "mode": mode,
        }
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_deliberation_output_written_to_state(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response("tech"))
        agent = DeliberationAgent(client)
        state = make_state(
            lexical_label="tech", contextual_label="tech", logic_label="sports"
        )
        state = agent.run(state)
        assert state.deliberation_output is not None

    def test_correct_label_stored(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response("sports", 0.75))
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="sports", contextual_label="tech")
        state = agent.run(state)
        assert state.deliberation_output.recommended_label == "sports"

    def test_correct_confidence_stored(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response("tech", 0.91))
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        state = agent.run(state)
        assert state.deliberation_output.confidence == pytest.approx(0.91)

    def test_justification_stored(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response("tech"))
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        state = agent.run(state)
        assert state.deliberation_output.justification != ""

    def test_mode_recommendation_stored(self):
        client = MockLLMClient(
            mode="fixed", fixed_response=valid_response("tech", mode="recommendation")
        )
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        state = agent.run(state)
        assert state.deliberation_output.mode == "recommendation"

    def test_mode_justification_stored(self):
        client = MockLLMClient(
            mode="fixed", fixed_response=valid_response("tech", mode="justification")
        )
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        state = agent.run(state)
        assert state.deliberation_output.mode == "justification"

    def test_history_event_appended(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response("tech"))
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        initial_history_len = len(state.history)
        state = agent.run(state)
        assert len(state.history) == initial_history_len + 1

    def test_history_component_name(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response("tech"))
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        state = agent.run(state)
        assert state.history[-1].component == "DeliberationAgent"

    def test_history_outputs_contain_label(self):
        client = MockLLMClient(mode="fixed", fixed_response=valid_response("finance"))
        agent = DeliberationAgent(client)
        state = make_state(contextual_label="finance")
        state = agent.run(state)
        assert state.history[-1].outputs["recommended_label"] == "finance"

    def test_markdown_fences_stripped(self):
        raw = "```json\n" + valid_response("tech") + "\n```"
        client = MockLLMClient(mode="fixed", fixed_response=raw)
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        state = agent.run(state)
        assert state.deliberation_output is not None
        assert state.deliberation_output.recommended_label == "tech"


# ---------------------------------------------------------------------------
# Parse error tests
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_invalid_json_leaves_no_output(self):
        client = MockLLMClient(mode="fixed", fixed_response="not-json!")
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        state = agent.run(state)
        assert state.deliberation_output is None

    def test_missing_key_leaves_no_output(self):
        resp = json.dumps(
            {"recommended_label": "tech", "confidence": 0.8}
        )  # missing justification + mode
        client = MockLLMClient(mode="fixed", fixed_response=resp)
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        state = agent.run(state)
        assert state.deliberation_output is None

    def test_invalid_label_leaves_no_output(self):
        resp = json.dumps(
            {
                "recommended_label": "not_a_real_label",
                "confidence": 0.8,
                "justification": "test",
                "mode": "recommendation",
            }
        )
        client = MockLLMClient(mode="fixed", fixed_response=resp)
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        state = agent.run(state)
        assert state.deliberation_output is None

    def test_confidence_above_one_leaves_no_output(self):
        resp = json.dumps(
            {
                "recommended_label": "tech",
                "confidence": 1.5,
                "justification": "test",
                "mode": "recommendation",
            }
        )
        client = MockLLMClient(mode="fixed", fixed_response=resp)
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        state = agent.run(state)
        assert state.deliberation_output is None

    def test_confidence_below_zero_leaves_no_output(self):
        resp = json.dumps(
            {
                "recommended_label": "tech",
                "confidence": -0.1,
                "justification": "test",
                "mode": "recommendation",
            }
        )
        client = MockLLMClient(mode="fixed", fixed_response=resp)
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        state = agent.run(state)
        assert state.deliberation_output is None

    def test_invalid_mode_leaves_no_output(self):
        resp = json.dumps(
            {
                "recommended_label": "tech",
                "confidence": 0.8,
                "justification": "test",
                "mode": "invented_mode",
            }
        )
        client = MockLLMClient(mode="fixed", fixed_response=resp)
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        state = agent.run(state)
        assert state.deliberation_output is None

    def test_parse_error_appends_history(self):
        client = MockLLMClient(mode="fixed", fixed_response="bad json")
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        initial_len = len(state.history)
        state = agent.run(state)
        assert len(state.history) == initial_len + 1
        assert state.history[-1].outputs.get("fallback") is True
        assert "error" in state.history[-1].outputs


# ---------------------------------------------------------------------------
# LLM error tests
# ---------------------------------------------------------------------------


class TestLLMError:
    def test_llm_error_propagates(self):
        client = MockLLMClient(mode="raise_on_call")
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        with pytest.raises(LLMClientError):
            agent.run(state)


# ---------------------------------------------------------------------------
# Vote collection / prompt content tests
# ---------------------------------------------------------------------------


class TestVoteCollection:
    def test_all_three_agent_slots_appear_in_prompt(self):
        call_log: list[str] = []
        client = MockLLMClient(
            mode="fixed",
            fixed_response=valid_response("tech"),
            call_log=call_log,
        )
        agent = DeliberationAgent(client)
        state = make_state(
            lexical_label="tech",
            contextual_label="sports",
            logic_label="tech",
        )
        agent.run(state)
        prompt = call_log[0]
        assert "lexical" in prompt
        assert "contextual" in prompt
        assert "logic" in prompt

    def test_missing_agent_slot_omitted_from_prompt(self):
        call_log: list[str] = []
        client = MockLLMClient(
            mode="fixed",
            fixed_response=valid_response("tech"),
            call_log=call_log,
        )
        agent = DeliberationAgent(client)
        # Only lexical is set; contextual and logic are absent.
        state = make_state(lexical_label="tech")
        agent.run(state)
        prompt = call_log[0]
        assert "lexical" in prompt
        assert "contextual" not in prompt
        assert "logic" not in prompt

    def test_no_agents_uses_fallback_text(self):
        call_log: list[str] = []
        client = MockLLMClient(
            mode="fixed",
            fixed_response=valid_response("tech"),
            call_log=call_log,
        )
        agent = DeliberationAgent(client)
        state = make_state()  # no specialist outputs at all
        agent.run(state)
        assert "no agent votes available" in call_log[0]

    def test_long_notes_truncated_in_prompt(self):
        call_log: list[str] = []
        client = MockLLMClient(
            mode="fixed",
            fixed_response=valid_response("tech"),
            call_log=call_log,
        )
        agent = DeliberationAgent(client)
        state = make_state(lexical_label="tech")
        state.lexical_output.notes = "word " * 200  # far exceeds 120 chars
        agent.run(state)
        assert "..." in call_log[0]


# ---------------------------------------------------------------------------
# Prompt template unit tests
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_labels_appear_in_prompt(self):
        prompt = build_user_prompt(
            task_name="topic",
            labels=["tech", "sports"],
            label_descriptions={"tech": "Technology.", "sports": "Sports."},
            agent_votes=[],
        )
        assert "tech" in prompt
        assert "sports" in prompt

    def test_task_name_in_prompt(self):
        prompt = build_user_prompt(
            task_name="my_special_task",
            labels=["tech"],
            label_descriptions={},
            agent_votes=[],
        )
        assert "my_special_task" in prompt

    def test_agent_votes_rendered(self):
        prompt = build_user_prompt(
            task_name="topic",
            labels=["tech", "sports"],
            label_descriptions={},
            agent_votes=[
                ("lexical", "tech", 0.8, "keyword hit"),
                ("contextual", "sports", 0.6, ""),
            ],
        )
        assert "lexical" in prompt
        assert "contextual" in prompt
        assert "0.800" in prompt

    def test_empty_votes_fallback_message(self):
        prompt = build_user_prompt(
            task_name="topic",
            labels=["tech"],
            label_descriptions={},
            agent_votes=[],
        )
        assert "no agent votes available" in prompt

    def test_none_label_rendered_as_na(self):
        prompt = build_user_prompt(
            task_name="topic",
            labels=["tech"],
            label_descriptions={},
            agent_votes=[("lexical", None, None, "")],
        )
        assert "n/a" in prompt


# ---------------------------------------------------------------------------
# Consensus integration tests
# ---------------------------------------------------------------------------


class TestConsensusDeliberationIntegration:
    """Verify ConsensusAgent correctly includes the deliberation vote."""

    def _make_consensus_state(
        self,
        lexical_label: str = "tech",
        lexical_conf: float = 0.7,
        delib_label: str | None = None,
        delib_conf: float | None = None,
    ) -> PipelineState:
        from src.state.schema import AgentOutput, DeliberationOutput, ModelOutput

        state = PipelineState(
            metadata=StateMetadata(sample_id="cons-delib-01"),
            input_text="test",
            task_config=TaskConfig(
                task_name="topic_classification",
                labels=list(LABELS),
                label_descriptions=DESCRIPTIONS,
            ),
        )
        state.lexical_output = AgentOutput(
            agent_name="LexicalAgent",
            model_output=ModelOutput(label=lexical_label, confidence=lexical_conf),
        )
        if delib_label is not None:
            state.deliberation_output = DeliberationOutput(
                recommended_label=delib_label,
                confidence=delib_conf,
                justification="test justification",
                mode="recommendation",
            )
        return state

    def test_deliberation_vote_shifts_result(self):
        """When deliberation disagrees with lexical but has high weight, it wins."""
        from src.agents.consensus_agent import ConsensusAgent

        # lexical: tech@0.7, deliberation: finance@0.95 with weight 3.0
        state = self._make_consensus_state(
            lexical_label="tech",
            lexical_conf=0.7,
            delib_label="finance",
            delib_conf=0.95,
        )
        agent = ConsensusAgent(
            weights={"lexical": 1.0, "contextual": 0.0, "logic": 0.0, "deliberation": 3.0}
        )
        state = agent.run(state)
        # deliberation contribution: 3.0 * 0.95 = 2.85 > lexical: 1.0 * 0.7 = 0.70
        assert state.consensus_output.label == "finance"

    def test_deliberation_weight_zero_ignores_output(self):
        """When deliberation weight is 0, its output has no effect."""
        from src.agents.consensus_agent import ConsensusAgent

        state = self._make_consensus_state(
            lexical_label="tech",
            lexical_conf=0.9,
            delib_label="finance",
            delib_conf=1.0,
        )
        agent = ConsensusAgent(
            weights={"lexical": 1.0, "deliberation": 0.0}
        )
        state = agent.run(state)
        assert state.consensus_output.label == "tech"

    def test_no_deliberation_output_skipped(self):
        """When state.deliberation_output is None, consensus ignores deliberation slot."""
        from src.agents.consensus_agent import ConsensusAgent

        state = self._make_consensus_state(
            lexical_label="tech",
            lexical_conf=0.8,
            # no deliberation_output
        )
        agent = ConsensusAgent(weights={"deliberation": 2.0})
        state = agent.run(state)
        # Should still produce a valid result based on lexical only.
        assert state.consensus_output is not None
        assert state.consensus_output.label == "tech"

    def test_deliberation_rationale_included(self):
        """Deliberation vote details appear in the consensus rationale string."""
        from src.agents.consensus_agent import ConsensusAgent

        state = self._make_consensus_state(
            lexical_label="tech",
            lexical_conf=0.7,
            delib_label="tech",
            delib_conf=0.9,
        )
        agent = ConsensusAgent(
            weights={"lexical": 1.0, "deliberation": 1.5}
        )
        state = agent.run(state)
        assert "deliberation" in state.consensus_output.rationale
