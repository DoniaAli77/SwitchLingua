"""Tests for LLMLexicalAgent, LLMLogicAgent, and their orchestrator routing.

Coverage
--------
* LLMLexicalAgent happy path: writes state.lexical_output with valid label
* LLMLexicalAgent invalid JSON: fallback applied, output written
* LLMLexicalAgent invalid label: fallback applied, valid label used
* LLMLexicalAgent history events appended
* LLMLogicAgent happy path: writes state.logic_output with valid label
* LLMLogicAgent invalid JSON: fallback applied, output written
* LLMLogicAgent invalid label: fallback applied, valid label used
* LLMLogicAgent history events appended
* paper_style does NOT call LLMLexicalAgent or LLMLogicAgent
* full_agentic uses LLMLexicalAgent and LLMLogicAgent when provided
* full_agentic falls back to LexicalAgent when llm_lexical_agent is None
* full_agentic falls back to LogicAgent when llm_logic_agent is None
* all outputs use valid task labels
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from src.agents.llm_lexical_agent import LLMLexicalAgent, LLMLexicalParseError
from src.agents.llm_logic_agent import LLMLogicAgent, LLMLogicParseError
from src.llm.base_client import LLMClient, LLMClientError
from src.llm.mock_client import MockLLMClient
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.router import Router
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    StateMetadata,
    TaskConfig,
)

# ---------------------------------------------------------------------------
# Shared labels / helpers
# ---------------------------------------------------------------------------

_LABELS = ["tech", "sports", "health"]
_DESCS = {
    "tech": "software apps AI programming",
    "sports": "match team player score",
    "health": "exercise diet fitness vitamins",
}


def _make_state(
    pipeline_mode: str = "full_agentic",
    *,
    input_text: str = "The software app crashed after the update",
    threshold: float = 0.99,
) -> PipelineState:
    return PipelineState(
        metadata=StateMetadata(sample_id="test_llm"),
        input_text=input_text,
        task_config=TaskConfig(
            task_name="topic",
            labels=_LABELS,
            label_descriptions=_DESCS,
            threshold=threshold,
            pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
        ),
    )


def _valid_json(label: str = "tech", confidence: float = 0.88) -> str:
    return json.dumps({
        "label": label,
        "confidence": confidence,
        "reasoning": "The text mentions software and apps.",
        "evidence": ["software", "app", "update"],
    })


# ---------------------------------------------------------------------------
# Stub LLMClient that returns a fixed string
# ---------------------------------------------------------------------------

class _FixedClient(LLMClient):
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response


class _RaisingClient(LLMClient):
    def generate(self, prompt: str) -> str:
        raise LLMClientError("network error")


# ---------------------------------------------------------------------------
# Stand-in agents for orchestrator routing tests
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _RecorderAgent:
    name: str
    calls: List[str]

    def run(self, state: PipelineState) -> PipelineState:
        self.calls.append(self.name)
        return state


@dataclass(slots=True)
class _PrimaryRecorder:
    calls: List[str]
    confidence: float = 0.05  # below threshold → always escalates

    def run(self, state: PipelineState) -> PipelineState:
        self.calls.append("primary")
        labels = state.task_config.labels
        n = len(labels)
        rest = round((1.0 - self.confidence) / max(n - 1, 1), 6)
        probs = {lbl: self.confidence if lbl == labels[0] else rest for lbl in labels}
        state.primary_model_output = ModelOutput(
            label=labels[0],
            confidence=self.confidence,
            probabilities=probs,
            raw_text=state.input_text,
        )
        return state


@dataclass(slots=True)
class _RouterRecorder:
    calls: List[str]

    def run(self, state: PipelineState) -> PipelineState:
        self.calls.append("router")
        return Router().run(state)


def _build_orch(
    calls: List[str],
    *,
    llm_lexical_agent=None,
    llm_logic_agent=None,
    paper_contextual_agent=None,
) -> PipelineOrchestrator:
    return PipelineOrchestrator(
        primary_classifier=_PrimaryRecorder(calls=calls),
        router=_RouterRecorder(calls=calls),
        lexical_agent=_RecorderAgent("lexical", calls),
        contextual_agent=_RecorderAgent("contextual", calls),
        logic_agent=_RecorderAgent("logic", calls),
        consensus_agent=_RecorderAgent("consensus", calls),
        explainability_agent=_RecorderAgent("explainability", calls),
        llm_lexical_agent=llm_lexical_agent,
        llm_logic_agent=llm_logic_agent,
        paper_contextual_agent=paper_contextual_agent,
    )


# ===========================================================================
# LLMLexicalAgent unit tests
# ===========================================================================

class TestLLMLexicalAgent:

    def _agent(self, response: str = "") -> LLMLexicalAgent:
        return LLMLexicalAgent(llm_client=_FixedClient(response or _valid_json("tech")))

    def test_happy_path_writes_lexical_output(self) -> None:
        agent = self._agent(_valid_json("tech", 0.88))
        state = _make_state()
        state = agent.run(state)
        assert state.lexical_output is not None

    def test_happy_path_valid_label(self) -> None:
        agent = self._agent(_valid_json("sports", 0.75))
        state = _make_state()
        state = agent.run(state)
        assert state.lexical_output.model_output.label == "sports"

    def test_happy_path_confidence_stored(self) -> None:
        agent = self._agent(_valid_json("health", 0.72))
        state = _make_state()
        state = agent.run(state)
        assert abs(state.lexical_output.model_output.confidence - 0.72) < 1e-9

    def test_happy_path_probabilities_sum_to_one(self) -> None:
        agent = self._agent(_valid_json("tech", 0.9))
        state = _make_state()
        state = agent.run(state)
        probs = state.lexical_output.model_output.probabilities
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_happy_path_all_labels_in_probabilities(self) -> None:
        agent = self._agent(_valid_json("tech", 0.8))
        state = _make_state()
        state = agent.run(state)
        assert set(state.lexical_output.model_output.probabilities.keys()) == set(_LABELS)

    def test_happy_path_evidence_in_features(self) -> None:
        agent = self._agent(_valid_json("tech", 0.8))
        state = _make_state()
        state = agent.run(state)
        assert "evidence" in state.lexical_output.features
        assert isinstance(state.lexical_output.features["evidence"], list)

    def test_happy_path_agent_name(self) -> None:
        agent = self._agent()
        assert agent.name == "LLMLexicalAgent"

    def test_happy_path_history_appended(self) -> None:
        agent = self._agent(_valid_json("tech"))
        state = _make_state()
        state = agent.run(state)
        names = [e.component for e in state.history]
        assert "LLMLexicalAgent" in names

    def test_happy_path_history_not_fallback(self) -> None:
        agent = self._agent(_valid_json("tech"))
        state = _make_state()
        state = agent.run(state)
        event = next(e for e in state.history if e.component == "LLMLexicalAgent")
        assert event.outputs.get("fallback") is False

    def test_invalid_json_triggers_abstain(self) -> None:
        agent = LLMLexicalAgent(llm_client=_FixedClient("not json at all"))
        state = _make_state()
        state = agent.run(state)
        assert state.lexical_output is not None
        assert state.lexical_output.features.get("abstained") is True

    def test_invalid_json_abstains_label_none(self) -> None:
        agent = LLMLexicalAgent(llm_client=_FixedClient("not json at all"))
        state = _make_state()
        state = agent.run(state)
        # Abstain, not labels[0]:
        assert state.lexical_output.model_output.label is None
        assert state.lexical_output.model_output.probabilities == {}

    def test_invalid_label_triggers_abstain(self) -> None:
        bad = json.dumps({
            "label": "UNKNOWN_LABEL",
            "confidence": 0.9,
            "reasoning": "test",
            "evidence": [],
        })
        agent = LLMLexicalAgent(llm_client=_FixedClient(bad))
        state = _make_state()
        state = agent.run(state)
        assert state.lexical_output.features.get("abstained") is True

    def test_invalid_label_abstains_label_none(self) -> None:
        bad = json.dumps({
            "label": "INVENTED",
            "confidence": 0.9,
            "reasoning": "test",
            "evidence": [],
        })
        agent = LLMLexicalAgent(llm_client=_FixedClient(bad))
        state = _make_state()
        state = agent.run(state)
        assert state.lexical_output.model_output.label is None  # NOT labels[0]

    def test_missing_key_triggers_abstain(self) -> None:
        incomplete = json.dumps({"label": "tech", "confidence": 0.8})
        agent = LLMLexicalAgent(llm_client=_FixedClient(incomplete))
        state = _make_state()
        state = agent.run(state)
        assert state.lexical_output.features.get("abstained") is True

    def test_llm_client_error_propagates(self) -> None:
        agent = LLMLexicalAgent(llm_client=_RaisingClient())
        state = _make_state()
        with pytest.raises(LLMClientError):
            agent.run(state)

    def test_label_echo_client_integration(self) -> None:
        """MockLLMClient in label_echo mode picks a valid label from the prompt."""
        agent = LLMLexicalAgent(
            llm_client=MockLLMClient(mode="label_echo", allowed_labels=_LABELS)
        )
        state = _make_state()
        state = agent.run(state)
        assert state.lexical_output.model_output.label in _LABELS


# ===========================================================================
# LLMLogicAgent unit tests
# ===========================================================================

class TestLLMLogicAgent:

    def _agent(self, response: str = "") -> LLMLogicAgent:
        return LLMLogicAgent(llm_client=_FixedClient(response or _valid_json("tech")))

    def test_happy_path_writes_logic_output(self) -> None:
        agent = self._agent(_valid_json("tech", 0.85))
        state = _make_state()
        state = agent.run(state)
        assert state.logic_output is not None

    def test_happy_path_valid_label(self) -> None:
        agent = self._agent(_valid_json("sports", 0.78))
        state = _make_state()
        state = agent.run(state)
        assert state.logic_output.model_output.label == "sports"

    def test_happy_path_confidence_stored(self) -> None:
        agent = self._agent(_valid_json("health", 0.65))
        state = _make_state()
        state = agent.run(state)
        assert abs(state.logic_output.model_output.confidence - 0.65) < 1e-9

    def test_happy_path_probabilities_sum_to_one(self) -> None:
        agent = self._agent(_valid_json("tech", 0.9))
        state = _make_state()
        state = agent.run(state)
        probs = state.logic_output.model_output.probabilities
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_happy_path_all_labels_in_probabilities(self) -> None:
        agent = self._agent(_valid_json("health", 0.7))
        state = _make_state()
        state = agent.run(state)
        assert set(state.logic_output.model_output.probabilities.keys()) == set(_LABELS)

    def test_happy_path_agent_name(self) -> None:
        agent = self._agent()
        assert agent.name == "LLMLogicAgent"

    def test_happy_path_history_appended(self) -> None:
        agent = self._agent(_valid_json("tech"))
        state = _make_state()
        state = agent.run(state)
        names = [e.component for e in state.history]
        assert "LLMLogicAgent" in names

    def test_happy_path_history_not_fallback(self) -> None:
        agent = self._agent(_valid_json("sports"))
        state = _make_state()
        state = agent.run(state)
        event = next(e for e in state.history if e.component == "LLMLogicAgent")
        assert event.outputs.get("fallback") is False

    def test_invalid_json_triggers_abstain(self) -> None:
        agent = LLMLogicAgent(llm_client=_FixedClient("{broken"))
        state = _make_state()
        state = agent.run(state)
        assert state.logic_output.features.get("abstained") is True

    def test_invalid_json_abstains_label_none(self) -> None:
        agent = LLMLogicAgent(llm_client=_FixedClient("{broken"))
        state = _make_state()
        state = agent.run(state)
        assert state.logic_output.model_output.label is None
        assert state.logic_output.model_output.probabilities == {}

    def test_invalid_label_triggers_abstain(self) -> None:
        bad = json.dumps({
            "label": "BADLABEL",
            "confidence": 0.9,
            "reasoning": "test",
            "evidence": [],
        })
        agent = LLMLogicAgent(llm_client=_FixedClient(bad))
        state = _make_state()
        state = agent.run(state)
        assert state.logic_output.features.get("abstained") is True

    def test_invalid_label_abstains_label_none(self) -> None:
        bad = json.dumps({
            "label": "INVENTED",
            "confidence": 0.9,
            "reasoning": "test",
            "evidence": [],
        })
        agent = LLMLogicAgent(llm_client=_FixedClient(bad))
        state = _make_state()
        state = agent.run(state)
        assert state.logic_output.model_output.label is None  # NOT labels[0]

    def test_llm_client_error_propagates(self) -> None:
        agent = LLMLogicAgent(llm_client=_RaisingClient())
        state = _make_state()
        with pytest.raises(LLMClientError):
            agent.run(state)

    def test_label_echo_client_integration(self) -> None:
        agent = LLMLogicAgent(
            llm_client=MockLLMClient(mode="label_echo", allowed_labels=_LABELS)
        )
        state = _make_state()
        state = agent.run(state)
        assert state.logic_output.model_output.label in _LABELS


# ===========================================================================
# Orchestrator routing tests
# ===========================================================================

class TestOrchestratorLLMSpecialistRouting:
    """Verify paper_style vs full_agentic specialist agent routing."""

    # ------------------------------------------------------------------
    # paper_style must NOT call LLMLexicalAgent or LLMLogicAgent
    # ------------------------------------------------------------------

    def test_paper_style_does_not_call_llm_lexical(self) -> None:
        calls: List[str] = []
        llm_lex = _RecorderAgent("llm_lexical", calls)
        orch = _build_orch(calls, llm_lexical_agent=llm_lex)
        state = _make_state("paper_style")
        orch.run(state)
        assert "llm_lexical" not in calls

    def test_paper_style_does_not_call_llm_logic(self) -> None:
        calls: List[str] = []
        llm_log = _RecorderAgent("llm_logic", calls)
        orch = _build_orch(calls, llm_logic_agent=llm_log)
        state = _make_state("paper_style")
        orch.run(state)
        assert "llm_logic" not in calls

    def test_paper_style_calls_keyword_lexical(self) -> None:
        calls: List[str] = []
        orch = _build_orch(calls)
        state = _make_state("paper_style")
        orch.run(state)
        assert "lexical" in calls

    def test_paper_style_calls_regex_logic(self) -> None:
        calls: List[str] = []
        orch = _build_orch(calls)
        state = _make_state("paper_style")
        orch.run(state)
        assert "logic" in calls

    # ------------------------------------------------------------------
    # full_agentic uses LLM agents when provided
    # ------------------------------------------------------------------

    def test_full_agentic_calls_llm_lexical_when_provided(self) -> None:
        calls: List[str] = []
        llm_lex = _RecorderAgent("llm_lexical", calls)
        orch = _build_orch(calls, llm_lexical_agent=llm_lex)
        state = _make_state("full_agentic")
        orch.run(state)
        assert "llm_lexical" in calls

    def test_full_agentic_calls_llm_logic_when_provided(self) -> None:
        calls: List[str] = []
        llm_log = _RecorderAgent("llm_logic", calls)
        orch = _build_orch(calls, llm_logic_agent=llm_log)
        state = _make_state("full_agentic")
        orch.run(state)
        assert "llm_logic" in calls

    def test_full_agentic_skips_keyword_lexical_when_llm_provided(self) -> None:
        calls: List[str] = []
        llm_lex = _RecorderAgent("llm_lexical", calls)
        orch = _build_orch(calls, llm_lexical_agent=llm_lex)
        state = _make_state("full_agentic")
        orch.run(state)
        assert "lexical" not in calls

    def test_full_agentic_skips_regex_logic_when_llm_provided(self) -> None:
        calls: List[str] = []
        llm_log = _RecorderAgent("llm_logic", calls)
        orch = _build_orch(calls, llm_logic_agent=llm_log)
        state = _make_state("full_agentic")
        orch.run(state)
        assert "logic" not in calls

    # ------------------------------------------------------------------
    # full_agentic falls back when LLM agents are None
    # ------------------------------------------------------------------

    def test_full_agentic_fallback_to_keyword_lexical_when_none(self) -> None:
        calls: List[str] = []
        orch = _build_orch(calls, llm_lexical_agent=None)
        state = _make_state("full_agentic")
        orch.run(state)
        assert "lexical" in calls

    def test_full_agentic_fallback_to_regex_logic_when_none(self) -> None:
        calls: List[str] = []
        orch = _build_orch(calls, llm_logic_agent=None)
        state = _make_state("full_agentic")
        orch.run(state)
        assert "logic" in calls

    def test_full_agentic_always_calls_contextual(self) -> None:
        calls: List[str] = []
        orch = _build_orch(calls)
        state = _make_state("full_agentic")
        orch.run(state)
        assert "contextual" in calls

    # ------------------------------------------------------------------
    # Stage ordering for full_agentic with both LLM agents provided
    # ------------------------------------------------------------------

    def test_full_agentic_stage_order_with_both_llm_agents(self) -> None:
        calls: List[str] = []
        llm_lex = _RecorderAgent("llm_lexical", calls)
        llm_log = _RecorderAgent("llm_logic", calls)
        orch = _build_orch(calls, llm_lexical_agent=llm_lex, llm_logic_agent=llm_log)
        state = _make_state("full_agentic")
        orch.run(state)
        # Extract specialist stage calls only
        specialist_calls = [
            c for c in calls
            if c in {"llm_lexical", "llm_logic", "contextual", "consensus", "explainability"}
        ]
        assert specialist_calls == [
            "llm_lexical", "llm_logic", "contextual", "consensus", "explainability"
        ]
