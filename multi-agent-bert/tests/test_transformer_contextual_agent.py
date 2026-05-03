"""Tests for TransformerContextualAgent and its integration with PipelineOrchestrator.

Coverage
--------
* tfidf mode: valid label output, sensible ranking, output fields, edge cases
* embedding mode: graceful fallback when transformers/torch unavailable
* Orchestrator routing:
    - paper_style uses paper_contextual_agent when provided
    - paper_style does NOT call the LLM contextual_agent when paper_contextual_agent set
    - paper_style falls back to contextual_agent when paper_contextual_agent is None
    - full_agentic always uses contextual_agent, ignores paper_contextual_agent
    - both paths write state.contextual_output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from src.agents.transformer_contextual_agent import TransformerContextualAgent
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
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------

_TECH_LABELS = ["tech", "sports", "health"]
_TECH_DESCS = {
    "tech": "software technology programming computers apps",
    "sports": "football match team player game score",
    "health": "exercise diet fitness nutrition vitamins",
}
_TECH_INPUT = "The new software update fixed the bugs in the app"


def _make_topic_state(
    pipeline_mode: str = "paper_style",
    *,
    input_text: str = _TECH_INPUT,
    threshold: float = 0.99,  # always forces escalation
    labels: Optional[List[str]] = None,
    label_descriptions: Optional[dict] = None,
) -> PipelineState:
    return PipelineState(
        metadata=StateMetadata(sample_id="tc_test"),
        input_text=input_text,
        task_config=TaskConfig(
            task_name="topic",
            labels=labels or _TECH_LABELS,
            label_descriptions=label_descriptions or _TECH_DESCS,
            threshold=threshold,
            pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
        ),
    )


# ---------------------------------------------------------------------------
# Minimal stand-in agents for orchestrator tests
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _RecorderAgent:
    """Records calls without modifying state."""

    name: str
    calls: List[str]

    def run(self, state: PipelineState) -> PipelineState:
        self.calls.append(self.name)
        return state


@dataclass(slots=True)
class _PrimaryRecorder:
    """Sets primary_model_output with given confidence."""

    calls: List[str]
    confidence: float = 0.2  # always below threshold=0.99 → forces escalation

    def run(self, state: PipelineState) -> PipelineState:
        self.calls.append("primary")
        labels = state.task_config.labels
        n = len(labels)
        probs = {labels[0]: self.confidence}
        rest = round((1.0 - self.confidence) / max(n - 1, 1), 6)
        for lbl in labels[1:]:
            probs[lbl] = rest
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


class _LLMCallSpy:
    """Contextual-agent stand-in that records if run() was invoked."""

    def __init__(self, label: str = "tech") -> None:
        self.was_called = False
        self._label = label

    def run(self, state: PipelineState) -> PipelineState:
        self.was_called = True
        state.contextual_output = AgentOutput(
            agent_name="LLMCallSpy",
            model_output=ModelOutput(label=self._label, confidence=0.9),
        )
        return state


def _build_orch(
    calls: List[str],
    *,
    paper_contextual_agent=None,
    contextual_agent=None,
) -> PipelineOrchestrator:
    if contextual_agent is None:
        contextual_agent = _RecorderAgent("contextual", calls)
    return PipelineOrchestrator(
        primary_classifier=_PrimaryRecorder(calls=calls),
        router=_RouterRecorder(calls=calls),
        lexical_agent=_RecorderAgent("lexical", calls),
        contextual_agent=contextual_agent,
        logic_agent=_RecorderAgent("logic", calls),
        consensus_agent=_RecorderAgent("consensus", calls),
        explainability_agent=_RecorderAgent("explainability", calls),
        paper_contextual_agent=paper_contextual_agent,
    )


# ===========================================================================
# TransformerContextualAgent — tfidf mode
# ===========================================================================

class TestTransformerContextualAgentTfidf:
    """Unit tests for the default tfidf operating mode."""

    def _agent(self) -> TransformerContextualAgent:
        return TransformerContextualAgent(mode="tfidf")

    def _state(self, **kwargs) -> PipelineState:
        return _make_topic_state(**kwargs)

    # --- basic correctness ------------------------------------------------

    def test_tfidf_returns_valid_label(self):
        out = self._agent().run(self._state())
        assert out.contextual_output is not None
        assert out.contextual_output.model_output.label in _TECH_LABELS

    def test_tfidf_tech_sentence_ranks_tech_first(self):
        """Input dominated by software/app vocabulary must prefer 'tech'."""
        out = self._agent().run(self._state())
        assert out.contextual_output.model_output.label == "tech"

    def test_tfidf_writes_contextual_output(self):
        state = self._state()
        assert state.contextual_output is None
        out = self._agent().run(state)
        assert isinstance(out.contextual_output, AgentOutput)

    # --- numeric invariants ------------------------------------------------

    def test_confidence_in_unit_interval(self):
        out = self._agent().run(self._state())
        conf = out.contextual_output.model_output.confidence
        assert conf is not None
        assert 0.0 <= conf <= 1.0

    def test_probabilities_sum_to_one(self):
        out = self._agent().run(self._state())
        probs = out.contextual_output.model_output.probabilities
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_all_labels_present_in_probabilities(self):
        out = self._agent().run(self._state())
        probs = out.contextual_output.model_output.probabilities
        assert set(probs.keys()) == set(_TECH_LABELS)

    # --- features & notes --------------------------------------------------

    def test_features_contain_similarity_scores_for_all_labels(self):
        out = self._agent().run(self._state())
        features = out.contextual_output.features
        assert "similarity_scores" in features
        assert set(features["similarity_scores"].keys()) == set(_TECH_LABELS)

    def test_features_contain_effective_mode(self):
        out = self._agent().run(self._state())
        assert out.contextual_output.features["effective_mode"] == "tfidf"

    def test_notes_mention_tfidf_mode(self):
        out = self._agent().run(self._state())
        assert "tfidf" in out.contextual_output.notes

    def test_agent_name_set_correctly(self):
        out = self._agent().run(self._state())
        assert out.contextual_output.agent_name == "TransformerContextualAgent"

    # --- history -----------------------------------------------------------

    def test_appends_history_event(self):
        state = self._state()
        assert len(state.history) == 0
        out = self._agent().run(state)
        assert len(out.history) == 1

    def test_history_event_component_is_agent_name(self):
        out = self._agent().run(self._state())
        assert out.history[0].component == "TransformerContextualAgent"

    def test_history_event_outputs_contain_label_and_mode(self):
        out = self._agent().run(self._state())
        outputs = out.history[0].outputs
        assert "label" in outputs
        assert "effective_mode" in outputs
        assert outputs["effective_mode"] == "tfidf"

    def test_history_event_outputs_contain_similarity_scores(self):
        out = self._agent().run(self._state())
        outputs = out.history[0].outputs
        assert "similarity_scores" in outputs
        assert set(outputs["similarity_scores"].keys()) == set(_TECH_LABELS)

    def test_history_summary_mentions_non_finetuned(self):
        out = self._agent().run(self._state())
        assert "contextual similarity" in out.history[0].summary

    # --- does not call LLM -------------------------------------------------

    def test_does_not_require_llm_client(self):
        """TransformerContextualAgent must run without any LLMClient attribute."""
        agent = self._agent()
        assert not hasattr(agent, "llm_client"), (
            "TransformerContextualAgent must not hold an LLMClient reference"
        )

    def test_does_not_call_llm_when_transformers_unavailable(self):
        """Even in embedding mode, must never call an LLMClient."""
        agent = TransformerContextualAgent(mode="embedding")
        assert not hasattr(agent, "llm_client")

    # --- edge cases --------------------------------------------------------

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown mode"):
            TransformerContextualAgent(mode="bert_finetune")

    def test_uniform_fallback_when_zero_vocabulary_overlap(self):
        """All-unknown tokens → uniform distribution; result is still a valid label."""
        state = PipelineState(
            metadata=StateMetadata(sample_id="ov"),
            input_text="xyzzy qqqq aaaa bbbb 1234 zzzz",
            task_config=TaskConfig(
                task_name="topic",
                labels=["tech", "sports"],
                label_descriptions={
                    "tech": "software technology programming",
                    "sports": "football match team",
                },
            ),
        )
        out = self._agent().run(state)
        assert out.contextual_output.model_output.label in ["tech", "sports"]
        probs = out.contextual_output.model_output.probabilities
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_missing_description_falls_back_to_label_name(self):
        """Labels with no entry in label_descriptions use the label name as description."""
        state = PipelineState(
            metadata=StateMetadata(sample_id="nd"),
            input_text="playing football in the park",
            task_config=TaskConfig(
                task_name="topic",
                labels=["tech", "sports"],
                label_descriptions={},
            ),
        )
        out = self._agent().run(state)
        assert out.contextual_output is not None
        assert out.contextual_output.model_output.label in ["tech", "sports"]

    def test_single_label_returns_confidence_one(self):
        state = PipelineState(
            metadata=StateMetadata(sample_id="sl"),
            input_text="software bugs",
            task_config=TaskConfig(
                task_name="topic",
                labels=["tech"],
                label_descriptions={"tech": "software technology"},
            ),
        )
        out = self._agent().run(state)
        assert out.contextual_output.model_output.label == "tech"
        assert abs(out.contextual_output.model_output.confidence - 1.0) < 1e-4


# ===========================================================================
# TransformerContextualAgent — embedding mode
# ===========================================================================

class TestTransformerContextualAgentEmbedding:
    """embedding mode must always produce a valid result — tfidf fallback ensures this."""

    def _state(self) -> PipelineState:
        return _make_topic_state()

    def test_embedding_mode_does_not_crash(self):
        """Must succeed regardless of whether transformers/torch are installed."""
        agent = TransformerContextualAgent(mode="embedding")
        out = agent.run(self._state())
        assert out.contextual_output is not None
        assert out.contextual_output.model_output.label in _TECH_LABELS

    def test_embedding_mode_features_include_effective_mode(self):
        """features['effective_mode'] must be present; value is 'tfidf' or 'embedding'."""
        agent = TransformerContextualAgent(mode="embedding")
        out = agent.run(self._state())
        mode = out.contextual_output.features["effective_mode"]
        assert mode in ("tfidf", "embedding")

    def test_embedding_mode_writes_contextual_output(self):
        agent = TransformerContextualAgent(mode="embedding")
        out = agent.run(self._state())
        assert isinstance(out.contextual_output, AgentOutput)
        assert out.contextual_output.model_output.label in _TECH_LABELS

    def test_embedding_probabilities_sum_to_one(self):
        agent = TransformerContextualAgent(mode="embedding")
        out = agent.run(self._state())
        probs = out.contextual_output.model_output.probabilities
        assert abs(sum(probs.values()) - 1.0) < 1e-4


# ===========================================================================
# PipelineOrchestrator — paper_contextual_agent routing
# ===========================================================================

class TestOrchestratorPaperContextualAgentRouting:
    """Verify that PipelineOrchestrator correctly routes to paper_contextual_agent."""

    # --- paper_style + paper_contextual_agent provided --------------------

    def test_paper_style_calls_paper_contextual_agent(self):
        """paper_contextual_agent.run must be invoked in paper_style mode."""
        calls: List[str] = []
        paper_calls: List[str] = []
        paper_agent = _RecorderAgent("paper_contextual", paper_calls)
        llm_spy = _LLMCallSpy()

        orch = _build_orch(calls, paper_contextual_agent=paper_agent, contextual_agent=llm_spy)
        orch.run(_make_topic_state("paper_style"))

        assert "paper_contextual" in paper_calls

    def test_paper_style_does_not_call_llm_contextual(self):
        """LLM contextual_agent must NOT be invoked when paper_contextual_agent is set."""
        calls: List[str] = []
        paper_calls: List[str] = []
        paper_agent = _RecorderAgent("paper_contextual", paper_calls)
        llm_spy = _LLMCallSpy()

        orch = _build_orch(calls, paper_contextual_agent=paper_agent, contextual_agent=llm_spy)
        orch.run(_make_topic_state("paper_style"))

        assert not llm_spy.was_called

    # --- paper_style without paper_contextual_agent (backward compat) -----

    def test_paper_style_falls_back_to_contextual_agent_when_no_paper_agent(self):
        """When paper_contextual_agent=None, paper_style uses contextual_agent."""
        calls: List[str] = []
        orch = _build_orch(calls, paper_contextual_agent=None)
        orch.run(_make_topic_state("paper_style"))
        assert "contextual" in calls

    # --- full_agentic always uses contextual_agent ------------------------

    def test_full_agentic_uses_llm_contextual_agent_not_paper_agent(self):
        """full_agentic must use contextual_agent even when paper_contextual_agent is set."""
        calls: List[str] = []
        paper_calls: List[str] = []
        llm_spy = _LLMCallSpy()
        paper_agent = _RecorderAgent("paper_contextual", paper_calls)

        orch = _build_orch(calls, paper_contextual_agent=paper_agent, contextual_agent=llm_spy)
        orch.run(_make_topic_state("full_agentic"))

        assert llm_spy.was_called
        assert "paper_contextual" not in paper_calls

    # --- contextual_output written on both paths --------------------------

    def test_paper_style_transformer_agent_writes_contextual_output(self):
        """TransformerContextualAgent wired as paper_contextual_agent must write contextual_output."""
        calls: List[str] = []
        transformer_agent = TransformerContextualAgent(mode="tfidf")

        orch = _build_orch(calls, paper_contextual_agent=transformer_agent)
        out = orch.run(_make_topic_state("paper_style"))

        assert out.contextual_output is not None
        assert out.contextual_output.model_output.label in _TECH_LABELS

    def test_full_agentic_llm_spy_writes_contextual_output(self):
        """LLM contextual_agent in full_agentic must write contextual_output."""
        calls: List[str] = []
        llm_spy = _LLMCallSpy(label="tech")

        orch = _build_orch(calls, contextual_agent=llm_spy)
        out = orch.run(_make_topic_state("full_agentic"))

        assert out.contextual_output is not None
        assert out.contextual_output.model_output.label == "tech"

    # --- tfidf output label validity --------------------------------------

    def test_tfidf_label_is_always_in_task_labels(self):
        """TransformerContextualAgent must only return labels from task_config.labels."""
        agent = TransformerContextualAgent(mode="tfidf")
        for _ in range(5):  # test several short inputs
            for text in [
                "software update",
                "football goal",
                "diet nutrition",
                "xyzzy unknown",
                "شركة سوق",
            ]:
                state = PipelineState(
                    metadata=StateMetadata(sample_id="lv"),
                    input_text=text,
                    task_config=TaskConfig(
                        task_name="topic",
                        labels=_TECH_LABELS,
                        label_descriptions=_TECH_DESCS,
                    ),
                )
                out = agent.run(state)
                assert out.contextual_output.model_output.label in _TECH_LABELS, (
                    f"Unexpected label for input {text!r}: "
                    f"{out.contextual_output.model_output.label!r}"
                )
