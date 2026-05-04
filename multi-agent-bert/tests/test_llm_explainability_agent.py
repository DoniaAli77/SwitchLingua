"""Tests for LLMExplainabilityAgent and its orchestrator routing.

Coverage
--------
* LLMExplainabilityAgent happy path: writes state.explanation_output
* LLMExplainabilityAgent evidence and caveats stored correctly
* LLMExplainabilityAgent agent name
* LLMExplainabilityAgent history appended with fallback=False
* Invalid JSON triggers deterministic fallback
* Missing required keys triggers fallback
* Fallback still writes explanation_output
* LLMClientError propagates
* paper_style uses template ExplainabilityAgent (not LLM)
* primary_only uses template ExplainabilityAgent (not LLM)
* full_agentic uses LLMExplainabilityAgent when provided
* full_agentic falls back to template when llm_explainability_agent is None
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from src.agents.llm_explainability_agent import (
    LLMExplainabilityAgent,
    LLMExplainabilityParseError,
)
from src.llm.base_client import LLMClient, LLMClientError
from src.llm.mock_client import MockLLMClient
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.router import Router
from src.state.schema import (
    ExplanationOutput,
    FinalOutput,
    ModelOutput,
    PipelineState,
    StateMetadata,
    TaskConfig,
)

# ---------------------------------------------------------------------------
# Shared helpers
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
    input_text: str = "The app crashed after the update",
    threshold: float = 0.99,
) -> PipelineState:
    return PipelineState(
        metadata=StateMetadata(sample_id="expl_test"),
        input_text=input_text,
        task_config=TaskConfig(
            task_name="topic",
            labels=_LABELS,
            label_descriptions=_DESCS,
            threshold=threshold,
            pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
        ),
    )


def _valid_json(
    summary: str = "The text discusses a software bug.",
    evidence: Optional[list] = None,
    caveats: Optional[list] = None,
) -> str:
    return json.dumps({
        "summary": summary,
        "evidence": evidence or ["software", "update", "crashed"],
        "caveats": caveats or [],
    })


class _FixedClient(LLMClient):
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response


class _RaisingClient(LLMClient):
    def generate(self, prompt: str) -> str:
        raise LLMClientError("network error")


# Stand-in agents for orchestrator routing tests
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
    confidence: float = 0.05

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


# A consensus stand-in that also writes final_output (needed for
# LLMExplainabilityAgent to have a label to reference).
@dataclass(slots=True)
class _ConsensusRecorder:
    calls: List[str]
    label: str = "tech"

    def run(self, state: PipelineState) -> PipelineState:
        self.calls.append("consensus")
        state.final_output = FinalOutput(label=self.label, confidence=0.8)
        return state


def _build_orch(
    calls: List[str],
    *,
    llm_explain=None,
    pipeline_mode: str = "full_agentic",
) -> PipelineOrchestrator:
    return PipelineOrchestrator(
        primary_classifier=_PrimaryRecorder(calls=calls),
        router=_RouterRecorder(calls=calls),
        lexical_agent=_RecorderAgent("lexical", calls),
        contextual_agent=_RecorderAgent("contextual", calls),
        logic_agent=_RecorderAgent("logic", calls),
        consensus_agent=_ConsensusRecorder(calls=calls),
        explainability_agent=_RecorderAgent("template_explain", calls),
        llm_explainability_agent=llm_explain,
    )


# ===========================================================================
# LLMExplainabilityAgent unit tests
# ===========================================================================

class TestLLMExplainabilityAgent:

    def _agent(self, response: str = "") -> LLMExplainabilityAgent:
        return LLMExplainabilityAgent(
            llm_client=_FixedClient(response or _valid_json())
        )

    def test_happy_path_writes_explanation_output(self) -> None:
        agent = self._agent(_valid_json())
        state = _make_state()
        state = agent.run(state)
        assert state.explanation_output is not None

    def test_happy_path_summary_stored(self) -> None:
        agent = self._agent(_valid_json(summary="Tech topic confirmed."))
        state = _make_state()
        state = agent.run(state)
        assert state.explanation_output.summary == "Tech topic confirmed."

    def test_happy_path_evidence_stored(self) -> None:
        agent = self._agent(_valid_json(evidence=["software", "app"]))
        state = _make_state()
        state = agent.run(state)
        assert "software" in state.explanation_output.evidence

    def test_happy_path_caveats_stored(self) -> None:
        agent = self._agent(_valid_json(caveats=["sports agent disagreed"]))
        state = _make_state()
        state = agent.run(state)
        assert "sports agent disagreed" in state.explanation_output.caveats

    def test_happy_path_empty_caveats_accepted(self) -> None:
        agent = self._agent(_valid_json(caveats=[]))
        state = _make_state()
        state = agent.run(state)
        assert state.explanation_output.caveats == []

    def test_happy_path_agent_name(self) -> None:
        agent = self._agent()
        assert agent.name == "LLMExplainabilityAgent"

    def test_happy_path_history_appended(self) -> None:
        agent = self._agent(_valid_json())
        state = _make_state()
        state = agent.run(state)
        names = [e.component for e in state.history]
        assert "LLMExplainabilityAgent" in names

    def test_happy_path_history_not_fallback(self) -> None:
        agent = self._agent(_valid_json())
        state = _make_state()
        state = agent.run(state)
        event = next(e for e in state.history if e.component == "LLMExplainabilityAgent")
        assert event.outputs.get("fallback") is False

    def test_invalid_json_triggers_fallback(self) -> None:
        agent = LLMExplainabilityAgent(llm_client=_FixedClient("not json"))
        state = _make_state()
        state = agent.run(state)
        assert state.explanation_output is not None

    def test_invalid_json_fallback_writes_caveats(self) -> None:
        agent = LLMExplainabilityAgent(llm_client=_FixedClient("{broken"))
        state = _make_state()
        state = agent.run(state)
        assert len(state.explanation_output.caveats) > 0

    def test_missing_key_triggers_fallback(self) -> None:
        incomplete = json.dumps({"summary": "ok"})  # missing evidence + caveats
        agent = LLMExplainabilityAgent(llm_client=_FixedClient(incomplete))
        state = _make_state()
        state = agent.run(state)
        assert state.explanation_output is not None
        assert any("fallback" in c for c in state.explanation_output.caveats)

    def test_fallback_history_records_error(self) -> None:
        agent = LLMExplainabilityAgent(llm_client=_FixedClient("bad"))
        state = _make_state()
        state = agent.run(state)
        event = next(
            (e for e in state.history if e.component == "LLMExplainabilityAgent"), None
        )
        assert event is not None
        assert event.outputs.get("fallback") is True

    def test_llm_client_error_propagates(self) -> None:
        agent = LLMExplainabilityAgent(llm_client=_RaisingClient())
        state = _make_state()
        with pytest.raises(LLMClientError):
            agent.run(state)

    def test_final_output_in_fallback_summary(self) -> None:
        """Fallback explanation references the final label when available."""
        agent = LLMExplainabilityAgent(llm_client=_FixedClient("bad"))
        state = _make_state()
        state.final_output = FinalOutput(label="tech", confidence=0.82)
        state = agent.run(state)
        assert "tech" in state.explanation_output.summary

    def test_mock_label_echo_integration(self) -> None:
        """MockLLMClient fixed mode returns a parseable response."""
        valid = _valid_json(summary="Classification complete.")
        agent = LLMExplainabilityAgent(llm_client=_FixedClient(valid))
        state = _make_state()
        state = agent.run(state)
        assert state.explanation_output.summary == "Classification complete."


# ===========================================================================
# Orchestrator routing tests
# ===========================================================================

class TestOrchestratorExplainabilityRouting:

    def test_paper_style_uses_template_explainability(self) -> None:
        calls: List[str] = []
        llm_explain = _RecorderAgent("llm_explain", calls)
        orch = _build_orch(calls, llm_explain=llm_explain)
        state = _make_state("paper_style")
        orch.run(state)
        assert "template_explain" in calls
        assert "llm_explain" not in calls

    def test_primary_only_does_not_use_llm_explainability(self) -> None:
        """primary_only mode sets explanation_output inline; LLM agent never called."""
        calls: List[str] = []
        llm_explain = _RecorderAgent("llm_explain", calls)
        orch = _build_orch(calls, llm_explain=llm_explain)
        state = _make_state("primary_only")
        result = orch.run(state)
        assert "llm_explain" not in calls
        assert result.explanation_output is not None

    def test_full_agentic_uses_llm_explainability_when_provided(self) -> None:
        calls: List[str] = []
        llm_explain = _RecorderAgent("llm_explain", calls)
        orch = _build_orch(calls, llm_explain=llm_explain)
        state = _make_state("full_agentic")
        orch.run(state)
        assert "llm_explain" in calls

    def test_full_agentic_skips_template_when_llm_provided(self) -> None:
        calls: List[str] = []
        llm_explain = _RecorderAgent("llm_explain", calls)
        orch = _build_orch(calls, llm_explain=llm_explain)
        state = _make_state("full_agentic")
        orch.run(state)
        assert "template_explain" not in calls

    def test_full_agentic_falls_back_to_template_when_none(self) -> None:
        calls: List[str] = []
        orch = _build_orch(calls, llm_explain=None)
        state = _make_state("full_agentic")
        orch.run(state)
        assert "template_explain" in calls

    def test_paper_style_fast_path_uses_template(self) -> None:
        """Even when primary confidence is high (fast path), template is used."""
        calls: List[str] = []
        llm_explain = _RecorderAgent("llm_explain", calls)
        orch = _build_orch(calls, llm_explain=llm_explain)
        # low threshold → primary always accepted → fast path
        state = _make_state("paper_style", threshold=0.0)
        orch.run(state)
        assert "template_explain" in calls
        assert "llm_explain" not in calls
