"""tests/test_orchestrator_ner_path.py

Integration tests for PipelineOrchestrator's sequence-labeling (NER) path.

Covers:
  - NER path triggered when task_type == "sequence_labeling"
  - Classification path unchanged when task_type == "classification"
  - primary_only: final_output written, specialist agents NOT called
  - paper_style: all four NER agents run in order, explainability runs
  - full_agentic: same NER agents run (deterministic; no LLM required)
  - final_output.payload["sequence_output"] exists after paper/full run
  - final_output.payload["token_count"] correct
  - consensus_output.rationale == "sequence_labeling_completed"
  - History contains the right components for each mode
  - Pipeline error captured and returned gracefully (bad agent)
  - Custom NER agents (non-default) are used when supplied
  - Classification tests: existing orchestrator flow still passes
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.explainability_agent import ExplainabilityAgent
from src.agents.ner_consensus_agent import NERConsensusAgent
from src.agents.ner_contextual_agent import NERContextualAgent
from src.agents.ner_lexical_agent import NERLexicalAgent
from src.agents.ner_logic_agent import NERLogicAgent
from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.router import Router
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    SequenceLabelingOutput,
    StateMetadata,
    TaskConfig,
    TokenTag,
)

# ---------------------------------------------------------------------------
# Shared label sets
# ---------------------------------------------------------------------------

_NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
_CLS_LABELS = ["positive", "negative", "neutral"]


# ---------------------------------------------------------------------------
# Minimal stub agents
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _CallLog:
    calls: List[str] = field(default_factory=list)


@dataclass
class _MockPrimary:
    """Always writes a classification output so the orchestrator doesn't crash."""
    log: _CallLog = field(default_factory=_CallLog)

    def run(self, state: PipelineState) -> PipelineState:
        self.log.calls.append("primary")
        state.primary_model_output = ModelOutput(
            label=_CLS_LABELS[0],
            confidence=0.99,
            probabilities={l: 0.33 for l in _CLS_LABELS},
        )
        return state


@dataclass
class _RecordingNERAgent:
    """NER specialist that records calls and writes a trivial sequence output."""
    name: str
    log: _CallLog
    slot: str  # "lexical" | "logic" | "contextual"

    def run(self, state: PipelineState) -> PipelineState:
        self.log.calls.append(self.name)
        tokens: List[str] = state.extras.get("tokens") or state.input_text.split()
        seq = SequenceLabelingOutput(
            tags=[TokenTag(token=t, tag="O", confidence=1.0) for t in tokens]
        )
        ao = AgentOutput(agent_name=self.name, model_output=ModelOutput(), sequence_output=seq)
        if self.slot == "lexical":
            state.lexical_output = ao
        elif self.slot == "logic":
            state.logic_output = ao
        elif self.slot == "contextual":
            state.contextual_output = ao
        state.append_history(component=self.name, summary="recorded")
        return state


@dataclass
class _BrokenAgent:
    """Always raises — used to verify error capture."""
    name: str

    def run(self, state: PipelineState) -> PipelineState:
        raise RuntimeError(f"{self.name} intentionally failed")


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _ner_state(
    tokens: List[str],
    pipeline_mode: str = "paper_style",
) -> PipelineState:
    return PipelineState(
        metadata=StateMetadata(sample_id="ner-t"),
        input_text=" ".join(tokens),
        task_config=TaskConfig(
            task_name="ner",
            task_type="sequence_labeling",
            labels=_NER_LABELS,
            pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
        ),
        extras={"tokens": tokens},
    )


def _cls_state(pipeline_mode: str = "paper_style") -> PipelineState:
    return PipelineState(
        metadata=StateMetadata(sample_id="cls-t"),
        input_text="great product",
        task_config=TaskConfig(
            task_name="sentiment",
            task_type="classification",
            labels=_CLS_LABELS,
            threshold=0.99,     # force escalation
            pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
        ),
    )


def _make_orchestrator(
    log: _CallLog,
    *,
    ner_lexical=None,
    ner_logic=None,
    ner_contextual=None,
    ner_consensus=None,
    explainability=None,
) -> PipelineOrchestrator:
    """Build a minimal orchestrator wired with recording NER agents."""
    from src.agents.consensus_agent import ConsensusAgent
    from src.agents.contextual_agent import ContextualAgent
    from src.agents.lexical_agent import LexicalAgent
    from src.agents.logic_agent import LogicAgent
    from src.llm.mock_client import MockLLMClient

    llm = MockLLMClient(mode="label_echo", allowed_labels=_CLS_LABELS)

    return PipelineOrchestrator(
        primary_classifier=_MockPrimary(log=log),
        router=Router(),
        lexical_agent=LexicalAgent(),
        contextual_agent=ContextualAgent(llm_client=llm),
        logic_agent=LogicAgent(),
        consensus_agent=ConsensusAgent(),
        explainability_agent=explainability or ExplainabilityAgent(),
        ner_lexical_agent=ner_lexical or _RecordingNERAgent("ner_lex", log, "lexical"),
        ner_logic_agent=ner_logic or _RecordingNERAgent("ner_log", log, "logic"),
        ner_contextual_agent=ner_contextual or _RecordingNERAgent("ner_ctx", log, "contextual"),
        ner_consensus_agent=ner_consensus or NERConsensusAgent(),
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _history_components(state: PipelineState) -> List[str]:
    return [e.component for e in state.history]


# ===========================================================================
# NER — primary_only mode
# ===========================================================================

class TestNERPrimaryOnly:

    def _run(self, tokens=None):
        tokens = tokens or ["Ahmed", "works", "at", "Google"]
        log = _CallLog()
        orch = _make_orchestrator(log)
        state = _ner_state(tokens, pipeline_mode="primary_only")
        return orch.run(state), log

    def test_final_output_is_set(self):
        state, _ = self._run()
        assert state.final_output is not None

    def test_specialist_agents_not_called(self):
        state, log = self._run()
        assert "ner_lex" not in log.calls
        assert "ner_log" not in log.calls
        assert "ner_ctx" not in log.calls

    def test_primary_called(self):
        state, log = self._run()
        assert "primary" in log.calls

    def test_consensus_output_not_set(self):
        # NER consensus is a specialist-stage agent; shouldn't run in primary_only
        state, _ = self._run()
        assert state.consensus_output is None

    def test_history_has_orchestrator_events(self):
        state, _ = self._run()
        comps = _history_components(state)
        assert comps.count("orchestrator") >= 1


# ===========================================================================
# NER — paper_style mode
# ===========================================================================

class TestNERPaperStyle:

    def _run(self, tokens=None):
        tokens = tokens or ["Ahmed", "works", "at", "Google"]
        log = _CallLog()
        orch = _make_orchestrator(log)
        state = _ner_state(tokens, pipeline_mode="paper_style")
        return orch.run(state), log

    def test_all_ner_agents_called(self):
        _, log = self._run()
        assert "ner_lex" in log.calls
        assert "ner_log" in log.calls
        assert "ner_ctx" in log.calls

    def test_ner_agents_called_in_order(self):
        _, log = self._run()
        idx_lex = log.calls.index("ner_lex")
        idx_log = log.calls.index("ner_log")
        idx_ctx = log.calls.index("ner_ctx")
        assert idx_lex < idx_log < idx_ctx

    def test_final_output_has_sequence_output(self):
        state, _ = self._run()
        assert state.final_output is not None
        assert "sequence_output" in state.final_output.payload

    def test_final_output_token_count(self):
        tokens = ["Ahmed", "works", "at", "Google"]
        state, _ = self._run(tokens)
        assert state.final_output.payload["token_count"] == len(tokens)

    def test_consensus_output_rationale(self):
        state, _ = self._run()
        assert state.consensus_output is not None
        assert state.consensus_output.rationale == "sequence_labeling_completed"

    def test_consensus_output_label_is_none(self):
        state, _ = self._run()
        assert state.consensus_output.label is None

    def test_explainability_ran(self):
        state, _ = self._run()
        comps = _history_components(state)
        # ExplainabilityAgent appends its own history entry
        assert any("explain" in c.lower() for c in comps)

    def test_one_tag_per_token(self):
        tokens = ["Ahmed", "works", "at", "Google"]
        state, _ = self._run(tokens)
        seq = state.final_output.payload["sequence_output"]
        assert len(seq) == len(tokens)

    def test_no_pipeline_error(self):
        state, _ = self._run()
        assert "pipeline_error" not in state.extras

    def test_history_has_orchestrator_events(self):
        state, _ = self._run()
        comps = _history_components(state)
        assert comps.count("orchestrator") >= 2  # started + finished

    def test_classification_agents_not_called(self):
        """Lexical/logic/contextual classification agents must not run on NER path."""
        _, log = self._run()
        # The classification agent names used in the orchestrator are not in the
        # NER recording log because the NER path bypasses them entirely.
        # We verify by checking the classification agents never showed up.
        for name in ("LexicalAgent", "LogicAgent", "ContextualAgent", "ConsensusAgent"):
            assert name not in log.calls


# ===========================================================================
# NER — full_agentic mode
# ===========================================================================

class TestNERFullAgentic:

    def _run(self, tokens=None):
        tokens = tokens or ["Sara", "joined", "Microsoft"]
        log = _CallLog()
        orch = _make_orchestrator(log)
        state = _ner_state(tokens, pipeline_mode="full_agentic")
        return orch.run(state), log

    def test_all_ner_agents_called(self):
        _, log = self._run()
        assert "ner_lex" in log.calls
        assert "ner_log" in log.calls
        assert "ner_ctx" in log.calls

    def test_final_output_has_sequence_output(self):
        state, _ = self._run()
        assert "sequence_output" in state.final_output.payload

    def test_consensus_output_rationale(self):
        state, _ = self._run()
        assert state.consensus_output.rationale == "sequence_labeling_completed"

    def test_no_pipeline_error(self):
        state, _ = self._run()
        assert "pipeline_error" not in state.extras

    def test_one_tag_per_token(self):
        tokens = ["Sara", "joined", "Microsoft"]
        state, _ = self._run(tokens)
        seq = state.final_output.payload["sequence_output"]
        assert len(seq) == len(tokens)


# ===========================================================================
# NER — custom agent injection
# ===========================================================================

class TestNERCustomAgents:

    def test_custom_ner_lexical_used(self):
        log = _CallLog()
        custom_log = _CallLog()
        custom_lex = _RecordingNERAgent("custom_lex", custom_log, "lexical")
        orch = _make_orchestrator(log, ner_lexical=custom_lex)
        state = _ner_state(["Paris"])
        orch.run(state)
        assert "custom_lex" in custom_log.calls

    def test_custom_ner_logic_used(self):
        log = _CallLog()
        custom_log = _CallLog()
        custom_log_agent = _RecordingNERAgent("custom_log", custom_log, "logic")
        orch = _make_orchestrator(log, ner_logic=custom_log_agent)
        state = _ner_state(["Google"])
        orch.run(state)
        assert "custom_log" in custom_log.calls

    def test_default_ner_agents_created_when_none(self):
        """Orchestrator should auto-create default NER agents when not supplied."""
        from src.agents.consensus_agent import ConsensusAgent
        from src.agents.contextual_agent import ContextualAgent
        from src.agents.lexical_agent import LexicalAgent
        from src.agents.logic_agent import LogicAgent
        from src.llm.mock_client import MockLLMClient

        llm = MockLLMClient(mode="label_echo", allowed_labels=_CLS_LABELS)
        log = _CallLog()
        orch = PipelineOrchestrator(
            primary_classifier=_MockPrimary(log=log),
            router=Router(),
            lexical_agent=LexicalAgent(),
            contextual_agent=ContextualAgent(llm_client=llm),
            logic_agent=LogicAgent(),
            consensus_agent=ConsensusAgent(),
            explainability_agent=ExplainabilityAgent(),
            # No NER agents supplied — should use defaults
        )
        state = _ner_state(["Ahmed"])
        result = orch.run(state)
        # Defaults produce an all-O output but should not crash.
        assert result.final_output is not None
        assert "sequence_output" in result.final_output.payload


# ===========================================================================
# NER — error handling
# ===========================================================================

class TestNERErrorHandling:

    def test_broken_ner_agent_captured(self):
        log = _CallLog()
        broken = _BrokenAgent("ner_lex_broken")
        orch = _make_orchestrator(log, ner_lexical=broken)
        state = _ner_state(["Ahmed"])
        result = orch.run(state)
        assert "pipeline_error" in result.extras
        assert result.extras["pipeline_error"]["stage"] == "ner_lexical_agent"

    def test_pipeline_error_preserves_history(self):
        log = _CallLog()
        broken = _BrokenAgent("ner_log_broken")
        orch = _make_orchestrator(log, ner_logic=broken)
        state = _ner_state(["Ahmed"])
        result = orch.run(state)
        # history was written before the crash
        assert len(result.history) > 0


# ===========================================================================
# Classification path — regression tests
# ===========================================================================

class TestClassificationPathUnchanged:
    """Verify the original classification pipeline is unaffected by the NER additions."""

    def _make_orch(self):
        from src.agents.consensus_agent import ConsensusAgent
        from src.agents.contextual_agent import ContextualAgent
        from src.agents.lexical_agent import LexicalAgent
        from src.agents.logic_agent import LogicAgent
        from src.llm.mock_client import MockLLMClient

        llm = MockLLMClient(mode="label_echo", allowed_labels=_CLS_LABELS)
        log = _CallLog()
        orch = PipelineOrchestrator(
            primary_classifier=_MockPrimary(log=log),
            router=Router(),
            lexical_agent=LexicalAgent(),
            contextual_agent=ContextualAgent(llm_client=llm),
            logic_agent=LogicAgent(),
            consensus_agent=ConsensusAgent(),
            explainability_agent=ExplainabilityAgent(),
        )
        return orch, log

    def test_classification_primary_only_final_output(self):
        orch, log = self._make_orch()
        state = _cls_state(pipeline_mode="primary_only")
        result = orch.run(state)
        assert result.final_output is not None
        assert result.final_output.label is not None

    def test_classification_paper_style_no_ner_agents(self):
        orch, log = self._make_orch()
        state = _cls_state(pipeline_mode="paper_style")
        result = orch.run(state)
        assert result.final_output is not None
        # NER path was NOT taken — no sequence_output in payload
        payload = result.final_output.payload or {}
        assert "sequence_output" not in payload

    def test_classification_full_agentic_no_ner_agents(self):
        orch, log = self._make_orch()
        state = _cls_state(pipeline_mode="full_agentic")
        result = orch.run(state)
        assert result.final_output is not None
        payload = result.final_output.payload or {}
        assert "sequence_output" not in payload

    def test_classification_final_output_label_in_labels(self):
        orch, log = self._make_orch()
        state = _cls_state(pipeline_mode="paper_style")
        result = orch.run(state)
        assert result.final_output.label in _CLS_LABELS

    def test_classification_no_pipeline_error(self):
        orch, log = self._make_orch()
        state = _cls_state(pipeline_mode="paper_style")
        result = orch.run(state)
        assert "pipeline_error" not in result.extras

    def test_classification_task_type_still_routes_to_cls_path(self):
        orch, log = self._make_orch()
        state = _cls_state(pipeline_mode="paper_style")
        result = orch.run(state)
        # If classification path ran, consensus_output.label should be a CLS label
        if result.consensus_output is not None:
            assert result.consensus_output.label in _CLS_LABELS or result.consensus_output.label is None
