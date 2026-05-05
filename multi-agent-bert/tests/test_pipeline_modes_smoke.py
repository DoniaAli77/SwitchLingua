"""Smoke tests for the three pipeline modes.

These tests verify pipeline *integrity* — that each mode:
  - produces the expected outputs
  - wires the correct concrete agent class for each role
  - does NOT invoke agents that belong to a different mode

Agent identity is verified via ``state.history`` component names, which each
agent writes using ``self.name`` (defaulting to ``self.__class__.__name__``).

Expected history component names
---------------------------------
  primary_classifier          → "primary_classifier"
  router                      → "router"
  LexicalAgent                → "LexicalAgent"
  LogicAgent                  → "LogicAgent"
  ContextualAgent (LLM)       → "ContextualAgent"
  TransformerContextualAgent  → "TransformerContextualAgent"
  ConsensusAgent              → "ConsensusAgent"
  ExplainabilityAgent         → "ExplainabilityAgent"
  LLMLexicalAgent             → "LLMLexicalAgent"
  LLMLogicAgent               → "LLMLogicAgent"
  LLMExplainabilityAgent      → "LLMExplainabilityAgent"
  DeliberationAgent           → "DeliberationAgent"

Labels used: business, education, sports, tech (topic classification).

A high routing threshold (0.99) ensures the mock primary classifier's
heuristic confidence always falls below it, forcing escalation in both
paper_style and full_agentic modes.

Accuracy is never tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_pipeline import build_orchestrator
from src.state.schema import PipelineState, StateMetadata, TaskConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_LABELS = ["business", "education", "sports", "tech"]
_DESCS = {
    "business": "companies markets products corporate strategy",
    "education": "learning schools universities courses exams",
    "sports": "match team player score goals coach",
    "tech": "software apps AI machine learning cloud",
}
_TEXT = "The new app uses machine learning to predict sports scores."


def _make_state(pipeline_mode: str, *, enable_deliberation: bool = False) -> PipelineState:
    return PipelineState(
        metadata=StateMetadata(sample_id=f"smoke_{pipeline_mode}"),
        input_text=_TEXT,
        task_config=TaskConfig(
            task_name="topic",
            labels=_LABELS,
            label_descriptions=_DESCS,
            threshold=0.99,          # high threshold → heuristic primary always escalates
            pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
            enable_deliberation=enable_deliberation,
        ),
    )


def _build(pipeline_mode: str, *, enable_deliberation: bool = False) -> PipelineState:
    """Build a fully-wired orchestrator, run it, and return the result state."""
    task_config = TaskConfig(
        task_name="topic",
        labels=_LABELS,
        label_descriptions=_DESCS,
        threshold=0.99,
        pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
        enable_deliberation=enable_deliberation,
    )
    orch = build_orchestrator(
        task_config=task_config,
        threshold=0.99,
        enable_deliberation=enable_deliberation,
    )
    state = _make_state(pipeline_mode, enable_deliberation=enable_deliberation)
    return orch.run(state)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _history_components(state: PipelineState) -> list[str]:
    return [e.component for e in state.history]


# ===========================================================================
# primary_only
# ===========================================================================

class TestPrimaryOnly:

    @pytest.fixture(scope="class")
    def result(self) -> PipelineState:
        return _build("primary_only")

    # -- Outputs ----------------------------------------------------------

    def test_final_output_not_none(self, result: PipelineState) -> None:
        assert result.final_output is not None

    def test_final_label_in_labels(self, result: PipelineState) -> None:
        assert result.final_output.label in _LABELS

    def test_explanation_output_not_none(self, result: PipelineState) -> None:
        """primary_only mode writes an inline explanation without running any agent."""
        assert result.explanation_output is not None

    def test_explanation_output_indicates_primary_only(self, result: PipelineState) -> None:
        """Explanation summary must signal that specialist agents were skipped."""
        summary = result.explanation_output.summary.lower()
        assert "primary" in summary or "skipped" in summary

    def test_routing_info_is_none(self, result: PipelineState) -> None:
        """Router is skipped in primary_only mode."""
        assert result.routing_info is None

    def test_lexical_output_is_none(self, result: PipelineState) -> None:
        assert result.lexical_output is None

    def test_contextual_output_is_none(self, result: PipelineState) -> None:
        assert result.contextual_output is None

    def test_logic_output_is_none(self, result: PipelineState) -> None:
        assert result.logic_output is None

    def test_consensus_output_is_none(self, result: PipelineState) -> None:
        assert result.consensus_output is None

    def test_no_pipeline_error(self, result: PipelineState) -> None:
        assert "pipeline_error" not in result.extras

    # -- History: what ran ------------------------------------------------

    def test_primary_classifier_in_history(self, result: PipelineState) -> None:
        assert "primary_classifier" in _history_components(result)

    # -- History: what must NOT have run ----------------------------------

    def test_router_not_in_history(self, result: PipelineState) -> None:
        assert "router" not in _history_components(result)

    def test_no_specialist_agents_in_history(self, result: PipelineState) -> None:
        """None of the escalation-path agents should appear in history."""
        should_be_absent = {
            "LexicalAgent", "LLMLexicalAgent",
            "LogicAgent", "LLMLogicAgent",
            "ContextualAgent", "TransformerContextualAgent",
            "DeliberationAgent",
            "ConsensusAgent",
        }
        ran = set(_history_components(result))
        assert ran.isdisjoint(should_be_absent), (
            f"Unexpected agents in primary_only history: {ran & should_be_absent}"
        )


# ===========================================================================
# paper_style
# ===========================================================================

class TestPaperStyle:

    @pytest.fixture(scope="class")
    def result(self) -> PipelineState:
        return _build("paper_style")

    # -- Outputs ----------------------------------------------------------

    def test_routing_decision_is_escalate(self, result: PipelineState) -> None:
        assert result.routing_info is not None
        assert result.routing_info.decision == "escalate"

    def test_lexical_output_not_none(self, result: PipelineState) -> None:
        assert result.lexical_output is not None

    def test_logic_output_not_none(self, result: PipelineState) -> None:
        assert result.logic_output is not None

    def test_contextual_output_not_none(self, result: PipelineState) -> None:
        assert result.contextual_output is not None

    def test_consensus_output_not_none(self, result: PipelineState) -> None:
        assert result.consensus_output is not None

    def test_final_output_not_none(self, result: PipelineState) -> None:
        assert result.final_output is not None

    def test_final_label_in_labels(self, result: PipelineState) -> None:
        assert result.final_output.label in _LABELS

    def test_no_deliberation_output(self, result: PipelineState) -> None:
        """paper_style never runs deliberation."""
        assert result.deliberation_output is None

    def test_no_pipeline_error(self, result: PipelineState) -> None:
        assert "pipeline_error" not in result.extras

    # -- History: concrete classes that must have run ---------------------

    def test_history_contains_lexical_agent(self, result: PipelineState) -> None:
        """paper_style uses the keyword-based LexicalAgent, not LLMLexicalAgent."""
        assert "LexicalAgent" in _history_components(result)

    def test_history_contains_logic_agent(self, result: PipelineState) -> None:
        """paper_style uses the regex-based LogicAgent, not LLMLogicAgent."""
        assert "LogicAgent" in _history_components(result)

    def test_history_contains_transformer_contextual_agent(self, result: PipelineState) -> None:
        """paper_style uses TransformerContextualAgent as the contextual agent."""
        assert "TransformerContextualAgent" in _history_components(result)

    def test_history_contains_consensus_agent(self, result: PipelineState) -> None:
        assert "ConsensusAgent" in _history_components(result)

    def test_history_contains_explainability_agent(self, result: PipelineState) -> None:
        """paper_style uses the template ExplainabilityAgent, not LLMExplainabilityAgent."""
        assert "ExplainabilityAgent" in _history_components(result)

    # -- History: LLM / deliberation agents must NOT have run ------------

    def test_llm_lexical_agent_not_in_history(self, result: PipelineState) -> None:
        assert "LLMLexicalAgent" not in _history_components(result)

    def test_llm_logic_agent_not_in_history(self, result: PipelineState) -> None:
        assert "LLMLogicAgent" not in _history_components(result)

    def test_llm_explainability_not_in_history(self, result: PipelineState) -> None:
        assert "LLMExplainabilityAgent" not in _history_components(result)

    def test_deliberation_agent_not_in_history(self, result: PipelineState) -> None:
        assert "DeliberationAgent" not in _history_components(result)


# ===========================================================================
# full_agentic — without deliberation
# ===========================================================================

class TestFullAgenticNoDeliberation:

    @pytest.fixture(scope="class")
    def result(self) -> PipelineState:
        return _build("full_agentic", enable_deliberation=False)

    # -- Outputs ----------------------------------------------------------

    def test_routing_decision_is_escalate(self, result: PipelineState) -> None:
        assert result.routing_info is not None
        assert result.routing_info.decision == "escalate"

    def test_lexical_output_not_none(self, result: PipelineState) -> None:
        assert result.lexical_output is not None

    def test_logic_output_not_none(self, result: PipelineState) -> None:
        assert result.logic_output is not None

    def test_contextual_output_not_none(self, result: PipelineState) -> None:
        assert result.contextual_output is not None

    def test_consensus_output_not_none(self, result: PipelineState) -> None:
        assert result.consensus_output is not None

    def test_final_output_not_none(self, result: PipelineState) -> None:
        assert result.final_output is not None

    def test_final_label_in_labels(self, result: PipelineState) -> None:
        assert result.final_output.label in _LABELS

    def test_deliberation_output_is_none(self, result: PipelineState) -> None:
        """enable_deliberation=False → deliberation stage skipped."""
        assert result.deliberation_output is None

    def test_no_pipeline_error(self, result: PipelineState) -> None:
        assert "pipeline_error" not in result.extras

    # -- History: concrete classes that must have run ---------------------

    def test_history_contains_llm_lexical_agent(self, result: PipelineState) -> None:
        """full_agentic uses LLMLexicalAgent, not the keyword-based LexicalAgent."""
        assert "LLMLexicalAgent" in _history_components(result)

    def test_history_contains_llm_logic_agent(self, result: PipelineState) -> None:
        """full_agentic uses LLMLogicAgent, not the regex-based LogicAgent."""
        assert "LLMLogicAgent" in _history_components(result)

    def test_history_contains_contextual_agent(self, result: PipelineState) -> None:
        """full_agentic uses the LLM-backed ContextualAgent."""
        assert "ContextualAgent" in _history_components(result)

    def test_history_contains_consensus_agent(self, result: PipelineState) -> None:
        assert "ConsensusAgent" in _history_components(result)

    def test_history_contains_llm_explainability_agent(self, result: PipelineState) -> None:
        """full_agentic uses LLMExplainabilityAgent when provided."""
        assert "LLMExplainabilityAgent" in _history_components(result)

    # -- History: wrong-mode agents must NOT have run --------------------

    def test_deliberation_agent_not_in_history(self, result: PipelineState) -> None:
        """enable_deliberation=False → DeliberationAgent must not run."""
        assert "DeliberationAgent" not in _history_components(result)

    def test_transformer_contextual_not_in_history(self, result: PipelineState) -> None:
        """TransformerContextualAgent belongs to paper_style only."""
        assert "TransformerContextualAgent" not in _history_components(result)

    def test_keyword_lexical_not_in_history(self, result: PipelineState) -> None:
        """LexicalAgent (keyword-based) must not run when LLMLexicalAgent is wired."""
        assert "LexicalAgent" not in _history_components(result)

    def test_regex_logic_not_in_history(self, result: PipelineState) -> None:
        """LogicAgent (regex-based) must not run when LLMLogicAgent is wired."""
        assert "LogicAgent" not in _history_components(result)


# ===========================================================================
# full_agentic — with deliberation
# ===========================================================================

class TestFullAgenticWithDeliberation:

    @pytest.fixture(scope="class")
    def result(self) -> PipelineState:
        return _build("full_agentic", enable_deliberation=True)

    # -- Outputs ----------------------------------------------------------

    def test_routing_decision_is_escalate(self, result: PipelineState) -> None:
        assert result.routing_info is not None
        assert result.routing_info.decision == "escalate"

    def test_lexical_output_not_none(self, result: PipelineState) -> None:
        assert result.lexical_output is not None

    def test_logic_output_not_none(self, result: PipelineState) -> None:
        assert result.logic_output is not None

    def test_contextual_output_not_none(self, result: PipelineState) -> None:
        assert result.contextual_output is not None

    def test_consensus_output_not_none(self, result: PipelineState) -> None:
        assert result.consensus_output is not None

    def test_final_output_not_none(self, result: PipelineState) -> None:
        assert result.final_output is not None

    def test_final_label_in_labels(self, result: PipelineState) -> None:
        assert result.final_output.label in _LABELS

    def test_deliberation_output_not_none(self, result: PipelineState) -> None:
        """enable_deliberation=True → deliberation_output should be written."""
        assert result.deliberation_output is not None

    def test_no_pipeline_error(self, result: PipelineState) -> None:
        assert "pipeline_error" not in result.extras

    # -- History: concrete classes that must have run ---------------------

    def test_history_contains_llm_lexical_agent(self, result: PipelineState) -> None:
        assert "LLMLexicalAgent" in _history_components(result)

    def test_history_contains_llm_logic_agent(self, result: PipelineState) -> None:
        assert "LLMLogicAgent" in _history_components(result)

    def test_history_contains_contextual_agent(self, result: PipelineState) -> None:
        assert "ContextualAgent" in _history_components(result)

    def test_history_contains_deliberation_agent(self, result: PipelineState) -> None:
        """enable_deliberation=True → DeliberationAgent must appear in history."""
        assert "DeliberationAgent" in _history_components(result)

    def test_history_contains_consensus_agent(self, result: PipelineState) -> None:
        assert "ConsensusAgent" in _history_components(result)

    def test_history_contains_llm_explainability_agent(self, result: PipelineState) -> None:
        assert "LLMExplainabilityAgent" in _history_components(result)

    # -- History: wrong-mode agents must NOT have run --------------------

    def test_transformer_contextual_not_in_history(self, result: PipelineState) -> None:
        """TransformerContextualAgent belongs to paper_style only."""
        assert "TransformerContextualAgent" not in _history_components(result)

    def test_keyword_lexical_not_in_history(self, result: PipelineState) -> None:
        assert "LexicalAgent" not in _history_components(result)

    def test_regex_logic_not_in_history(self, result: PipelineState) -> None:
        assert "LogicAgent" not in _history_components(result)
