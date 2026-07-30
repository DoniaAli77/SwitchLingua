"""tests/test_llm_ner_agent.py

Offline tests for LLMNERAgent using MockLLMClient (no network). Mirrors the
style of test_llm_specialist_agents.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.llm_ner_agent import LLMNERAgent
from src.llm.base_client import LLMClient
from src.llm.mock_client import MockLLMClient
from src.state.schema import (
    PipelineState, StateMetadata, TaskConfig,
)

_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class _CannedClient(LLMClient):
    """Returns a caller-supplied raw string, records the prompt."""
    def __init__(self, raw: str):
        self.raw = raw
        self.calls = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.raw


def _ner_state(tokens, task_type="sequence_labeling"):
    return PipelineState(
        metadata=StateMetadata(sample_id="t"),
        input_text=" ".join(tokens),
        task_config=TaskConfig(task_name="ner", task_type=task_type, labels=_LABELS),
        extras={"tokens": tokens},
    )


def _resp(tags, reasoning="found entities"):
    return json.dumps({"tags": tags, "reasoning": reasoning})


# ===========================================================================
# Happy path
# ===========================================================================

class TestHappyPath:

    def test_writes_contextual_output(self):
        tokens = ["Ahmed", "at", "Google"]
        agent = LLMNERAgent(_CannedClient(_resp(["B-PER", "O", "B-ORG"])))
        st = agent.run(_ner_state(tokens))
        assert st.contextual_output is not None
        seq = st.contextual_output.sequence_output
        assert [t.tag for t in seq.tags] == ["B-PER", "O", "B-ORG"]
        assert [t.token for t in seq.tags] == tokens

    def test_arabic_tokens_tagged(self):
        tokens = ["مريم", "من", "القاهرة"]
        agent = LLMNERAgent(_CannedClient(_resp(["B-PER", "O", "B-LOC"])))
        st = agent.run(_ner_state(tokens))
        assert [t.tag for t in st.contextual_output.sequence_output.tags] == ["B-PER", "O", "B-LOC"]

    def test_appends_history(self):
        agent = LLMNERAgent(_CannedClient(_resp(["O"])))
        st = agent.run(_ner_state(["x"]))
        assert any(e.component == "LLMNERAgent" for e in st.history)

    def test_markdown_fenced_json_parsed(self):
        raw = "```json\n" + _resp(["B-PER", "O"]) + "\n```"
        agent = LLMNERAgent(_CannedClient(raw))
        st = agent.run(_ner_state(["Sara", "left"]))
        assert [t.tag for t in st.contextual_output.sequence_output.tags] == ["B-PER", "O"]


# ===========================================================================
# Robustness
# ===========================================================================

class TestRobustness:

    def test_skips_on_classification(self):
        agent = LLMNERAgent(_CannedClient(_resp(["B-PER"])))
        st = agent.run(_ner_state(["Ahmed"], task_type="classification"))
        assert st.contextual_output is None

    def test_too_few_tags_padded(self):
        tokens = ["a", "b", "c", "d"]
        agent = LLMNERAgent(_CannedClient(_resp(["B-PER"])))  # only 1 tag
        st = agent.run(_ner_state(tokens))
        tags = st.contextual_output.sequence_output.tags
        assert len(tags) == 4
        assert [t.tag for t in tags] == ["B-PER", "O", "O", "O"]

    def test_too_many_tags_truncated(self):
        tokens = ["a", "b"]
        agent = LLMNERAgent(_CannedClient(_resp(["B-PER", "I-PER", "B-ORG", "O"])))
        st = agent.run(_ner_state(tokens))
        assert len(st.contextual_output.sequence_output.tags) == 2

    def test_invalid_tag_mapped_to_O(self):
        agent = LLMNERAgent(_CannedClient(_resp(["B-PER", "B-MISC"])))
        st = agent.run(_ner_state(["Ahmed", "Cairo"]))
        assert [t.tag for t in st.contextual_output.sequence_output.tags] == ["B-PER", "O"]

    def test_unparseable_falls_back_all_O(self):
        agent = LLMNERAgent(_CannedClient("not json at all"))
        st = agent.run(_ner_state(["a", "b", "c"]))
        seq = st.contextual_output.sequence_output
        assert [t.tag for t in seq.tags] == ["O", "O", "O"]  # no crash
        assert "parse failed" in seq.notes

    def test_missing_tags_key_falls_back(self):
        agent = LLMNERAgent(_CannedClient(json.dumps({"reasoning": "x"})))
        st = agent.run(_ner_state(["a"]))
        assert st.contextual_output.sequence_output.tags[0].tag == "O"


# ===========================================================================
# Slot routing
# ===========================================================================

class TestSlotRouting:

    def test_lexical_slot(self):
        agent = LLMNERAgent(_CannedClient(_resp(["B-PER"])), output_slot="lexical")
        st = agent.run(_ner_state(["Ahmed"]))
        assert st.lexical_output is not None
        assert st.contextual_output is None

    def test_invalid_slot_raises(self):
        with pytest.raises(ValueError):
            LLMNERAgent(_CannedClient(_resp(["O"])), output_slot="bogus")


# ===========================================================================
# Integration: full_agentic pipeline (primary → router → LLM agent → consensus)
# ===========================================================================

class TestPipelineIntegration:

    def _orch(self, llm_raw, tagger, threshold, model_weight=1.0):
        from src.agents.consensus_agent import ConsensusAgent
        from src.agents.contextual_agent import ContextualAgent
        from src.agents.explainability_agent import ExplainabilityAgent
        from src.agents.lexical_agent import LexicalAgent
        from src.agents.logic_agent import LogicAgent
        from src.agents.ner_consensus_agent import NERConsensusAgent
        from src.agents.ner_lexical_agent import NERLexicalAgent
        from src.agents.ner_logic_agent import NERLogicAgent
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.pipeline.router import Router
        from src.state.schema import ModelOutput

        class _P:
            def run(self, s):
                s.primary_model_output = ModelOutput(label="O", confidence=0.9,
                                                     probabilities={"O": 1.0})
                return s

        llm = MockLLMClient(mode="label_echo", allowed_labels=["positive"])
        return PipelineOrchestrator(
            primary_classifier=_P(), router=Router(),
            lexical_agent=LexicalAgent(), contextual_agent=ContextualAgent(llm_client=llm),
            logic_agent=LogicAgent(), consensus_agent=ConsensusAgent(),
            explainability_agent=ExplainabilityAgent(),
            # LLM NER agent occupies the contextual slot; heuristic lex/logic empty.
            ner_lexical_agent=NERLexicalAgent(),
            ner_logic_agent=NERLogicAgent(),
            ner_contextual_agent=LLMNERAgent(_CannedClient(llm_raw), output_slot="contextual"),
            # "LLM instead of heuristics": disable the empty heuristic slots so
            # only the XLM-R primary (model) and the LLM (contextual) vote.
            ner_consensus_agent=NERConsensusAgent(
                weights={"model": model_weight, "lexical": 0.0, "logic": 0.0}),
            ner_primary=tagger,
        )

    def test_escalated_case_uses_llm_agent(self):
        # Stub XLM-R primary with LOW confidence so the router escalates,
        # and a WRONG primary tag, so we can see the LLM correct it.
        from tests.test_ner_transformer_agent import _StubTagger  # reuse stub
        tagger = _StubTagger({"مريم": "O"}, conf=0.40)  # primary misses the Arabic name
        orch = self._orch(_resp(["B-PER", "O"]), tagger, threshold=0.95, model_weight=1.0)
        state = _ner_state(["مريم", "left"])
        state.task_config.pipeline_mode = "full_agentic"
        state.task_config.threshold = 0.95
        result = orch.run(state)
        assert result.routing_info.decision == "escalate"
        # LLM (contextual) says B-PER conf 0.9; primary says O conf 0.40.
        seq = result.final_output.payload["sequence_output"]
        assert seq[0]["tag"] == "B-PER"  # LLM rescued the Arabic entity
