"""tests/test_llm_ner_reflection_agent.py

Offline tests for LLMNERReflectionAgent (review-and-correct) using canned LLM
responses — no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.llm_ner_reflection_agent import LLMNERReflectionAgent
from src.llm.base_client import LLMClient
from src.state.schema import (
    AgentOutput, ModelOutput, PipelineState, SequenceLabelingOutput,
    StateMetadata, TaskConfig, TokenTag,
)

_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class _CannedClient(LLMClient):
    def __init__(self, raw): self.raw, self.calls = raw, []
    def generate(self, prompt): self.calls.append(prompt); return self.raw


def _resp(tags, reasoning="corrected"):
    return json.dumps({"tags": tags, "reasoning": reasoning})


def _state_with_draft(tokens, draft_tags, task_type="sequence_labeling"):
    st = PipelineState(
        metadata=StateMetadata(sample_id="t"),
        input_text=" ".join(tokens),
        task_config=TaskConfig(task_name="ner", task_type=task_type, labels=_LABELS),
        extras={"tokens": tokens},
    )
    if draft_tags is not None:
        st.ner_model_output = AgentOutput(
            agent_name="primary", model_output=ModelOutput(),
            sequence_output=SequenceLabelingOutput(
                tags=[TokenTag(token=t, tag=g, confidence=0.7)
                      for t, g in zip(tokens, draft_tags)]))
    return st


class TestReflection:

    def test_corrects_missed_arabic_entity(self):
        # Primary missed the Arabic name (draft O); LLM reviewer fixes it.
        tokens = ["مريم", "من", "القاهرة"]
        draft = ["O", "O", "B-LOC"]
        agent = LLMNERReflectionAgent(_CannedClient(_resp(["B-PER", "O", "B-LOC"])))
        st = agent.run(_state_with_draft(tokens, draft))
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["B-PER", "O", "B-LOC"]
        assert st.contextual_output.sequence_output.features["tokens_changed"] == 1

    def test_keeps_correct_draft_unchanged(self):
        tokens = ["Ahmed", "left"]
        draft = ["B-PER", "O"]
        agent = LLMNERReflectionAgent(_CannedClient(_resp(["B-PER", "O"], "no changes")))
        st = agent.run(_state_with_draft(tokens, draft))
        assert st.contextual_output.sequence_output.features["tokens_changed"] == 0

    def test_draft_shown_in_prompt(self):
        tokens = ["Ahmed"]
        client = _CannedClient(_resp(["B-PER"]))
        agent = LLMNERReflectionAgent(client)
        agent.run(_state_with_draft(tokens, ["O"]))
        # The primary's draft tag must appear in the prompt sent to the LLM.
        assert "Ahmed -> O" in client.calls[0]

    def test_parse_failure_keeps_draft(self):
        tokens = ["Ahmed", "at", "Google"]
        draft = ["B-PER", "O", "B-ORG"]
        agent = LLMNERReflectionAgent(_CannedClient("garbage not json"))
        st = agent.run(_state_with_draft(tokens, draft))
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == draft  # unchanged fallback
        assert "kept primary draft" in st.contextual_output.sequence_output.notes

    def test_no_primary_draft_defaults_all_O(self):
        tokens = ["a", "b"]
        agent = LLMNERReflectionAgent(_CannedClient(_resp(["B-PER", "O"])))
        st = agent.run(_state_with_draft(tokens, draft_tags=None))  # no primary
        assert [t.tag for t in st.contextual_output.sequence_output.tags] == ["B-PER", "O"]

    def test_skips_on_classification(self):
        agent = LLMNERReflectionAgent(_CannedClient(_resp(["B-PER"])))
        st = agent.run(_state_with_draft(["Ahmed"], ["O"], task_type="classification"))
        assert st.contextual_output is None

    def test_invalid_correction_mapped_to_O(self):
        tokens = ["Ahmed", "Cairo"]
        agent = LLMNERReflectionAgent(_CannedClient(_resp(["B-PER", "B-MISC"])))
        st = agent.run(_state_with_draft(tokens, ["O", "O"]))
        assert [t.tag for t in st.contextual_output.sequence_output.tags] == ["B-PER", "O"]


class TestReflectionInPipeline:

    def test_reflection_becomes_final_output(self):
        """Reflection agent as sole specialist → its correction is the final answer."""
        from src.agents.consensus_agent import ConsensusAgent
        from src.agents.contextual_agent import ContextualAgent
        from src.agents.explainability_agent import ExplainabilityAgent
        from src.agents.lexical_agent import LexicalAgent
        from src.agents.logic_agent import LogicAgent
        from src.agents.ner_consensus_agent import NERConsensusAgent
        from src.llm.mock_client import MockLLMClient
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.pipeline.router import Router
        from tests.test_ner_transformer_agent import _StubTagger

        class _P:
            def run(self, s):
                s.primary_model_output = ModelOutput(label="O", confidence=0.9,
                                                     probabilities={"O": 1.0})
                return s

        # Primary (XLM-R stub) MISSES the Arabic name with low confidence → escalate.
        tagger = _StubTagger({"مريم": "O"}, conf=0.40)
        llm = MockLLMClient(mode="label_echo", allowed_labels=["positive"])
        orch = PipelineOrchestrator(
            primary_classifier=_P(), router=Router(),
            lexical_agent=LexicalAgent(), contextual_agent=ContextualAgent(llm_client=llm),
            logic_agent=LogicAgent(), consensus_agent=ConsensusAgent(),
            explainability_agent=ExplainabilityAgent(),
            ner_contextual_agent=LLMNERReflectionAgent(_CannedClient(_resp(["B-PER", "O"]))),
            # Consensus uses ONLY the reflection (contextual) output as the final answer.
            ner_consensus_agent=NERConsensusAgent(
                weights={"model": 0.0, "lexical": 0.0, "logic": 0.0, "contextual": 1.0}),
            ner_primary=tagger,
        )
        st = PipelineState(
            metadata=StateMetadata(sample_id="t"), input_text="مريم left",
            task_config=TaskConfig(task_name="ner", task_type="sequence_labeling",
                                   labels=_LABELS, pipeline_mode="full_agentic", threshold=0.95),
            extras={"tokens": ["مريم", "left"]})
        res = orch.run(st)
        assert res.routing_info.decision == "escalate"
        seq = res.final_output.payload["sequence_output"]
        assert seq[0]["tag"] == "B-PER"  # reflection rescued the Arabic entity
