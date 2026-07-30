"""tests/test_ner_transformer_agent.py

Offline tests for the real NER primary model (TransformerNERTagger) and its
integration into the NER path. These NEVER download a model — the tagger is
stubbed by overriding ``tag()`` and marking it loaded, so CI stays hermetic.

Covers:
  - Pure helpers: sub-word→word alignment and tag normalisation.
  - run(state): writes state.ner_model_output; skips on classification tasks.
  - NERConsensusAgent reads the new "model" slot and it can win with weight.
  - Orchestrator NER path: primary_only finalizes the model's output;
    paper_style runs the model in the primary position before specialists.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.ner_consensus_agent import NERConsensusAgent
from src.models.transformer_ner_tagger import TransformerNERTagger
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    SequenceLabelingOutput,
    StateMetadata,
    TaskConfig,
    TokenTag,
)

_NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


# ---------------------------------------------------------------------------
# Stub tagger: canned per-token tags, no model download
# ---------------------------------------------------------------------------

class _StubTagger(TransformerNERTagger):
    """TransformerNERTagger whose tag() returns a caller-supplied tag map.

    ``conf`` sets the confidence on every emitted token (drives the NER router's
    min-confidence gate in tests).
    """

    def __init__(self, tag_map: dict, conf: float = 0.95, **kwargs):
        super().__init__(checkpoint="stub://model", **kwargs)
        self._tag_map = tag_map
        self._conf = conf
        self._loaded = True  # bypass load()

    def tag(self, tokens: List[str], task_labels: Optional[List[str]] = None):
        valid = set(task_labels) if task_labels else set(_NER_LABELS)
        tags = []
        for t in tokens:
            raw = self._tag_map.get(t, "O")
            tag = raw if raw in valid else "O"
            tags.append(TokenTag(token=t, tag=tag, confidence=self._conf))
        return SequenceLabelingOutput(tags=tags, notes="stub")


def _ner_state(tokens, mode="paper_style", threshold=0.5):
    return PipelineState(
        metadata=StateMetadata(sample_id="t"),
        input_text=" ".join(tokens),
        task_config=TaskConfig(task_name="ner", task_type="sequence_labeling",
                               labels=_NER_LABELS, pipeline_mode=mode, threshold=threshold),
        extras={"tokens": tokens},
    )


# ===========================================================================
# Pure helpers
# ===========================================================================

class TestPureHelpers:

    def test_align_first_subword_takes_first(self):
        # words: 0 -> subwords [1,2], 1 -> [3], 2 -> [4,5]; specials are None
        word_ids = [None, 0, 0, 1, 2, 2, None]
        pred_ids = [9, 3, 4, 5, 6, 7, 9]
        probs = [0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.1]
        out = TransformerNERTagger._align_first_subword(word_ids, pred_ids, probs, 3)
        assert out == [(3, 0.9), (5, 0.7), (6, 0.6)]

    def test_align_handles_missing_word(self):
        # word index 2 never appears -> defaults to (0, 0.0)
        out = TransformerNERTagger._align_first_subword([None, 0, 1], [0, 5, 6], [0.0, 0.9, 0.8], 3)
        assert out[2] == (0, 0.0)

    def test_normalise_exact_match(self):
        assert TransformerNERTagger._normalise_tag("B-PER", set(_NER_LABELS)) == "B-PER"

    def test_normalise_case_fold(self):
        assert TransformerNERTagger._normalise_tag("b-per", set(_NER_LABELS)) == "B-PER"

    def test_normalise_bare_type(self):
        assert TransformerNERTagger._normalise_tag("PER", set(_NER_LABELS)) == "B-PER"

    def test_normalise_unknown_to_O(self):
        assert TransformerNERTagger._normalise_tag("B-MISC", set(_NER_LABELS)) == "O"


# ===========================================================================
# run(state)
# ===========================================================================

class TestRunState:

    def test_writes_ner_model_output(self):
        tagger = _StubTagger({"Ahmed": "B-PER", "Google": "B-ORG"})
        state = _ner_state(["Ahmed", "works", "Google"])
        out = tagger.run(state)
        assert out.ner_model_output is not None
        seq = out.ner_model_output.sequence_output
        assert [t.tag for t in seq.tags] == ["B-PER", "O", "B-ORG"]

    def test_one_tag_per_token(self):
        tagger = _StubTagger({})
        state = _ner_state(["a", "b", "c", "d"])
        out = tagger.run(state)
        assert len(out.ner_model_output.sequence_output.tags) == 4

    def test_skips_on_classification_task(self):
        tagger = _StubTagger({"Ahmed": "B-PER"})
        state = PipelineState(
            metadata=StateMetadata(sample_id="c"),
            input_text="great",
            task_config=TaskConfig(task_name="sent", task_type="classification",
                                   labels=["positive", "negative"]),
        )
        out = tagger.run(state)
        assert out.ner_model_output is None  # untouched on classification

    def test_appends_history(self):
        tagger = _StubTagger({})
        state = _ner_state(["x"])
        out = tagger.run(state)
        assert any(e.component == "ner_primary_model" for e in out.history)


# ===========================================================================
# Consensus "model" slot
# ===========================================================================

class TestConsensusModelSlot:

    def _state_with_model_and_heuristics(self, model_tag, heur_tag):
        state = _ner_state(["Google"])
        state.ner_model_output = AgentOutput(
            agent_name="m", model_output=ModelOutput(),
            sequence_output=SequenceLabelingOutput(
                tags=[TokenTag(token="Google", tag=model_tag, confidence=0.95)]),
        )
        # Three heuristic slots all say heur_tag.
        for attr in ("lexical_output", "logic_output", "contextual_output"):
            setattr(state, attr, AgentOutput(
                agent_name=attr, model_output=ModelOutput(),
                sequence_output=SequenceLabelingOutput(
                    tags=[TokenTag(token="Google", tag=heur_tag, confidence=1.0)])))
        return state

    def test_model_wins_with_high_weight(self):
        # 3 heuristics say O; model says B-ORG. With model weight 5 it wins.
        state = self._state_with_model_and_heuristics("B-ORG", "O")
        NERConsensusAgent(weights={"model": 5.0}).run(state)
        seq = state.final_output.payload["sequence_output"]
        assert seq[0]["tag"] == "B-ORG"

    def test_model_outvoted_at_neutral_weight(self):
        # At equal weights, 3 heuristics (O) outweigh 1 model (B-ORG).
        state = self._state_with_model_and_heuristics("B-ORG", "O")
        NERConsensusAgent().run(state)  # all weights 1.0
        seq = state.final_output.payload["sequence_output"]
        assert seq[0]["tag"] == "O"

    def test_model_only_when_heuristics_zeroed(self):
        state = self._state_with_model_and_heuristics("B-ORG", "O")
        NERConsensusAgent(weights={"lexical": 0, "logic": 0, "contextual": 0}).run(state)
        seq = state.final_output.payload["sequence_output"]
        assert seq[0]["tag"] == "B-ORG"


# ===========================================================================
# Orchestrator NER path with the model wired in
# ===========================================================================

def _make_orch(ner_primary, consensus_weights=None):
    from src.agents.consensus_agent import ConsensusAgent
    from src.agents.contextual_agent import ContextualAgent
    from src.agents.explainability_agent import ExplainabilityAgent
    from src.agents.lexical_agent import LexicalAgent
    from src.agents.logic_agent import LogicAgent
    from src.llm.mock_client import MockLLMClient
    from src.pipeline.orchestrator import PipelineOrchestrator
    from src.pipeline.router import Router

    class _P:
        def run(self, s):
            s.primary_model_output = ModelOutput(label="O", confidence=0.9,
                                                 probabilities={"O": 1.0})
            return s

    llm = MockLLMClient(mode="label_echo", allowed_labels=["positive", "negative"])
    return PipelineOrchestrator(
        primary_classifier=_P(), router=Router(),
        lexical_agent=LexicalAgent(), contextual_agent=ContextualAgent(llm_client=llm),
        logic_agent=LogicAgent(), consensus_agent=ConsensusAgent(),
        explainability_agent=ExplainabilityAgent(),
        ner_consensus_agent=NERConsensusAgent(weights=consensus_weights or {}),
        ner_primary=ner_primary,
    )


def _components(state):
    return [e.component for e in state.history]


class TestOrchestratorNERPrimary:

    def test_primary_only_uses_model_output(self):
        tagger = _StubTagger({"Ahmed": "B-PER", "Google": "B-ORG"})
        orch = _make_orch(tagger)
        state = _ner_state(["Ahmed", "at", "Google"], mode="primary_only")
        result = orch.run(state)
        seq = result.final_output.payload["sequence_output"]
        assert [t["tag"] for t in seq] == ["B-PER", "O", "B-ORG"]
        assert result.final_output.payload["source"] == "ner_primary_model"

    def test_no_pipeline_error(self):
        tagger = _StubTagger({})
        orch = _make_orch(tagger)
        state = _ner_state(["a", "b"], mode="paper_style")
        result = orch.run(state)
        assert "pipeline_error" not in result.extras


class TestNERRouter:
    """primary → router → (accept | escalate → agents → consensus)."""

    def test_accept_path_skips_agents(self):
        # High confidence (0.95) >= threshold 0.5 → accept, skip specialists.
        tagger = _StubTagger({"Ahmed": "B-PER", "Google": "B-ORG"}, conf=0.95)
        orch = _make_orch(tagger)
        state = _ner_state(["Ahmed", "at", "Google"], mode="paper_style", threshold=0.5)
        result = orch.run(state)
        assert result.routing_info.decision == "accept_primary"
        assert "ner_router" in _components(result)
        assert result.lexical_output is None         # specialist agents skipped
        assert result.consensus_output is None
        assert result.final_output.payload["source"] == "ner_primary_accepted"
        seq = result.final_output.payload["sequence_output"]
        assert [t["tag"] for t in seq] == ["B-PER", "O", "B-ORG"]

    def test_escalate_path_runs_agents(self):
        # Low confidence (0.40) < threshold 0.5 → escalate to specialists+consensus.
        tagger = _StubTagger({"Ahmed": "B-PER", "Google": "B-ORG"}, conf=0.40)
        orch = _make_orch(tagger, consensus_weights={"model": 100.0})
        state = _ner_state(["Ahmed", "at", "Google"], mode="paper_style", threshold=0.5)
        result = orch.run(state)
        assert result.routing_info.decision == "escalate"
        assert result.lexical_output is not None      # specialist agents ran
        assert result.consensus_output is not None
        # Model still wins the vote (weight 5) so tags match the primary.
        seq = result.final_output.payload["sequence_output"]
        assert [t["tag"] for t in seq] == ["B-PER", "O", "B-ORG"]

    def test_threshold_one_forces_escalation(self):
        # threshold 1.0 → even 0.95 tokens escalate (min-confidence gate).
        tagger = _StubTagger({"Ahmed": "B-PER"}, conf=0.95)
        orch = _make_orch(tagger)
        state = _ner_state(["Ahmed"], mode="paper_style", threshold=1.0)
        result = orch.run(state)
        assert result.routing_info.decision == "escalate"
        assert result.consensus_output is not None
