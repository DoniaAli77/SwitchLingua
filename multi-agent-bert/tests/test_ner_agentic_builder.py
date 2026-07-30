"""tests/test_ner_agentic_builder.py

Offline tests for the canonical agentic NER orchestrator builder.
Verifies the standard lineup (XLM-R primary -> router -> LLM specialist ->
consensus, rule agents OFF) is wired correctly, using stubs (no network/model).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.llm_ner_agent import LLMNERAgent
from src.agents.llm_ner_reflection_agent import LLMNERReflectionAgent
from src.llm.base_client import LLMClient
from src.pipeline.ner_agentic import (
    SPECIALISTS,
    agentic_ner_task_config,
    build_agentic_ner_orchestrator,
)
from src.state.schema import PipelineState, StateMetadata

from tests.test_ner_transformer_agent import _StubTagger

_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class _Canned(LLMClient):
    def __init__(self, raw): self.raw = raw
    def generate(self, prompt): return self.raw


def _resp(tags):
    return json.dumps({"tags": tags, "reasoning": "x"})


def _state(tokens, threshold):
    return PipelineState(
        metadata=StateMetadata(sample_id="t"),
        input_text=" ".join(tokens),
        task_config=agentic_ner_task_config(_LABELS, threshold=threshold),
        extras={"tokens": tokens},
    )


def test_default_specialist_is_reflector():
    orch = build_agentic_ner_orchestrator(_StubTagger({}), _Canned(_resp(["O"])))
    assert isinstance(orch._ner_contextual, LLMNERReflectionAgent)


def test_tagger_specialist_selected():
    orch = build_agentic_ner_orchestrator(
        _StubTagger({}), _Canned(_resp(["O"])), specialist="tagger")
    assert isinstance(orch._ner_contextual, LLMNERAgent)
    assert not isinstance(orch._ner_contextual, LLMNERReflectionAgent)


def test_rule_agents_off_in_consensus():
    orch = build_agentic_ner_orchestrator(_StubTagger({}), _Canned(_resp(["O"])))
    w = orch._ner_consensus.weights
    assert w["lexical"] == 0.0 and w["logic"] == 0.0
    assert w["model"] > 0.0 and w["contextual"] > 0.0


def test_invalid_specialist_raises():
    with pytest.raises(ValueError):
        build_agentic_ner_orchestrator(_StubTagger({}), _Canned(_resp(["O"])),
                                       specialist="bogus")


def test_specialists_constant():
    assert set(SPECIALISTS) == {"reflector", "tagger"}


def test_end_to_end_reflector_rescues_arabic():
    # Primary misses the Arabic name with low confidence -> escalate -> reflector fixes.
    tagger = _StubTagger({"مريم": "O"}, conf=0.40)
    orch = build_agentic_ner_orchestrator(tagger, _Canned(_resp(["B-PER", "O"])))
    res = orch.run(_state(["مريم", "left"], threshold=0.95))
    assert res.routing_info.decision == "escalate"
    seq = res.final_output.payload["sequence_output"]
    assert seq[0]["tag"] == "B-PER"


def test_confident_primary_accepted_no_llm():
    # High primary confidence -> router accepts -> specialist never consulted.
    tagger = _StubTagger({"Ahmed": "B-PER"}, conf=0.99)
    orch = build_agentic_ner_orchestrator(tagger, _Canned(_resp(["WRONG"])))
    res = orch.run(_state(["Ahmed"], threshold=0.95))
    assert res.routing_info.decision == "accept_primary"
    seq = res.final_output.payload["sequence_output"]
    assert seq[0]["tag"] == "B-PER"  # primary's answer, LLM not used
