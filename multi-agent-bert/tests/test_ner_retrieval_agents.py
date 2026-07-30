"""tests/test_ner_retrieval_agents.py — offline tests for the verification and
gazetteer agents."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.ner_retrieval_agents import (
    LLMNERVerifyAgent, NERGazetteerAgent, build_gazetteer_from_conll,
)
from src.llm.base_client import LLMClient
from src.state.schema import (
    AgentOutput, ModelOutput, PipelineState, SequenceLabelingOutput,
    StateMetadata, TaskConfig, TokenTag,
)

_LABELS = ["O", "PERS", "LOC", "ORG", "MISC"]


class _Canned(LLMClient):
    def __init__(self, raw): self.raw, self.calls = raw, []
    def generate(self, prompt): self.calls.append(prompt); return self.raw


def _state(tokens, contextual=None):
    st = PipelineState(metadata=StateMetadata(sample_id="t"), input_text=" ".join(tokens),
        task_config=TaskConfig(task_name="ner", task_type="sequence_labeling", labels=_LABELS),
        extras={"tokens": tokens})
    if contextual is not None:
        st.contextual_output = AgentOutput(agent_name="s", model_output=ModelOutput(),
            sequence_output=SequenceLabelingOutput(
                tags=[TokenTag(token=t, tag=g, confidence=0.9) for t, g in zip(tokens, contextual)]))
    return st


# --- Verification -----------------------------------------------------------

class TestVerify:
    def test_removes_rejected_entity(self):
        toks = ["في", "Cairo"]
        # draft over-tagged 'في' (a preposition) as LOC; LLM confirms only Cairo.
        st = _state(toks, contextual=["LOC", "LOC"])
        agent = LLMNERVerifyAgent(_Canned(json.dumps({"entities": [{"text": "Cairo", "type": "LOC"}]})))
        agent.run(st)
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["O", "LOC"]   # 'في' removed, Cairo kept

    def test_keeps_all_when_confirmed(self):
        toks = ["Ahmed", "left"]
        st = _state(toks, contextual=["PERS", "O"])
        agent = LLMNERVerifyAgent(_Canned(json.dumps({"entities": [{"text": "Ahmed", "type": "PERS"}]})))
        agent.run(st)
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["PERS", "O"]

    def test_no_entities_passthrough_no_call(self):
        toks = ["hello", "world"]
        client = _Canned("{}")
        st = _state(toks, contextual=["O", "O"])
        LLMNERVerifyAgent(client).run(st)
        assert client.calls == []

    def test_parse_fail_keeps_all(self):
        toks = ["Google", "rocks"]
        st = _state(toks, contextual=["ORG", "O"])
        LLMNERVerifyAgent(_Canned("garbage")).run(st)
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["ORG", "O"]


# --- Gazetteer --------------------------------------------------------------

def _identity_t2t(tag):
    return tag if tag == "O" else (tag.split("-", 1)[1] if "-" in tag else tag)


class TestGazetteer:
    def _gaz(self):
        train = [
            {"tokens": ["أولمبيك", "خريبكة", "فاز"], "tags": ["ORG", "ORG", "O"]},
            {"tokens": ["زار", "Cairo"], "tags": ["O", "LOC"]},
        ]
        return build_gazetteer_from_conll(train, _identity_t2t)

    def test_build_gazetteer(self):
        gaz = self._gaz()
        assert gaz["أولمبيك خريبكة"] == "ORG"
        assert gaz["cairo"] == "LOC"

    def test_augment_fills_only_O(self):
        gaz = self._gaz()
        toks = ["أولمبيك", "خريبكة", "ضد", "Cairo"]
        # draft missed the club (O O) but tagged nothing; gazetteer fills both.
        st = _state(toks, contextual=["O", "O", "O", "O"])
        NERGazetteerAgent(gaz, source_slot="contextual", mode="augment").run(st)
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["ORG", "ORG", "O", "LOC"]

    def test_augment_does_not_override_draft(self):
        gaz = self._gaz()
        toks = ["Cairo"]
        st = _state(toks, contextual=["PERS"])   # draft says PERS
        NERGazetteerAgent(gaz, source_slot="contextual", mode="augment").run(st)
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["PERS"]   # augment keeps draft's non-O

    def test_overwrite_mode(self):
        gaz = self._gaz()
        toks = ["Cairo"]
        st = _state(toks, contextual=["PERS"])
        NERGazetteerAgent(gaz, source_slot="contextual", mode="overwrite").run(st)
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["LOC"]    # gazetteer wins

    def test_standalone_no_source(self):
        gaz = self._gaz()
        toks = ["زار", "Cairo"]
        st = _state(toks)
        NERGazetteerAgent(gaz, source_slot=None).run(st)
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["O", "LOC"]
