"""tests/test_ner_panel_agents.py — offline tests for the debate & disambiguation
NER panel agents (canned LLM responses; no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.llm_ner_panel_agents import LLMNERDebateAgent, LLMNERDisambiguationAgent
from src.llm.base_client import LLMClient
from src.state.schema import (
    AgentOutput, ModelOutput, PipelineState, SequenceLabelingOutput,
    StateMetadata, TaskConfig, TokenTag,
)

_LABELS = ["O", "PERS", "LOC", "ORG", "MISC"]


class _Canned(LLMClient):
    def __init__(self, raw): self.raw, self.calls = raw, []
    def generate(self, prompt): self.calls.append(prompt); return self.raw


def _resp(tags): return json.dumps({"tags": tags, "reasoning": "x"})


def _state(tokens, model=None, contextual=None):
    st = PipelineState(
        metadata=StateMetadata(sample_id="t"), input_text=" ".join(tokens),
        task_config=TaskConfig(task_name="ner", task_type="sequence_labeling", labels=_LABELS),
        extras={"tokens": tokens})
    def _ao(tags):
        return AgentOutput(agent_name="s", model_output=ModelOutput(),
            sequence_output=SequenceLabelingOutput(
                tags=[TokenTag(token=t, tag=g, confidence=0.9) for t, g in zip(tokens, tags)]))
    if model is not None:
        st.ner_model_output = _ao(model)
    if contextual is not None:
        st.contextual_output = _ao(contextual)
    return st


# --- Debate -----------------------------------------------------------------

class TestDebate:
    def test_resolves_only_disagreement(self):
        tokens = ["Ahmed", "visited", "Cairo"]
        # A(model): Ahmed=PERS, Cairo=ORG ; B(contextual): Ahmed=PERS, Cairo=LOC -> disagree on Cairo
        st = _state(tokens, model=["PERS", "O", "ORG"], contextual=["PERS", "O", "LOC"])
        agent = LLMNERDebateAgent(_Canned(_resp(["PERS", "O", "LOC"])))
        agent.run(st)
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["PERS", "O", "LOC"]  # Cairo resolved to LOC

    def test_no_disagreement_no_llm_call(self):
        tokens = ["Ahmed", "left"]
        client = _Canned(_resp(["WRONG", "WRONG"]))
        st = _state(tokens, model=["PERS", "O"], contextual=["PERS", "O"])
        LLMNERDebateAgent(client).run(st)
        assert client.calls == []  # judge not consulted
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["PERS", "O"]

    def test_agreement_positions_untouched(self):
        tokens = ["Sara", "and", "Google"]
        st = _state(tokens, model=["PERS", "O", "LOC"], contextual=["PERS", "O", "ORG"])
        # LLM tries to also change position 0, but debate must keep agreements.
        agent = LLMNERDebateAgent(_Canned(_resp(["ORG", "O", "ORG"])))
        agent.run(st)
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out[0] == "PERS"      # agreement preserved despite LLM
        assert out[2] == "ORG"       # disagreement resolved

    def test_skips_on_classification(self):
        st = _state(["x"], model=["PERS"], contextual=["O"])
        st.task_config.task_type = "classification"
        LLMNERDebateAgent(_Canned(_resp(["PERS"]))).run(st)
        # contextual_output unchanged length-wise; agent skipped (no debate write)
        assert st.contextual_output.sequence_output.tags[0].tag == "O"


# --- Disambiguation ---------------------------------------------------------

class TestDisambiguation:
    def test_retypes_entity(self):
        tokens = ["Amman", "is", "big"]
        # draft marks Amman as PERS (wrong); disambiguation should retype to LOC.
        st = _state(tokens, contextual=["PERS", "O", "O"])
        agent = LLMNERDisambiguationAgent(_Canned(_resp(["LOC", "O", "O"])))
        agent.run(st)
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["LOC", "O", "O"]

    def test_does_not_add_entities(self):
        tokens = ["the", "Nile"]
        # draft has NO entities; LLM tries to add one -> disambiguation must not.
        st = _state(tokens, contextual=["O", "O"])
        client = _Canned(_resp(["O", "LOC"]))
        LLMNERDisambiguationAgent(client).run(st)
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["O", "O"]     # no entity added
        assert client.calls == []    # nothing to disambiguate -> no call

    def test_does_not_delete_entities(self):
        tokens = ["Google", "rocks"]
        st = _state(tokens, contextual=["ORG", "O"])
        # LLM returns O for the entity; disambiguation keeps it (never deletes).
        agent = LLMNERDisambiguationAgent(_Canned(_resp(["O", "O"])))
        agent.run(st)
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out[0] == "ORG"

    def test_falls_back_to_model_slot(self):
        tokens = ["Paris"]
        st = _state(tokens, model=["PERS"])   # only model slot populated
        agent = LLMNERDisambiguationAgent(_Canned(_resp(["LOC"])))
        agent.run(st)
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["LOC"]
