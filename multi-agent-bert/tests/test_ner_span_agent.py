"""tests/test_ner_span_agent.py — tests for the deterministic span aligner and
the span-extraction agent (offline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.llm_ner_span_agent import LLMNERSpanAgent, align_entities_to_tokens
from src.llm.base_client import LLMClient
from src.state.schema import PipelineState, StateMetadata, TaskConfig

_V = {"O", "PERS", "LOC", "ORG", "MISC"}


class _Canned(LLMClient):
    def __init__(self, raw): self.raw, self.calls = raw, []
    def generate(self, prompt): self.calls.append(prompt); return self.raw


# --- the deterministic aligner (pure function) ------------------------------

class TestAligner:
    def test_single_token_entity(self):
        toks = ["قال", "Nasrallah", "شيئا"]
        ents = [{"text": "Nasrallah", "type": "PERS"}]
        assert align_entities_to_tokens(toks, ents, _V) == ["O", "PERS", "O"]

    def test_multiword_entity_lands_exactly(self):
        # The drift case: "الجنوب اللبناني" must land on its own tokens, not neighbors.
        toks = ["مع", "مقاتليه", "في", "الجنوب", "اللبناني", "،"]
        ents = [{"text": "الجنوب اللبناني", "type": "LOC"}]
        assert align_entities_to_tokens(toks, ents, _V) == ["O", "O", "O", "LOC", "LOC", "O"]

    def test_type_alias_mapped(self):
        # LLM emits standard PER; corpus uses PERS.
        toks = ["Sara"]
        ents = [{"text": "Sara", "type": "PER"}]
        assert align_entities_to_tokens(toks, ents, _V) == ["PERS"]

    def test_unlocatable_entity_dropped(self):
        toks = ["hello", "world"]
        ents = [{"text": "Paris", "type": "LOC"}]   # not in tokens
        assert align_entities_to_tokens(toks, ents, _V) == ["O", "O"]

    def test_punctuation_insensitive_match(self):
        toks = ["visited", "Cairo", "."]
        ents = [{"text": "Cairo.", "type": "LOC"}]
        assert align_entities_to_tokens(toks, ents, _V) == ["O", "LOC", "O"]

    def test_longer_entity_wins_over_subword(self):
        toks = ["New", "York", "city"]
        ents = [{"text": "York", "type": "LOC"}, {"text": "New York", "type": "LOC"}]
        # "New York" (longer) claims both; "York" alone can't re-claim.
        assert align_entities_to_tokens(toks, ents, _V) == ["LOC", "LOC", "O"]

    def test_two_occurrences_first_unclaimed(self):
        toks = ["Cairo", "and", "Cairo"]
        ents = [{"text": "Cairo", "type": "LOC"}, {"text": "Cairo", "type": "LOC"}]
        assert align_entities_to_tokens(toks, ents, _V) == ["LOC", "O", "LOC"]

    def test_invalid_type_dropped(self):
        toks = ["Google"]
        ents = [{"text": "Google", "type": "NONSENSE"}]
        assert align_entities_to_tokens(toks, ents, _V) == ["O"]


# --- the agent --------------------------------------------------------------

def _state(tokens):
    return PipelineState(
        metadata=StateMetadata(sample_id="t"), input_text=" ".join(tokens),
        task_config=TaskConfig(task_name="ner", task_type="sequence_labeling",
                               labels=["O", "PERS", "LOC", "ORG", "MISC"]),
        extras={"tokens": tokens})


class TestAgent:
    def test_writes_aligned_tags(self):
        toks = ["Ahmed", "went", "to", "Cairo"]
        raw = json.dumps({"entities": [{"text": "Ahmed", "type": "PER"},
                                       {"text": "Cairo", "type": "LOC"}], "reasoning": "x"})
        st = LLMNERSpanAgent(_Canned(raw)).run(_state(toks))
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["PERS", "O", "O", "LOC"]

    def test_parse_failure_all_O(self):
        st = LLMNERSpanAgent(_Canned("not json")).run(_state(["a", "b"]))
        out = [t.tag for t in st.contextual_output.sequence_output.tags]
        assert out == ["O", "O"]

    def test_skips_on_classification(self):
        st = _state(["x"])
        st.task_config.task_type = "classification"
        LLMNERSpanAgent(_Canned("{}")).run(st)
        assert st.contextual_output is None
