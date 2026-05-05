"""tests/test_ner_consensus_agent.py

Tests for NERConsensusAgent (src/agents/ner_consensus_agent.py).

Covers:
  - Basic happy path (all 3 slots agree)
  - Majority vote (2 of 3 disagree)
  - Weighted voting (heavier slot wins against majority)
  - Tie-breaking: equal scores → earliest label in task_config.labels wins
  - state.final_output.payload["sequence_output"] structure
  - state.final_output.payload["token_count"]
  - state.consensus_output fields (label=None, rationale, votes keys)
  - Only lexical slot present
  - Only logic slot present
  - Only contextual slot present
  - No slot present → all tokens get O at confidence 0
  - Token count mismatch between slots (shorter slot skips position)
  - Token resolution priority: extras["tokens"] > sequence_output tokens > split
  - Tokens taken from sequence_output when extras["tokens"] absent
  - Whitespace-split fallback
  - Skip when task_type != "sequence_labeling"
  - History written on run path
  - History written on skip path
  - Zero-weight slot is silenced
  - Confidence clamped to [0, 1]
  - Single-token input
  - Multi-token input
  - _vote_token and _extract_seq_output helpers
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.ner_consensus_agent import (
    NERConsensusAgent,
    _extract_seq_output,
    _vote_token,
)
from src.state.schema import (
    AgentOutput,
    ConsensusOutput,
    FinalOutput,
    ModelOutput,
    PipelineState,
    SequenceLabelingOutput,
    StateMetadata,
    TaskConfig,
    TokenTag,
)

# ---------------------------------------------------------------------------
# BIO label set
# ---------------------------------------------------------------------------

_NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ner_state(
    tokens: list[str],
    task_type: str = "sequence_labeling",
    put_tokens_in_extras: bool = True,
) -> PipelineState:
    extras = {"tokens": tokens} if put_tokens_in_extras else {}
    return PipelineState(
        metadata=StateMetadata(sample_id="t"),
        input_text=" ".join(tokens),
        task_config=TaskConfig(
            task_name="ner",
            task_type=task_type,
            labels=_NER_LABELS,
        ),
        extras=extras,
    )


def _seq_out(tags: list[str], tokens: list[str], conf: float = 0.9) -> SequenceLabelingOutput:
    return SequenceLabelingOutput(
        tags=[TokenTag(token=t, tag=g, confidence=conf) for t, g in zip(tokens, tags)]
    )


def _agent_out(name: str, tags: list[str], tokens: list[str], conf: float = 0.9) -> AgentOutput:
    return AgentOutput(
        agent_name=name,
        model_output=ModelOutput(),
        sequence_output=_seq_out(tags, tokens, conf=conf),
    )


def _final_tags(state: PipelineState) -> list[str]:
    return [e["tag"] for e in state.final_output.payload["sequence_output"]]


def _final_tokens(state: PipelineState) -> list[str]:
    return [e["token"] for e in state.final_output.payload["sequence_output"]]


def _history_components(state: PipelineState) -> list[str]:
    return [e.component for e in state.history]


# ===========================================================================
# Helper function tests
# ===========================================================================

class TestExtractSeqOutput:

    def test_returns_none_for_none(self):
        assert _extract_seq_output(None) is None

    def test_returns_none_when_sequence_output_absent(self):
        ao = AgentOutput("a", model_output=ModelOutput())
        assert _extract_seq_output(ao) is None

    def test_returns_sequence_output(self):
        so = _seq_out(["O"], ["x"])
        ao = AgentOutput("a", model_output=ModelOutput(), sequence_output=so)
        assert _extract_seq_output(ao) is so


class TestVoteToken:

    def test_single_slot_wins(self):
        slots = {
            "lexical": _seq_out(["B-PER"], ["Ahmed"]),
            "contextual": None,
            "logic": None,
        }
        result = _vote_token(0, slots, {"lexical": 1.0, "contextual": 1.0, "logic": 1.0}, _NER_LABELS)
        assert result.tag == "B-PER"

    def test_all_agree(self):
        so = _seq_out(["B-ORG"], ["Google"])
        slots = {"lexical": so, "contextual": so, "logic": so}
        result = _vote_token(0, slots, {"lexical": 1.0, "contextual": 1.0, "logic": 1.0}, _NER_LABELS)
        assert result.tag == "B-ORG"

    def test_majority_wins(self):
        slots = {
            "lexical":     _seq_out(["B-PER"], ["Ahmed"]),
            "contextual":  _seq_out(["B-PER"], ["Ahmed"]),
            "logic":       _seq_out(["O"],     ["Ahmed"]),
        }
        result = _vote_token(0, slots, {"lexical": 1.0, "contextual": 1.0, "logic": 1.0}, _NER_LABELS)
        assert result.tag == "B-PER"

    def test_weighted_minority_wins(self):
        # logic says B-LOC at weight 5.0; lexical+contextual say O at weight 1 each
        slots = {
            "lexical":     _seq_out(["O"],     ["Paris"], conf=0.9),
            "contextual":  _seq_out(["O"],     ["Paris"], conf=0.9),
            "logic":       _seq_out(["B-LOC"], ["Paris"], conf=0.9),
        }
        weights = {"lexical": 1.0, "contextual": 1.0, "logic": 5.0}
        result = _vote_token(0, slots, weights, _NER_LABELS)
        assert result.tag == "B-LOC"

    def test_tie_broken_by_label_order(self):
        # "O" and "B-PER" get equal scores; "O" is first in _NER_LABELS → wins
        slots = {
            "lexical":     _seq_out(["O"],     ["x"], conf=0.9),
            "contextual":  _seq_out(["B-PER"], ["x"], conf=0.9),
            "logic":       None,
        }
        result = _vote_token(0, slots, {"lexical": 1.0, "contextual": 1.0, "logic": 1.0}, _NER_LABELS)
        assert result.tag == "O"

    def test_no_slots_returns_o(self):
        slots = {"lexical": None, "contextual": None, "logic": None}
        result = _vote_token(0, slots, {"lexical": 1.0, "contextual": 1.0, "logic": 1.0}, _NER_LABELS)
        assert result.tag == "O"
        assert result.confidence == 0.0

    def test_position_out_of_range_skipped(self):
        # Position 2 but slot only has 1 token → slot contributes no vote
        slots = {
            "lexical":    _seq_out(["B-PER"], ["Ahmed"]),     # only index 0
            "contextual": _seq_out(["O", "O", "B-LOC"], ["a", "b", "c"]),
            "logic":      None,
        }
        result = _vote_token(2, slots, {"lexical": 1.0, "contextual": 1.0, "logic": 1.0}, _NER_LABELS)
        assert result.tag == "B-LOC"  # only contextual voted

    def test_zero_weight_slot_silenced(self):
        slots = {
            "lexical":     _seq_out(["B-PER"], ["Ahmed"], conf=0.95),
            "contextual":  _seq_out(["O"],     ["Ahmed"], conf=0.95),
            "logic":       None,
        }
        # Zero-weight lexical — contextual O should win
        result = _vote_token(0, slots, {"lexical": 0.0, "contextual": 1.0, "logic": 0.0}, _NER_LABELS)
        assert result.tag == "O"

    def test_confidence_normalised(self):
        # Single slot at conf 0.8 → final_conf should be 0.8
        slots = {
            "lexical":    _seq_out(["B-PER"], ["Ahmed"], conf=0.8),
            "contextual": None,
            "logic":      None,
        }
        result = _vote_token(0, slots, {"lexical": 1.0, "contextual": 1.0, "logic": 1.0}, _NER_LABELS)
        assert result.confidence == pytest.approx(0.8, abs=1e-5)


# ===========================================================================
# NERConsensusAgent integration tests
# ===========================================================================

class TestNERConsensusAgentBasic:

    def test_all_agree_single_token(self):
        tokens = ["Google"]
        state = _ner_state(tokens)
        state.lexical_output    = _agent_out("lex", ["B-ORG"], tokens)
        state.contextual_output = _agent_out("ctx", ["B-ORG"], tokens)
        state.logic_output      = _agent_out("log", ["B-ORG"], tokens)
        NERConsensusAgent().run(state)
        assert _final_tags(state) == ["B-ORG"]

    def test_all_agree_multi_token(self):
        tokens = ["Ahmed", "works", "at", "Google"]
        expected = ["B-PER", "O", "O", "B-ORG"]
        state = _ner_state(tokens)
        state.lexical_output    = _agent_out("lex", expected, tokens)
        state.contextual_output = _agent_out("ctx", expected, tokens)
        state.logic_output      = _agent_out("log", expected, tokens)
        NERConsensusAgent().run(state)
        assert _final_tags(state) == expected

    def test_majority_vote_multi_token(self):
        tokens = ["Ahmed", "went"]
        state = _ner_state(tokens)
        # 2 of 3 say B-PER for "Ahmed"; 2 of 3 say O for "went"
        state.lexical_output    = _agent_out("lex", ["B-PER", "O"], tokens)
        state.contextual_output = _agent_out("ctx", ["B-PER", "O"], tokens)
        state.logic_output      = _agent_out("log", ["O",     "O"], tokens)
        NERConsensusAgent().run(state)
        assert _final_tags(state) == ["B-PER", "O"]

    def test_weighted_minority_wins(self):
        tokens = ["Paris"]
        state = _ner_state(tokens)
        state.lexical_output    = _agent_out("lex", ["O"],     tokens, conf=0.9)
        state.contextual_output = _agent_out("ctx", ["O"],     tokens, conf=0.9)
        state.logic_output      = _agent_out("log", ["B-LOC"], tokens, conf=0.9)
        NERConsensusAgent(weights={"logic": 5.0}).run(state)
        assert _final_tags(state) == ["B-LOC"]


class TestNERConsensusAgentOutputStructure:

    def _run(self, tokens, tags):
        state = _ner_state(tokens)
        state.lexical_output = _agent_out("lex", tags, tokens)
        NERConsensusAgent().run(state)
        return state

    def test_final_output_is_set(self):
        state = self._run(["Ahmed"], ["B-PER"])
        assert state.final_output is not None

    def test_payload_has_sequence_output_key(self):
        state = self._run(["Ahmed"], ["B-PER"])
        assert "sequence_output" in state.final_output.payload

    def test_payload_token_count(self):
        tokens = ["Ahmed", "works", "at", "Google"]
        state = self._run(tokens, ["B-PER", "O", "O", "B-ORG"])
        assert state.final_output.payload["token_count"] == 4

    def test_payload_sequence_output_length(self):
        tokens = ["Ahmed", "works"]
        state = self._run(tokens, ["B-PER", "O"])
        assert len(state.final_output.payload["sequence_output"]) == 2

    def test_payload_entry_has_token_tag_confidence(self):
        state = self._run(["Ahmed"], ["B-PER"])
        entry = state.final_output.payload["sequence_output"][0]
        assert "token" in entry
        assert "tag" in entry
        assert "confidence" in entry

    def test_payload_token_strings_preserved(self):
        tokens = ["Ahmed", "works"]
        state = self._run(tokens, ["B-PER", "O"])
        assert _final_tokens(state) == tokens

    def test_consensus_output_label_is_none(self):
        state = self._run(["Ahmed"], ["B-PER"])
        assert state.consensus_output.label is None

    def test_consensus_output_confidence_is_none(self):
        state = self._run(["Ahmed"], ["B-PER"])
        assert state.consensus_output.confidence is None

    def test_consensus_output_rationale(self):
        state = self._run(["Ahmed"], ["B-PER"])
        assert state.consensus_output.rationale == "sequence_labeling_completed"

    def test_consensus_output_votes_token_count(self):
        tokens = ["Ahmed", "works"]
        state = self._run(tokens, ["B-PER", "O"])
        assert state.consensus_output.votes["token_count"] == "2"

    def test_consensus_output_votes_tag_distribution(self):
        tokens = ["Ahmed", "works"]
        state = self._run(tokens, ["B-PER", "O"])
        dist = state.consensus_output.votes["tag_distribution"]
        assert "B-PER" in dist
        assert "O" in dist


class TestNERConsensusAgentPartialSlots:

    def test_only_lexical_slot(self):
        tokens = ["Google"]
        state = _ner_state(tokens)
        state.lexical_output = _agent_out("lex", ["B-ORG"], tokens)
        NERConsensusAgent().run(state)
        assert _final_tags(state) == ["B-ORG"]

    def test_only_logic_slot(self):
        tokens = ["Paris"]
        state = _ner_state(tokens)
        state.logic_output = _agent_out("log", ["B-LOC"], tokens)
        NERConsensusAgent().run(state)
        assert _final_tags(state) == ["B-LOC"]

    def test_only_contextual_slot(self):
        tokens = ["Ahmed"]
        state = _ner_state(tokens)
        state.contextual_output = _agent_out("ctx", ["B-PER"], tokens)
        NERConsensusAgent().run(state)
        assert _final_tags(state) == ["B-PER"]

    def test_no_slots_all_o(self):
        tokens = ["some", "text"]
        state = _ner_state(tokens)
        NERConsensusAgent().run(state)
        assert _final_tags(state) == ["O", "O"]

    def test_shorter_slot_does_not_crash(self):
        """Slot with fewer tokens than canonical list is safely skipped per position."""
        tokens = ["Ahmed", "works", "at", "Google"]
        state = _ner_state(tokens)
        # lexical has all 4; logic only has 2
        state.lexical_output = _agent_out("lex", ["B-PER", "O", "O", "B-ORG"], tokens)
        state.logic_output   = _agent_out("log", ["B-PER", "O"], tokens[:2])
        NERConsensusAgent().run(state)
        tags = _final_tags(state)
        assert len(tags) == 4
        assert tags[0] == "B-PER"
        assert tags[3] == "B-ORG"


class TestNERConsensusAgentTokenResolution:

    def test_extras_tokens_used(self):
        tokens = ["Ahmed", "works"]
        state = _ner_state(tokens, put_tokens_in_extras=True)
        state.lexical_output = _agent_out("lex", ["B-PER", "O"], tokens)
        NERConsensusAgent().run(state)
        assert _final_tokens(state) == tokens

    def test_sequence_output_tokens_used_when_no_extras(self):
        tokens = ["Google", "Inc"]
        state = _ner_state(tokens, put_tokens_in_extras=False)
        state.lexical_output = _agent_out("lex", ["B-ORG", "I-ORG"], tokens)
        NERConsensusAgent().run(state)
        assert _final_tokens(state) == tokens

    def test_whitespace_split_fallback(self):
        state = _ner_state(["hello", "world"], put_tokens_in_extras=False)
        # No agent outputs — fallback to whitespace split of input_text
        NERConsensusAgent().run(state)
        assert _final_tokens(state) == ["hello", "world"]

    def test_single_token(self):
        tokens = ["Paris"]
        state = _ner_state(tokens)
        state.lexical_output = _agent_out("lex", ["B-LOC"], tokens)
        NERConsensusAgent().run(state)
        assert len(state.final_output.payload["sequence_output"]) == 1
        assert _final_tags(state) == ["B-LOC"]


class TestNERConsensusAgentHistory:

    def test_run_writes_history(self):
        tokens = ["Ahmed"]
        state = _ner_state(tokens)
        state.lexical_output = _agent_out("lex", ["B-PER"], tokens)
        NERConsensusAgent().run(state)
        assert "NERConsensusAgent" in [e.component for e in state.history]

    def test_history_includes_token_count(self):
        tokens = ["Ahmed", "works"]
        state = _ner_state(tokens)
        state.lexical_output = _agent_out("lex", ["B-PER", "O"], tokens)
        NERConsensusAgent().run(state)
        ev = next(e for e in state.history if e.component == "NERConsensusAgent")
        assert ev.outputs["token_count"] == 2


class TestNERConsensusAgentSkip:

    def test_skipped_for_classification(self):
        tokens = ["hello"]
        state = _ner_state(tokens, task_type="classification")
        NERConsensusAgent().run(state)
        assert state.final_output is None
        assert state.consensus_output is None

    def test_skip_writes_history(self):
        tokens = ["hello"]
        state = _ner_state(tokens, task_type="classification")
        NERConsensusAgent().run(state)
        assert "NERConsensusAgent" in [e.component for e in state.history]

    def test_skip_does_not_overwrite_existing_final_output(self):
        tokens = ["hello"]
        state = _ner_state(tokens, task_type="classification")
        existing = FinalOutput(label="positive", confidence=0.9)
        state.final_output = existing
        NERConsensusAgent().run(state)
        assert state.final_output.label == "positive"


class TestNERConsensusAgentWeights:

    def test_zero_weight_slot_silenced(self):
        tokens = ["Ahmed"]
        state = _ner_state(tokens)
        state.lexical_output    = _agent_out("lex", ["B-PER"], tokens, conf=0.95)
        state.contextual_output = _agent_out("ctx", ["O"],     tokens, conf=0.95)
        # Zero out lexical → contextual O wins
        NERConsensusAgent(weights={"lexical": 0.0}).run(state)
        assert _final_tags(state) == ["O"]

    def test_custom_weight_overrides_majority(self):
        tokens = ["Paris"]
        state = _ner_state(tokens)
        state.lexical_output    = _agent_out("lex", ["O"],     tokens, conf=0.9)
        state.contextual_output = _agent_out("ctx", ["O"],     tokens, conf=0.9)
        state.logic_output      = _agent_out("log", ["B-LOC"], tokens, conf=0.9)
        # Give logic weight 10 → it overrides 2-vs-1
        NERConsensusAgent(weights={"logic": 10.0}).run(state)
        assert _final_tags(state) == ["B-LOC"]

    def test_tie_broken_by_label_order(self):
        # Equal weights: lexical says O, contextual says B-PER
        # O comes first in _NER_LABELS → O wins tie
        tokens = ["x"]
        state = _ner_state(tokens)
        state.lexical_output    = _agent_out("lex", ["O"],     tokens, conf=0.9)
        state.contextual_output = _agent_out("ctx", ["B-PER"], tokens, conf=0.9)
        NERConsensusAgent(weights={"lexical": 1.0, "contextual": 1.0, "logic": 0.0}).run(state)
        assert _final_tags(state) == ["O"]
