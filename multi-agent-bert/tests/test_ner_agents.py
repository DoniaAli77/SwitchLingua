"""tests/test_ner_agents.py

Tests for the three NER specialist agents:
  * NERLexicalAgent   (src/agents/ner_lexical_agent.py)
  * NERLogicAgent     (src/agents/ner_logic_agent.py)
  * NERContextualAgent(src/agents/ner_contextual_agent.py)

Each class of tests verifies:
  - One tag is produced per token (core invariant)
  - Correct output slot is populated (lexical/logic/contextual_output)
  - sequence_output is set; model_output.label is NOT set
  - Tokens from state.extras["tokens"] are preferred over whitespace split
  - Whitespace-split fallback works when extras["tokens"] is absent
  - Specific tagging correctness (gazetteers, regexes, capitalisation)
  - B-/I- continuation logic
  - Agent is skipped (returns state unchanged + history note) when
    task_type != "sequence_labeling"
  - History event is appended on both run and skip paths
  - Bad regex patterns are skipped without crashing (NERLogicAgent only)
  - Arabic tokens in extras["tokens"] are preserved
  - Labels not in the valid set fall back to O
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.ner_contextual_agent import NERContextualAgent
from src.agents.ner_lexical_agent import NERLexicalAgent
from src.agents.ner_logic_agent import NERLogicAgent
from src.state.schema import PipelineState, StateMetadata, TaskConfig

# ---------------------------------------------------------------------------
# BIO label set used for all NER tests
# ---------------------------------------------------------------------------

_NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ner_state(
    text: str,
    tokens: list[str] | None = None,
    task_type: str = "sequence_labeling",
) -> PipelineState:
    """Return a PipelineState configured for NER."""
    extras = {}
    if tokens is not None:
        extras["tokens"] = tokens
    return PipelineState(
        metadata=StateMetadata(sample_id="ner-test"),
        input_text=text,
        task_config=TaskConfig(
            task_name="ner",
            task_type=task_type,
            labels=_NER_LABELS,
            label_descriptions={lbl: lbl for lbl in _NER_LABELS},
        ),
        extras=extras,
    )


def _tags(token_tags) -> list[str]:
    return [tt.tag for tt in token_tags]


def _history_components(state: PipelineState) -> list[str]:
    return [e.component for e in state.history]


# ===========================================================================
# NERLexicalAgent
# ===========================================================================

class TestNERLexicalAgent:

    _GAZETTEER = {
        "PER": ["Ahmed", "Sara", "أحمد"],
        "ORG": ["Google", "OpenAI", "Microsoft"],
        "LOC": ["Paris", "Cairo", "London"],
    }

    def _agent(self) -> NERLexicalAgent:
        return NERLexicalAgent(gazetteer=self._GAZETTEER)

    # -- Core invariant --

    def test_one_tag_per_token_extras(self):
        tokens = ["Ahmed", "works", "at", "Google"]
        state = self._agent().run(_ner_state("Ahmed works at Google", tokens=tokens))
        tags = state.lexical_output.sequence_output.tags
        assert len(tags) == len(tokens)

    def test_one_tag_per_token_whitespace_split(self):
        text = "Sara joined Microsoft yesterday"
        state = self._agent().run(_ner_state(text))
        tags = state.lexical_output.sequence_output.tags
        assert len(tags) == len(text.split())

    def test_one_tag_per_token_single_token(self):
        state = self._agent().run(_ner_state("Paris", tokens=["Paris"]))
        assert len(state.lexical_output.sequence_output.tags) == 1

    # -- Output slot --

    def test_lexical_output_populated(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert state.lexical_output is not None

    def test_sequence_output_set(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert state.lexical_output.sequence_output is not None

    def test_model_output_label_not_set(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert state.lexical_output.model_output.label is None

    # -- Tagging correctness --

    def test_known_person_tagged_b_per(self):
        state = self._agent().run(_ner_state("Ahmed works", tokens=["Ahmed", "works"]))
        assert _tags(state.lexical_output.sequence_output.tags)[0] == "B-PER"

    def test_unknown_token_tagged_o(self):
        state = self._agent().run(_ner_state("works", tokens=["works"]))
        assert _tags(state.lexical_output.sequence_output.tags)[0] == "O"

    def test_org_token_tagged_b_org(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert _tags(state.lexical_output.sequence_output.tags)[0] == "B-ORG"

    def test_loc_token_tagged_b_loc(self):
        state = self._agent().run(_ner_state("Paris", tokens=["Paris"]))
        assert _tags(state.lexical_output.sequence_output.tags)[0] == "B-LOC"

    def test_arabic_token_in_extras(self):
        tokens = ["أحمد", "يعمل"]
        state = self._agent().run(_ner_state("أحمد يعمل", tokens=tokens))
        tags = _tags(state.lexical_output.sequence_output.tags)
        assert len(tags) == 2
        assert tags[0] == "B-PER"
        assert tags[1] == "O"

    def test_case_insensitive_lookup(self):
        state = self._agent().run(_ner_state("ahmed", tokens=["ahmed"]))
        # "ahmed" lower == "ahmed"; gazetteer has "Ahmed" — match expected
        assert _tags(state.lexical_output.sequence_output.tags)[0] == "B-PER"

    # -- B/I continuation --

    def test_no_continuation_across_different_types(self):
        tokens = ["Ahmed", "Google"]
        state = self._agent().run(_ner_state("Ahmed Google", tokens=tokens))
        tags = _tags(state.lexical_output.sequence_output.tags)
        assert tags == ["B-PER", "B-ORG"]

    def test_o_resets_continuation(self):
        tokens = ["Ahmed", "works", "Sara"]
        state = self._agent().run(_ner_state("Ahmed works Sara", tokens=tokens))
        tags = _tags(state.lexical_output.sequence_output.tags)
        assert tags == ["B-PER", "O", "B-PER"]

    # -- Skip on non-NER task --

    def test_skipped_for_classification_task(self):
        state = _ner_state("hello world", tokens=["hello", "world"],
                           task_type="classification")
        after = self._agent().run(state)
        assert after.lexical_output is None

    def test_skip_writes_history(self):
        state = _ner_state("hello", tokens=["hello"], task_type="classification")
        after = self._agent().run(state)
        assert "NERLexicalAgent" in _history_components(after)

    # -- History on run path --

    def test_run_writes_history(self):
        state = _ner_state("Ahmed", tokens=["Ahmed"])
        after = self._agent().run(state)
        assert "NERLexicalAgent" in _history_components(after)

    # -- Empty gazetteer --

    def test_empty_gazetteer_all_o(self):
        agent = NERLexicalAgent(gazetteer={})
        tokens = ["Ahmed", "works", "at", "Google"]
        state = agent.run(_ner_state("Ahmed works at Google", tokens=tokens))
        assert all(tt.tag == "O" for tt in state.lexical_output.sequence_output.tags)


# ===========================================================================
# NERLogicAgent
# ===========================================================================

class TestNERLogicAgent:

    _RULES = {
        "PER": [r"(Dr|Mr|Ms)\.?\s+\w+", r"Ahmed|Sara|أحمد"],
        "ORG": [r"Google|OpenAI|Microsoft|Inc\.?"],
        "LOC": [r"Paris|Cairo|London|New\s+York"],
    }

    def _agent(self) -> NERLogicAgent:
        return NERLogicAgent(rule_map=self._RULES)

    # -- Core invariant --

    def test_one_tag_per_token_extras(self):
        tokens = ["Ahmed", "works", "at", "Google"]
        state = self._agent().run(_ner_state("Ahmed works at Google", tokens=tokens))
        assert len(state.logic_output.sequence_output.tags) == len(tokens)

    def test_one_tag_per_token_whitespace_split(self):
        text = "Ahmed went to Paris"
        state = self._agent().run(_ner_state(text))
        assert len(state.logic_output.sequence_output.tags) == len(text.split())

    def test_one_tag_per_token_single_token(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert len(state.logic_output.sequence_output.tags) == 1

    # -- Output slot --

    def test_logic_output_populated(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert state.logic_output is not None

    def test_sequence_output_set(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert state.logic_output.sequence_output is not None

    def test_model_output_label_not_set(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert state.logic_output.model_output.label is None

    # -- Tagging correctness --

    def test_org_regex_match(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert _tags(state.logic_output.sequence_output.tags)[0] == "B-ORG"

    def test_per_regex_match(self):
        state = self._agent().run(_ner_state("Ahmed", tokens=["Ahmed"]))
        assert _tags(state.logic_output.sequence_output.tags)[0] == "B-PER"

    def test_loc_regex_match(self):
        state = self._agent().run(_ner_state("Paris", tokens=["Paris"]))
        assert _tags(state.logic_output.sequence_output.tags)[0] == "B-LOC"

    def test_unmatched_token_is_o(self):
        state = self._agent().run(_ner_state("works", tokens=["works"]))
        assert _tags(state.logic_output.sequence_output.tags)[0] == "O"

    def test_arabic_token_matches_per_rule(self):
        tokens = ["أحمد", "يعمل"]
        state = self._agent().run(_ner_state("أحمد يعمل", tokens=tokens))
        tags = _tags(state.logic_output.sequence_output.tags)
        assert len(tags) == 2
        assert tags[0] == "B-PER"

    # -- B/I continuation --

    def test_o_resets_continuation(self):
        tokens = ["Ahmed", "went", "to", "Paris"]
        state = self._agent().run(_ner_state("Ahmed went to Paris", tokens=tokens))
        tags = _tags(state.logic_output.sequence_output.tags)
        assert tags[0] == "B-PER"
        assert tags[1] == "O"
        assert tags[2] == "O"
        assert tags[3] == "B-LOC"

    # -- Skip on non-NER task --

    def test_skipped_for_classification_task(self):
        state = _ner_state("hello", tokens=["hello"], task_type="classification")
        after = self._agent().run(state)
        assert after.logic_output is None

    def test_skip_writes_history(self):
        state = _ner_state("hello", tokens=["hello"], task_type="classification")
        after = self._agent().run(state)
        assert "NERLogicAgent" in _history_components(after)

    # -- History on run path --

    def test_run_writes_history(self):
        state = _ner_state("Google", tokens=["Google"])
        after = self._agent().run(state)
        assert "NERLogicAgent" in _history_components(after)

    # -- Bad regex is silently skipped --

    def test_bad_regex_does_not_crash(self):
        agent = NERLogicAgent(rule_map={"PER": [r"[invalid("]})
        # Should construct without error; bad pattern skipped.
        state = agent.run(_ner_state("Ahmed", tokens=["Ahmed"]))
        # All O because the only rule was invalid.
        assert all(tt.tag == "O" for tt in state.logic_output.sequence_output.tags)

    # -- Empty rule map --

    def test_empty_rules_all_o(self):
        agent = NERLogicAgent(rule_map={})
        tokens = ["Ahmed", "Google"]
        state = agent.run(_ner_state("Ahmed Google", tokens=tokens))
        assert all(tt.tag == "O" for tt in state.logic_output.sequence_output.tags)


# ===========================================================================
# NERContextualAgent
# ===========================================================================

class TestNERContextualAgent:

    _KNOWN = {
        "google": "ORG",
        "paris":  "LOC",
        "ahmed":  "PER",
        "أحمد":   "PER",
    }

    def _agent(self) -> NERContextualAgent:
        return NERContextualAgent(known_entities=self._KNOWN)

    # -- Core invariant --

    def test_one_tag_per_token_extras(self):
        tokens = ["Ahmed", "works", "at", "Google"]
        state = self._agent().run(_ner_state("Ahmed works at Google", tokens=tokens))
        assert len(state.contextual_output.sequence_output.tags) == len(tokens)

    def test_one_tag_per_token_whitespace_split(self):
        text = "Ahmed went to Paris today"
        state = self._agent().run(_ner_state(text))
        assert len(state.contextual_output.sequence_output.tags) == len(text.split())

    def test_one_tag_per_token_single_token(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert len(state.contextual_output.sequence_output.tags) == 1

    # -- Output slot --

    def test_contextual_output_populated(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert state.contextual_output is not None

    def test_sequence_output_set(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert state.contextual_output.sequence_output is not None

    def test_model_output_label_not_set(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert state.contextual_output.model_output.label is None

    # -- Tagging correctness (known_entities) --

    def test_known_org_tagged(self):
        state = self._agent().run(_ner_state("Google", tokens=["Google"]))
        assert _tags(state.contextual_output.sequence_output.tags)[0] == "B-ORG"

    def test_known_loc_tagged(self):
        state = self._agent().run(_ner_state("Paris", tokens=["Paris"]))
        assert _tags(state.contextual_output.sequence_output.tags)[0] == "B-LOC"

    def test_known_per_tagged(self):
        state = self._agent().run(_ner_state("ahmed", tokens=["ahmed"]))
        assert _tags(state.contextual_output.sequence_output.tags)[0] == "B-PER"

    def test_known_arabic_per_tagged(self):
        tokens = ["أحمد", "يعمل"]
        state = self._agent().run(_ner_state("أحمد يعمل", tokens=tokens))
        tags = _tags(state.contextual_output.sequence_output.tags)
        assert len(tags) == 2
        assert tags[0] == "B-PER"

    # -- Capitalisation heuristic --

    def test_capitalised_unknown_token_tagged_per(self):
        # "London" is not in known_entities but starts with uppercase.
        state = self._agent().run(_ner_state("London", tokens=["London"]))
        assert _tags(state.contextual_output.sequence_output.tags)[0] == "B-PER"

    def test_lowercase_unknown_not_tagged(self):
        state = self._agent().run(_ner_state("works", tokens=["works"]))
        assert _tags(state.contextual_output.sequence_output.tags)[0] == "O"

    # -- Arabic token not capitalised → O (unless in known_entities) --

    def test_arabic_unknown_token_is_o(self):
        tokens = ["يعمل"]
        state = self._agent().run(_ner_state("يعمل", tokens=tokens))
        assert _tags(state.contextual_output.sequence_output.tags)[0] == "O"

    # -- B/I continuation --

    def test_o_resets_continuation(self):
        tokens = ["Ahmed", "likes", "Paris"]
        state = self._agent().run(_ner_state("Ahmed likes Paris", tokens=tokens))
        tags = _tags(state.contextual_output.sequence_output.tags)
        assert tags[0] == "B-PER"
        assert tags[1] == "O"
        assert tags[2] == "B-LOC"

    # -- Skip on non-NER task --

    def test_skipped_for_classification_task(self):
        state = _ner_state("hello", tokens=["hello"], task_type="classification")
        after = self._agent().run(state)
        assert after.contextual_output is None

    def test_skip_writes_history(self):
        state = _ner_state("hello", tokens=["hello"], task_type="classification")
        after = self._agent().run(state)
        assert "NERContextualAgent" in _history_components(after)

    # -- History on run path --

    def test_run_writes_history(self):
        state = _ner_state("Google", tokens=["Google"])
        after = self._agent().run(state)
        assert "NERContextualAgent" in _history_components(after)

    # -- Empty known_entities --

    def test_empty_known_entities_capitalisation_only(self):
        agent = NERContextualAgent(known_entities={})
        tokens = ["Ahmed", "works"]
        state = agent.run(_ner_state("Ahmed works", tokens=tokens))
        tags = _tags(state.contextual_output.sequence_output.tags)
        assert tags[0] == "B-PER"   # capitalised heuristic
        assert tags[1] == "O"       # lowercase

    def test_known_entities_case_insensitive(self):
        agent = NERContextualAgent(known_entities={"GOOGLE": "ORG"})
        state = agent.run(_ner_state("google", tokens=["google"]))
        assert _tags(state.contextual_output.sequence_output.tags)[0] == "B-ORG"
