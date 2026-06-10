"""Unit tests for src.agents.lexical_agent.LexicalAgent."""

from __future__ import annotations

import pytest

from src.agents.lexical_agent import LexicalAgent, _NO_MATCH_NOTE
from src.state.schema import PipelineState, StateMetadata, TaskConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(text: str, labels: list[str] | None = None) -> PipelineState:
    """Return a minimal PipelineState with the given input text."""
    if labels is None:
        labels = ["tech", "sports", "finance"]
    return PipelineState(
        metadata=StateMetadata(sample_id="test-001"),
        input_text=text,
        task_config=TaskConfig(
            task_name="topic_classification",
            labels=labels,
        ),
    )


KEYWORD_MAP = {
    "tech": ["software", "AI", "startup", "cloud"],
    "sports": ["match", "league", "goal", "tournament"],
    "finance": ["stocks", "bank", "earnings"],
}


# ---------------------------------------------------------------------------
# English (Latin-script) matching
# ---------------------------------------------------------------------------

class TestEnglishMatching:
    def test_single_label_all_keywords_match(self):
        agent = LexicalAgent(KEYWORD_MAP)
        state = agent.run(make_state("AI software drives the startup boom"))
        out = state.lexical_output
        assert out is not None
        assert out.model_output.label == "tech"
        # 3 tech hits, 0 others → confidence == 1.0
        assert out.model_output.confidence == pytest.approx(1.0)

    def test_partial_match_correct_label(self):
        agent = LexicalAgent(KEYWORD_MAP)
        state = agent.run(make_state("The bank reported strong earnings this quarter"))
        out = state.lexical_output
        assert out.model_output.label == "finance"
        assert out.model_output.probabilities["finance"] > 0.5

    def test_whole_word_not_substring(self):
        # "AI" must not match "FAIL" or "rail"
        agent = LexicalAgent({"tech": ["AI"], "sports": ["match"]})
        state = agent.run(make_state("rail infrastructure FAILED today", ["tech", "sports"]))
        # No whole-word "AI" → abstain (no vote), not a first-label fallback
        out = state.lexical_output
        assert out.model_output.label is None
        assert out.model_output.probabilities == {}
        assert out.features.get("abstained") is True
        assert out.notes == _NO_MATCH_NOTE

    def test_case_insensitive_latin(self):
        agent = LexicalAgent({"tech": ["software"], "sports": ["match"]})
        state = agent.run(make_state("I love SOFTWARE engineering"))
        out = state.lexical_output
        assert out.model_output.label == "tech"
        assert out.model_output.confidence == pytest.approx(1.0)

    def test_multiple_labels_tie_broken_by_first(self):
        # sports gets 1 hit, tech gets 1 hit — tie → first label in list wins
        labels = ["tech", "sports"]
        agent = LexicalAgent({"tech": ["software"], "sports": ["match"]})
        state = agent.run(make_state("software match result", labels))
        out = state.lexical_output
        assert out.model_output.label == "tech"          # first in labels list
        assert out.model_output.confidence == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Arabic (non-ASCII) matching
# ---------------------------------------------------------------------------

class TestArabicMatching:
    def test_arabic_keyword_match(self):
        agent = LexicalAgent({"tech": ["تقنية", "برمجة"], "sports": ["رياضة"]})
        state = agent.run(make_state("أنا أعمل في برمجة التطبيقات", ["tech", "sports"]))
        out = state.lexical_output
        assert out.model_output.label == "tech"
        assert out.model_output.confidence == pytest.approx(1.0)

    def test_arabic_multi_word_keyword(self):
        agent = LexicalAgent({"tech": ["ذكاء اصطناعي"], "sports": ["كرة القدم"]})
        state = agent.run(
            make_state("يُعدّ الذكاء الاصطناعي محور التطور", ["tech", "sports"])
        )
        # "ذكاء اصطناعي" as substring — will NOT match "الذكاء الاصطناعي"
        # This confirms the agent doesn't add phantom matches; no match → abstain.
        out = state.lexical_output
        assert out.model_output.label is None  # abstain, not first-label
        assert out.notes == _NO_MATCH_NOTE

    def test_exact_arabic_substring_match(self):
        agent = LexicalAgent({"tech": ["برمجة"], "sports": ["رياضة"]})
        state = agent.run(make_state("برمجة الأنظمة", ["tech", "sports"]))
        out = state.lexical_output
        assert out.model_output.label == "tech"
        assert out.model_output.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Code-switched (Arabic + English) text
# ---------------------------------------------------------------------------

class TestCodeSwitchedMatching:
    def test_mixed_language_keywords_both_match(self):
        agent = LexicalAgent({"tech": ["software", "تقنية"], "sports": ["match"]})
        state = agent.run(make_state("استخدمت software جديد في التقنية"))
        out = state.lexical_output
        assert out.model_output.label == "tech"
        # 2 tech hits → confidence 1.0
        assert out.model_output.confidence == pytest.approx(1.0)

    def test_mixed_language_split_across_labels(self):
        agent = LexicalAgent({"tech": ["AI"], "sports": ["رياضة"]})
        state = agent.run(make_state("أحب رياضة AI"))
        out = state.lexical_output
        # 1 hit each → tie → first label wins
        assert out.model_output.label == "tech"
        assert out.model_output.probabilities["tech"] == pytest.approx(0.5)
        assert out.model_output.probabilities["sports"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# No-match fallback
# ---------------------------------------------------------------------------

class TestNoMatchFallback:
    def test_no_match_abstains_with_empty_probabilities(self):
        agent = LexicalAgent({"tech": ["quantum"], "sports": ["polo"], "finance": ["bitcoin"]})
        state = agent.run(make_state("today is a nice day"))
        out = state.lexical_output
        assert out.model_output.label is None
        assert out.model_output.probabilities == {}   # no fake uniform distribution
        assert out.features.get("abstained") is True

    def test_no_match_returns_abstain_not_first_label(self):
        labels = ["finance", "tech", "sports"]
        agent = LexicalAgent({})
        state = agent.run(make_state("nothing relevant here", labels))
        out = state.lexical_output
        assert out.model_output.label is None         # NOT labels[0] ('finance')
        assert out.features.get("abstained") is True

    def test_no_match_note_set(self):
        agent = LexicalAgent({})
        state = agent.run(make_state("irrelevant", ["tech", "sports"]))
        assert state.lexical_output.notes == _NO_MATCH_NOTE

    def test_empty_keyword_map_abstains(self):
        agent = LexicalAgent({})
        state = agent.run(make_state("any text", ["tech"]))
        out = state.lexical_output
        assert out.model_output.label is None         # no keywords → abstain
        assert out.features.get("abstained") is True


# ---------------------------------------------------------------------------
# Output structure completeness
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_evidence_format_label_colon_keyword(self):
        agent = LexicalAgent({"tech": ["AI", "cloud"], "sports": ["match"]})
        state = agent.run(make_state("AI runs in the cloud"))
        evidence = state.lexical_output.features["matched_evidence"]
        assert "tech:AI" in evidence
        assert "tech:cloud" in evidence
        assert not any(e.startswith("sports:") for e in evidence)

    def test_raw_scores_present(self):
        agent = LexicalAgent({"tech": ["AI"], "sports": ["match", "goal"]})
        state = agent.run(make_state("AI goal match"))
        scores = state.lexical_output.features["raw_scores"]
        assert scores["tech"] == 1
        assert scores["sports"] == 2

    def test_probabilities_sum_to_one(self):
        agent = LexicalAgent(KEYWORD_MAP)
        state = agent.run(make_state("AI and stocks and match"))
        probs = state.lexical_output.model_output.probabilities
        assert sum(probs.values()) == pytest.approx(1.0, rel=1e-5)

    def test_raw_text_preserved_in_output(self):
        text = "AI drives innovation"
        agent = LexicalAgent(KEYWORD_MAP)
        state = agent.run(make_state(text))
        assert state.lexical_output.model_output.raw_text == text

    def test_agent_name_in_output(self):
        agent = LexicalAgent(KEYWORD_MAP, name="my_lexical")
        state = agent.run(make_state("AI"))
        assert state.lexical_output.agent_name == "my_lexical"

    def test_reasoning_string_populated_on_match(self):
        agent = LexicalAgent({"tech": ["AI"], "sports": []})
        state = agent.run(make_state("AI", ["tech", "sports"]))
        notes = state.lexical_output.notes
        assert "tech" in notes
        assert "%" in notes  # percentage present


# ---------------------------------------------------------------------------
# Validation hooks
# ---------------------------------------------------------------------------

class TestValidation:
    def test_raises_on_blank_input_text(self):
        agent = LexicalAgent(KEYWORD_MAP)
        with pytest.raises(ValueError, match="input_text"):
            agent.execute(make_state("   "))

    def test_raises_on_empty_labels(self):
        agent = LexicalAgent(KEYWORD_MAP)
        state = make_state("AI", labels=[])
        with pytest.raises(ValueError, match="labels"):
            agent.execute(state)
