"""Unit tests for src.agents.logic_agent.LogicAgent."""

from __future__ import annotations

import pytest

from src.agents.logic_agent import LogicAgent, _NO_MATCH_NOTE
from src.state.schema import PipelineState, StateMetadata, TaskConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(text: str, labels: list[str] | None = None) -> PipelineState:
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


RULE_MAP = {
    "tech": [r"\b(app|api|sdk|AI|cloud)\b", r"\b(software|startup)\b"],
    "finance": [r"\b(GDP|IPO|ROI|ETF)\b", r"\b(stocks|bank|earnings)\b"],
    "sports": [r"\b(goal|match|league|tournament)\b"],
}


# ---------------------------------------------------------------------------
# Single-label clear match
# ---------------------------------------------------------------------------

class TestSingleLabelMatch:
    def test_tech_rules_fire(self):
        agent = LogicAgent(RULE_MAP)
        state = agent.run(make_state("The new API and SDK are live on the cloud"))
        out = state.logic_output
        assert out.model_output.label == "tech"
        assert out.model_output.confidence == pytest.approx(1.0)

    def test_finance_acronym_rule_fires(self):
        agent = LogicAgent(RULE_MAP)
        state = agent.run(make_state("IPO valuation above expected GDP growth"))
        out = state.logic_output
        assert out.model_output.label == "finance"
        # Both finance rules fired → 2/2 hits for finance, 0 others
        assert out.model_output.confidence == pytest.approx(1.0)

    def test_sports_rule_fires(self):
        agent = LogicAgent(RULE_MAP)
        state = agent.run(make_state("The league match ended in a goal"))
        out = state.logic_output
        assert out.model_output.label == "sports"

    def test_case_insensitive_match(self):
        agent = LogicAgent({"tech": [r"\bSOFTWARE\b"], "sports": [r"\bmatch\b"]})
        state = agent.run(make_state("software engineering is great", ["tech", "sports"]))
        assert state.logic_output.model_output.label == "tech"


# ---------------------------------------------------------------------------
# Multi-label scoring and tiebreak
# ---------------------------------------------------------------------------

class TestMultiLabelScoring:
    def test_scores_split_between_labels(self):
        agent = LogicAgent(RULE_MAP)
        # "bank" fires finance, "match" fires sports
        state = agent.run(make_state("bank match results"))
        out = state.logic_output
        probs = out.model_output.probabilities
        assert probs["finance"] == pytest.approx(0.5)
        assert probs["sports"] == pytest.approx(0.5)
        assert probs["tech"] == pytest.approx(0.0)

    def test_tie_broken_by_first_label_in_list(self):
        # tech and finance each get 1 hit; "tech" comes first in labels list
        labels = ["tech", "finance", "sports"]
        agent = LogicAgent({"tech": [r"\bAPI\b"], "finance": [r"\bbank\b"]})
        state = agent.run(make_state("API bank", labels))
        assert state.logic_output.model_output.label == "tech"

    def test_probabilities_sum_to_one(self):
        agent = LogicAgent(RULE_MAP)
        state = agent.run(make_state("AI stocks match"))
        probs = state.logic_output.model_output.probabilities
        assert sum(probs.values()) == pytest.approx(1.0, rel=1e-5)

    def test_dominant_label_wins_with_more_rules(self):
        # tech has 2 patterns; finance has 1. Three tech rules fire, one finance rule fires.
        agent = LogicAgent(RULE_MAP)
        state = agent.run(make_state("app SDK cloud startup"))
        out = state.logic_output
        assert out.model_output.label == "tech"
        assert out.model_output.probabilities["tech"] > out.model_output.probabilities["finance"]


# ---------------------------------------------------------------------------
# No-match fallback
# ---------------------------------------------------------------------------

class TestNoMatchFallback:
    def test_no_rules_fire_abstains_with_empty_probabilities(self):
        agent = LogicAgent({"tech": [r"\bquantum\b"], "sports": [r"\bpolo\b"]})
        state = agent.run(make_state("today is a lovely day", ["tech", "sports"]))
        out = state.logic_output
        assert out.model_output.label is None
        assert out.model_output.probabilities == {}
        assert out.features.get("abstained") is True

    def test_no_match_returns_abstain_not_first_label(self):
        labels = ["finance", "tech", "sports"]
        agent = LogicAgent({})
        state = agent.run(make_state("nothing relevant", labels))
        out = state.logic_output
        assert out.model_output.label is None         # NOT labels[0] ('finance')
        assert out.features.get("abstained") is True

    def test_fallback_note_set(self):
        agent = LogicAgent({})
        state = agent.run(make_state("irrelevant", ["tech", "sports"]))
        assert state.logic_output.notes == _NO_MATCH_NOTE

    def test_empty_rule_map_abstains(self):
        agent = LogicAgent({})
        state = agent.run(make_state("any text", ["tech"]))
        out = state.logic_output
        assert out.model_output.label is None         # no rules → abstain
        assert out.features.get("abstained") is True


# ---------------------------------------------------------------------------
# Triggered rules output
# ---------------------------------------------------------------------------

class TestTriggeredRules:
    def test_triggered_rules_format_label_colon_pattern(self):
        agent = LogicAgent({"tech": [r"\bAPI\b"], "sports": [r"\bmatch\b"]})
        state = agent.run(make_state("API call", ["tech", "sports"]))
        triggered = state.logic_output.features["triggered_rules"]
        assert triggered == [r"tech:\bAPI\b"]

    def test_multiple_rules_all_listed(self):
        agent = LogicAgent(RULE_MAP)
        state = agent.run(make_state("app SDK cloud"))
        triggered = state.logic_output.features["triggered_rules"]
        # First tech pattern fires once (app, SDK, cloud → one regex match counts as 1 trigger)
        assert any(t.startswith("tech:") for t in triggered)

    def test_no_trigger_abstains(self):
        agent = LogicAgent({"tech": [r"\bquantum\b"]})
        state = agent.run(make_state("nothing here", ["tech"]))
        out = state.logic_output
        assert out.model_output.label is None          # no rule fired → abstain
        assert out.features.get("abstained") is True

    def test_raw_scores_present_and_accurate(self):
        agent = LogicAgent({"tech": [r"\bAPI\b", r"\bSDK\b"], "finance": [r"\bbank\b"]})
        state = agent.run(make_state("API SDK bank", ["tech", "finance"]))
        scores = state.logic_output.features["raw_scores"]
        assert scores["tech"] == 2
        assert scores["finance"] == 1


# ---------------------------------------------------------------------------
# Unicode / Arabic rules
# ---------------------------------------------------------------------------

class TestUnicodeRules:
    def test_arabic_regex_rule_fires(self):
        agent = LogicAgent({"tech": [r"تقنية|برمجة"], "sports": [r"رياضة|فريق"]})
        state = agent.run(make_state("أعمل في مجال برمجة", ["tech", "sports"]))
        assert state.logic_output.model_output.label == "tech"

    def test_arabic_no_match_fallback(self):
        agent = LogicAgent({"tech": [r"ذكاء"], "sports": [r"كرة"]})
        state = agent.run(make_state("مرحبا بالجميع", ["tech", "sports"]))
        assert state.logic_output.notes == _NO_MATCH_NOTE

    def test_mixed_script_rules(self):
        # Rule uses both Arabic and ASCII patterns in same label
        agent = LogicAgent({"tech": [r"برمجة|software"], "sports": [r"match"]})
        state = agent.run(make_state("software برمجة", ["tech", "sports"]))
        out = state.logic_output
        assert out.model_output.label == "tech"
        assert out.model_output.probabilities["tech"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Invalid regex handling
# ---------------------------------------------------------------------------

class TestInvalidRegex:
    def test_invalid_pattern_skipped_not_raised(self):
        # "[invalid" is an unterminated character class — should not raise at construction
        agent = LogicAgent({"tech": [r"[invalid", r"\bAPI\b"], "sports": []})
        state = agent.run(make_state("API call", ["tech", "sports"]))
        # Valid rule still fires
        assert state.logic_output.model_output.label == "tech"

    def test_all_invalid_patterns_give_fallback(self):
        agent = LogicAgent({"tech": [r"[bad"], "sports": [r"(?P<x>(?P<x>))"]})
        state = agent.run(make_state("some text", ["tech", "sports"]))
        assert state.logic_output.notes == _NO_MATCH_NOTE


# ---------------------------------------------------------------------------
# Validation hooks
# ---------------------------------------------------------------------------

class TestValidation:
    def test_raises_on_blank_input_text(self):
        agent = LogicAgent(RULE_MAP)
        with pytest.raises(ValueError, match="input_text"):
            agent.execute(make_state("   "))

    def test_raises_on_empty_labels(self):
        agent = LogicAgent(RULE_MAP)
        state = make_state("API", labels=[])
        with pytest.raises(ValueError, match="labels"):
            agent.execute(state)

    def test_rules_for_unknown_label_ignored(self):
        # Rule for "unknown" is not in task_config.labels → should not raise, just skip
        agent = LogicAgent({"unknown": [r"\bfoo\b"], "tech": [r"\bAPI\b"]})
        state = agent.run(make_state("API foo", ["tech", "sports"]))
        assert state.logic_output.model_output.label == "tech"


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_agent_name_in_output(self):
        agent = LogicAgent(RULE_MAP, name="my_logic")
        state = agent.run(make_state("API"))
        assert state.logic_output.agent_name == "my_logic"

    def test_raw_text_preserved(self):
        text = "AI drives IPO valuation"
        agent = LogicAgent(RULE_MAP)
        state = agent.run(make_state(text))
        assert state.logic_output.model_output.raw_text == text

    def test_reasoning_string_on_match(self):
        agent = LogicAgent({"tech": [r"\bAPI\b"], "sports": [r"\bmatch\b"]})
        state = agent.run(make_state("API call", ["tech", "sports"]))
        notes = state.logic_output.notes
        assert "tech" in notes
        assert "%" in notes
