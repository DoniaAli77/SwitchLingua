"""Tests for src/config/loader.py — task-aware config loading.

Coverage
--------
* Loading topic_classification returns topic labels and filled keyword/rule maps
* Loading sentiment_classification returns sentiment labels and filled keyword/rule maps
* active_task override switches tasks correctly
* CLI threshold override applies over config value
* CLI pipeline_mode override applies over config value
* keyword_map contains only labels belonging to the active task
* rule_map contains only labels belonging to the active task
* Missing config file raises FileNotFoundError
* Unknown active_task raises KeyError
* No active_task in config or arg raises ValueError
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from src.config.loader import TaskBundle, load_task_bundle

# ---------------------------------------------------------------------------
# Path to the real project config — tests run from multi-agent-bert/
# ---------------------------------------------------------------------------
_REAL_CONFIG = Path(__file__).parent.parent / "src" / "config" / "default.yaml"

# ---------------------------------------------------------------------------
# Minimal in-memory YAML fixture helpers
# ---------------------------------------------------------------------------

_MINIMAL_YAML = textwrap.dedent("""\
    active_task: sentiment_classification
    tasks:
      sentiment_classification:
        task_type: classification
        labels: [positive, negative, neutral]
        label_descriptions:
          positive: "Good"
          negative: "Bad"
          neutral:  "Meh"
        label_knowledge:
          positive:
            keywords_l1: [great, love]
            keywords_l2: [ممتاز, رائع]
            regex_rules: ["\\\\b(great|love)\\\\b"]
          negative:
            keywords_l1: [bad, hate]
            keywords_l2: [سيء, كريه]
            regex_rules: ["\\\\b(bad|hate)\\\\b"]
          neutral:
            keywords_l1: [okay, fine]
            keywords_l2: [عادي, مقبول]
            regex_rules: ["\\\\b(okay|fine)\\\\b"]
      topic_classification:
        task_type: classification
        labels: [tech, sports, health]
        label_descriptions:
          tech:    "Technology"
          sports:  "Sports"
          health:  "Health"
        label_knowledge:
          tech:
            keywords_l1: [software, app]
            keywords_l2: [برنامج, تطبيق]
            regex_rules: ["\\\\b(software|app)\\\\b"]
          sports:
            keywords_l1: [match, team]
            keywords_l2: [ماتش, فريق]
            regex_rules: ["\\\\b(match|team)\\\\b"]
          health:
            keywords_l1: [fitness, diet]
            keywords_l2: [صحة, لياقة]
            regex_rules: ["\\\\b(fitness|diet)\\\\b"]
    language_pair:
      pair_name: en-ar
      l1: en
      l2: ar
    execution:
      threshold: 0.6
      pipeline_mode: full_agentic
      enable_deliberation: false
      contextual_use_prior_outputs: false
""")


@pytest.fixture()
def mini_config(tmp_path: Path) -> Path:
    """Write the minimal YAML to a temp file and return its path."""
    p = tmp_path / "test_config.yaml"
    p.write_text(_MINIMAL_YAML, encoding="utf-8")
    return p


# ===========================================================================
# sentiment_classification
# ===========================================================================

class TestSentimentTask:

    def test_returns_sentiment_labels(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="sentiment_classification")
        assert set(bundle.task_config.labels) == {"positive", "negative", "neutral"}

    def test_active_task_stored(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="sentiment_classification")
        assert bundle.active_task == "sentiment_classification"

    def test_keyword_map_has_sentiment_labels(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="sentiment_classification")
        assert set(bundle.keyword_map.keys()) == {"positive", "negative", "neutral"}

    def test_rule_map_has_sentiment_labels(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="sentiment_classification")
        assert set(bundle.rule_map.keys()) == {"positive", "negative", "neutral"}

    def test_keyword_map_merges_l1_and_l2(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="sentiment_classification")
        # "great" (l1) and "ممتاز" (l2) should both be in positive keywords
        assert "great" in bundle.keyword_map["positive"]
        assert "ممتاز" in bundle.keyword_map["positive"]

    def test_label_descriptions_populated(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="sentiment_classification")
        assert bundle.task_config.label_descriptions["positive"] == "Good"

    def test_task_type_is_classification(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="sentiment_classification")
        assert bundle.task_config.task_type == "classification"


# ===========================================================================
# topic_classification
# ===========================================================================

class TestTopicTask:

    def test_returns_topic_labels(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="topic_classification")
        assert set(bundle.task_config.labels) == {"tech", "sports", "health"}

    def test_keyword_map_has_topic_labels(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="topic_classification")
        assert set(bundle.keyword_map.keys()) == {"tech", "sports", "health"}

    def test_rule_map_has_topic_labels(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="topic_classification")
        assert set(bundle.rule_map.keys()) == {"tech", "sports", "health"}

    def test_keyword_map_merges_l1_and_l2(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="topic_classification")
        assert "software" in bundle.keyword_map["tech"]
        assert "برنامج" in bundle.keyword_map["tech"]

    def test_active_task_stored(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="topic_classification")
        assert bundle.active_task == "topic_classification"


# ===========================================================================
# active_task override
# ===========================================================================

class TestActiveTaskOverride:

    def test_override_switches_from_default(self, mini_config: Path) -> None:
        """Default is sentiment; override should load topic."""
        default_bundle = load_task_bundle(mini_config)
        override_bundle = load_task_bundle(mini_config, active_task="topic_classification")
        assert default_bundle.active_task == "sentiment_classification"
        assert override_bundle.active_task == "topic_classification"

    def test_override_changes_labels(self, mini_config: Path) -> None:
        default_bundle = load_task_bundle(mini_config)
        override_bundle = load_task_bundle(mini_config, active_task="topic_classification")
        assert set(default_bundle.task_config.labels) != set(override_bundle.task_config.labels)

    def test_override_changes_keyword_map_keys(self, mini_config: Path) -> None:
        sent_bundle = load_task_bundle(mini_config, active_task="sentiment_classification")
        topic_bundle = load_task_bundle(mini_config, active_task="topic_classification")
        assert set(sent_bundle.keyword_map.keys()) != set(topic_bundle.keyword_map.keys())


# ===========================================================================
# CLI execution overrides
# ===========================================================================

class TestExecutionOverrides:

    def test_threshold_override_applied(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, threshold=0.99)
        assert bundle.task_config.threshold == pytest.approx(0.99)

    def test_threshold_config_default_used_when_no_override(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config)
        assert bundle.task_config.threshold == pytest.approx(0.6)

    def test_pipeline_mode_override_applied(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, pipeline_mode="paper_style")
        assert bundle.task_config.pipeline_mode == "paper_style"

    def test_pipeline_mode_config_default_used_when_no_override(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config)
        assert bundle.task_config.pipeline_mode == "full_agentic"

    def test_enable_deliberation_override(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, enable_deliberation=True)
        assert bundle.task_config.enable_deliberation is True

    def test_enable_deliberation_config_default(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config)
        assert bundle.task_config.enable_deliberation is False

    def test_contextual_use_prior_override(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, contextual_use_prior_outputs=True)
        assert bundle.task_config.contextual_use_prior_outputs is True


# ===========================================================================
# Isolation — no labels from other tasks leak into maps
# ===========================================================================

class TestLabelIsolation:

    def test_no_topic_labels_in_sentiment_keyword_map(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="sentiment_classification")
        topic_labels = {"tech", "sports", "health"}
        assert topic_labels.isdisjoint(bundle.keyword_map.keys())

    def test_no_sentiment_labels_in_topic_keyword_map(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="topic_classification")
        sentiment_labels = {"positive", "negative", "neutral"}
        assert sentiment_labels.isdisjoint(bundle.keyword_map.keys())

    def test_no_topic_labels_in_sentiment_rule_map(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="sentiment_classification")
        topic_labels = {"tech", "sports", "health"}
        assert topic_labels.isdisjoint(bundle.rule_map.keys())

    def test_no_sentiment_labels_in_topic_rule_map(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="topic_classification")
        sentiment_labels = {"positive", "negative", "neutral"}
        assert sentiment_labels.isdisjoint(bundle.rule_map.keys())

    def test_keyword_map_keys_subset_of_active_labels(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="topic_classification")
        assert set(bundle.keyword_map.keys()).issubset(set(bundle.task_config.labels))

    def test_rule_map_keys_subset_of_active_labels(self, mini_config: Path) -> None:
        bundle = load_task_bundle(mini_config, active_task="topic_classification")
        assert set(bundle.rule_map.keys()).issubset(set(bundle.task_config.labels))


# ===========================================================================
# Error handling
# ===========================================================================

class TestErrorHandling:

    def test_missing_config_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_task_bundle(tmp_path / "nonexistent.yaml")

    def test_unknown_active_task_raises_key_error(self, mini_config: Path) -> None:
        with pytest.raises(KeyError, match="unknown_task"):
            load_task_bundle(mini_config, active_task="unknown_task")

    def test_no_active_task_raises_value_error(self, tmp_path: Path) -> None:
        p = tmp_path / "no_task.yaml"
        p.write_text("tasks:\n  foo:\n    labels: [a]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="active_task"):
            load_task_bundle(p)


# ===========================================================================
# Integration — real config/default.yaml
# ===========================================================================

@pytest.mark.skipif(
    not _REAL_CONFIG.exists(),
    reason="config/default.yaml not present",
)
class TestRealConfig:

    def test_real_config_loads_sentiment(self) -> None:
        bundle = load_task_bundle(_REAL_CONFIG, active_task="sentiment_classification")
        assert "positive" in bundle.task_config.labels
        assert "negative" in bundle.task_config.labels
        assert "neutral" in bundle.task_config.labels

    def test_real_config_loads_topic(self) -> None:
        bundle = load_task_bundle(_REAL_CONFIG, active_task="topic_classification")
        expected = {"business", "education", "health", "shopping", "medical",
                    "sports", "tech", "finance", "social"}
        assert set(bundle.task_config.labels) == expected

    def test_real_topic_keyword_map_has_all_labels(self) -> None:
        bundle = load_task_bundle(_REAL_CONFIG, active_task="topic_classification")
        assert set(bundle.keyword_map.keys()) == set(bundle.task_config.labels)

    def test_real_topic_rule_map_has_all_labels(self) -> None:
        bundle = load_task_bundle(_REAL_CONFIG, active_task="topic_classification")
        assert set(bundle.rule_map.keys()) == set(bundle.task_config.labels)

    def test_real_sentiment_keyword_map_has_all_labels(self) -> None:
        bundle = load_task_bundle(_REAL_CONFIG, active_task="sentiment_classification")
        assert set(bundle.keyword_map.keys()) == set(bundle.task_config.labels)

    def test_real_config_default_active_task(self) -> None:
        bundle = load_task_bundle(_REAL_CONFIG)
        # default.yaml sets active_task: topic_classification
        assert bundle.active_task == "topic_classification"
