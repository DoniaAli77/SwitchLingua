"""tests/test_task_aware_config_modes.py

Task-awareness smoke tests: verify that the pipeline is correctly configured
and behaves consistently for *both* supported tasks when loaded from
``src/config/default.yaml``.

For each task (topic_classification, sentiment_classification) we check:
  1. ``load_task_bundle`` returns the correct labels for the active task.
  2. ``keyword_map`` keys are a subset of the task labels.
  3. ``rule_map`` keys are a subset of the task labels.
  4. A sample run in ``paper_style`` produces a valid label for that task.
  5. A sample run in ``full_agentic`` produces a valid label for that task.
  6. The final label never belongs to the *other* task's label set.

Accuracy is not tested here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_pipeline import build_orchestrator
from src.config.loader import TaskBundle, load_task_bundle
from src.state.schema import PipelineState, StateMetadata

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "src" / "config" / "default.yaml"

_TOPIC_LABELS = {
    "business", "education", "health", "shopping", "medical",
    "sports", "tech", "finance", "social",
}
_SENTIMENT_LABELS = {"positive", "negative", "neutral"}

# One representative sample per task (Arabic-English code-switched).
_TOPIC_TEXT    = "The new software update uses AI, التطبيق يستخدم الذكاء الاصطناعي."
_SENTIMENT_TEXT = "The product is great, الجودة ممتازة وأنا سعيد جداً بالشراء."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(bundle: TaskBundle, pipeline_mode: str, *, enable_deliberation: bool = False) -> PipelineState:
    """Build an orchestrator from *bundle* and run one sample through it."""
    task_config = bundle.task_config.__class__(
        task_name=bundle.task_config.task_name,
        labels=bundle.task_config.labels,
        label_descriptions=bundle.task_config.label_descriptions,
        threshold=0.99,               # force escalation for non-primary_only modes
        pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
        enable_deliberation=enable_deliberation,
    )
    orch = build_orchestrator(
        task_config=task_config,
        threshold=0.99,
        enable_deliberation=enable_deliberation,
        keyword_map=bundle.keyword_map,
        rule_map=bundle.rule_map,
    )
    text = _TOPIC_TEXT if "topic" in bundle.active_task else _SENTIMENT_TEXT
    state = PipelineState(
        input_text=text,
        task_config=task_config,
        metadata=StateMetadata(sample_id=f"task_aware_{bundle.active_task}_{pipeline_mode}"),
    )
    return orch.run(state)


# ===========================================================================
# topic_classification
# ===========================================================================

@pytest.fixture(scope="module")
def topic_bundle() -> TaskBundle:
    return load_task_bundle(_CONFIG_PATH, active_task="topic_classification", threshold=0.99)


class TestTopicConfigLoading:

    def test_active_task_name(self, topic_bundle: TaskBundle) -> None:
        assert topic_bundle.active_task == "topic_classification"

    def test_labels_match_topic_set(self, topic_bundle: TaskBundle) -> None:
        assert set(topic_bundle.task_config.labels) == _TOPIC_LABELS

    def test_keyword_map_keys_subset_of_labels(self, topic_bundle: TaskBundle) -> None:
        extra = set(topic_bundle.keyword_map) - set(topic_bundle.task_config.labels)
        assert not extra, f"keyword_map has keys not in topic labels: {extra}"

    def test_rule_map_keys_subset_of_labels(self, topic_bundle: TaskBundle) -> None:
        extra = set(topic_bundle.rule_map) - set(topic_bundle.task_config.labels)
        assert not extra, f"rule_map has keys not in topic labels: {extra}"

    def test_keyword_map_nonempty(self, topic_bundle: TaskBundle) -> None:
        assert len(topic_bundle.keyword_map) > 0

    def test_rule_map_nonempty(self, topic_bundle: TaskBundle) -> None:
        assert len(topic_bundle.rule_map) > 0

    def test_no_sentiment_labels_in_config(self, topic_bundle: TaskBundle) -> None:
        assert not _SENTIMENT_LABELS.intersection(topic_bundle.task_config.labels)


class TestTopicPaperStyle:

    @pytest.fixture(scope="class")
    def result(self, topic_bundle: TaskBundle) -> PipelineState:
        return _run(topic_bundle, "paper_style")

    def test_no_pipeline_error(self, result: PipelineState) -> None:
        assert "pipeline_error" not in result.extras

    def test_final_output_not_none(self, result: PipelineState) -> None:
        assert result.final_output is not None

    def test_final_label_in_topic_labels(self, result: PipelineState) -> None:
        assert result.final_output.label in _TOPIC_LABELS

    def test_final_label_not_sentiment(self, result: PipelineState) -> None:
        """Output label must never belong to the sentiment task."""
        assert result.final_output.label not in _SENTIMENT_LABELS

    def test_escalation_occurred(self, result: PipelineState) -> None:
        assert result.routing_info is not None
        assert result.routing_info.decision == "escalate"


class TestTopicFullAgentic:

    @pytest.fixture(scope="class")
    def result(self, topic_bundle: TaskBundle) -> PipelineState:
        return _run(topic_bundle, "full_agentic")

    def test_no_pipeline_error(self, result: PipelineState) -> None:
        assert "pipeline_error" not in result.extras

    def test_final_output_not_none(self, result: PipelineState) -> None:
        assert result.final_output is not None

    def test_final_label_in_topic_labels(self, result: PipelineState) -> None:
        assert result.final_output.label in _TOPIC_LABELS

    def test_final_label_not_sentiment(self, result: PipelineState) -> None:
        assert result.final_output.label not in _SENTIMENT_LABELS

    def test_escalation_occurred(self, result: PipelineState) -> None:
        assert result.routing_info is not None
        assert result.routing_info.decision == "escalate"


# ===========================================================================
# sentiment_classification
# ===========================================================================

@pytest.fixture(scope="module")
def sentiment_bundle() -> TaskBundle:
    return load_task_bundle(_CONFIG_PATH, active_task="sentiment_classification", threshold=0.99)


class TestSentimentConfigLoading:

    def test_active_task_name(self, sentiment_bundle: TaskBundle) -> None:
        assert sentiment_bundle.active_task == "sentiment_classification"

    def test_labels_match_sentiment_set(self, sentiment_bundle: TaskBundle) -> None:
        assert set(sentiment_bundle.task_config.labels) == _SENTIMENT_LABELS

    def test_keyword_map_keys_subset_of_labels(self, sentiment_bundle: TaskBundle) -> None:
        extra = set(sentiment_bundle.keyword_map) - set(sentiment_bundle.task_config.labels)
        assert not extra, f"keyword_map has keys not in sentiment labels: {extra}"

    def test_rule_map_keys_subset_of_labels(self, sentiment_bundle: TaskBundle) -> None:
        extra = set(sentiment_bundle.rule_map) - set(sentiment_bundle.task_config.labels)
        assert not extra, f"rule_map has keys not in sentiment labels: {extra}"

    def test_keyword_map_nonempty(self, sentiment_bundle: TaskBundle) -> None:
        assert len(sentiment_bundle.keyword_map) > 0

    def test_rule_map_nonempty(self, sentiment_bundle: TaskBundle) -> None:
        assert len(sentiment_bundle.rule_map) > 0

    def test_no_topic_labels_in_config(self, sentiment_bundle: TaskBundle) -> None:
        assert not _TOPIC_LABELS.intersection(sentiment_bundle.task_config.labels)

    def test_label_descriptions_bilingual(self, sentiment_bundle: TaskBundle) -> None:
        """Every sentiment label description should contain at least one Arabic word."""
        for lbl, desc in sentiment_bundle.task_config.label_descriptions.items():
            has_arabic = any("\u0600" <= ch <= "\u06ff" for ch in desc)
            assert has_arabic, f"No Arabic content in label_description for '{lbl}': {desc!r}"


class TestSentimentPaperStyle:

    @pytest.fixture(scope="class")
    def result(self, sentiment_bundle: TaskBundle) -> PipelineState:
        return _run(sentiment_bundle, "paper_style")

    def test_no_pipeline_error(self, result: PipelineState) -> None:
        assert "pipeline_error" not in result.extras

    def test_final_output_not_none(self, result: PipelineState) -> None:
        assert result.final_output is not None

    def test_final_label_in_sentiment_labels(self, result: PipelineState) -> None:
        assert result.final_output.label in _SENTIMENT_LABELS

    def test_final_label_not_topic(self, result: PipelineState) -> None:
        """Output label must never belong to the topic task."""
        assert result.final_output.label not in _TOPIC_LABELS

    def test_escalation_occurred(self, result: PipelineState) -> None:
        assert result.routing_info is not None
        assert result.routing_info.decision == "escalate"


class TestSentimentFullAgentic:

    @pytest.fixture(scope="class")
    def result(self, sentiment_bundle: TaskBundle) -> PipelineState:
        return _run(sentiment_bundle, "full_agentic")

    def test_no_pipeline_error(self, result: PipelineState) -> None:
        assert "pipeline_error" not in result.extras

    def test_final_output_not_none(self, result: PipelineState) -> None:
        assert result.final_output is not None

    def test_final_label_in_sentiment_labels(self, result: PipelineState) -> None:
        assert result.final_output.label in _SENTIMENT_LABELS

    def test_final_label_not_topic(self, result: PipelineState) -> None:
        assert result.final_output.label not in _TOPIC_LABELS

    def test_escalation_occurred(self, result: PipelineState) -> None:
        assert result.routing_info is not None
        assert result.routing_info.decision == "escalate"
