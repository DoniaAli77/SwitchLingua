"""tests/test_ner_config.py

Tests for NER config support.

Verifies:
  1. ``load_task_bundle`` loads the ner task without error.
  2. ``TaskConfig.task_type`` is ``"sequence_labeling"``.
  3. NER labels exactly match the expected BIO tag set.
  4. ``evaluate_pipeline.main()`` refuses NER with exit code 1 and a
     clear message — never crashing with label/output errors.
  5. topic_classification and sentiment_classification are unaffected.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import load_task_bundle

_CONFIG = Path(__file__).parent.parent / "src" / "config" / "default.yaml"

_EXPECTED_NER_LABELS = {"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"}


# ===========================================================================
# Config loading
# ===========================================================================

class TestNERConfigLoading:

    @pytest.fixture(scope="class")
    def ner_bundle(self):
        return load_task_bundle(_CONFIG, active_task="ner")

    def test_loads_without_error(self, ner_bundle) -> None:
        assert ner_bundle is not None

    def test_active_task_name(self, ner_bundle) -> None:
        assert ner_bundle.active_task == "ner"

    def test_task_type_is_sequence_labeling(self, ner_bundle) -> None:
        assert ner_bundle.task_config.task_type == "sequence_labeling"

    def test_labels_match_bio_set(self, ner_bundle) -> None:
        assert set(ner_bundle.task_config.labels) == _EXPECTED_NER_LABELS

    def test_all_bio_labels_have_descriptions(self, ner_bundle) -> None:
        descs = ner_bundle.task_config.label_descriptions
        for lbl in _EXPECTED_NER_LABELS:
            assert lbl in descs and descs[lbl], (
                f"Missing or empty label_description for '{lbl}'"
            )

    def test_keyword_map_is_empty_for_ner(self, ner_bundle) -> None:
        """sequence_labeling tasks have no keyword map — not applicable."""
        assert ner_bundle.keyword_map == {}

    def test_rule_map_is_empty_for_ner(self, ner_bundle) -> None:
        """sequence_labeling tasks have no rule map — not applicable."""
        assert ner_bundle.rule_map == {}

    def test_task_name_stored_on_config(self, ner_bundle) -> None:
        assert ner_bundle.task_config.task_name == "ner"


# ===========================================================================
# evaluate_pipeline safety guard
# ===========================================================================

class TestEvaluatePipelineNERGuard:

    def test_main_returns_1_for_ner(self, tmp_path, caplog) -> None:
        """evaluate_pipeline.main() must return 1 (not raise) when NER is requested."""
        from evaluate_pipeline import main

        # Create a minimal one-sample JSONL dataset so the loader doesn't fail.
        dataset = tmp_path / "dummy_ner.jsonl"
        dataset.write_text(
            '{"id": "t1", "text": "Ahmed works at Google.", "label": "O"}\n',
            encoding="utf-8",
        )

        exit_code = main([
            "--config", str(_CONFIG),
            "--active_task", "ner",
            "--dataset", str(dataset),
            "--mode", "full_pipeline",
            "--pipeline_mode", "primary_only",
            "--output_dir", str(tmp_path / "ner_out"),
        ])

        assert exit_code == 1

    def test_main_logs_dataset_error_for_wrong_format(self, tmp_path, caplog) -> None:
        """Providing a classification-format dataset for an NER task must log an error.

        NER evaluation is now implemented.  A dataset that lacks the
        required 'tokens' and 'tags' fields causes load_sequence_labeling_dataset
        to reject all lines and raise ValueError, which main() catches and
        logs before returning 1.
        """
        from evaluate_pipeline import main

        dataset = tmp_path / "dummy_ner.jsonl"
        dataset.write_text(
            '{"id": "t1", "text": "Ahmed works at Google.", "label": "O"}\n',
            encoding="utf-8",
        )

        with caplog.at_level(logging.ERROR, logger="evaluate_pipeline"):
            main([
                "--config", str(_CONFIG),
                "--active_task", "ner",
                "--dataset", str(dataset),
                "--mode", "full_pipeline",
                "--pipeline_mode", "primary_only",
                "--output_dir", str(tmp_path / "ner_out2"),
            ])

        assert any(
            "valid samples" in rec.message.lower() or "no valid" in rec.message.lower()
            for rec in caplog.records
        ), (
            "Expected error about invalid dataset format. "
            f"Got: {[r.message for r in caplog.records]}"
        )

    def test_no_output_files_written_for_ner(self, tmp_path) -> None:
        """Guard must fire before any output files are written."""
        from evaluate_pipeline import main

        dataset = tmp_path / "dummy_ner.jsonl"
        dataset.write_text(
            '{"id": "t1", "text": "Ahmed works at Google.", "label": "O"}\n',
            encoding="utf-8",
        )
        out_dir = tmp_path / "ner_out3"

        main([
            "--config", str(_CONFIG),
            "--active_task", "ner",
            "--dataset", str(dataset),
            "--mode", "full_pipeline",
            "--pipeline_mode", "primary_only",
            "--output_dir", str(out_dir),
        ])

        # Output directory should not have been created / should be empty.
        assert not out_dir.exists() or not any(out_dir.iterdir()), (
            "evaluate_pipeline wrote output files for a sequence_labeling task."
        )


# ===========================================================================
# Regression: existing tasks still pass
# ===========================================================================

class TestClassificationTasksUnaffected:

    def test_topic_classification_task_type(self) -> None:
        bundle = load_task_bundle(_CONFIG, active_task="topic_classification")
        assert bundle.task_config.task_type == "classification"

    def test_sentiment_classification_task_type(self) -> None:
        bundle = load_task_bundle(_CONFIG, active_task="sentiment_classification")
        assert bundle.task_config.task_type == "classification"

    def test_topic_labels_unchanged(self) -> None:
        bundle = load_task_bundle(_CONFIG, active_task="topic_classification")
        expected = {
            "business", "education", "health", "shopping", "medical",
            "sports", "tech", "finance", "social",
        }
        assert set(bundle.task_config.labels) == expected

    def test_sentiment_labels_unchanged(self) -> None:
        bundle = load_task_bundle(_CONFIG, active_task="sentiment_classification")
        assert set(bundle.task_config.labels) == {"positive", "negative", "neutral"}
