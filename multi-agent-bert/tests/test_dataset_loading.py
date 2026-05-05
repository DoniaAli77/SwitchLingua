"""tests/test_dataset_loading.py

Tests for dataset loading helpers in evaluate_pipeline.py:

  * load_classification_dataset — existing classification format (text + label)
  * load_sequence_labeling_dataset — NER format (text + tokens + tags)
  * load_dataset — backward-compat alias for load_classification_dataset

Covers:
  - Valid inputs (happy path)
  - Missing required keys
  - JSON parse errors
  - len(tokens) != len(tags)
  - Unknown tags
  - File not found
  - Empty / all-skipped file
  - empty valid_tags guard
  - Backward-compat alias behaviour
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_pipeline import (
    load_classification_dataset,
    load_dataset,
    load_sequence_labeling_dataset,
)

_VALID_NER_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path: Path, lines: list) -> Path:
    """Write a list of objects (or raw strings) to a temp JSONL file."""
    p = tmp_path / "dataset.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for item in lines:
            fh.write((item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)) + "\n")
    return p


# ===========================================================================
# load_classification_dataset
# ===========================================================================

class TestLoadClassificationDataset:

    def test_valid_samples_returned(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "s1", "text": "hello", "label": "positive"},
            {"id": "s2", "text": "world", "label": "negative"},
        ])
        samples = load_classification_dataset(str(p))
        assert len(samples) == 2
        assert samples[0]["label"] == "positive"
        assert samples[1]["label"] == "negative"

    def test_id_field_preserved(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "abc", "text": "hi", "label": "neutral"},
        ])
        samples = load_classification_dataset(str(p))
        assert samples[0]["id"] == "abc"

    def test_missing_label_key_skipped(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "s1", "text": "hello", "label": "positive"},
            {"id": "s2", "text": "bad"},                          # no label
        ])
        samples = load_classification_dataset(str(p))
        assert len(samples) == 1
        assert samples[0]["id"] == "s1"

    def test_missing_text_key_skipped(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "s1", "text": "hello", "label": "positive"},
            {"id": "s2", "label": "negative"},                    # no text
        ])
        samples = load_classification_dataset(str(p))
        assert len(samples) == 1

    def test_json_parse_error_line_skipped(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            '{"id": "s1", "text": "ok", "label": "positive"}',
            "not valid json{{{",
        ])
        samples = load_classification_dataset(str(p))
        assert len(samples) == 1

    def test_blank_lines_ignored(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "s1", "text": "hello", "label": "positive"},
            "",
            {"id": "s2", "text": "world", "label": "negative"},
        ])
        samples = load_classification_dataset(str(p))
        assert len(samples) == 2

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_classification_dataset(str(tmp_path / "nonexistent.jsonl"))

    def test_all_lines_invalid_raises_value_error(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "s1", "text": "no label here"},
        ])
        with pytest.raises(ValueError, match="No valid samples"):
            load_classification_dataset(str(p))

    def test_arabic_english_text_preserved(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "s1", "text": "أنا happy", "label": "positive"},
        ])
        samples = load_classification_dataset(str(p))
        assert samples[0]["text"] == "أنا happy"


# ===========================================================================
# load_sequence_labeling_dataset
# ===========================================================================

class TestLoadSequenceLabelingDataset:

    def test_valid_sample_returned(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {
                "id": "n1",
                "text": "Ahmed works at Google",
                "tokens": ["Ahmed", "works", "at", "Google"],
                "tags":   ["B-PER", "O", "O", "B-ORG"],
            }
        ])
        samples = load_sequence_labeling_dataset(str(p), _VALID_NER_TAGS)
        assert len(samples) == 1
        assert samples[0]["tokens"] == ["Ahmed", "works", "at", "Google"]
        assert samples[0]["tags"]   == ["B-PER", "O", "O", "B-ORG"]

    def test_multiple_valid_samples(self, tmp_path):
        rows = [
            {"id": f"n{i}", "text": "x", "tokens": ["x"], "tags": ["O"]}
            for i in range(5)
        ]
        p = _write_jsonl(tmp_path, rows)
        samples = load_sequence_labeling_dataset(str(p), _VALID_NER_TAGS)
        assert len(samples) == 5

    def test_id_field_preserved(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "ner-42", "text": "Paris", "tokens": ["Paris"], "tags": ["B-LOC"]},
        ])
        samples = load_sequence_labeling_dataset(str(p), _VALID_NER_TAGS)
        assert samples[0]["id"] == "ner-42"

    def test_length_mismatch_skipped(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "good", "text": "ok", "tokens": ["ok"], "tags": ["O"]},
            {
                "id": "bad",
                "text": "mismatch",
                "tokens": ["a", "b", "c"],
                "tags":   ["O", "O"],          # len 2 != len 3
            },
        ])
        samples = load_sequence_labeling_dataset(str(p), _VALID_NER_TAGS)
        assert len(samples) == 1
        assert samples[0]["id"] == "good"

    def test_unknown_tag_skipped(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "good", "text": "ok", "tokens": ["ok"],  "tags": ["O"]},
            {"id": "bad",  "text": "x",  "tokens": ["x"],   "tags": ["B-MISC"]},  # not in set
        ])
        samples = load_sequence_labeling_dataset(str(p), _VALID_NER_TAGS)
        assert len(samples) == 1
        assert samples[0]["id"] == "good"

    def test_missing_tokens_key_skipped(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "good", "text": "ok", "tokens": ["ok"], "tags": ["O"]},
            {"id": "bad",  "text": "x",  "tags": ["O"]},       # no tokens
        ])
        samples = load_sequence_labeling_dataset(str(p), _VALID_NER_TAGS)
        assert len(samples) == 1

    def test_missing_tags_key_skipped(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "good", "text": "ok", "tokens": ["ok"], "tags": ["O"]},
            {"id": "bad",  "text": "x",  "tokens": ["x"]},     # no tags
        ])
        samples = load_sequence_labeling_dataset(str(p), _VALID_NER_TAGS)
        assert len(samples) == 1

    def test_missing_text_key_skipped(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "good", "text": "ok", "tokens": ["ok"], "tags": ["O"]},
            {"id": "bad",  "tokens": ["x"], "tags": ["O"]},    # no text
        ])
        samples = load_sequence_labeling_dataset(str(p), _VALID_NER_TAGS)
        assert len(samples) == 1

    def test_json_parse_error_skipped(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            '{"id":"good","text":"ok","tokens":["ok"],"tags":["O"]}',
            "{{not json",
        ])
        samples = load_sequence_labeling_dataset(str(p), _VALID_NER_TAGS)
        assert len(samples) == 1

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_sequence_labeling_dataset(str(tmp_path / "nope.jsonl"), _VALID_NER_TAGS)

    def test_all_lines_invalid_raises_value_error(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "bad", "text": "x", "tokens": ["a", "b"], "tags": ["O"]},  # mismatch
        ])
        with pytest.raises(ValueError, match="No valid samples"):
            load_sequence_labeling_dataset(str(p), _VALID_NER_TAGS)

    def test_empty_valid_tags_raises_value_error(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "n1", "text": "x", "tokens": ["x"], "tags": ["O"]},
        ])
        with pytest.raises(ValueError, match="valid_tags must not be empty"):
            load_sequence_labeling_dataset(str(p), [])

    def test_blank_lines_ignored(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "n1", "text": "x", "tokens": ["x"], "tags": ["O"]},
            "",
            {"id": "n2", "text": "y", "tokens": ["y"], "tags": ["O"]},
        ])
        samples = load_sequence_labeling_dataset(str(p), _VALID_NER_TAGS)
        assert len(samples) == 2

    def test_arabic_english_tokens_preserved(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {
                "id": "ar1",
                "text": "أحمد works",
                "tokens": ["أحمد", "works"],
                "tags":   ["B-PER", "O"],
            }
        ])
        samples = load_sequence_labeling_dataset(str(p), _VALID_NER_TAGS)
        assert samples[0]["tokens"][0] == "أحمد"

    def test_single_tag_subset_valid(self, tmp_path):
        """Passing a subset of labels (e.g. only O and B-PER) is also valid."""
        p = _write_jsonl(tmp_path, [
            {"id": "n1", "text": "Ahmed", "tokens": ["Ahmed"], "tags": ["B-PER"]},
        ])
        samples = load_sequence_labeling_dataset(str(p), ["O", "B-PER", "I-PER"])
        assert len(samples) == 1

    def test_multiple_unknown_tags_reported(self, tmp_path):
        """All-invalid file still raises ValueError, not silently returns empty."""
        p = _write_jsonl(tmp_path, [
            {"id": "bad", "text": "x", "tokens": ["x", "y"], "tags": ["B-MISC", "I-MISC"]},
        ])
        with pytest.raises(ValueError, match="No valid samples"):
            load_sequence_labeling_dataset(str(p), _VALID_NER_TAGS)


# ===========================================================================
# load_dataset (backward-compat alias)
# ===========================================================================

class TestLoadDatasetAlias:

    def test_alias_returns_same_as_classification(self, tmp_path):
        p = _write_jsonl(tmp_path, [
            {"id": "s1", "text": "hello", "label": "positive"},
        ])
        assert load_dataset(str(p)) == load_classification_dataset(str(p))

    def test_alias_does_not_accept_ner_only_lines(self, tmp_path):
        """NER-format lines lack 'label', so they're skipped — no surprises."""
        p = _write_jsonl(tmp_path, [
            {"id": "n1", "text": "x", "tokens": ["x"], "tags": ["O"]},
        ])
        with pytest.raises(ValueError, match="No valid samples"):
            load_dataset(str(p))
