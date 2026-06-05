"""Lightweight offline tests for scripts/finetune_transformer_classifier.py.

Only the stdlib-only helpers are tested (arg parsing, data loading, label maps,
metrics). The module is loaded from its file path so the test does not depend on
scripts/ being a package, and importing it must NOT require torch/transformers
(they are lazy-imported inside main(), which is never called here).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "finetune_transformer_classifier.py"
_spec = importlib.util.spec_from_file_location("finetune_transformer_classifier", _SCRIPT)
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)  # raises if the module needs torch at import time


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self):
        args = ft.parse_args(["--train", "t.jsonl", "--output_dir", "out"])
        assert args.base_checkpoint == "bert-base-multilingual-cased"
        assert args.labels == ["positive", "negative", "neutral"]
        assert args.text_col == "text" and args.label_col == "label"
        assert args.dev is None

    def test_overrides(self):
        args = ft.parse_args([
            "--train", "tr.csv", "--dev", "dv.csv",
            "--text_col", "comment", "--label_col", "sentiment",
            "--base_checkpoint", "xlm-roberta-base",
            "--output_dir", "ckpt", "--epochs", "3", "--batch_size", "8",
        ])
        assert args.base_checkpoint == "xlm-roberta-base"
        assert args.text_col == "comment" and args.label_col == "sentiment"
        assert args.epochs == 3.0 and args.batch_size == 8

    def test_train_required(self):
        with pytest.raises(SystemExit):
            ft.parse_args(["--output_dir", "out"])


# ---------------------------------------------------------------------------
# load_examples
# ---------------------------------------------------------------------------

class TestLoadExamples:
    def test_jsonl(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text(
            '{"text": "great يا bosy", "label": "positive"}\n'
            '{"text": "سيء bad", "label": "negative"}\n'
            "\n",  # blank line ignored
            encoding="utf-8",
        )
        rows = ft.load_examples(str(p), labels=["positive", "negative", "neutral"])
        assert rows == [("great يا bosy", "positive"), ("سيء bad", "negative")]

    def test_csv_with_columns(self, tmp_path):
        p = tmp_path / "d.csv"
        p.write_text(
            "comment,sentiment\n"
            '"hello عربي",positive\n'
            '"   ",neutral\n'  # empty text dropped
            "bad,unknownlabel\n",  # filtered by label set
            encoding="utf-8",
        )
        rows = ft.load_examples(
            str(p), text_col="comment", label_col="sentiment",
            labels=["positive", "negative", "neutral"],
        )
        assert rows == [("hello عربي", "positive")]

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ft.load_examples("nope.jsonl")

    def test_no_valid_rows_raises(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text('{"text": "  ", "label": "positive"}\n', encoding="utf-8")
        with pytest.raises(ValueError):
            ft.load_examples(str(p), labels=["positive"])


# ---------------------------------------------------------------------------
# build_label_maps
# ---------------------------------------------------------------------------

class TestBuildLabelMaps:
    def test_order_preserved(self):
        l2i, i2l = ft.build_label_maps(["positive", "negative", "neutral"])
        assert l2i == {"positive": 0, "negative": 1, "neutral": 2}
        assert i2l == {0: "positive", 1: "negative", 2: "neutral"}

    def test_duplicates_raise(self):
        with pytest.raises(ValueError):
            ft.build_label_maps(["positive", "positive"])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ft.build_label_maps([])


# ---------------------------------------------------------------------------
# compute_classification_metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_perfect(self):
        labels = ["positive", "negative", "neutral"]
        y = [0, 1, 2, 0, 1, 2]
        m = ft.compute_classification_metrics(y, y, labels)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["weighted_f1"] == 1.0
        assert {pc["label"] for pc in m["per_class"]} == set(labels)

    def test_known_confusion(self):
        labels = ["pos", "neg"]
        # gold: pos,pos,neg,neg ; pred: pos,neg,neg,neg
        m = ft.compute_classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], labels)
        assert m["accuracy"] == pytest.approx(0.75)
        pos = next(pc for pc in m["per_class"] if pc["label"] == "pos")
        assert pos["precision"] == pytest.approx(1.0)   # 1 tp, 0 fp
        assert pos["recall"] == pytest.approx(0.5)      # 1 tp, 1 fn
        assert pos["support"] == 2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ft.compute_classification_metrics([0, 1], [0], ["a", "b"])
