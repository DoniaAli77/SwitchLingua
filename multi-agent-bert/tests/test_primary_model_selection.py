"""Tests for the --primary_model selection seam in evaluate_pipeline.

These tests verify that the pipeline can choose between the mock and the real
transformer primary classifier *without ever downloading a model*. The
transformer branch is exercised purely through monkeypatching, so the suite
stays offline and does not require torch / transformers to be installed.
"""

from __future__ import annotations

import pytest

import evaluate_pipeline
from src.models.mock_primary_classifier import MockPrimaryClassifier
from src.models.primary_transformer_classifier import PrimaryTransformerClassifier
from src.state.schema import TaskConfig


# ---------------------------------------------------------------------------
# build_primary_classifier — mock branch (default, no downloads)
# ---------------------------------------------------------------------------

class TestPrimaryModelMock:
    def test_default_is_mock(self):
        """With no arguments the factory returns the mock classifier."""
        clf = evaluate_pipeline.build_primary_classifier()
        assert isinstance(clf, MockPrimaryClassifier)

    def test_explicit_mock_returns_mock(self):
        """Selecting 'mock' returns a MockPrimaryClassifier in heuristic mode."""
        clf = evaluate_pipeline.build_primary_classifier("mock")
        assert isinstance(clf, MockPrimaryClassifier)
        assert clf.mode == "heuristic"

    def test_mock_ignores_checkpoint(self):
        """A checkpoint passed alongside 'mock' is harmless and ignored."""
        clf = evaluate_pipeline.build_primary_classifier(
            "mock", transformer_checkpoint="some/model"
        )
        assert isinstance(clf, MockPrimaryClassifier)


# ---------------------------------------------------------------------------
# build_primary_classifier — transformer branch (mocked, no downloads)
# ---------------------------------------------------------------------------

class TestPrimaryModelTransformer:
    def test_transformer_routes_to_real_class_without_download(self, monkeypatch):
        """Selecting 'transformer' calls PrimaryTransformerClassifier.from_pretrained
        with the given checkpoint — verified via monkeypatch, no real model loaded."""
        sentinel = object()
        captured = {}

        def fake_from_pretrained(*, checkpoint, label_map, device):
            captured["checkpoint"] = checkpoint
            captured["label_map"] = label_map
            captured["device"] = device
            return sentinel

        monkeypatch.setattr(
            PrimaryTransformerClassifier,
            "from_pretrained",
            staticmethod(fake_from_pretrained),
        )

        result = evaluate_pipeline.build_primary_classifier(
            "transformer",
            transformer_checkpoint="bert-base-multilingual-cased",
            device="cpu",
            label_map={0: "negative", 1: "neutral", 2: "positive"},
        )

        assert result is sentinel
        assert captured["checkpoint"] == "bert-base-multilingual-cased"
        assert captured["device"] == "cpu"
        assert captured["label_map"] == {0: "negative", 1: "neutral", 2: "positive"}

    def test_transformer_without_checkpoint_raises(self):
        """'transformer' without a checkpoint fails fast and downloads nothing."""
        with pytest.raises(ValueError, match="requires a checkpoint"):
            evaluate_pipeline.build_primary_classifier("transformer")

    def test_unknown_primary_model_raises(self):
        with pytest.raises(ValueError, match="Unknown primary_model"):
            evaluate_pipeline.build_primary_classifier("bogus")


# ---------------------------------------------------------------------------
# build_orchestrator — injected primary is honoured, default stays mock
# ---------------------------------------------------------------------------

class TestOrchestratorPrimaryInjection:
    @staticmethod
    def _task_config() -> TaskConfig:
        return TaskConfig(
            task_name="sentiment_classification",
            labels=["positive", "negative", "neutral"],
            label_descriptions={"positive": "p", "negative": "n", "neutral": "x"},
            threshold=0.6,
            enable_deliberation=False,
            pipeline_mode="primary_only",
        )

    def test_default_orchestrator_uses_mock_primary(self):
        """Existing callers that omit primary_classifier still get the mock."""
        orch = evaluate_pipeline.build_orchestrator(
            task_config=self._task_config(),
            threshold=0.6,
            enable_deliberation=False,
        )
        assert isinstance(orch._primary, MockPrimaryClassifier)

    def test_injected_primary_is_used(self):
        """A primary_classifier passed in is wired into the orchestrator as-is."""
        injected = MockPrimaryClassifier(mode="fixed", fixed_label="positive")
        orch = evaluate_pipeline.build_orchestrator(
            task_config=self._task_config(),
            threshold=0.6,
            enable_deliberation=False,
            primary_classifier=injected,
        )
        assert orch._primary is injected
