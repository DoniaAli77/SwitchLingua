"""Unit tests for src/evaluation/evaluator.py."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from typing import Optional
from unittest.mock import MagicMock

import pytest

from src.evaluation.evaluator import (
    EvalReport,
    Evaluator,
    PerClassMetrics,
    SampleResult,
    _accuracy,
    _macro_f1,
    _per_class_metrics,
)
from src.state.schema import (
    AgentOutput,
    ConsensusOutput,
    FinalOutput,
    ModelOutput,
    PipelineState,
    RoutingInfo,
    StateMetadata,
    TaskConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABELS = ["positive", "negative", "neutral"]


def make_task_config(labels=None) -> TaskConfig:
    return TaskConfig(
        task_name="test_task",
        labels=labels if labels is not None else list(LABELS),
        label_descriptions={lbl: lbl for lbl in (labels or LABELS)},
    )


def make_dataset(items=None):
    if items is None:
        items = [
            {"id": "s1", "text": "great", "label": "positive"},
            {"id": "s2", "text": "terrible", "label": "negative"},
            {"id": "s3", "text": "okay", "label": "neutral"},
        ]
    return items


class _FakePrimaryClassifier:
    """Classifier that predicts a hard-coded label."""

    def __init__(self, label: str, confidence: float = 0.9):
        self.label = label
        self.confidence = confidence

    def run(self, state: PipelineState) -> PipelineState:
        state.primary_model = ModelOutput(
            label=self.label,
            confidence=self.confidence,
            probabilities={lbl: (self.confidence if lbl == self.label else 0.05)
                           for lbl in state.task_config.labels},
            raw_text=state.input_text,
        )
        return state


class _FakeOrchestrator:
    """Orchestrator stub that predicts a given label and marks escalated."""

    def __init__(
        self,
        label: str,
        confidence: float = 0.9,
        escalated: bool = False,
        error: bool = False,
    ):
        self.label = label
        self.confidence = confidence
        self.escalated = escalated
        self.error = error

    def run(self, state: PipelineState) -> PipelineState:
        if self.error:
            raise RuntimeError("Simulated pipeline error")
        state.primary_model = ModelOutput(
            label=self.label,
            confidence=self.confidence,
            probabilities={lbl: (self.confidence if lbl == self.label else 0.05)
                           for lbl in state.task_config.labels},
            raw_text=state.input_text,
        )
        decision = "escalate" if self.escalated else "accept_primary"
        state.routing_info = RoutingInfo(
            threshold=state.task_config.threshold,
            decision=decision,
        )
        state.final_output = FinalOutput(
            label=self.label,
            confidence=self.confidence,
        )
        return state


class _ModeAwareOrchestrator:
    """Orchestrator stub that respects task_config.pipeline_mode for escalation."""

    def run(self, state: PipelineState) -> PipelineState:
        state.primary_model = ModelOutput(
            label="positive",
            confidence=0.91,
            probabilities={lbl: (0.91 if lbl == "positive" else 0.045)
                           for lbl in state.task_config.labels},
            raw_text=state.input_text,
        )
        if getattr(state.task_config, "pipeline_mode", "full_agentic") == "primary_only":
            state.final_output = FinalOutput(label="positive", confidence=0.91)
            return state

        state.routing_info = RoutingInfo(
            threshold=state.task_config.threshold,
            decision="escalate",
        )
        state.final_output = FinalOutput(label="positive", confidence=0.91)
        return state


# ---------------------------------------------------------------------------
# Metric helper tests
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_perfect(self):
        assert _accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_zero(self):
        assert _accuracy(["a", "b"], ["b", "a"]) == 0.0

    def test_partial(self):
        assert _accuracy(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(2 / 3, abs=1e-5)

    def test_empty(self):
        assert _accuracy([], []) == 0.0


class TestPerClassMetrics:
    def test_perfect_prediction(self):
        y_true = ["positive", "negative", "neutral"]
        y_pred = ["positive", "negative", "neutral"]
        metrics = _per_class_metrics(y_true, y_pred, LABELS)
        for m in metrics:
            assert m.f1 == pytest.approx(1.0)
            assert m.precision == pytest.approx(1.0)
            assert m.recall == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = ["positive", "positive", "positive"]
        y_pred = ["negative", "negative", "negative"]
        metrics = _per_class_metrics(y_true, y_pred, LABELS)
        pos_m = next(m for m in metrics if m.label == "positive")
        assert pos_m.f1 == 0.0
        assert pos_m.recall == 0.0

    def test_support_counts(self):
        y_true = ["positive", "positive", "negative"]
        y_pred = ["positive", "negative", "negative"]
        metrics = _per_class_metrics(y_true, y_pred, LABELS)
        pos_m = next(m for m in metrics if m.label == "positive")
        assert pos_m.support == 2

    def test_labels_in_output(self):
        metrics = _per_class_metrics(["positive"], ["positive"], LABELS)
        assert {m.label for m in metrics} == set(LABELS)


class TestMacroF1:
    def test_all_perfect(self):
        per_class = [
            PerClassMetrics("positive", 1.0, 1.0, 1.0, 1),
            PerClassMetrics("negative", 1.0, 1.0, 1.0, 1),
        ]
        assert _macro_f1(per_class) == 1.0

    def test_empty(self):
        assert _macro_f1([]) == 0.0

    def test_average(self):
        per_class = [
            PerClassMetrics("a", 1.0, 1.0, 0.8, 1),
            PerClassMetrics("b", 1.0, 1.0, 0.4, 1),
        ]
        assert _macro_f1(per_class) == pytest.approx(0.6, abs=1e-5)


# ---------------------------------------------------------------------------
# Evaluator — constructor validation
# ---------------------------------------------------------------------------

class TestEvaluatorInit:
    def test_full_pipeline_requires_orchestrator(self):
        with pytest.raises(ValueError, match="orchestrator"):
            Evaluator(task_config=make_task_config(), mode="full_pipeline")

    def test_primary_only_requires_classifier_or_orchestrator(self):
        with pytest.raises(ValueError, match="primary_classifier"):
            Evaluator(task_config=make_task_config(), mode="primary_only")

    def test_primary_only_with_classifier_ok(self):
        clf = _FakePrimaryClassifier("positive")
        ev = Evaluator(
            task_config=make_task_config(),
            primary_classifier=clf,
            mode="primary_only",
        )
        assert ev.mode == "primary_only"

    def test_full_pipeline_with_orchestrator_ok(self):
        orch = _FakeOrchestrator("positive")
        ev = Evaluator(
            task_config=make_task_config(),
            orchestrator=orch,
            mode="full_pipeline",
        )
        assert ev.mode == "full_pipeline"


# ---------------------------------------------------------------------------
# Evaluator — primary_only mode
# ---------------------------------------------------------------------------

class TestPrimaryOnly:
    def _make_evaluator(self, label="positive"):
        clf = _FakePrimaryClassifier(label)
        return Evaluator(
            task_config=make_task_config(),
            primary_classifier=clf,
            mode="primary_only",
            run_id="test_primary",
        )

    def test_returns_eval_report(self):
        ev = self._make_evaluator("positive")
        report = ev.evaluate(make_dataset())
        assert isinstance(report, EvalReport)

    def test_sample_count(self):
        ev = self._make_evaluator("positive")
        report = ev.evaluate(make_dataset())
        assert report.num_samples == 3

    def test_mode_in_report(self):
        ev = self._make_evaluator("positive")
        report = ev.evaluate(make_dataset())
        assert report.mode == "primary_only"

    def test_perfect_accuracy_when_all_correct(self):
        dataset = [
            {"id": "s1", "text": "t", "label": "positive"},
            {"id": "s2", "text": "t", "label": "positive"},
        ]
        ev = self._make_evaluator("positive")
        report = ev.evaluate(dataset)
        assert report.accuracy == 1.0

    def test_zero_accuracy_when_all_wrong(self):
        dataset = [
            {"id": "s1", "text": "t", "label": "negative"},
            {"id": "s2", "text": "t", "label": "neutral"},
        ]
        ev = self._make_evaluator("positive")
        report = ev.evaluate(dataset)
        assert report.accuracy == 0.0

    def test_per_class_labels_present(self):
        ev = self._make_evaluator("positive")
        report = ev.evaluate(make_dataset())
        labels_in_report = {m.label for m in report.per_class}
        assert labels_in_report == set(LABELS)

    def test_no_escalation_in_primary_only(self):
        ev = self._make_evaluator("positive")
        report = ev.evaluate(make_dataset())
        assert report.escalation_rate == 0.0
        assert report.escalated_count == 0

    def test_probabilities_in_samples(self):
        ev = self._make_evaluator("positive")
        report = ev.evaluate(make_dataset())
        for s in report.samples:
            assert isinstance(s.probabilities, dict)

    def test_empty_dataset_raises(self):
        ev = self._make_evaluator()
        with pytest.raises(ValueError, match="empty"):
            ev.evaluate([])


# ---------------------------------------------------------------------------
# Evaluator — full_pipeline mode
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def _make_evaluator(self, label="positive", escalated=False, error=False):
        orch = _FakeOrchestrator(label, escalated=escalated, error=error)
        return Evaluator(
            task_config=make_task_config(),
            orchestrator=orch,
            mode="full_pipeline",
            run_id="test_full",
        )

    def test_returns_eval_report(self):
        ev = self._make_evaluator("positive")
        report = ev.evaluate(make_dataset())
        assert isinstance(report, EvalReport)

    def test_final_output_label_used(self):
        dataset = [{"id": "x", "text": "hi", "label": "positive"}]
        ev = self._make_evaluator("positive")
        report = ev.evaluate(dataset)
        assert report.samples[0].predicted_label == "positive"

    def test_escalation_rate_computed(self):
        dataset = [
            {"id": "s1", "text": "hi", "label": "positive"},
            {"id": "s2", "text": "hi", "label": "positive"},
        ]
        orch = _FakeOrchestrator("positive", escalated=True)
        ev = Evaluator(
            task_config=make_task_config(),
            orchestrator=orch,
            mode="full_pipeline",
            run_id="test_esc",
        )
        report = ev.evaluate(dataset)
        assert report.escalation_rate == 1.0
        assert report.escalated_count == 2

    def test_non_escalated_samples_not_counted(self):
        dataset = [{"id": "x", "text": "hi", "label": "positive"}]
        orch = _FakeOrchestrator("positive", escalated=False)
        ev = Evaluator(
            task_config=make_task_config(),
            orchestrator=orch,
            mode="full_pipeline",
        )
        report = ev.evaluate(dataset)
        assert report.escalated_count == 0

    def test_pipeline_error_captured(self):
        dataset = [{"id": "x", "text": "hi", "label": "positive"}]
        ev = self._make_evaluator(error=True)
        report = ev.evaluate(dataset)
        assert report.samples[0].pipeline_error is not None

    def test_error_sample_excluded_from_metrics(self):
        dataset = [{"id": "x", "text": "hi", "label": "positive"}]
        ev = self._make_evaluator(error=True)
        report = ev.evaluate(dataset)
        assert report.meta["error_samples"] == 1
        assert report.meta["valid_samples"] == 0

    def test_selected_pipeline_mode_primary_only_has_zero_escalation(self):
        cfg = make_task_config()
        cfg.pipeline_mode = "primary_only"
        ev = Evaluator(
            task_config=cfg,
            orchestrator=_ModeAwareOrchestrator(),
            mode="full_pipeline",
            run_id="test_mode_primary_only",
        )
        report = ev.evaluate(make_dataset())
        assert report.escalation_rate == 0.0
        assert report.escalated_count == 0
        assert report.meta["pipeline_mode"] == "primary_only"


# ---------------------------------------------------------------------------
# Evaluator — save() output files
# ---------------------------------------------------------------------------

class TestSave:
    def _make_report(self, mode="primary_only") -> tuple[Evaluator, EvalReport]:
        clf = _FakePrimaryClassifier("positive")
        ev = Evaluator(
            task_config=make_task_config(),
            primary_classifier=clf,
            mode=mode,
            run_id="save_test",
        )
        report = ev.evaluate(make_dataset())
        return ev, report

    def test_save_creates_four_files(self):
        ev, report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ev.save(report, output_dir=tmpdir)
            for key, path in paths.items():
                assert os.path.exists(path), f"Missing file for key={key}: {path}"

    def test_predictions_json_valid(self):
        ev, report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ev.save(report, output_dir=tmpdir)
            with open(paths["predictions_json"], encoding="utf-8") as fh:
                data = json.load(fh)
            assert len(data) == report.num_samples
            assert "sample_id" in data[0]
            assert "predicted_label" in data[0]
            assert "true_label" in data[0]
            assert "correct" in data[0]

    def test_predictions_csv_valid(self):
        ev, report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ev.save(report, output_dir=tmpdir)
            with open(paths["predictions_csv"], encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            assert len(rows) == report.num_samples
            assert "sample_id" in rows[0]
            assert "correct" in rows[0]

    def test_metrics_json_valid(self):
        ev, report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ev.save(report, output_dir=tmpdir)
            with open(paths["metrics_json"], encoding="utf-8") as fh:
                data = json.load(fh)
            assert "accuracy" in data
            assert "macro_f1" in data
            assert "escalation_rate" in data
            assert "per_class" in data
            assert isinstance(data["per_class"], list)

    def test_metrics_csv_has_summary_row(self):
        ev, report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ev.save(report, output_dir=tmpdir)
            with open(paths["metrics_csv"], encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            labels_in_csv = [r["label"] for r in rows]
            assert "__summary__" in labels_in_csv

    def test_metrics_csv_one_row_per_class(self):
        ev, report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ev.save(report, output_dir=tmpdir)
            with open(paths["metrics_csv"], encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            class_rows = [r for r in rows if r["label"] != "__summary__"]
            assert len(class_rows) == len(LABELS)

    def test_output_dir_created_if_missing(self):
        ev, report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "nested", "output")
            ev.save(report, output_dir=new_dir)
            assert os.path.isdir(new_dir)

    def test_run_id_override_in_save(self):
        ev, report = self._make_report()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ev.save(report, output_dir=tmpdir, run_id="custom_run")
            for path in paths.values():
                assert "custom_run" in os.path.basename(path)

    def test_run_id_in_report(self):
        _, report = self._make_report()
        assert report.run_id == "save_test"

    # ------------------------------------------------------------------
    # pipeline_mode visibility in saved outputs
    # ------------------------------------------------------------------

    def _make_report_with_mode(self, pipeline_mode: str):
        """Build a report whose task_config carries the given pipeline_mode."""
        cfg = make_task_config()
        cfg.pipeline_mode = pipeline_mode  # type: ignore[attr-defined]
        clf = _FakePrimaryClassifier("positive")
        ev = Evaluator(
            task_config=cfg,
            primary_classifier=clf,
            mode="primary_only",
            run_id=f"pm_{pipeline_mode}",
        )
        report = ev.evaluate(make_dataset())
        return ev, report

    def test_predictions_json_contains_pipeline_mode(self):
        ev, report = self._make_report_with_mode("paper_style")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ev.save(report, output_dir=tmpdir)
            with open(paths["predictions_json"], encoding="utf-8") as fh:
                data = json.load(fh)
        assert all(row.get("pipeline_mode") == "paper_style" for row in data)

    def test_predictions_csv_has_pipeline_mode_column(self):
        ev, report = self._make_report_with_mode("primary_only")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ev.save(report, output_dir=tmpdir)
            with open(paths["predictions_csv"], encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        assert "pipeline_mode" in rows[0]
        assert all(r["pipeline_mode"] == "primary_only" for r in rows)

    def test_metrics_csv_summary_has_pipeline_mode(self):
        ev, report = self._make_report_with_mode("full_agentic")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ev.save(report, output_dir=tmpdir)
            with open(paths["metrics_csv"], encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        summary = next(r for r in rows if r["label"] == "__summary__")
        assert summary["pipeline_mode"] == "full_agentic"

    def test_metrics_csv_per_class_rows_have_empty_pipeline_mode(self):
        ev, report = self._make_report_with_mode("paper_style")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ev.save(report, output_dir=tmpdir)
            with open(paths["metrics_csv"], encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        class_rows = [r for r in rows if r["label"] != "__summary__"]
        assert all(r["pipeline_mode"] == "" for r in class_rows)


# ---------------------------------------------------------------------------
# CLI pipeline_mode override
# ---------------------------------------------------------------------------

class TestCLIPipelineModeOverride:
    """Verify that --pipeline_mode CLI arg is wired into TaskConfig correctly."""

    def test_default_pipeline_mode_is_full_agentic(self, tmp_path):
        """When --pipeline_mode is omitted, TaskConfig receives full_agentic."""
        import sys
        import importlib
        import evaluate_pipeline  # noqa: F401 — import from project root

        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text(
            '{"id":"1","text":"hello","label":"positive"}\n', encoding="utf-8"
        )
        output_dir = str(tmp_path / "out")

        captured_configs = []
        _original_evaluator = evaluate_pipeline.Evaluator

        class _CapturingEvaluator(_original_evaluator):
            def __init__(self, task_config, **kwargs):
                captured_configs.append(task_config)
                super().__init__(task_config=task_config, **kwargs)

        evaluate_pipeline.Evaluator = _CapturingEvaluator
        try:
            evaluate_pipeline.main([
                "--dataset", str(dataset_file),
                "--mode", "primary_only",
                "--output_dir", output_dir,
            ])
        finally:
            evaluate_pipeline.Evaluator = _original_evaluator

        assert captured_configs, "Evaluator was never instantiated"
        assert captured_configs[0].pipeline_mode == "full_agentic"

    def test_cli_pipeline_mode_overrides_to_paper_style(self, tmp_path):
        """When --pipeline_mode paper_style is given, TaskConfig receives paper_style."""
        import evaluate_pipeline  # noqa: F401

        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text(
            '{"id":"1","text":"hello","label":"positive"}\n', encoding="utf-8"
        )
        output_dir = str(tmp_path / "out")

        captured_configs = []
        _original_evaluator = evaluate_pipeline.Evaluator

        class _CapturingEvaluator(_original_evaluator):
            def __init__(self, task_config, **kwargs):
                captured_configs.append(task_config)
                super().__init__(task_config=task_config, **kwargs)

        evaluate_pipeline.Evaluator = _CapturingEvaluator
        try:
            evaluate_pipeline.main([
                "--dataset", str(dataset_file),
                "--mode", "primary_only",
                "--output_dir", output_dir,
                "--pipeline_mode", "paper_style",
            ])
        finally:
            evaluate_pipeline.Evaluator = _original_evaluator

        assert captured_configs[0].pipeline_mode == "paper_style"

    def test_cli_pipeline_mode_primary_only_reflected_in_saved_metrics(self, tmp_path):
        """End-to-end: --pipeline_mode primary_only appears in saved metrics JSON."""
        import evaluate_pipeline  # noqa: F401

        dataset_file = tmp_path / "data.jsonl"
        dataset_file.write_text(
            '{"id":"1","text":"hello","label":"positive"}\n', encoding="utf-8"
        )
        output_dir = str(tmp_path / "out")

        evaluate_pipeline.main([
            "--dataset", str(dataset_file),
            "--mode", "primary_only",
            "--output_dir", output_dir,
            "--pipeline_mode", "primary_only",
        ])

        import glob
        metrics_files = glob.glob(str(tmp_path / "out" / "*metrics*.json"))
        assert metrics_files, "No metrics JSON file written"
        with open(metrics_files[0], encoding="utf-8") as fh:
            data = json.load(fh)
        assert data.get("meta", {}).get("pipeline_mode") == "primary_only"


# ---------------------------------------------------------------------------
# MockLLMClient mode validation in evaluate_pipeline.build_orchestrator
# ---------------------------------------------------------------------------

class TestBuildOrchestratorMockLLMClient:
    """Verify build_orchestrator passes a valid MockLLMClient mode."""

    def _make_task_config(self, labels=None):
        from src.state.schema import TaskConfig
        labels = labels or ["positive", "negative", "neutral"]
        return TaskConfig(
            task_name="test",
            labels=labels,
            label_descriptions={lbl: lbl for lbl in labels},
        )

    def test_build_orchestrator_uses_label_echo_mode(self):
        """build_orchestrator must wire MockLLMClient with mode='label_echo'."""
        import evaluate_pipeline
        from src.llm.mock_client import MockLLMClient

        task_config = self._make_task_config()
        orchestrator = evaluate_pipeline.build_orchestrator(
            task_config=task_config,
            threshold=0.5,
            enable_deliberation=False,
        )
        llm_client = orchestrator._contextual.llm_client
        assert isinstance(llm_client, MockLLMClient)
        assert llm_client.mode == "label_echo"

    def test_build_orchestrator_passes_allowed_labels(self):
        """build_orchestrator must pass task_config.labels as allowed_labels."""
        import evaluate_pipeline
        from src.llm.mock_client import MockLLMClient

        labels = ["sports", "tech", "health"]
        task_config = self._make_task_config(labels=labels)
        orchestrator = evaluate_pipeline.build_orchestrator(
            task_config=task_config,
            threshold=0.5,
            enable_deliberation=False,
        )
        llm_client = orchestrator._contextual.llm_client
        assert isinstance(llm_client, MockLLMClient)
        assert set(llm_client.allowed_labels) == set(labels)

    def test_build_orchestrator_contextual_agent_returns_valid_label(self):
        """ContextualAgent must return a label from task_config.labels (not 'unknown')."""
        import evaluate_pipeline
        from src.state.schema import (
            PipelineState,
            StateMetadata,
            ModelOutput,
            AgentOutput,
            RoutingInfo,
        )

        labels = ["business", "finance", "tech"]
        task_config = self._make_task_config(labels=labels)
        orchestrator = evaluate_pipeline.build_orchestrator(
            task_config=task_config,
            threshold=0.5,
            enable_deliberation=False,
        )

        # Build minimal state that triggers ContextualAgent
        state = PipelineState(
            input_text="buy stocks tech business",
            task_config=task_config,
            metadata=StateMetadata(sample_id="t1"),
        )
        state.lexical_output = AgentOutput(
            agent_name="lexical",
            model_output=ModelOutput(label=labels[0], confidence=0.4),
        )
        state.logic_output = AgentOutput(
            agent_name="logic",
            model_output=ModelOutput(label=labels[1], confidence=0.4),
        )
        state.routing_info = RoutingInfo(threshold=0.5, decision="escalate")

        state = orchestrator._contextual.run(state)
        output_label = state.contextual_output.model_output.label
        assert output_label in labels, (
            f"ContextualAgent returned '{output_label}' which is not in {labels}"
        )


# ---------------------------------------------------------------------------
# build_agent_knowledge_maps
# ---------------------------------------------------------------------------

class TestBuildAgentKnowledgeMaps:
    """Verify build_agent_knowledge_maps returns correct, filtered maps."""

    def test_sentiment_labels_return_sentiment_maps(self):
        """Sentiment labels must resolve to the sentiment maps."""
        import evaluate_pipeline

        km, rm = evaluate_pipeline.build_agent_knowledge_maps(
            ["positive", "negative", "neutral"]
        )
        assert set(km.keys()) == {"positive", "negative", "neutral"}
        assert set(rm.keys()) == {"positive", "negative", "neutral"}
        assert "great" in km["positive"]
        assert any("great" in p or "excellent" in p for p in rm["positive"])

    def test_topic_labels_return_topic_maps(self):
        """Topic labels must resolve to the topic maps."""
        import evaluate_pipeline

        labels = ["business", "tech", "health"]
        km, rm = evaluate_pipeline.build_agent_knowledge_maps(labels)
        assert set(km.keys()) == set(labels)
        assert set(rm.keys()) == set(labels)
        assert "company" in km["business"]
        assert "software" in km["tech"]
        assert "doctor" in km["health"]

    def test_arabic_keywords_present_for_topic_labels(self):
        """Topic maps must contain Arabic keywords for code-switched input."""
        import evaluate_pipeline

        km, _ = evaluate_pipeline.build_agent_knowledge_maps(
            ["education", "finance", "social"]
        )
        # education → تعليم
        assert any("\u062a\u0639\u0644\u064a\u0645" in kw for kw in km["education"])
        # finance → مال
        assert any("\u0645\u0627\u0644" in kw for kw in km["finance"])
        # social → اجتماعي
        assert any("\u0627\u062c\u062a\u0645\u0627\u0639\u064a" in kw for kw in km["social"])

    def test_maps_contain_only_active_labels(self):
        """Returned maps must not contain labels absent from the input list."""
        import evaluate_pipeline

        labels = ["tech", "sports"]
        km, rm = evaluate_pipeline.build_agent_knowledge_maps(labels)
        assert set(km.keys()) == set(labels)
        assert set(rm.keys()) == set(labels)
        assert "business" not in km
        assert "positive" not in rm

    def test_unknown_labels_are_silently_omitted(self):
        """Labels with no known map entry are omitted without raising."""
        import evaluate_pipeline

        km, rm = evaluate_pipeline.build_agent_knowledge_maps(
            ["tech", "nonexistent_label_xyz"]
        )
        assert "tech" in km
        assert "nonexistent_label_xyz" not in km
        assert "nonexistent_label_xyz" not in rm

    def test_all_nine_topic_labels_are_covered(self):
        """All nine topic labels must have entries in both maps."""
        import evaluate_pipeline

        topic_labels = [
            "business", "education", "health", "shopping", "medical",
            "sports", "tech", "finance", "social",
        ]
        km, rm = evaluate_pipeline.build_agent_knowledge_maps(topic_labels)
        assert set(km.keys()) == set(topic_labels), (
            f"Missing from keyword_map: {set(topic_labels) - set(km.keys())}"
        )
        assert set(rm.keys()) == set(topic_labels), (
            f"Missing from rule_map: {set(topic_labels) - set(rm.keys())}"
        )

    def test_build_orchestrator_wires_task_aware_maps(self):
        """build_orchestrator must give LexicalAgent and LogicAgent task-filtered maps."""
        import evaluate_pipeline
        from src.state.schema import TaskConfig

        labels = ["tech", "finance"]
        task_config = TaskConfig(
            task_name="test",
            labels=labels,
            label_descriptions={lbl: lbl for lbl in labels},
        )
        orch = evaluate_pipeline.build_orchestrator(
            task_config=task_config,
            threshold=0.5,
            enable_deliberation=False,
        )
        lexical_km = orch._lexical.keyword_map
        # LogicAgent compiles patterns into _compiled — check labels present there
        logic_labels = {rule.label for rule in orch._logic._compiled}
        assert set(lexical_km.keys()) == set(labels)
        assert logic_labels == set(labels)
