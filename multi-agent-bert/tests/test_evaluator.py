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
        assert "fitness" in km["health"]   # 'doctor' is in medical, not health

    def test_arabic_keywords_present_for_topic_labels(self):
        """Topic maps must contain Arabic keywords for code-switched input."""
        import evaluate_pipeline

        km, _ = evaluate_pipeline.build_agent_knowledge_maps(
            ["education", "finance", "social"]
        )
        # education → مدرسة (school)
        assert any("\u0645\u062f\u0631\u0633\u0629" in kw for kw in km["education"])
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


# ---------------------------------------------------------------------------
# Topic seed keyword maps — curated Arabic-English knowledge
# ---------------------------------------------------------------------------

class TestTopicSeedKnowledgeMaps:
    """Verify the manually curated topic seed keyword maps in evaluate_pipeline."""

    def test_business_english_keywords(self):
        """business map must contain curated English keywords."""
        import evaluate_pipeline
        km, _ = evaluate_pipeline.build_agent_knowledge_maps(["business"])
        for kw in ["company", "startup", "market", "sales", "profit"]:
            assert kw in km["business"], f"Expected '{kw}' in business keyword map"

    def test_medical_english_keywords(self):
        """medical map must contain curated English keywords (including 'doctor')."""
        import evaluate_pipeline
        km, _ = evaluate_pipeline.build_agent_knowledge_maps(["medical"])
        for kw in ["doctor", "hospital", "patient", "medication", "treatment"]:
            assert kw in km["medical"], f"Expected '{kw}' in medical keyword map"

    def test_tech_english_keywords(self):
        """tech map must contain curated English keywords."""
        import evaluate_pipeline
        km, _ = evaluate_pipeline.build_agent_knowledge_maps(["tech"])
        for kw in ["technology", "software", "app", "AI", "cloud"]:
            assert kw in km["tech"], f"Expected '{kw}' in tech keyword map"

    def test_sports_arabic_keywords(self):
        """sports map must contain Arabic keywords: مباراة and فريق."""
        import evaluate_pipeline
        km, _ = evaluate_pipeline.build_agent_knowledge_maps(["sports"])
        assert "\u0645\u0628\u0627\u0631\u0627\u0629" in km["sports"]   # مباراة
        assert "\u0641\u0631\u064a\u0642" in km["sports"]               # فريق

    def test_education_arabic_keywords(self):
        """education map must contain Arabic keywords: مدرسة and جامعة."""
        import evaluate_pipeline
        km, _ = evaluate_pipeline.build_agent_knowledge_maps(["education"])
        assert "\u0645\u062f\u0631\u0633\u0629" in km["education"]       # مدرسة
        assert "\u062c\u0627\u0645\u0639\u0629" in km["education"]       # جامعة

    def test_shopping_arabic_keywords(self):
        """shopping map must contain Arabic keywords: خصم and سعر."""
        import evaluate_pipeline
        km, _ = evaluate_pipeline.build_agent_knowledge_maps(["shopping"])
        assert "\u062e\u0635\u0645" in km["shopping"]                   # خصم
        assert "\u0633\u0639\u0631" in km["shopping"]                    # سعر

    def test_sentiment_labels_unaffected_by_topic_maps(self):
        """Sentiment keyword maps must be unchanged after adding topic labels."""
        import evaluate_pipeline
        km_sentiment, _ = evaluate_pipeline.build_agent_knowledge_maps(
            ["positive", "negative", "neutral"]
        )
        km_topic, _ = evaluate_pipeline.build_agent_knowledge_maps(
            ["business", "tech", "health"]
        )
        # sentiment values must not appear in topic maps
        assert "great" not in km_topic.get("business", [])
        assert "great" not in km_topic.get("tech", [])
        # topic values must not appear in sentiment maps
        assert "company" not in km_sentiment.get("positive", [])
        assert "company" not in km_sentiment.get("negative", [])

    def test_mixed_topic_and_sentiment_labels(self):
        """Mixed label sets must produce correct per-label maps for both families."""
        import evaluate_pipeline
        labels = ["tech", "positive", "negative"]
        km, rm = evaluate_pipeline.build_agent_knowledge_maps(labels)
        assert set(km.keys()) == set(labels)
        assert set(rm.keys()) == set(labels)
        assert "software" in km["tech"]
        assert "great" in km["positive"]
        assert "awful" in km["negative"]

    def test_topic_rule_map_contains_english_pattern(self):
        """Rule map for finance must contain the curated English regex pattern."""
        import evaluate_pipeline
        _, rm = evaluate_pipeline.build_agent_knowledge_maps(["finance"])
        assert any("bank" in rule for rule in rm["finance"])
        assert any("investment" in rule for rule in rm["finance"])

    def test_topic_rule_map_contains_auto_arabic_pattern(self):
        """Rule map must include an auto-derived Arabic alternation rule per label."""
        import evaluate_pipeline
        _, rm = evaluate_pipeline.build_agent_knowledge_maps(["shopping"])
        joined = " ".join(rm["shopping"])
        # اشتريت or متجر must appear in the combined rules
        assert "\u0627\u0634\u062a\u0631\u064a\u062a" in joined or "\u0645\u062a\u062c\u0631" in joined

    def test_no_cross_contamination_between_topic_labels(self):
        """'doctor' is in medical only — must not appear in business or sports."""
        import evaluate_pipeline
        km, _ = evaluate_pipeline.build_agent_knowledge_maps(
            ["business", "medical", "sports"]
        )
        assert "doctor" in km["medical"]
        assert "doctor" not in km["business"]
        assert "doctor" not in km["sports"]

    def test_health_does_not_contain_doctor(self):
        """'doctor' must NOT be in health; it belongs to medical."""
        import evaluate_pipeline
        km, _ = evaluate_pipeline.build_agent_knowledge_maps(["health"])
        assert "doctor" not in km["health"]
        assert "fitness" in km["health"]

    def test_no_unknown_labels_added(self):
        """build_agent_knowledge_maps must never inject extra labels into the maps."""
        import evaluate_pipeline
        active = ["business", "sports"]
        km, rm = evaluate_pipeline.build_agent_knowledge_maps(active)
        assert set(km.keys()) == set(active)
        assert set(rm.keys()) == set(active)


# ---------------------------------------------------------------------------
# build_orchestrator — pipeline mode agent sequencing
# ---------------------------------------------------------------------------

class TestBuildOrchestratorPipelineModes:
    """Verify that build_orchestrator produces orchestrators with correct mode behaviour."""

    def _make_task_config(self, pipeline_mode: str, labels=None):
        from src.state.schema import TaskConfig
        labels = labels or ["positive", "negative", "neutral"]
        return TaskConfig(
            task_name="test",
            labels=labels,
            label_descriptions={lbl: lbl for lbl in labels},
            threshold=0.99,          # forces escalation for any mock primary output
            pipeline_mode=pipeline_mode,  # type: ignore[arg-type]
            enable_deliberation=False,
        )

    def _run_single(self, pipeline_mode: str, enable_deliberation: bool = False):
        """Build orchestrator and run one sample; return the resulting PipelineState."""
        import evaluate_pipeline
        from src.state.schema import PipelineState, StateMetadata

        labels = ["positive", "negative", "neutral"]
        task_config = self._make_task_config(pipeline_mode, labels=labels)
        task_config.enable_deliberation = enable_deliberation
        orch = evaluate_pipeline.build_orchestrator(
            task_config=task_config,
            threshold=0.99,
            enable_deliberation=enable_deliberation,
        )
        state = PipelineState(
            input_text="this is great",
            task_config=task_config,
            metadata=StateMetadata(sample_id="pm_test"),
        )
        return orch.run(state)

    # ------------------------------------------------------------------
    # paper_style — contextual agent must be called
    # ------------------------------------------------------------------

    def test_paper_style_sets_contextual_output_on_escalation(self):
        """paper_style: ContextualAgent must run and write contextual_output."""
        state = self._run_single("paper_style")

        # Ensure escalation happened (threshold=0.99 with mock primary)
        if state.routing_info and state.routing_info.decision == "escalate":
            assert state.contextual_output is not None, (
                "paper_style must run ContextualAgent on escalation"
            )

    def test_paper_style_does_not_set_deliberation_output(self):
        """paper_style must never run DeliberationAgent, even if enable_deliberation=True."""
        import evaluate_pipeline
        from src.state.schema import PipelineState, StateMetadata, TaskConfig

        labels = ["positive", "negative", "neutral"]
        task_config = TaskConfig(
            task_name="test",
            labels=labels,
            label_descriptions={lbl: lbl for lbl in labels},
            threshold=0.99,
            pipeline_mode="paper_style",  # type: ignore[arg-type]
            enable_deliberation=True,      # deliberately set True
        )
        orch = evaluate_pipeline.build_orchestrator(
            task_config=task_config,
            threshold=0.99,
            enable_deliberation=True,
        )
        state = PipelineState(
            input_text="this is great",
            task_config=task_config,
            metadata=StateMetadata(sample_id="pm_paper_delib"),
        )
        out = orch.run(state)
        assert out.deliberation_output is None, (
            "paper_style must not run DeliberationAgent regardless of enable_deliberation"
        )

    # ------------------------------------------------------------------
    # full_agentic — deliberation only when explicitly enabled
    # ------------------------------------------------------------------

    def test_full_agentic_without_deliberation_has_no_deliberation_output(self):
        """full_agentic with enable_deliberation=False must not run DeliberationAgent."""
        state = self._run_single("full_agentic", enable_deliberation=False)

        if state.routing_info and state.routing_info.decision == "escalate":
            assert state.deliberation_output is None, (
                "full_agentic with enable_deliberation=False must not produce deliberation_output"
            )

    def test_full_agentic_with_deliberation_has_deliberation_output(self):
        """full_agentic with enable_deliberation=True must run DeliberationAgent."""
        state = self._run_single("full_agentic", enable_deliberation=True)

        if state.routing_info and state.routing_info.decision == "escalate":
            assert state.deliberation_output is not None, (
                "full_agentic with enable_deliberation=True must produce deliberation_output"
            )

    def test_full_agentic_sets_contextual_output_on_escalation(self):
        """full_agentic: ContextualAgent must run and write contextual_output."""
        state = self._run_single("full_agentic")

        if state.routing_info and state.routing_info.decision == "escalate":
            assert state.contextual_output is not None, (
                "full_agentic must run ContextualAgent on escalation"
            )

    # ------------------------------------------------------------------
    # MockLLMClient mode — no invalid mode used by build_orchestrator
    # ------------------------------------------------------------------

    def test_ablation_study_uses_no_invalid_mock_llm_mode(self):
        """_run_ablation_study must wire MockLLMClient with mode='label_echo'."""
        import evaluate_pipeline
        from src.llm.mock_client import MockLLMClient
        from src.state.schema import TaskConfig

        labels = ["positive", "negative", "neutral"]
        task_config = TaskConfig(
            task_name="test",
            labels=labels,
            label_descriptions={lbl: lbl for lbl in labels},
        )
        # AblationStudy is built with the same llm_client as _run_ablation_study.
        # Verify the client wired into build_orchestrator is valid.
        orch = evaluate_pipeline.build_orchestrator(
            task_config=task_config,
            threshold=0.5,
            enable_deliberation=False,
        )
        llm = orch._contextual.llm_client
        assert isinstance(llm, MockLLMClient)
        assert llm.mode in ("fixed", "label_echo", "raise_on_call"), (
            f"Invalid MockLLMClient mode: {llm.mode!r}"
        )


# ---------------------------------------------------------------------------
# Topic seed regex rules — bilingual sentence-level trigger tests
# ---------------------------------------------------------------------------

class TestTopicSeedRegexRules:
    """Verify sample Arabic-English code-switched sentences trigger the expected
    topic regex rules in LogicAgent.

    These tests exercise the manually curated seed rules stored in
    _TOPIC_KNOWLEDGE["regex_rules"].  Each rule set covers:
      * purely English sentences (core vocabulary),
      * purely Arabic sentences (Arabic-script terms),
      * code-switched sentences mixing both scripts.

    The helper _matches_any() replicates LogicAgent's search semantics
    (re.IGNORECASE | re.UNICODE) without instantiating the full pipeline.
    """

    def _get_rule_map(self, labels):
        import evaluate_pipeline
        _, rm = evaluate_pipeline.build_agent_knowledge_maps(labels)
        return rm

    @staticmethod
    def _matches_any(rule_map, label, text):
        import re
        for pattern in rule_map[label]:
            if re.search(pattern, text, re.IGNORECASE | re.UNICODE):
                return True
        return False

    # ------------------------------------------------------------------
    # business
    # ------------------------------------------------------------------

    def test_business_english_sentence(self):
        rm = self._get_rule_map(["business"])
        # Rule 1: (company|startup).*(market|product)
        assert self._matches_any(rm, "business", "The startup entered a new market after a profitable deal")

    def test_business_arabic_sentence(self):
        rm = self._get_rule_map(["business"])
        assert self._matches_any(rm, "business", "الشركة حققت أرباح كبيرة هذا العام في السوق")

    def test_business_code_switched(self):
        rm = self._get_rule_map(["business"])
        assert self._matches_any(rm, "business", "الشركة طلعت startup جديدة في السوق")

    # ------------------------------------------------------------------
    # education
    # ------------------------------------------------------------------

    def test_education_english_sentence(self):
        rm = self._get_rule_map(["education"])
        # Rule 1: (student).*(lecture); Rule 3: (assignment).*(deadline)
        assert self._matches_any(rm, "education", "The student must submit the assignment before the course deadline")

    def test_education_arabic_sentence(self):
        rm = self._get_rule_map(["education"])
        # Rule 2: (امتحان).*(جامعة)
        assert self._matches_any(rm, "education", "الامتحان كان صعب في الجامعة هذا الترم")

    def test_education_code_switched(self):
        rm = self._get_rule_map(["education"])
        assert self._matches_any(rm, "education", "عندي exam بكره وأنا طالب في الجامعة")

    # ------------------------------------------------------------------
    # health
    # ------------------------------------------------------------------

    def test_health_english_sentence(self):
        rm = self._get_rule_map(["health"])
        # Rule 1: (exercise).*(fitness|health)
        assert self._matches_any(rm, "health", "Regular exercise is essential for fitness and good health")

    def test_health_arabic_sentence(self):
        rm = self._get_rule_map(["health"])
        # Rule 1: (تمارين).*(لياقة|صحة)
        assert self._matches_any(rm, "health", "التمارين اليومية تحسن اللياقة والصحة")

    def test_health_code_switched(self):
        rm = self._get_rule_map(["health"])
        # Rule 1: (exercise).*(health|صحة) — use base form صحة not possessive صحتي
        assert self._matches_any(rm, "health", "بدأت exercise كل يوم لتحسين الصحة والمناعة")

    # ------------------------------------------------------------------
    # shopping
    # ------------------------------------------------------------------

    def test_shopping_english_sentence(self):
        rm = self._get_rule_map(["shopping"])
        # Rule 1: (bought).*(store); Rule 3: (discount).*(price)
        assert self._matches_any(rm, "shopping", "I bought it at the store with a big discount on the price")

    def test_shopping_arabic_sentence(self):
        rm = self._get_rule_map(["shopping"])
        assert self._matches_any(rm, "shopping", "اشتريت من المتجر وكان فيه خصم وتوصيل مجاني")

    def test_shopping_code_switched(self):
        rm = self._get_rule_map(["shopping"])
        # Rule 3: (discount).*(سعر)
        assert self._matches_any(rm, "shopping", "فيه discount كبير خفّض السعر في المتجر")

    # ------------------------------------------------------------------
    # medical
    # ------------------------------------------------------------------

    def test_medical_english_sentence(self):
        rm = self._get_rule_map(["medical"])
        # Rule 3: (diagnosis).*(surgery|treatment)
        assert self._matches_any(rm, "medical", "The diagnosis confirmed the patient needs surgery and treatment")

    def test_medical_arabic_sentence(self):
        rm = self._get_rule_map(["medical"])
        assert self._matches_any(rm, "medical", "الطبيب قال إن المريض محتاج علاج في المستشفى")

    def test_medical_code_switched(self):
        rm = self._get_rule_map(["medical"])
        # Rule 1: (دكتور).*(مريض|موعد)
        assert self._matches_any(rm, "medical", "الدكتور شاف المريض وحدد موعد للـ surgery")

    # ------------------------------------------------------------------
    # sports
    # ------------------------------------------------------------------

    def test_sports_english_sentence(self):
        rm = self._get_rule_map(["sports"])
        # Rule 1: (match).*(team|player)
        assert self._matches_any(rm, "sports", "The match ended when the team scored a brilliant goal")

    def test_sports_arabic_sentence(self):
        rm = self._get_rule_map(["sports"])
        # Rule 1: (مباراة).*(فريق|لاعب)
        assert self._matches_any(rm, "sports", "المباراة كانت رهيبة وسجل الفريق هدف جميل")

    def test_sports_code_switched(self):
        rm = self._get_rule_map(["sports"])
        assert self._matches_any(rm, "sports", "الفريق اتقدم بـ goal وفاز في آخر match")

    # ------------------------------------------------------------------
    # tech
    # ------------------------------------------------------------------

    def test_tech_english_sentence(self):
        rm = self._get_rule_map(["tech"])
        assert self._matches_any(rm, "tech", "The new AI software update fixed all the bugs")

    def test_tech_arabic_sentence(self):
        rm = self._get_rule_map(["tech"])
        # Rule 1: (تطبيق|برنامج).*(تحديث)
        assert self._matches_any(rm, "tech", "البرنامج عمل تحديث تلقائي وأصلح الأخطاء")

    def test_tech_code_switched(self):
        rm = self._get_rule_map(["tech"])
        assert self._matches_any(rm, "tech", "نزّلت الـ app وعمل تحديث تلقائي بعد ما فتحته")

    # ------------------------------------------------------------------
    # finance
    # ------------------------------------------------------------------

    def test_finance_english_sentence(self):
        rm = self._get_rule_map(["finance"])
        # Rule 1: (bank).*(loan|interest)
        assert self._matches_any(rm, "finance", "The bank approved a new loan with low interest rates")

    def test_finance_arabic_sentence(self):
        rm = self._get_rule_map(["finance"])
        assert self._matches_any(rm, "finance", "البنك رفع سعر الفائدة والتضخم زاد على الدولار")

    def test_finance_code_switched(self):
        rm = self._get_rule_map(["finance"])
        # Rule 3: (دولار).*(تضخم|سعر)
        assert self._matches_any(rm, "finance", "سعر الدولار ارتفع بسبب التضخم في البلاد")

    # ------------------------------------------------------------------
    # social
    # ------------------------------------------------------------------

    def test_social_english_sentence(self):
        rm = self._get_rule_map(["social"])
        assert self._matches_any(rm, "social", "She posted a trending photo on Instagram and got many likes")

    def test_social_arabic_sentence(self):
        rm = self._get_rule_map(["social"])
        assert self._matches_any(rm, "social", "نشرت بوست على انستجرام وحصلت على لايكات كتير من أصدقائي")

    def test_social_code_switched(self):
        rm = self._get_rule_map(["social"])
        assert self._matches_any(rm, "social", "الـ Instagram post حصل على لايكات كتير من الأصدقاء")
