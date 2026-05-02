"""Evaluation framework for the multi-agent text-classification pipeline.

Evaluates two modes:

* **primary only** – uses ``state.primary_model_output.label`` as the
  prediction, ignoring all specialist agents.
* **full pipeline** – uses ``state.final_output.label`` after the entire
  pipeline has run.

Metrics computed
----------------
- Accuracy
- Macro-averaged F1
- Per-class F1 (precision, recall, F1 for every label)
- Escalation rate  (fraction of samples routed to specialist agents)
- Accuracy on the escalated subset

Outputs
-------
- ``{output_dir}/{run_id}_predictions.json``  – per-sample predictions & metadata
- ``{output_dir}/{run_id}_predictions.csv``   – same as a flat CSV
- ``{output_dir}/{run_id}_metrics.json``      – aggregate + per-class metrics
- ``{output_dir}/{run_id}_metrics.csv``       – one row per label + one summary row

Usage
-----
.. code-block:: python

    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator(orchestrator=orchestrator, mode="full_pipeline")
    report = evaluator.evaluate(dataset)      # list of {"text":…, "label":…}
    evaluator.save(report, output_dir="results/", run_id="exp_01")
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from src.state.schema import (
    ModelOutput,
    PipelineState,
    StateMetadata,
    TaskConfig,
)

log = logging.getLogger(__name__)

EvalMode = Literal["primary_only", "full_pipeline"]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    """Prediction result for a single sample."""

    sample_id: str
    input_text: str
    true_label: str
    predicted_label: Optional[str]
    confidence: Optional[float]
    probabilities: Dict[str, float]
    escalated: bool
    pipeline_error: Optional[str]
    mode: EvalMode


@dataclass
class PerClassMetrics:
    """Precision, recall, and F1 for one label."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int  # number of true positives in ground truth


@dataclass
class EvalReport:
    """Full evaluation report returned by :meth:`Evaluator.evaluate`."""

    mode: EvalMode
    run_id: str
    timestamp: str
    num_samples: int
    accuracy: float
    macro_f1: float
    per_class: List[PerClassMetrics]
    escalation_rate: float
    escalated_accuracy: float
    escalated_count: int
    samples: List[SampleResult]
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metric helpers (no external deps)
# ---------------------------------------------------------------------------


def _confusion(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> Dict[str, Dict[str, int]]:
    """Return a confusion dict: counts[true][pred] = n."""
    counts: Dict[str, Dict[str, int]] = {lbl: {l: 0 for l in labels} for lbl in labels}
    for t, p in zip(y_true, y_pred):
        if t in counts and p in counts:
            counts[t][p] += 1
    return counts


def _per_class_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> List[PerClassMetrics]:
    """Compute precision, recall, F1 per class."""
    counts = _confusion(y_true, y_pred, labels)
    results: List[PerClassMetrics] = []
    for lbl in labels:
        tp = counts[lbl][lbl]
        fp = sum(counts[t][lbl] for t in labels if t != lbl)
        fn = sum(counts[lbl][p] for p in labels if p != lbl)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results.append(
            PerClassMetrics(
                label=lbl,
                precision=round(precision, 6),
                recall=round(recall, 6),
                f1=round(f1, 6),
                support=support,
            )
        )
    return results


def _accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return round(correct / len(y_true), 6)


def _macro_f1(per_class: List[PerClassMetrics]) -> float:
    if not per_class:
        return 0.0
    return round(sum(m.f1 for m in per_class) / len(per_class), 6)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Runs the pipeline over a dataset and computes evaluation metrics.

    Parameters
    ----------
    orchestrator:
        A fully-configured :class:`~src.pipeline.orchestrator.PipelineOrchestrator`
        instance (or any object with a ``run(state) -> state`` method).
        Pass ``None`` when ``mode="primary_only"`` and you supply a separate
        ``primary_classifier``.
    primary_classifier:
        Any object with a ``run(state) -> state`` method.  Only used when
        ``mode="primary_only"`` **and** ``orchestrator`` is ``None``.
        When ``orchestrator`` is set this argument is ignored.
    mode:
        ``"primary_only"``   – classify with the primary model only and read
                               ``state.primary_model_output.label``.
        ``"full_pipeline"``  – run the full orchestrator and read
                               ``state.final_output.label``.
    task_config:
        :class:`~src.state.schema.TaskConfig` to use for every sample.
        The ``labels`` list drives the per-class metric computation.
    run_id:
        Identifier for this evaluation run.  Defaults to a UTC timestamp.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        orchestrator=None,
        primary_classifier=None,
        mode: EvalMode = "full_pipeline",
        run_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if mode == "full_pipeline" and orchestrator is None:
            raise ValueError(
                "mode='full_pipeline' requires an orchestrator. "
                "Pass orchestrator=... or switch to mode='primary_only'."
            )
        if mode == "primary_only" and orchestrator is None and primary_classifier is None:
            raise ValueError(
                "mode='primary_only' requires either orchestrator or primary_classifier."
            )

        self.orchestrator = orchestrator
        self.primary_classifier = primary_classifier
        self.mode: EvalMode = mode
        self.task_config = task_config
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.logger = logger or log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset: List[Dict[str, str]],
    ) -> EvalReport:
        """Evaluate the pipeline over ``dataset``.

        Parameters
        ----------
        dataset:
            A list of dicts, each with at minimum:
            ``{"text": "...", "label": "..."}``
            Optional key ``"id"`` overrides the auto-generated sample id.

        Returns
        -------
        EvalReport
            Full report including per-sample predictions and aggregate metrics.
        """
        if not dataset:
            raise ValueError("dataset is empty.")

        self.logger.info(
            "Evaluator — mode=%s run_id=%s samples=%d",
            self.mode, self.run_id, len(dataset),
        )

        samples: List[SampleResult] = []
        for idx, item in enumerate(dataset):
            result = self._run_one(idx, item)
            samples.append(result)

        return self._compute_report(samples)

    def save(
        self,
        report: EvalReport,
        output_dir: str = "results",
        run_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Persist ``report`` to disk as JSON and CSV files.

        Parameters
        ----------
        report:
            The :class:`EvalReport` returned by :meth:`evaluate`.
        output_dir:
            Directory where output files are written.  Created if missing.
        run_id:
            Override the run id used in file names.  Defaults to
            ``report.run_id``.

        Returns
        -------
        dict
            Mapping of ``{"predictions_json", "predictions_csv",
            "metrics_json", "metrics_csv"}`` to their absolute paths.
        """
        rid = run_id or report.run_id
        os.makedirs(output_dir, exist_ok=True)

        paths = {
            "predictions_json": os.path.join(output_dir, f"{rid}_predictions.json"),
            "predictions_csv":  os.path.join(output_dir, f"{rid}_predictions.csv"),
            "metrics_json":     os.path.join(output_dir, f"{rid}_metrics.json"),
            "metrics_csv":      os.path.join(output_dir, f"{rid}_metrics.csv"),
        }

        self._save_predictions_json(report, paths["predictions_json"])
        self._save_predictions_csv(report, paths["predictions_csv"])
        self._save_metrics_json(report, paths["metrics_json"])
        self._save_metrics_csv(report, paths["metrics_csv"])

        self.logger.info("Evaluator — saved outputs to '%s'", output_dir)
        for key, path in paths.items():
            self.logger.info("  %s: %s", key, path)
        return paths

    # ------------------------------------------------------------------
    # Internal – pipeline execution
    # ------------------------------------------------------------------

    def _run_one(self, idx: int, item: Dict[str, str]) -> SampleResult:
        """Run the pipeline on a single dataset item and return a SampleResult."""
        text = item.get("text", "")
        true_label = item.get("label", "")
        sample_id = item.get("id", f"sample_{idx:05d}")

        state = PipelineState(
            metadata=StateMetadata(sample_id=sample_id),
            input_text=text,
            task_config=self.task_config,
        )

        runner = self.orchestrator if self.orchestrator is not None else self.primary_classifier

        try:
            state = runner.run(state)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Evaluator — sample %s FAILED: %s", sample_id, exc
            )
            return SampleResult(
                sample_id=sample_id,
                input_text=text,
                true_label=true_label,
                predicted_label=None,
                confidence=None,
                probabilities={},
                escalated=False,
                pipeline_error=str(exc),
                mode=self.mode,
            )

        predicted_label, confidence, probabilities = self._extract_prediction(state)
        escalated = self._is_escalated(state)
        pipeline_error = (
            state.extras.get("pipeline_error", {}).get("message")
            if isinstance(state.extras.get("pipeline_error"), dict)
            else None
        )

        return SampleResult(
            sample_id=sample_id,
            input_text=text,
            true_label=true_label,
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
            escalated=escalated,
            pipeline_error=pipeline_error,
            mode=self.mode,
        )

    def _extract_prediction(
        self, state: PipelineState
    ) -> tuple[Optional[str], Optional[float], Dict[str, float]]:
        """Return (label, confidence, probabilities) according to eval mode."""
        if self.mode == "primary_only":
            out: ModelOutput = state.primary_model_output
            return out.label, out.confidence, dict(out.probabilities)

        # full_pipeline
        if state.final_output is not None:
            fo = state.final_output
            # Probabilities live in primary_model_output; copy them for richer output.
            probs = dict(state.primary_model_output.probabilities)
            return fo.label, fo.confidence, probs

        # Fallback: final_output not set (pipeline error mid-way)
        out = state.primary_model_output
        return out.label, out.confidence, dict(out.probabilities)

    @staticmethod
    def _is_escalated(state: PipelineState) -> bool:
        """Return True if the sample was routed to specialist agents."""
        if state.routing_info is not None:
            return state.routing_info.decision == "escalate"
        # If routing info is missing, infer from presence of specialist output.
        return any(
            x is not None
            for x in (
                state.lexical_output,
                state.contextual_output,
                state.logic_output,
            )
        )

    # ------------------------------------------------------------------
    # Internal – metric computation
    # ------------------------------------------------------------------

    def _compute_report(self, samples: List[SampleResult]) -> EvalReport:
        labels = self.task_config.labels

        # Only include samples that have both true and predicted labels.
        valid = [
            s for s in samples
            if s.predicted_label is not None and s.true_label
        ]
        y_true = [s.true_label for s in valid]
        y_pred = [s.predicted_label for s in valid]  # type: ignore[misc]

        per_class = _per_class_metrics(y_true, y_pred, labels)
        accuracy = _accuracy(y_true, y_pred)
        macro_f1 = _macro_f1(per_class)

        # Escalation metrics.
        escalated = [s for s in valid if s.escalated]
        esc_rate = round(len(escalated) / len(valid), 6) if valid else 0.0
        esc_true = [s.true_label for s in escalated]
        esc_pred = [s.predicted_label for s in escalated]  # type: ignore[misc]
        esc_acc = _accuracy(esc_true, esc_pred) if escalated else 0.0

        return EvalReport(
            mode=self.mode,
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            num_samples=len(samples),
            accuracy=accuracy,
            macro_f1=macro_f1,
            per_class=per_class,
            escalation_rate=esc_rate,
            escalated_accuracy=esc_acc,
            escalated_count=len(escalated),
            samples=samples,
            meta={
                "labels": labels,
                "pipeline_mode": getattr(self.task_config, "pipeline_mode", "full_agentic"),
                "valid_samples": len(valid),
                "error_samples": len(samples) - len(valid),
            },
        )

    # ------------------------------------------------------------------
    # Internal – serialisation
    # ------------------------------------------------------------------

    def _save_predictions_json(self, report: EvalReport, path: str) -> None:
        pipeline_mode = report.meta.get("pipeline_mode", "full_agentic")
        rows = [
            {
                "sample_id":       s.sample_id,
                "input_text":      s.input_text,
                "true_label":      s.true_label,
                "predicted_label": s.predicted_label,
                "confidence":      s.confidence,
                "probabilities":   s.probabilities,
                "escalated":       s.escalated,
                "correct":         s.predicted_label == s.true_label,
                "pipeline_error":  s.pipeline_error,
                "mode":            s.mode,
                "pipeline_mode":   pipeline_mode,
            }
            for s in report.samples
        ]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)

    def _save_predictions_csv(self, report: EvalReport, path: str) -> None:
        fieldnames = [
            "sample_id", "input_text", "true_label", "predicted_label",
            "confidence", "escalated", "correct", "pipeline_error", "mode",
            "pipeline_mode",
        ]
        pipeline_mode = report.meta.get("pipeline_mode", "full_agentic")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for s in report.samples:
                writer.writerow(
                    {
                        "sample_id":       s.sample_id,
                        "input_text":      s.input_text,
                        "true_label":      s.true_label,
                        "predicted_label": s.predicted_label,
                        "confidence":      s.confidence,
                        "escalated":       s.escalated,
                        "correct":         s.predicted_label == s.true_label,
                        "pipeline_error":  s.pipeline_error,
                        "mode":            s.mode,
                        "pipeline_mode":   pipeline_mode,
                    }
                )

    def _save_metrics_json(self, report: EvalReport, path: str) -> None:
        payload = {
            "run_id":              report.run_id,
            "mode":                report.mode,
            "timestamp":           report.timestamp,
            "num_samples":         report.num_samples,
            "accuracy":            report.accuracy,
            "macro_f1":            report.macro_f1,
            "escalation_rate":     report.escalation_rate,
            "escalated_count":     report.escalated_count,
            "escalated_accuracy":  report.escalated_accuracy,
            "per_class": [
                {
                    "label":     m.label,
                    "precision": m.precision,
                    "recall":    m.recall,
                    "f1":        m.f1,
                    "support":   m.support,
                }
                for m in report.per_class
            ],
            "meta": report.meta,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def _save_metrics_csv(self, report: EvalReport, path: str) -> None:
        fieldnames = [
            "label", "precision", "recall", "f1", "support",
            "accuracy", "macro_f1", "escalation_rate",
            "escalated_count", "escalated_accuracy", "pipeline_mode",
        ]
        pipeline_mode = report.meta.get("pipeline_mode", "full_agentic")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            # One row per class.
            for m in report.per_class:
                writer.writerow(
                    {
                        "label":              m.label,
                        "precision":          m.precision,
                        "recall":             m.recall,
                        "f1":                 m.f1,
                        "support":            m.support,
                        "accuracy":           "",
                        "macro_f1":           "",
                        "escalation_rate":    "",
                        "escalated_count":    "",
                        "escalated_accuracy": "",
                        "pipeline_mode":      "",
                    }
                )
            # Summary row.
            writer.writerow(
                {
                    "label":              "__summary__",
                    "precision":          "",
                    "recall":             "",
                    "f1":                 "",
                    "support":            report.num_samples,
                    "accuracy":           report.accuracy,
                    "macro_f1":           report.macro_f1,
                    "escalation_rate":    report.escalation_rate,
                    "escalated_count":    report.escalated_count,
                    "escalated_accuracy": report.escalated_accuracy,
                    "pipeline_mode":      pipeline_mode,
                }
            )
