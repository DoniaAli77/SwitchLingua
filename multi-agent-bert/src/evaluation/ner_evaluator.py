"""ner_evaluator.py — Evaluation framework for sequence-labeling (NER) tasks.

Evaluates a pipeline that produces BIO-tagged token sequences and computes:

* **Token accuracy** – proportion of tokens with the correct BIO tag.
* **Macro-averaged F1** – mean F1 across all BIO tags (including "O").
* **Per-tag metrics** – precision, recall, F1, and support for every tag.

Dataset format expected by :meth:`NEREvaluator.evaluate`
---------------------------------------------------------
A list of dicts, each with:

.. code-block:: json

    {"id": "001", "text": "Ahmed works at Google",
     "tokens": ["Ahmed", "works", "at", "Google"],
     "tags":   ["B-PER", "O", "O", "B-ORG"]}

The ``"id"`` field is optional; a sequential id is auto-assigned when absent.

Outputs written by :meth:`NEREvaluator.save`
--------------------------------------------
- ``{run_id}_ner_predictions.json``  – per-sample id/text/tokens/gold_tags/predicted_tags
- ``{run_id}_ner_predictions.csv``   – same as flat CSV (arrays JSON-encoded per cell)
- ``{run_id}_ner_metrics.json``      – aggregate + per-tag metrics
- ``{run_id}_ner_metrics.csv``       – one row per tag + one summary row
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.evaluator import (
    PerClassMetrics,
    _accuracy,  # noqa: F401  (re-exported for callers)
    _macro_f1,
    _per_class_metrics,
)
from src.state.schema import FinalOutput, PipelineState, StateMetadata, TaskConfig

log = logging.getLogger(__name__)

# NERPerTagMetrics is structurally identical to PerClassMetrics.
# We alias it so callers can use the NER-specific name without coupling to
# the classification evaluator.
NERPerTagMetrics = PerClassMetrics


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class NERSampleResult:
    """Prediction result for a single NER sample."""

    sample_id: str
    input_text: str
    tokens: List[str]
    gold_tags: List[str]
    predicted_tags: List[str]
    token_count: int
    correct_count: int          # tokens where gold_tag == predicted_tag
    pipeline_error: Optional[str]


@dataclass
class NERReport:
    """Full NER evaluation report returned by :meth:`NEREvaluator.evaluate`."""

    run_id: str
    timestamp: str
    num_samples: int
    num_tokens: int             # total tokens across all samples
    token_accuracy: float       # correct_tokens / num_tokens
    macro_f1: float             # macro avg F1 over all BIO tags
    per_tag: List[NERPerTagMetrics]
    samples: List[NERSampleResult]
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tag-alignment helper
# ---------------------------------------------------------------------------


def _align_tags(
    gold: List[str],
    pred: List[str],
    n: int,
) -> Tuple[List[str], List[str]]:
    """Return two tag lists of exactly length *n*.

    Pads short lists with ``"O"`` and truncates lists longer than *n*.
    This prevents crashes when the pipeline emits fewer/more tags than there
    are tokens in the gold annotation.
    """
    gold_aligned = (list(gold) + ["O"] * n)[:n]
    pred_aligned = (list(pred) + ["O"] * n)[:n]
    return gold_aligned, pred_aligned


# ---------------------------------------------------------------------------
# NEREvaluator
# ---------------------------------------------------------------------------


class NEREvaluator:
    """Evaluates a pipeline on sequence-labeling (NER) data.

    Parameters
    ----------
    task_config:
        Task configuration.  ``task_config.task_type`` must be
        ``"sequence_labeling"`` and ``task_config.labels`` must list all
        valid BIO tags (e.g. ``["O", "B-PER", "I-PER", ...]``).
    orchestrator:
        A fully-configured :class:`~src.pipeline.orchestrator.PipelineOrchestrator`
        with NER agents wired in.  Must expose a ``run(state) -> state`` method.
    run_id:
        Identifier for this run.  Defaults to a UTC timestamp.
    logger:
        Optional pre-configured logger.
    """

    def __init__(
        self,
        task_config: TaskConfig,
        orchestrator,
        run_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if orchestrator is None:
            raise ValueError("NEREvaluator requires an orchestrator.")
        self.task_config = task_config
        self.orchestrator = orchestrator
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.logger = logger or log

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, dataset: List[Dict]) -> NERReport:
        """Evaluate the NER pipeline over *dataset*.

        Parameters
        ----------
        dataset:
            A list of dicts, each with at minimum:
            ``{"text": "...", "tokens": [...], "tags": [...]}``
            Optional key ``"id"`` overrides the auto-generated sample id.

        Returns
        -------
        NERReport
            Full report including per-sample predictions and aggregate metrics.
        """
        if not dataset:
            raise ValueError("dataset is empty.")

        self.logger.info(
            "NEREvaluator — run_id=%s samples=%d",
            self.run_id,
            len(dataset),
        )

        samples: List[NERSampleResult] = []
        for idx, item in enumerate(dataset):
            result = self._run_one(idx, item)
            samples.append(result)

        return self._compute_report(samples)

    def save(
        self,
        report: NERReport,
        output_dir: str = "results",
        run_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Persist *report* to disk as JSON and CSV files.

        Parameters
        ----------
        report:
            The :class:`NERReport` returned by :meth:`evaluate`.
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
            "predictions_json": os.path.join(output_dir, f"{rid}_ner_predictions.json"),
            "predictions_csv":  os.path.join(output_dir, f"{rid}_ner_predictions.csv"),
            "metrics_json":     os.path.join(output_dir, f"{rid}_ner_metrics.json"),
            "metrics_csv":      os.path.join(output_dir, f"{rid}_ner_metrics.csv"),
        }

        self._save_predictions_json(report, paths["predictions_json"])
        self._save_predictions_csv(report, paths["predictions_csv"])
        self._save_metrics_json(report, paths["metrics_json"])
        self._save_metrics_csv(report, paths["metrics_csv"])

        self.logger.info("NEREvaluator — saved outputs to '%s'", output_dir)
        for key, path in paths.items():
            self.logger.info("  %s: %s", key, path)
        return paths

    # ------------------------------------------------------------------
    # Internal – pipeline execution
    # ------------------------------------------------------------------

    def _run_one(self, idx: int, item: Dict) -> NERSampleResult:
        """Run the NER pipeline on a single dataset item."""
        text: str = item.get("text", "")
        tokens: List[str] = list(item.get("tokens") or text.split())
        gold_tags: List[str] = list(item.get("tags") or [])
        sample_id: str = str(item.get("id", f"sample_{idx:05d}"))

        state = PipelineState(
            metadata=StateMetadata(sample_id=sample_id),
            input_text=text,
            task_config=self.task_config,
            extras={"tokens": tokens},
        )

        pipeline_error: Optional[str] = None
        try:
            state = self.orchestrator.run(state)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "NEREvaluator — sample %s FAILED: %s", sample_id, exc
            )
            pipeline_error = str(exc)

        # Extract structured pipeline error from state.extras when present.
        if pipeline_error is None:
            err = state.extras.get("pipeline_error")
            if isinstance(err, dict):
                pipeline_error = err.get("message") or err.get("stage")

        # Extract predicted tags (falls back to all-"O" when unavailable).
        predicted_tags = self._extract_predicted_tags(state, len(tokens))

        # Ensure both tag sequences have the same length as the token list.
        gold_tags, predicted_tags = _align_tags(gold_tags, predicted_tags, len(tokens))

        correct = sum(g == p for g, p in zip(gold_tags, predicted_tags))
        return NERSampleResult(
            sample_id=sample_id,
            input_text=text,
            tokens=tokens,
            gold_tags=gold_tags,
            predicted_tags=predicted_tags,
            token_count=len(tokens),
            correct_count=correct,
            pipeline_error=pipeline_error,
        )

    @staticmethod
    def _extract_predicted_tags(state: PipelineState, n_tokens: int) -> List[str]:
        """Return predicted BIO tags from *state.final_output*.

        The NERConsensusAgent writes tags as a list of
        ``{"token": str, "tag": str, "confidence": float}`` dicts under
        ``state.final_output.payload["sequence_output"]``.

        Falls back to all-``"O"`` when the key is missing (e.g. after a
        ``primary_only`` mode run that only writes a stub FinalOutput).
        """
        if (
            state.final_output is not None
            and isinstance(state.final_output.payload, dict)
            and "sequence_output" in state.final_output.payload
        ):
            seq_out = state.final_output.payload["sequence_output"]
            if isinstance(seq_out, list) and seq_out:
                return [
                    entry.get("tag", "O") if isinstance(entry, dict) else "O"
                    for entry in seq_out
                ]
        return ["O"] * n_tokens

    # ------------------------------------------------------------------
    # Internal – metric computation
    # ------------------------------------------------------------------

    def _compute_report(self, samples: List[NERSampleResult]) -> NERReport:
        labels = self.task_config.labels

        all_gold: List[str] = []
        all_pred: List[str] = []
        total_correct = 0
        total_tokens = 0

        for s in samples:
            all_gold.extend(s.gold_tags)
            all_pred.extend(s.predicted_tags)
            total_correct += s.correct_count
            total_tokens += s.token_count

        per_tag = _per_class_metrics(all_gold, all_pred, labels)
        token_acc = (
            round(total_correct / total_tokens, 6) if total_tokens > 0 else 0.0
        )
        macro_f1 = _macro_f1(per_tag)

        return NERReport(
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            num_samples=len(samples),
            num_tokens=total_tokens,
            token_accuracy=token_acc,
            macro_f1=macro_f1,
            per_tag=per_tag,
            samples=samples,
            meta={
                "labels": labels,
                "pipeline_mode": getattr(
                    self.task_config, "pipeline_mode", "full_agentic"
                ),
                "error_samples": sum(
                    1 for s in samples if s.pipeline_error is not None
                ),
            },
        )

    # ------------------------------------------------------------------
    # Internal – serialisation
    # ------------------------------------------------------------------

    def _save_predictions_json(self, report: NERReport, path: str) -> None:
        rows = [
            {
                "id":             s.sample_id,
                "text":           s.input_text,
                "tokens":         s.tokens,
                "gold_tags":      s.gold_tags,
                "predicted_tags": s.predicted_tags,
                "token_count":    s.token_count,
                "correct_count":  s.correct_count,
                "pipeline_error": s.pipeline_error,
            }
            for s in report.samples
        ]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)

    def _save_predictions_csv(self, report: NERReport, path: str) -> None:
        fieldnames = [
            "id", "text", "tokens", "gold_tags", "predicted_tags",
            "token_count", "correct_count", "pipeline_error",
        ]
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for s in report.samples:
                writer.writerow(
                    {
                        "id":             s.sample_id,
                        "text":           s.input_text,
                        # Arrays stored as JSON strings so the CSV stays flat.
                        "tokens":         json.dumps(s.tokens, ensure_ascii=False),
                        "gold_tags":      json.dumps(s.gold_tags, ensure_ascii=False),
                        "predicted_tags": json.dumps(
                            s.predicted_tags, ensure_ascii=False
                        ),
                        "token_count":    s.token_count,
                        "correct_count":  s.correct_count,
                        "pipeline_error": s.pipeline_error or "",
                    }
                )

    def _save_metrics_json(self, report: NERReport, path: str) -> None:
        payload = {
            "run_id":         report.run_id,
            "timestamp":      report.timestamp,
            "num_samples":    report.num_samples,
            "num_tokens":     report.num_tokens,
            "token_accuracy": report.token_accuracy,
            "macro_f1":       report.macro_f1,
            "per_tag": [
                {
                    "tag":       m.label,
                    "precision": m.precision,
                    "recall":    m.recall,
                    "f1":        m.f1,
                    "support":   m.support,
                }
                for m in report.per_tag
            ],
            "meta": report.meta,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def _save_metrics_csv(self, report: NERReport, path: str) -> None:
        fieldnames = [
            "tag", "precision", "recall", "f1", "support",
            "token_accuracy", "macro_f1", "pipeline_mode",
        ]
        pipeline_mode = report.meta.get("pipeline_mode", "full_agentic")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for m in report.per_tag:
                writer.writerow(
                    {
                        "tag":            m.label,
                        "precision":      m.precision,
                        "recall":         m.recall,
                        "f1":             m.f1,
                        "support":        m.support,
                        "token_accuracy": "",
                        "macro_f1":       "",
                        "pipeline_mode":  "",
                    }
                )
            # Summary row.
            writer.writerow(
                {
                    "tag":            "__summary__",
                    "precision":      "",
                    "recall":         "",
                    "f1":             "",
                    "support":        report.num_tokens,
                    "token_accuracy": report.token_accuracy,
                    "macro_f1":       report.macro_f1,
                    "pipeline_mode":  pipeline_mode,
                }
            )
