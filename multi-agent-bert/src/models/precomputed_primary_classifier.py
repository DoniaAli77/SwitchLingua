"""Frozen primary classifier backed by precomputed predictions (e.g. an external model).

Reads a CSV of per-sample predictions + probabilities and exposes the same
``run(state) -> state`` interface as
:class:`~src.models.primary_transformer_classifier.PrimaryTransformerClassifier`,
so the orchestrator/router/agents need no changes. No model is loaded or run; the
"prediction" is a lookup keyed by ``state.metadata.sample_id``.

Expected CSV columns:
``sample_id, text, true_label, pred_label, prob_negative, prob_neutral,
prob_positive, confidence`` (extra columns ignored). Probabilities are reported as
``{"negative": .., "neutral": .., "positive": ..}``.

Used for the Ahmed-as-frozen-primary experiment: Ahmed's provided aligned predictions
are used as a frozen primary; the agentic layer is evaluated on top of them. The
predictions are NOT modified, and this does NOT reproduce Ahmed's preprocessing or
training pipeline.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.state.schema import ModelOutput, PipelineState

log = logging.getLogger(__name__)

_PROB_COLS = {"negative": "prob_negative", "neutral": "prob_neutral", "positive": "prob_positive"}


class PrecomputedPrimaryClassifier:
    """Primary classifier that returns precomputed predictions by ``sample_id``."""

    def __init__(
        self,
        predictions_path: str,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.predictions_path = str(predictions_path)
        self.name = name or "PrecomputedPrimaryClassifier"
        self.logger = logger or log
        self._by_id: Dict[str, Dict[str, object]] = {}
        self._load()

    def _load(self) -> None:
        p = Path(self.predictions_path)
        if not p.exists():
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_path}")
        with open(p, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                sid = row["sample_id"]
                probs = {lbl: float(row[col]) for lbl, col in _PROB_COLS.items()}
                self._by_id[sid] = {
                    "label": row["pred_label"],
                    "confidence": float(row["confidence"]),
                    "probabilities": probs,
                }
        if not self._by_id:
            raise ValueError(f"No predictions loaded from '{self.predictions_path}'.")
        self.logger.info("%s: loaded %d precomputed predictions.", self.name, len(self._by_id))

    def predict(self, sample_id: str, task_labels: Optional[List[str]] = None) -> ModelOutput:
        if sample_id not in self._by_id:
            raise KeyError(
                f"{self.name}: no precomputed prediction for sample_id={sample_id!r}."
            )
        rec = self._by_id[sample_id]
        probs: Dict[str, float] = dict(rec["probabilities"])  # type: ignore[arg-type]
        if task_labels is not None:
            probs = {lbl: float(probs.get(lbl, 0.0)) for lbl in task_labels}
        return ModelOutput(
            label=str(rec["label"]),
            confidence=round(float(rec["confidence"]), 6),  # type: ignore[arg-type]
            probabilities=probs,
            raw_text="",
        )

    def run(self, state: PipelineState) -> PipelineState:
        """Look up the precomputed prediction by sample_id and write to state."""
        labels: List[str] = state.task_config.labels
        if not labels:
            raise ValueError(f"{self.name}: state.task_config.labels cannot be empty.")

        sid = state.metadata.sample_id
        output = self.predict(sample_id=sid, task_labels=labels)
        output.raw_text = state.input_text

        if output.label is None:
            raise ValueError(f"{self.name}: prediction returned no label.")  # pragma: no cover
        state.task_config.validate_label(output.label, field_name="primary_model.label")
        output.validate_labels(state.task_config, field_name="primary_model.label")

        state.primary_model = output
        state.append_history(
            component="primary_classifier",
            summary=(
                f"Precomputed '{output.label}' with confidence "
                f"{output.confidence:.3f} (source='{self.predictions_path}')"
            ),
            outputs={
                "label": output.label,
                "confidence": output.confidence,
                "probabilities": dict(output.probabilities),
                "source": "precomputed_external",
                "predictions_file": self.predictions_path,
            },
        )
        return state

    def __repr__(self) -> str:
        return (
            f"PrecomputedPrimaryClassifier(path={self.predictions_path!r}, "
            f"n={len(self._by_id)})"
        )
