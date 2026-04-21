"""Mock primary classifier for stateful multi-agent text classification pipelines."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from src.state.schema import ModelOutput, PipelineState

Mode = Literal["fixed", "random", "heuristic"]


@dataclass(slots=True)
class MockPrimaryClassifier:
    """Mock classifier supporting fixed, random, and keyword-assisted heuristics."""

    mode: Mode = "fixed"
    seed: Optional[int] = None

    fixed_label: Optional[str] = None
    fixed_confidence: float = 0.8
    fixed_probabilities: Optional[Dict[str, float]] = None

    keyword_label_map: Dict[str, List[str]] = field(default_factory=dict)

    def run(self, state: PipelineState) -> PipelineState:
        """Predict one label and write ModelOutput to state.primary_model."""

        labels = state.task_config.labels
        if not labels:
            raise ValueError("PipelineState.task_config.labels cannot be empty.")

        if self.mode == "fixed":
            output = self._predict_fixed(labels)
        elif self.mode == "random":
            output = self._predict_random(labels)
        elif self.mode == "heuristic":
            output = self._predict_heuristic(state.input_text, labels)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if output.label is None:
            raise ValueError("MockPrimaryClassifier produced no predicted label.")

        state.task_config.validate_label(output.label, field_name="primary_model.label")
        output.validate_labels(state.task_config, field_name="primary_model.label")

        state.primary_model = output
        state.append_history(
            component="primary_classifier",
            summary=(
                f"Predicted '{output.label}' with confidence "
                f"{output.confidence:.3f} (mode={self.mode})"
            ),
            outputs={
                "label": output.label,
                "confidence": output.confidence,
                "probabilities": dict(output.probabilities),
                "mode": self.mode,
            },
        )
        return state

    def _predict_fixed(self, labels: List[str]) -> ModelOutput:
        if self.fixed_label is None:
            label = labels[0]
        else:
            label = self.fixed_label

        probabilities = self._build_fixed_probabilities(labels, label)
        confidence = probabilities[label]
        return ModelOutput(label=label, confidence=confidence, probabilities=probabilities)

    def _predict_random(self, labels: List[str]) -> ModelOutput:
        rng = random.Random(self.seed)
        weights = [rng.random() for _ in labels]
        probabilities = self._normalize_from_weights(labels, weights)
        predicted_label = max(probabilities, key=probabilities.get)
        return ModelOutput(
            label=predicted_label,
            confidence=probabilities[predicted_label],
            probabilities=probabilities,
        )

    def _predict_heuristic(self, text: str, labels: List[str]) -> ModelOutput:
        if not self.keyword_label_map:
            return self._predict_random(labels)

        lowered = text.lower()
        scores: Dict[str, float] = {label: 0.0 for label in labels}

        for label, keywords in self.keyword_label_map.items():
            if label not in scores:
                continue
            for keyword in keywords:
                if keyword.lower() in lowered:
                    scores[label] += 1.0

        if max(scores.values()) == 0.0:
            return self._predict_random(labels)

        probabilities = self._normalize_from_weights(labels, [scores[label] for label in labels])
        predicted_label = max(probabilities, key=probabilities.get)
        return ModelOutput(
            label=predicted_label,
            confidence=probabilities[predicted_label],
            probabilities=probabilities,
        )

    def _build_fixed_probabilities(self, labels: List[str], label: str) -> Dict[str, float]:
        if self.fixed_probabilities:
            probabilities = {key: float(value) for key, value in self.fixed_probabilities.items()}
            missing_labels = [item for item in labels if item not in probabilities]
            for missing in missing_labels:
                probabilities[missing] = 0.0
            return self._normalize_probabilities(labels, probabilities)

        confidence = min(max(self.fixed_confidence, 0.0), 1.0)
        if len(labels) == 1:
            return {labels[0]: 1.0}

        remainder = (1.0 - confidence) / (len(labels) - 1)
        probabilities: Dict[str, float] = {}
        for item in labels:
            probabilities[item] = confidence if item == label else remainder
        return self._normalize_probabilities(labels, probabilities)

    @staticmethod
    def _normalize_from_weights(labels: List[str], weights: List[float]) -> Dict[str, float]:
        total = sum(weights)
        if total <= 0.0:
            uniform = 1.0 / len(labels)
            return {label: uniform for label in labels}
        return {label: weight / total for label, weight in zip(labels, weights)}

    @staticmethod
    def _normalize_probabilities(labels: List[str], probabilities: Dict[str, float]) -> Dict[str, float]:
        filtered = {label: max(float(probabilities.get(label, 0.0)), 0.0) for label in labels}
        total = sum(filtered.values())
        if total <= 0.0:
            uniform = 1.0 / len(labels)
            return {label: uniform for label in labels}
        return {label: value / total for label, value in filtered.items()}
