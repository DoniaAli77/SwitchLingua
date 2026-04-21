"""Routing component for the stateful multi-agent text classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from src.state.schema import FinalOutput, PipelineState, RoutingInfo


@dataclass(slots=True)
class Router:
    """Route pipeline flow based on primary model confidence and task threshold."""

    def run(self, state: PipelineState) -> PipelineState:
        """Set routing decision and optionally finalize from primary output."""

        if state.primary_model is None:
            raise ValueError("Missing primary model output in state.primary_model.")

        primary = state.primary_model
        if primary.label is None:
            raise ValueError("Primary model output is missing predicted label.")
        if primary.confidence is None:
            raise ValueError("Primary model output is missing confidence value.")

        threshold = state.task_config.threshold
        if threshold is None:
            raise ValueError("state.task_config.threshold must be configured.")

        state.task_config.validate_label(primary.label, field_name="primary_model.label")
        primary.validate_labels(state.task_config, field_name="primary_model.label")

        decision = "accept_primary" if primary.confidence >= threshold else "escalate"
        state.routing_info = RoutingInfo(threshold=threshold, decision=decision)

        if decision == "accept_primary":
            state.final_output = FinalOutput(
                label=primary.label,
                confidence=primary.confidence,
                payload={
                    "source": "primary_model",
                    "probabilities": dict(primary.probabilities),
                    "raw_text": primary.raw_text,
                },
            )

        state.append_history(
            component="router",
            summary=(
                f"Decision: {decision} "
                f"(confidence={primary.confidence:.3f} vs threshold={threshold:.3f})"
            ),
            outputs={
                "decision": decision,
                "threshold": threshold,
                "primary_label": primary.label,
                "primary_confidence": primary.confidence,
            },
        )
        return state
