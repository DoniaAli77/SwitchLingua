"""Example script showing both router branches: accept_primary and escalate."""

from __future__ import annotations

try:
    from src.pipeline.router import Router
    from src.state.schema import ModelOutput, PipelineState, StateMetadata, TaskConfig, utc_now
except ModuleNotFoundError:
    import pathlib
    import sys

    project_root = pathlib.Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.pipeline.router import Router
    from src.state.schema import ModelOutput, PipelineState, StateMetadata, TaskConfig, utc_now


def build_state(confidence: float) -> PipelineState:
    """Create a sample state with configurable primary confidence."""

    remaining = max(0.0, 1.0 - confidence)
    technology_prob = remaining / 2.0
    finance_prob = remaining / 2.0

    return PipelineState(
        metadata=StateMetadata(sample_id=f"sample-{confidence}", timestamp=utc_now()),
        input_text="Sample text for routing demo.",
        task_config=TaskConfig(
            task_name="topic_classification",
            labels=["sports", "technology", "finance"],
            label_descriptions={
                "sports": "Sports content.",
                "technology": "Technology content.",
                "finance": "Finance content.",
            },
            threshold=0.75,
        ),
        primary_model_output=ModelOutput(
            label="sports",
            confidence=confidence,
            probabilities={
                "sports": confidence,
                "technology": technology_prob,
                "finance": finance_prob,
            },
            raw_text="Primary model mock output",
        ),
    )


def main() -> None:
    router = Router()

    accepted = router.run(build_state(confidence=0.92))
    print("High confidence branch")
    print("decision:", accepted.routing_info.decision if accepted.routing_info else None)
    print("final_output:", accepted.final_output)

    escalated = router.run(build_state(confidence=0.42))
    print("\nLow confidence branch")
    print("decision:", escalated.routing_info.decision if escalated.routing_info else None)
    print("final_output:", escalated.final_output)


if __name__ == "__main__":
    main()
