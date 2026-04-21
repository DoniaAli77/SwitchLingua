"""Simple script to run the mock primary classifier on one sample state."""

from __future__ import annotations

try:
    from src.models.mock_primary_classifier import MockPrimaryClassifier
    from src.state.schema import PipelineState, StateMetadata, TaskConfig, utc_now
except ModuleNotFoundError:
    import pathlib
    import sys

    project_root = pathlib.Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from src.models.mock_primary_classifier import MockPrimaryClassifier
    from src.state.schema import PipelineState, StateMetadata, TaskConfig, utc_now


def main() -> None:
    state = PipelineState(
        metadata=StateMetadata(sample_id="topic-001", timestamp=utc_now()),
        input_text="The team won the tournament after a dramatic final match.",
        task_config=TaskConfig(
            task_name="topic_classification",
            labels=["sports", "technology", "finance"],
            label_descriptions={
                "sports": "Sports and competitions.",
                "technology": "Tech products and software.",
                "finance": "Markets and economic topics.",
            },
        ),
    )

    classifier = MockPrimaryClassifier(
        mode="heuristic",
        seed=7,
        keyword_label_map={
            "sports": ["team", "match", "tournament"],
            "technology": ["software", "AI", "chip"],
            "finance": ["stocks", "market", "bank"],
        },
    )

    updated_state = classifier.run(state)
    primary = updated_state.primary_model

    print("Predicted label:", primary.label)
    print("Confidence:", primary.confidence)
    print("Probabilities:", primary.probabilities)


if __name__ == "__main__":
    main()
