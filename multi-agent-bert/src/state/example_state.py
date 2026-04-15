"""Small example showing how to initialize a generic pipeline state."""

from __future__ import annotations

try:
    from src.state.schema import (
        AgentOutput,
        ConsensusOutput,
        ExplanationOutput,
        FinalOutput,
        ModelOutput,
        PipelineState,
        RoutingInfo,
        StateMetadata,
        TaskConfig,
        utc_now,
    )
except ModuleNotFoundError:
    from schema import (  # type: ignore
        AgentOutput,
        ConsensusOutput,
        ExplanationOutput,
        FinalOutput,
        ModelOutput,
        PipelineState,
        RoutingInfo,
        StateMetadata,
        TaskConfig,
        utc_now,
    )


def build_example_state() -> PipelineState:
    """Create a minimal state example for sentiment-like classification."""

    task_config = TaskConfig(
        task_name="sentiment",
        labels=["positive", "negative", "neutral"],
        label_descriptions={
            "positive": "Text expresses favorable sentiment.",
            "negative": "Text expresses unfavorable sentiment.",
            "neutral": "Text is balanced or emotionally neutral.",
        },
    )

    state = PipelineState(
        metadata=StateMetadata(sample_id="sample-001", timestamp=utc_now()),
        input_text="The product is easy to use and performs well.",
        task_config=task_config,
        primary_model_output=ModelOutput(
            label="positive",
            confidence=0.91,
            probabilities={"positive": 0.91, "negative": 0.03, "neutral": 0.06},
            raw_text="Primary model predicted positive sentiment.",
        ),
        routing_info=RoutingInfo(threshold=0.75, decision="use_all_specialists"),
        lexical_output=AgentOutput(
            agent_name="lexical",
            model_output=ModelOutput(label="positive", confidence=0.88),
            notes="Lexical cues favor positive sentiment.",
        ),
        contextual_output=AgentOutput(
            agent_name="contextual",
            model_output=ModelOutput(label="positive", confidence=0.86),
            notes="Context remains positive in user intent.",
        ),
        logic_output=AgentOutput(
            agent_name="logic",
            model_output=ModelOutput(label="neutral", confidence=0.52),
            notes="No strong contradiction, minor uncertainty.",
        ),
        consensus_output=ConsensusOutput(
            label="positive",
            confidence=0.89,
            votes={"lexical": "positive", "contextual": "positive", "logic": "neutral"},
            rationale="Majority supports positive label.",
        ),
        explanation_output=ExplanationOutput(
            summary="Positive language and context dominate.",
            evidence=["easy to use", "performs well"],
            caveats=["logic agent reported mild uncertainty"],
        ),
        final_output=FinalOutput(
            label="positive",
            confidence=0.89,
            payload={"task": "sentiment", "source": "example"},
        ),
    )

    state.validate_labels()
    return state


if __name__ == "__main__":
    example = build_example_state()
    print(f"Example state initialized for sample_id={example.metadata.sample_id}")
