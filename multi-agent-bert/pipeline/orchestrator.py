"""Pipeline orchestrator that passes shared state across components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agents import (
    ConsensusAgent,
    ContextualAgent,
    ExplainabilityAgent,
    LexicalAgent,
    LogicAgent,
    PrimaryClassifier,
    Router,
)
from models.state import InputState, PipelineState, RequestState


@dataclass(slots=True)
class PipelineOrchestrator:
    """Coordinates all pipeline components using a shared state object."""

    primary_classifier: PrimaryClassifier
    router: Router
    lexical_agent: LexicalAgent
    contextual_agent: ContextualAgent
    logic_agent: LogicAgent
    consensus_agent: ConsensusAgent
    explainability_agent: ExplainabilityAgent

    @classmethod
    def default(cls) -> "PipelineOrchestrator":
        """Create orchestrator with default concrete component instances."""

        return cls(
            primary_classifier=PrimaryClassifier(),
            router=Router(),
            lexical_agent=LexicalAgent(),
            contextual_agent=ContextualAgent(),
            logic_agent=LogicAgent(),
            consensus_agent=ConsensusAgent(),
            explainability_agent=ExplainabilityAgent(),
        )

    def run(
        self,
        text: str,
        request_id: Optional[str] = None,
        language: Optional[str] = None,
    ) -> PipelineState:
        """Run one full pipeline pass and return the final state.

        The call sequence is intentionally fixed here as a starter flow:
        primary -> router -> lexical/contextual/logic -> consensus -> explainability
        """

        state = PipelineState(
            request=RequestState(request_id=request_id, language=language),
            input=InputState(input_text=text),
        )

        state = self.primary_classifier.run(state)
        state = self.router.run(state)
        state = self.lexical_agent.run(state)
        state = self.contextual_agent.run(state)
        state = self.logic_agent.run(state)
        state = self.consensus_agent.run(state)
        state = self.explainability_agent.run(state)

        return state
