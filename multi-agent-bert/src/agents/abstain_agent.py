"""A specialist agent that always abstains, casting no vote.

Used by the two-agent sentiment variant B (Polarity + Contextual) to *remove*
the Lexical specialist without changing the orchestrator's fixed stage list: the
lexical stage runs this agent, which writes an abstaining ``AgentOutput``
(``label=None``) to its slot and makes **no LLM call**. ``ConsensusAgent`` already
excludes ``None``-label votes, so the effect is a clean drop of that agent's vote
with zero extra cost.

This is task-generic — no labels or dataset assumptions — and touches nothing
else (router, consensus, primary model all unchanged).
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agents._abstain import abstain_output
from src.agents.base_agent import BaseAgent
from src.state.schema import PipelineState

_ABSTAIN_NOTE = "AbstainAgent: agent disabled in this variant — no vote cast."


class AbstainAgent(BaseAgent[PipelineState]):
    """Writes an abstaining output to ``output_attr`` and makes no LLM call.

    Parameters
    ----------
    output_attr:
        State slot to write the abstain to (e.g. ``"lexical_output"``).
    name:
        Optional agent name (defaults to ``"AbstainAgent"``).
    """

    def __init__(
        self,
        output_attr: str,
        name: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(name=name or "AbstainAgent", logger=logger)
        self._output_attr = output_attr

    def run(self, state: PipelineState) -> PipelineState:
        setattr(state, self._output_attr, abstain_output(self.name, state, _ABSTAIN_NOTE))
        state.append_history(
            component=self.name,
            summary="Abstained (agent disabled in this variant).",
            outputs={"abstained": True},
        )
        return state
