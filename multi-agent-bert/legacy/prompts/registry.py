"""Prompt retrieval helpers."""

from __future__ import annotations

from typing import Dict

from prompts.templates import (
    CONSENSUS_AGENT_PROMPT,
    CONTEXTUAL_AGENT_PROMPT,
    EXPLAINABILITY_AGENT_PROMPT,
    LEXICAL_AGENT_PROMPT,
    LOGIC_AGENT_PROMPT,
    PRIMARY_CLASSIFIER_PROMPT,
    ROUTER_PROMPT,
)


def get_prompt_registry() -> Dict[str, str]:
    """Return prompt templates indexed by component name."""

    return {
        "primary_classifier": PRIMARY_CLASSIFIER_PROMPT,
        "router": ROUTER_PROMPT,
        "lexical_agent": LEXICAL_AGENT_PROMPT,
        "contextual_agent": CONTEXTUAL_AGENT_PROMPT,
        "logic_agent": LOGIC_AGENT_PROMPT,
        "consensus_agent": CONSENSUS_AGENT_PROMPT,
        "explainability_agent": EXPLAINABILITY_AGENT_PROMPT,
    }
