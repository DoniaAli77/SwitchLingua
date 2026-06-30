"""Sentiment agent-architecture variant resolver.

Selects which specialist trio the full_agentic sentiment pipeline uses:

- ``"default"`` → Lexical + Logic + Contextual (unchanged shipped behaviour).
- ``"lexical_polarity_contextual"`` → Lexical + **Polarity** + Contextual, i.e. the
  Logic agent is replaced by a sentiment :class:`~src.agents.polarity_agent.PolarityAgent`.

The variant is read from the ``SENTIMENT_AGENT_VARIANT`` environment variable (or
passed explicitly). It defaults to ``"default"``, so **unless explicitly enabled the
agent list is unchanged**. This is independent of ``SENTIMENT_PROMPT_VARIANT`` (the
prompt wording for Lexical/Contextual); the two flags can be combined.

This module does **not** touch the router, consensus, or primary model. The Polarity
agent writes the same ``state.logic_output`` slot as the Logic agent, so the swap is a
drop-in at the orchestrator's logic stage and requires no consensus change.
"""

from __future__ import annotations

import os

#: Environment variable that selects the active sentiment agent variant.
ENV_VAR = "SENTIMENT_AGENT_VARIANT"

#: The default (legacy) variant — preserves the original Lexical+Logic+Contextual trio.
DEFAULT_VARIANT = "default"

#: All recognised variant names.
#: - ``default``                       → Lexical + Logic + Contextual (A)
#: - ``polarity_contextual``           → Polarity + Contextual (B, 2 agents)
#: - ``lexical_polarity_contextual``   → Lexical + Polarity + Contextual (C)
#: - ``lexical_logic_contextual_polarity`` → Lexical + Logic + Contextual + Polarity (D, 4 agents)
#: - ``lexical_intent_polarity_contextual`` → Lexical + Intent + Polarity + Contextual (E, 4 agents)
VALID_VARIANTS = frozenset({
    "default",
    "polarity_contextual",
    "lexical_polarity_contextual",
    "lexical_logic_contextual_polarity",
    "lexical_intent_polarity_contextual",
})


def active_agent_variant(explicit: str | None = None) -> str:
    """Resolve the active sentiment agent variant.

    Resolution order:
    1. ``explicit`` argument, if provided (non-empty).
    2. The ``SENTIMENT_AGENT_VARIANT`` environment variable.
    3. ``"default"``.

    Raises ``ValueError`` for an unrecognised variant name so misconfiguration
    fails loudly rather than silently falling back to default behaviour.
    """
    raw = explicit if explicit else os.environ.get(ENV_VAR, DEFAULT_VARIANT)
    variant = (raw or DEFAULT_VARIANT).strip().lower()
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"Unknown sentiment_agent_variant {variant!r}. "
            f"Valid options: {sorted(VALID_VARIANTS)}."
        )
    return variant
