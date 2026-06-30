"""Sentiment prompt-variant resolver.

A single, clean gate that selects which system-prompt variant the three
sentiment specialist agents (lexical / logic / contextual) use.

The variant is read from the ``SENTIMENT_PROMPT_VARIANT`` environment variable
(or passed explicitly). It defaults to ``"default"``, so **unless the variant is
explicitly enabled, agent behaviour is unchanged** — the default system prompts
are byte-identical to the originals.

Enabling the experimental variant
----------------------------------
Set the environment variable before running the pipeline, e.g.::

    SENTIMENT_PROMPT_VARIANT=semantic_v1 python evaluate_pipeline.py ...

or pass ``--sentiment_prompt_variant semantic_v1`` to ``evaluate_pipeline.py``
(which sets the environment variable for you).

This module deliberately does **not** touch architecture, consensus, or routing.
It only decides which system-prompt string each sentiment agent sends.
"""

from __future__ import annotations

import os

#: Environment variable that selects the active sentiment prompt variant.
ENV_VAR = "SENTIMENT_PROMPT_VARIANT"

#: The default (legacy) variant — preserves original agent behaviour exactly.
DEFAULT_VARIANT = "default"

#: All recognised variant names.
VALID_VARIANTS = frozenset({"default", "semantic_v1"})


def active_variant(explicit: str | None = None) -> str:
    """Resolve the active sentiment prompt variant.

    Resolution order:
    1. ``explicit`` argument, if provided (non-empty).
    2. The ``SENTIMENT_PROMPT_VARIANT`` environment variable.
    3. ``"default"``.

    Raises ``ValueError`` for an unrecognised variant name so misconfiguration
    fails loudly rather than silently falling back to default behaviour.
    """
    raw = explicit if explicit else os.environ.get(ENV_VAR, DEFAULT_VARIANT)
    variant = (raw or DEFAULT_VARIANT).strip().lower()
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"Unknown sentiment_prompt_variant {variant!r}. "
            f"Valid options: {sorted(VALID_VARIANTS)}."
        )
    return variant
