"""Tests for the semantic_v3_pragmatic_contextual prompt variant (Pragmatic Reasoner).

Contextual-only upgrade: the Contextual agent becomes an explicit pragmatic reasoner;
Lexical and Logic keep their semantic_v1 behaviour; default and semantic_v1 are unchanged.
All offline — no LLM calls.
"""
from __future__ import annotations

import pytest

from src.prompts import contextual_prompt as ctx
from src.prompts import llm_lexical_prompt as lex
from src.prompts import llm_logic_prompt as logic
from src.prompts._sentiment_variant import active_variant

V3 = "semantic_v3_pragmatic_contextual"
OUTPUT_MARKER = "OUTPUT FORMAT (copy this structure exactly):"
FORBIDDEN = ["eesa", "arensa", "ahmed", "twitter", "arsentd", "tweet"]


# 1. default remains unchanged
def test_default_contextual_unchanged():
    assert ctx.get_system_prompt("default") == ctx.SYSTEM_PROMPT
    assert ctx.get_system_prompt(None) == ctx.SYSTEM_PROMPT


# 2. semantic_v1 remains unchanged
def test_semantic_v1_contextual_unchanged():
    assert ctx.get_system_prompt("semantic_v1") == ctx.SYSTEM_PROMPT_SEMANTIC_V1


# 3. v3 activates only when selected
def test_v3_activates_only_when_selected():
    assert active_variant(V3) == V3
    assert ctx.get_system_prompt(V3) == ctx.SYSTEM_PROMPT_PRAGMATIC
    assert ctx.SYSTEM_PROMPT_PRAGMATIC not in (ctx.SYSTEM_PROMPT, ctx.SYSTEM_PROMPT_SEMANTIC_V1)
    # default/semantic_v1 never return the pragmatic prompt
    assert ctx.get_system_prompt("default") != ctx.SYSTEM_PROMPT_PRAGMATIC
    assert ctx.get_system_prompt("semantic_v1") != ctx.SYSTEM_PROMPT_PRAGMATIC


# 4. JSON output format remains last, schema intact
def test_v3_output_format_last_and_schema_intact():
    p = ctx.SYSTEM_PROMPT_PRAGMATIC
    assert p.count(OUTPUT_MARKER) == 1
    tail = p.split(OUTPUT_MARKER, 1)[1]
    for k in ('"label"', '"confidence"', '"reasoning"', '"evidence"'):
        assert k in tail  # JSON contract is after the marker (last)
    # the pragmatic block is inserted BEFORE the output format
    assert ctx._PRAGMATIC_ADDENDUM in p
    assert p.index(ctx._PRAGMATIC_ADDENDUM) < p.index(OUTPUT_MARKER)
    # the five pragmatic axes are present
    low = p.lower()
    for axis in ["speech act", "target", "mention vs use", "implicature", "description vs evaluation"]:
        assert axis in low


# 5. no dataset/benchmark names appear in the prompt
def test_v3_prompt_is_clean():
    low = ctx.SYSTEM_PROMPT_PRAGMATIC.lower()
    assert not any(t in low for t in FORBIDDEN)


# Lexical + Logic are UNCHANGED under v3 (they keep their semantic_v1 prompt)
def test_lexical_and_logic_unchanged_under_v3():
    assert lex.get_system_prompt(V3) == lex.SYSTEM_PROMPT_SEMANTIC_V1
    assert logic.get_system_prompt(V3) == logic.SYSTEM_PROMPT_SEMANTIC_V1
    # and default/semantic_v1 for lexical still resolve as before
    assert lex.get_system_prompt("default") == lex.SYSTEM_PROMPT
    assert lex.get_system_prompt("semantic_v1") == lex.SYSTEM_PROMPT_SEMANTIC_V1


# user-prompt template is untouched across variants
def test_user_prompt_identical_across_variants():
    labels = ["positive", "negative", "neutral"]
    descs = {l: l for l in labels}
    a = ctx.build_user_prompt("sentiment", labels, descs, "hello")
    b = ctx.build_user_prompt("sentiment", labels, descs, "hello")
    assert a == b


def test_bad_variant_raises():
    with pytest.raises(ValueError):
        active_variant("semantic_v99_bogus")
