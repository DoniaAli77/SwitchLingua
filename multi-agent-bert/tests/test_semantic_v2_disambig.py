"""Tests for the semantic_v2_disambig prompt variant (general, non-dataset-specific).

Verifies: the variant resolves; all four agent prompts (Lexical/Polarity/Contextual/Intent)
switch to their disambig text under it; the disambig text encodes the RELATIONSHIP rule
(report/endorse/attack) and description-vs-evaluation; default and other variants are
unchanged; and no dataset/benchmark names leak into any prompt.
"""

from __future__ import annotations

import pytest

from src.prompts._sentiment_variant import active_variant, VALID_VARIANTS
from src.prompts import (
    llm_lexical_prompt as LEX,
    polarity_prompt as POL,
    contextual_prompt as CTX,
    intent_prompt as INT,
)

VAR = "semantic_v2_disambig"
_BANNED = ["eesa", "arensa", "ahmed", "twitter", "arsentd", "egyptian", "arabizi"]


def test_variant_registered_and_resolves():
    assert VAR in VALID_VARIANTS
    assert active_variant(VAR) == VAR


@pytest.mark.parametrize("mod", [LEX, POL, CTX, INT])
def test_each_agent_switches_to_disambig(mod, monkeypatch):
    # All four consult the active variant; the gate (INT) reads it from the env when its
    # own system_variant is None, so drive all four uniformly via the env var.
    monkeypatch.setenv("SENTIMENT_PROMPT_VARIANT", VAR)
    disambig = mod.get_system_prompt()
    assert disambig == mod.SYSTEM_PROMPT_DISAMBIG
    monkeypatch.setenv("SENTIMENT_PROMPT_VARIANT", "default")
    assert mod.get_system_prompt() != mod.SYSTEM_PROMPT_DISAMBIG


@pytest.mark.parametrize("mod", [LEX, POL, CTX, INT])
def test_disambig_encodes_relationship_rule(mod):
    p = mod.SYSTEM_PROMPT_DISAMBIG.lower()
    # the three-way relationship: report/count, endorse, object/attack
    assert "relationship" in p
    assert "dislike" in p  # platform action named generically (not dataset-specific)
    assert ("endors" in p or "celebrat" in p)
    assert ("object" in p or "attack" in p)


def test_description_vs_evaluation_in_polarity_and_contextual():
    for mod in (POL, CTX):
        p = mod.SYSTEM_PROMPT_DISAMBIG.lower()
        assert "described" in p or "description" in p or "describing" in p


@pytest.mark.parametrize("mod", [LEX, POL, CTX, INT])
def test_no_dataset_names_leak(mod):
    p = mod.SYSTEM_PROMPT_DISAMBIG.lower()
    assert not any(t in p for t in _BANNED)
    assert "json" in p


@pytest.mark.parametrize("mod", [LEX, POL, CTX])
def test_other_variants_unchanged(mod):
    # default must be unaffected by adding the new variant
    assert mod.get_system_prompt("default") == mod.SYSTEM_PROMPT
    # semantic_v1 only exists for Lexical and Contextual
    if hasattr(mod, "SYSTEM_PROMPT_SEMANTIC_V1"):
        assert mod.get_system_prompt("semantic_v1") == mod.SYSTEM_PROMPT_SEMANTIC_V1


def test_intent_default_and_selective_unchanged():
    assert INT.get_system_prompt(None) == INT.SYSTEM_PROMPT  # default (no env var set)
    assert INT.get_system_prompt("selective") == INT.SYSTEM_PROMPT_SELECTIVE


def test_intent_disambig_via_env(monkeypatch):
    # the gate reads the SENTIMENT_PROMPT_VARIANT env when system_variant is None
    monkeypatch.setenv("SENTIMENT_PROMPT_VARIANT", VAR)
    assert INT.get_system_prompt(None) == INT.SYSTEM_PROMPT_DISAMBIG
    # selective still wins over the env
    assert INT.get_system_prompt("selective") == INT.SYSTEM_PROMPT_SELECTIVE


def test_disambig_drops_platform_neutral_shortcut_in_lexical():
    # the lossy "treat platform words as WEAK cues" shortcut should NOT be the framing;
    # the relationship rule replaces it (built from default base, not semantic_v1)
    p = LEX.SYSTEM_PROMPT_DISAMBIG.lower()
    assert "not neutral by default" in p
