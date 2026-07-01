"""Tests for the four sentiment agent-architecture designs (A/B/C/D).

A = default                          → Lexical + Logic + Contextual (unchanged)
B = polarity_contextual              → Polarity + Contextual (Lexical abstains)
C = lexical_polarity_contextual      → Lexical + Polarity + Contextual
D = lexical_logic_contextual_polarity→ Lexical + Logic + Contextual + Polarity (4 votes)

All offline — no OpenAI, no training. Verifies wiring, the default-off polarity
consensus slot, the 4th-vote effect, B's abstaining Lexical, and that bad variant
names fail loudly.
"""

from __future__ import annotations

import pytest

from evaluate_pipeline import build_orchestrator, _build_consensus
from src.agents.abstain_agent import AbstainAgent
from src.agents.consensus_agent import ConsensusAgent
from src.agents.llm_lexical_agent import LLMLexicalAgent
from src.agents.llm_logic_agent import LLMLogicAgent
from src.agents.polarity_agent import PolarityAgent
from src.agents.intent_agent import IntentAgent
from src.prompts import intent_prompt

INTENTGATE = "lexical_polarity_contextual_intent_gate"
from src.agents._sentiment_agent_variant import active_agent_variant
from src.state.schema import (
    AgentOutput,
    ModelOutput,
    PipelineState,
    StateMetadata,
    TaskConfig,
)

LABELS = ["positive", "negative", "neutral"]


def _tc():
    return TaskConfig(
        task_name="sentiment_classification",
        labels=list(LABELS),
        label_descriptions={l: l for l in LABELS},
    )


def _orch(variant):
    return build_orchestrator(
        _tc(), threshold=0.7, enable_deliberation=False,
        consensus_primary_weight=1.0, sentiment_agent_variant=variant,
    )


def _vote(name, label, conf):
    return AgentOutput(agent_name=name, model_output=ModelOutput(label=label, confidence=conf))


# --------------------------------------------------------------------------- A
def test_default_is_lexical_logic_contextual():
    o = _orch("default")
    assert isinstance(o._llm_lexical, LLMLexicalAgent)
    assert isinstance(o._llm_logic, LLMLogicAgent)
    assert o._polarity is None
    assert o._consensus.weights["polarity"] == 0.0


# --------------------------------------------------------------------------- B
def test_B_polarity_contextual_lexical_abstains():
    o = _orch("polarity_contextual")
    assert isinstance(o._llm_lexical, AbstainAgent)   # Lexical removed
    assert isinstance(o._llm_logic, PolarityAgent)    # Polarity in logic slot
    assert o._polarity is None
    assert o._consensus.weights["polarity"] == 0.0    # no 4th vote


def test_B_abstain_agent_casts_no_vote():
    st = PipelineState(metadata=StateMetadata(sample_id="t"), input_text="hi", task_config=_tc())
    AbstainAgent(output_attr="lexical_output").run(st)
    assert st.lexical_output is not None
    assert st.lexical_output.model_output.label is None  # excluded by consensus


# --------------------------------------------------------------------------- C
def test_C_lexical_polarity_contextual():
    o = _orch("lexical_polarity_contextual")
    assert isinstance(o._llm_lexical, LLMLexicalAgent)
    assert isinstance(o._llm_logic, PolarityAgent)    # Logic replaced by Polarity
    assert o._polarity is None
    assert o._consensus.weights["polarity"] == 0.0


# --------------------------------------------------------------------------- D
def test_D_four_agent_wiring():
    o = _orch("lexical_logic_contextual_polarity")
    assert isinstance(o._llm_lexical, LLMLexicalAgent)
    assert isinstance(o._llm_logic, LLMLogicAgent)    # Logic kept
    assert isinstance(o._polarity, PolarityAgent)     # Polarity added as 4th
    assert o._polarity._output_attr == "polarity_output"
    assert o._consensus.weights["polarity"] == 1.0    # 4th vote active


def test_D_polarity_vote_changes_consensus():
    """The 4th (polarity) vote flips the outcome only when its weight is on."""
    def run(weights):
        st = PipelineState(metadata=StateMetadata(sample_id="t"), input_text="x", task_config=_tc())
        st.lexical_output = _vote("lex", "neutral", 0.8)
        st.logic_output = _vote("log", "neutral", 0.8)
        st.contextual_output = _vote("ctx", "positive", 0.8)
        st.polarity_output = _vote("pol", "positive", 0.9)
        return ConsensusAgent(weights=weights).run(st).final_output.label

    # primary off to isolate the agents; polarity OFF (default) → neutral wins (1.6 vs 0.8)
    assert run({"primary": 0.0, "polarity": 0.0}) == "neutral"
    # polarity ON (weight 1.0) → positive 0.8+0.9=1.7 beats neutral 1.6
    assert run({"primary": 0.0, "polarity": 1.0}) == "positive"


# --------------------------------------------------------------------------- E
def test_E_lexical_intent_polarity_contextual_wiring():
    o = _orch("lexical_intent_polarity_contextual")
    assert isinstance(o._llm_lexical, LLMLexicalAgent)
    assert isinstance(o._llm_logic, PolarityAgent)      # Polarity in logic slot
    assert isinstance(o._polarity, IntentAgent)         # Intent is the 4th agent
    assert o._polarity._output_attr == "polarity_output"
    assert o._consensus.weights["polarity"] == 1.0      # 4th vote active


def test_E_intent_prompt_is_clean_and_schema_preserving():
    sp = intent_prompt.SYSTEM_PROMPT
    low = sp.lower()
    assert not any(t in low for t in ["eesa", "arensa", "ahmed", "twitter", "arsentd", "tweet"])
    assert all(k in sp for k in ['"label"', '"confidence"', '"reasoning"', '"evidence"'])


# --------------------------------------------------------------------------- F
def test_F_intent_polarity_contextual_wiring():
    o = _orch("intent_polarity_contextual")
    assert isinstance(o._llm_lexical, AbstainAgent)     # Lexical removed (abstains)
    assert isinstance(o._llm_logic, PolarityAgent)      # Polarity in logic slot
    assert isinstance(o._polarity, IntentAgent)         # Intent is the 4th agent
    assert o._polarity._output_attr == "polarity_output"
    assert o._consensus.weights["polarity"] == 1.0      # 3 active votes: intent+polarity+contextual


# --------------------------------------------------------------------------- G
def test_G_intent_gate_wiring():
    o = _orch(INTENTGATE)
    assert isinstance(o._llm_lexical, LLMLexicalAgent)   # Lexical KEPT
    assert isinstance(o._llm_logic, PolarityAgent)       # Polarity in logic slot
    assert isinstance(o._polarity, IntentAgent)          # IntentGate agent
    assert o._consensus.weights["polarity"] == 0.0       # gate does NOT vote
    assert o._consensus._intent_gate is True             # guard active


def _gate_state(primary, agents_label, gate_label):
    st = PipelineState(metadata=StateMetadata(sample_id="t"), input_text="x", task_config=_tc())
    st.primary_model_output = ModelOutput(label=primary, confidence=0.5)
    st.lexical_output = _vote("lex", agents_label, 0.9)
    st.logic_output = _vote("pol", agents_label, 0.9)
    st.contextual_output = _vote("ctx", agents_label, 0.9)
    if gate_label is not None:
        st.polarity_output = _vote("IntentGate", gate_label, 0.8)
    return st


def test_G_gate_blocks_polar_override_of_neutral_primary():
    # primary neutral, agents override to positive, gate says neutral (no opinion) → blocked
    st = _gate_state("neutral", "positive", "neutral")
    out = ConsensusAgent(weights={"primary": 1.0}, intent_gate=True).run(st).final_output
    assert out.label == "neutral"  # override blocked, neutral protected


def test_G_gate_allows_override_when_opinion_expressed():
    # gate says positive (clear opinion) → does not block; override to positive stands
    st = _gate_state("neutral", "positive", "positive")
    out = ConsensusAgent(weights={"primary": 1.0}, intent_gate=True).run(st).final_output
    assert out.label == "positive"


def test_G_gate_off_by_default_reproduces_override():
    # same inputs, gate OFF → agents override the neutral primary to positive
    st = _gate_state("neutral", "positive", "neutral")
    out = ConsensusAgent(weights={"primary": 1.0}).run(st).final_output
    assert out.label == "positive"  # no guard → C-style override


def test_G_gate_does_not_touch_non_override():
    # consensus already agrees with primary → guard is a no-op
    st = _gate_state("neutral", "neutral", "neutral")
    out = ConsensusAgent(weights={"primary": 1.0}, intent_gate=True).run(st).final_output
    assert out.label == "neutral"


def test_default_consensus_has_no_gate():
    assert ConsensusAgent()._intent_gate is False


# --------------------------------------------------------------------------- G2
SELECTIVE = "lexical_polarity_contextual_selective_gate"


def test_G2_selective_gate_wiring():
    o = _orch(SELECTIVE)
    assert isinstance(o._llm_lexical, LLMLexicalAgent)   # Lexical kept (same as G)
    assert isinstance(o._llm_logic, PolarityAgent)
    assert isinstance(o._polarity, IntentAgent)
    assert o._polarity._system_variant == "selective"    # SELECTIVE gate prompt
    assert o._consensus.weights["polarity"] == 0.0        # still non-voting
    assert o._consensus._intent_gate is True


def test_G2_selective_prompt_differs_and_is_clean():
    default = intent_prompt.get_system_prompt(None)
    sel = intent_prompt.get_system_prompt("selective")
    assert sel != default
    assert sel == intent_prompt.SYSTEM_PROMPT_SELECTIVE
    low = sel.lower()
    assert not any(t in low for t in ["eesa", "arensa", "ahmed", "twitter", "arsentd", "tweet"])
    assert all(k in sel for k in ['"label"', '"confidence"', '"reasoning"', '"evidence"'])
    # the selective gate must mention meta/mention protection AND implicit-stance carve-outs
    assert "meta" in low and "mention" in low
    assert "implicit" in low and ("insult" in low or "praise" in low)


def test_G_still_uses_default_intent_prompt():
    # G unchanged: its IntentGate uses the default (non-selective) intent prompt
    o = _orch(INTENTGATE)
    assert o._polarity._system_variant is None


def test_default_state_has_polarity_slot_unused():
    """polarity_output exists and defaults None so default consensus is unaffected."""
    st = PipelineState(metadata=StateMetadata(sample_id="t"), input_text="x", task_config=_tc())
    assert st.polarity_output is None
    assert _build_consensus(None, 0.0).weights["polarity"] == 0.0


# ------------------------------------------------------------------- resolver
@pytest.mark.parametrize("v", [
    "default", "polarity_contextual",
    "lexical_polarity_contextual", "lexical_logic_contextual_polarity",
    "lexical_intent_polarity_contextual", "intent_polarity_contextual",
    "lexical_polarity_contextual_intent_gate",
    "lexical_polarity_contextual_selective_gate",
])
def test_valid_variants_resolve(v):
    assert active_agent_variant(v) == v


def test_bad_variant_raises():
    with pytest.raises(ValueError):
        active_agent_variant("not_a_variant")
    with pytest.raises(ValueError):
        build_orchestrator(_tc(), 0.7, False, sentiment_agent_variant="bogus")
