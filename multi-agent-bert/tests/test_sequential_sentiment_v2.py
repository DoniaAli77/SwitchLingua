"""Tests for sequential_sentiment_v2 (forward-pragmatics pipeline), all offline.

Covers: wiring & opt-in; Stage-2 emits NO sentiment label and names no dataset;
the feature-aware polarity stage sees the pragmatic JSON (not a prior label);
controller rules (no-opinion cross-checked gate / feature-aware trust / fallback);
malformed-JSON coercion; and an end-to-end escalated run.
"""

from __future__ import annotations

import json

import pytest

from evaluate_pipeline import build_orchestrator
from src.agents._sentiment_agent_variant import active_agent_variant
from src.agents.sequential_sentiment import SEQ_KEY
from src.agents.sequential_sentiment_v2 import (
    SeqV2IntentAgent,
    SeqV2PragmaticFeaturesAgent,
    SeqV2PolarityResolverAgent,
    SequentialControllerV2,
)
from src.llm.mock_client import MockLLMClient
from src.prompts import sequential_sentiment_v2_prompts as P2
from src.state.schema import ModelOutput, PipelineState, StateMetadata, TaskConfig

LABELS = ["positive", "negative", "neutral"]
_BANNED = ["eesa", "arensa", "ahmed", "twitter", "arsentd", "tweet"]


def _tc():
    return TaskConfig(task_name="sentiment_classification", labels=list(LABELS),
                      label_descriptions={l: l for l in LABELS})


def _state(text="the movie was great", primary=("neutral", 0.5)):
    st = PipelineState(metadata=StateMetadata(sample_id="t"), input_text=text, task_config=_tc())
    st.primary_model_output = ModelOutput(label=primary[0], confidence=primary[1])
    return st


def _seed(st, intent=None, pragmatic=None, polarity=None):
    store = {"stage_events": []}
    if intent is not None:
        store["intent"] = intent
    if pragmatic is not None:
        store["pragmatic"] = pragmatic
    if polarity is not None:
        store["polarity"] = polarity
    st.extras[SEQ_KEY] = store
    return st


def _features(**kw):
    base = {"speech_act": "evaluate", "target": "movie", "target_attribution": "author",
            "use_vs_mention": "use", "platform_meta": False,
            "description_vs_evaluation": "evaluation", "sarcasm_or_irony": False,
            "implicit_stance": "none", "stance_strength": "moderate",
            "confidence": 0.8, "evidence": ["great"]}
    base.update(kw)
    return base


# ------------------------------------------------------------------- wiring
def test_v2_wires_three_stages_and_controller():
    o = build_orchestrator(_tc(), threshold=0.7, enable_deliberation=False,
                           sentiment_agent_variant="sequential_sentiment_v2")
    assert isinstance(o._sequential_controller, SequentialControllerV2)
    names = [n for n, _ in o._sequential_stages]
    assert names == ["seq_intent_stage", "seq_pragmatic_features_stage", "seq_polarity_resolver_stage"]
    types = [type(a) for _, a in o._sequential_stages]
    assert types == [SeqV2IntentAgent, SeqV2PragmaticFeaturesAgent, SeqV2PolarityResolverAgent]


def test_v2_variant_resolves_and_default_unchanged():
    assert active_agent_variant("sequential_sentiment_v2") == "sequential_sentiment_v2"
    o = build_orchestrator(_tc(), threshold=0.7, enable_deliberation=False)
    assert o._sequential_controller is None  # default untouched


# ------------------------------------------------------------------- prompts
@pytest.mark.parametrize("sp", [
    P2.INTENT_V2_SYSTEM_PROMPT, P2.PRAGMATIC_FEATURES_SYSTEM_PROMPT, P2.POLARITY_RESOLVER_SYSTEM_PROMPT,
])
def test_v2_prompts_clean(sp):
    low = sp.lower()
    assert not any(t in low for t in _BANNED)
    assert "json" in low


def test_stage2_prompt_forbids_a_label_and_declares_features():
    sp = P2.PRAGMATIC_FEATURES_SYSTEM_PROMPT
    assert "do not output a sentiment label" in sp.lower()
    for k in ('"sarcasm_or_irony"', '"implicit_stance"', '"use_vs_mention"', '"stance_strength"'):
        assert k in sp


def test_stage3_resolver_is_not_a_reviewer():
    sp = P2.POLARITY_RESOLVER_SYSTEM_PROMPT.lower()
    # explicitly not reviewing/ratifying a prior label
    assert "not reviewing" in sp or "not review" in sp
    assert "pragmatic features" in sp


def test_stage3_user_prompt_embeds_features_not_a_prior_label():
    up = P2.build_polarity_resolver_user_prompt(
        "sentiment", LABELS, {l: l for l in LABELS}, "txt",
        {"opinion_expressed": True}, _features(sarcasm_or_irony=True))
    assert "Pragmatic features (JSON)" in up
    assert "sarcasm_or_irony" in up
    # no proposed polarity label field is fed in
    assert "keep_or_revise" not in up


# ------------------------------------------------------------------- stage agents
def test_stage2_features_parse_and_persist_no_label():
    st = _seed(_state(), intent={"opinion_expressed": True})
    SeqV2PragmaticFeaturesAgent(MockLLMClient(mode="fixed", fixed_response=json.dumps(_features()))).run(st)
    rec = st.extras[SEQ_KEY]["pragmatic"]
    assert "label" not in rec
    assert rec["use_vs_mention"] == "use" and rec["implicit_stance"] == "none"


def test_stage3_resolver_validates_label():
    st = _seed(_state(), intent={"opinion_expressed": True}, pragmatic=_features())
    resp = json.dumps({"label": "positive", "confidence": 0.8, "used_features": ["stance_strength"],
                       "reasoning": "praise", "evidence": ["great"]})
    SeqV2PolarityResolverAgent(MockLLMClient(mode="fixed", fixed_response=resp)).run(st)
    assert st.extras[SEQ_KEY]["polarity"]["label"] == "positive"


def test_stage2_malformed_coerces_safe_default_that_does_not_force_neutral():
    st = _seed(_state(), intent={"opinion_expressed": True})
    SeqV2PragmaticFeaturesAgent(MockLLMClient(mode="label_echo", allowed_labels=LABELS)).run(st)
    rec = st.extras[SEQ_KEY]["pragmatic"]
    assert rec["coerced"] is True
    # safe default must NOT satisfy the neutral gate (use + mixed)
    assert rec["use_vs_mention"] == "use" and rec["description_vs_evaluation"] == "mixed"


# ------------------------------------------------------------------- controller
def _run(st, **kw):
    return SequentialControllerV2(**kw).run(st).final_output.label


def test_v2_rule1_no_opinion_cross_checked_gate_neutral():
    st = _seed(_state(),
               intent={"opinion_expressed": False, "confidence": 0.9},
               pragmatic=_features(use_vs_mention="platform_meta", implicit_stance="none",
                                   description_vs_evaluation="description"),
               polarity={"label": "positive", "confidence": 0.8})
    assert _run(st) == "neutral"
    assert st.extras[SEQ_KEY]["decided_by"] == "intent_no_opinion"


def test_v2_gate_does_not_fire_when_pragmatic_sees_implicit_stance():
    # intent says no-opinion but features find an implicit stance → gate must NOT fire
    st = _seed(_state(),
               intent={"opinion_expressed": False, "confidence": 0.9},
               pragmatic=_features(use_vs_mention="use", implicit_stance="negative"),
               polarity={"label": "negative", "confidence": 0.8})
    assert _run(st) == "negative"
    assert st.extras[SEQ_KEY]["decided_by"] == "polarity_feature_aware"


def test_v2_rule2_trusts_feature_aware_polarity():
    st = _seed(_state(),
               intent={"opinion_expressed": True, "confidence": 0.9},
               pragmatic=_features(sarcasm_or_irony=True),
               polarity={"label": "negative", "confidence": 0.75})
    assert _run(st) == "negative"
    assert st.extras[SEQ_KEY]["decided_by"] == "polarity_feature_aware"


def test_v2_rule3_weak_polarity_falls_back_to_primary():
    st = _seed(_state(primary=("neutral", 0.55)),
               intent={"opinion_expressed": True, "confidence": 0.5},
               pragmatic=_features(),
               polarity={"label": "positive", "confidence": 0.30})  # < tau_low
    assert _run(st) == "neutral"
    assert st.extras[SEQ_KEY]["decided_by"] == "fallback_primary"


def test_v2_pure_sequential_ablation_keeps_polarity_on_weak():
    st = _seed(_state(primary=("negative", 0.9)),
               intent={"opinion_expressed": True, "confidence": 0.5},
               pragmatic=_features(),
               polarity={"label": "positive", "confidence": 0.30})
    assert _run(st, use_primary_fallback=False) == "positive"
    assert st.extras[SEQ_KEY]["decided_by"] == "fallback_polarity"


def test_v2_no_keep_revise_rule_exists():
    # v2 controller never emits a pragmatic_revision decision
    st = _seed(_state(),
               intent={"opinion_expressed": True, "confidence": 0.9},
               pragmatic=_features(), polarity={"label": "positive", "confidence": 0.9})
    SequentialControllerV2().run(st)
    assert st.extras[SEQ_KEY]["decided_by"] in (
        "intent_no_opinion", "polarity_feature_aware", "fallback_primary", "fallback_polarity")


# ------------------------------------------------------------------- integration
class _AlwaysEscalatePrimary:
    def run(self, state):
        state.primary_model_output = ModelOutput(
            label="neutral", confidence=0.10,
            probabilities={"positive": 0.3, "negative": 0.3, "neutral": 0.4},
            raw_text=state.input_text)
        return state


def test_v2_end_to_end_escalated_valid_label_and_trace():
    o = build_orchestrator(_tc(), threshold=0.7, enable_deliberation=False,
                           primary_classifier=_AlwaysEscalatePrimary(),
                           sentiment_agent_variant="sequential_sentiment_v2")
    out = o.run(_state())
    assert out.routing_info.decision == "escalate"
    assert out.final_output.label in LABELS
    store = out.extras[SEQ_KEY]
    assert {"intent", "pragmatic", "polarity", "decided_by", "final_label"} <= store.keys()
    assert out.final_output.payload["source"] == "sequential_sentiment_v2"
