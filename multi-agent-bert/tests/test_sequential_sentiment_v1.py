"""Tests for the sequential_sentiment_v1 staged pipeline (opt-in).

Covers, all offline (no OpenAI, no training):
* the default pipeline is unchanged and does NOT build the sequential path;
* the variant activates only when explicitly selected;
* each stage prompt is JSON-only and names no dataset/benchmark;
* per-stage malformed-JSON path retries then coerces a safe default (no crash);
* deterministic-controller rules a)-e);
* the pragmatic escape hatch and the pure-sequential (no primary fallback) ablation;
* an end-to-end escalated run produces a valid label and persists the full trace.
"""

from __future__ import annotations

import json

import pytest

from evaluate_pipeline import build_orchestrator
from src.agents._sentiment_agent_variant import active_agent_variant
from src.agents.sequential_sentiment import (
    SEQ_KEY,
    SeqIntentAgent,
    SeqPolarityAgent,
    SeqPragmaticAgent,
    SequentialController,
)
from src.llm.mock_client import MockLLMClient
from src.prompts import sequential_sentiment_prompts as P
from src.state.schema import ModelOutput, PipelineState, StateMetadata, TaskConfig

LABELS = ["positive", "negative", "neutral"]
_BANNED = ["eesa", "arensa", "ahmed", "twitter", "arsentd", "tweet"]


def _tc():
    return TaskConfig(
        task_name="sentiment_classification",
        labels=list(LABELS),
        label_descriptions={l: l for l in LABELS},
    )


def _state(text="the movie was great", primary=("neutral", 0.5)):
    st = PipelineState(metadata=StateMetadata(sample_id="t"), input_text=text, task_config=_tc())
    st.primary_model_output = ModelOutput(label=primary[0], confidence=primary[1])
    return st


def _seed(st, intent=None, polarity=None, pragmatic=None):
    store = {"stage_events": []}
    if intent is not None:
        store["intent"] = intent
    if polarity is not None:
        store["polarity"] = polarity
    if pragmatic is not None:
        store["pragmatic"] = pragmatic
    st.extras[SEQ_KEY] = store
    return st


# ---------------------------------------------------------------- default/wiring
def test_default_variant_has_no_sequential_path():
    o = build_orchestrator(_tc(), threshold=0.7, enable_deliberation=False)
    assert o._sequential_controller is None
    assert o._sequential_stages is None


def test_sequential_variant_wires_three_stages_and_controller():
    o = build_orchestrator(
        _tc(), threshold=0.7, enable_deliberation=False,
        sentiment_agent_variant="sequential_sentiment_v1",
    )
    assert isinstance(o._sequential_controller, SequentialController)
    assert o._sequential_stages is not None and len(o._sequential_stages) == 3
    names = [n for n, _ in o._sequential_stages]
    assert names == ["seq_intent_stage", "seq_polarity_stage", "seq_pragmatic_stage"]
    types = [type(a) for _, a in o._sequential_stages]
    assert types == [SeqIntentAgent, SeqPolarityAgent, SeqPragmaticAgent]


def test_variant_resolves_and_is_opt_in():
    assert active_agent_variant("sequential_sentiment_v1") == "sequential_sentiment_v1"
    # default resolution never selects it
    assert active_agent_variant("default") == "default"


def test_controller_threshold_config_is_exposed():
    o = build_orchestrator(
        _tc(), threshold=0.7, enable_deliberation=False,
        sentiment_agent_variant="sequential_sentiment_v1",
        sequential_config={"tau_intent": 0.8, "tau_revise": 0.7, "tau_low": 0.3,
                           "use_primary_fallback": False},
    )
    c = o._sequential_controller
    assert (c.tau_intent, c.tau_revise, c.tau_low) == (0.8, 0.7, 0.3)
    assert c.use_primary_fallback is False


# ------------------------------------------------------------------- clean prompts
@pytest.mark.parametrize("sp", [
    P.INTENT_SYSTEM_PROMPT, P.POLARITY_SYSTEM_PROMPT, P.PRAGMATIC_SYSTEM_PROMPT,
])
def test_stage_prompts_name_no_dataset(sp):
    low = sp.lower()
    assert not any(t in low for t in _BANNED)
    assert "json" in low  # JSON-only instruction present


def test_stage_prompts_declare_their_json_keys():
    assert all(k in P.INTENT_SYSTEM_PROMPT for k in
               ['"opinion_expressed"', '"speech_act"', '"use_vs_mention"', '"confidence"'])
    assert all(k in P.POLARITY_SYSTEM_PROMPT for k in ['"label"', '"confidence"', '"mixed"'])
    assert all(k in P.PRAGMATIC_SYSTEM_PROMPT for k in
               ['"keep_or_revise"', '"final_label"', '"confidence"'])


# ------------------------------------------------------------------- stage agents
def _intent_json(**kw):
    base = {"opinion_expressed": True, "target": "movie", "speech_act": "evaluate",
            "use_vs_mention": "use", "confidence": 0.9, "evidence": ["great"]}
    base.update(kw)
    return json.dumps(base)


def test_intent_stage_parses_and_persists():
    st = _state()
    SeqIntentAgent(MockLLMClient(mode="fixed", fixed_response=_intent_json())).run(st)
    rec = st.extras[SEQ_KEY]["intent"]
    assert rec["opinion_expressed"] is True
    assert rec["use_vs_mention"] == "use"
    assert rec["confidence"] == 0.9


def test_polarity_stage_validates_label_and_persists():
    st = _seed(_state(), intent={"opinion_expressed": True})
    resp = json.dumps({"label": "positive", "confidence": 0.8, "mixed": False,
                       "reasoning": "praise", "evidence": ["great"]})
    SeqPolarityAgent(MockLLMClient(mode="fixed", fixed_response=resp)).run(st)
    assert st.extras[SEQ_KEY]["polarity"]["label"] == "positive"


def test_pragmatic_stage_parses_keep_or_revise():
    st = _seed(_state(), intent={"opinion_expressed": True}, polarity={"label": "positive"})
    resp = json.dumps({"keep_or_revise": "revise", "final_label": "negative",
                       "confidence": 0.85, "reasoning": "sarcasm", "evidence": ["yeah right"]})
    SeqPragmaticAgent(MockLLMClient(mode="fixed", fixed_response=resp)).run(st)
    rec = st.extras[SEQ_KEY]["pragmatic"]
    assert rec["keep_or_revise"] == "revise"
    assert rec["final_label"] == "negative"


def test_intent_stage_malformed_retries_then_coerces_safe_default():
    st = _state()
    # label_echo returns the *classifier* schema, missing the intent keys → both
    # attempts fail → safe default coerced, no crash.
    SeqIntentAgent(MockLLMClient(mode="label_echo", allowed_labels=LABELS)).run(st)
    rec = st.extras[SEQ_KEY]["intent"]
    assert rec["coerced"] is True
    assert rec["opinion_expressed"] == "unclear"  # disables the no-opinion branch
    events = [e["event"] for e in st.extras[SEQ_KEY]["stage_events"]]
    assert "retry" in events and "coerced_default" in events


def test_polarity_stage_invalid_label_coerces_zero_confidence_default():
    st = _seed(_state(), intent={"opinion_expressed": True})
    resp = json.dumps({"label": "SUPER_POSITIVE", "confidence": 0.9,
                       "reasoning": "x", "evidence": []})
    SeqPolarityAgent(MockLLMClient(mode="fixed", fixed_response=resp)).run(st)
    rec = st.extras[SEQ_KEY]["polarity"]
    assert rec["coerced"] is True
    assert rec["confidence"] == 0.0
    assert rec["label"] in LABELS


# ------------------------------------------------------------------- controller a)-e)
def _run_controller(st, **kw):
    return SequentialController(**kw).run(st).final_output.label


def test_rule_a_no_opinion_high_conf_neutral():
    st = _seed(
        _state(),
        intent={"opinion_expressed": False, "confidence": 0.9},
        polarity={"label": "positive", "confidence": 0.9},
        pragmatic={"keep_or_revise": "keep", "final_label": "positive", "confidence": 0.5},
    )
    assert _run_controller(st) == "neutral"
    assert st.extras[SEQ_KEY]["decided_by"] == "intent_no_opinion"


def test_rule_b_no_opinion_but_pragmatic_confident_implicit_stance_overrides():
    st = _seed(
        _state(),
        intent={"opinion_expressed": False, "confidence": 0.9},
        polarity={"label": "neutral", "confidence": 0.4},
        pragmatic={"keep_or_revise": "revise", "final_label": "negative", "confidence": 0.9},
    )
    # escape hatch: confident non-neutral revision beats the no-opinion neutral branch
    assert _run_controller(st) == "negative"
    assert st.extras[SEQ_KEY]["decided_by"] == "pragmatic_revision"


def test_rule_c_pragmatic_keep_uses_polarity_label():
    st = _seed(
        _state(),
        intent={"opinion_expressed": True, "confidence": 0.9},
        polarity={"label": "positive", "confidence": 0.8},
        pragmatic={"keep_or_revise": "keep", "final_label": "positive", "confidence": 0.7},
    )
    assert _run_controller(st) == "positive"
    assert st.extras[SEQ_KEY]["decided_by"] == "polarity_kept"


def test_rule_d_pragmatic_confident_revision():
    st = _seed(
        _state(),
        intent={"opinion_expressed": True, "confidence": 0.9},
        polarity={"label": "positive", "confidence": 0.8},
        pragmatic={"keep_or_revise": "revise", "final_label": "negative", "confidence": 0.8},
    )
    assert _run_controller(st) == "negative"
    assert st.extras[SEQ_KEY]["decided_by"] == "pragmatic_revision"


def test_rule_e_weak_revision_falls_back_to_primary_when_polarity_weak():
    st = _seed(
        _state(primary=("neutral", 0.55)),
        intent={"opinion_expressed": True, "confidence": 0.5},
        polarity={"label": "positive", "confidence": 0.30},   # < tau_low
        pragmatic={"keep_or_revise": "revise", "final_label": "negative", "confidence": 0.30},
    )
    assert _run_controller(st) == "neutral"  # primary label
    assert st.extras[SEQ_KEY]["decided_by"] == "fallback_primary"


def test_rule_e_weak_revision_falls_back_to_polarity_when_polarity_strong():
    st = _seed(
        _state(primary=("neutral", 0.55)),
        intent={"opinion_expressed": True, "confidence": 0.5},
        polarity={"label": "positive", "confidence": 0.60},   # >= tau_low
        pragmatic={"keep_or_revise": "revise", "final_label": "negative", "confidence": 0.30},
    )
    assert _run_controller(st) == "positive"  # polarity label, weak revision discarded
    assert st.extras[SEQ_KEY]["decided_by"] == "fallback_polarity"


def test_pure_sequential_ablation_never_uses_primary():
    st = _seed(
        _state(primary=("negative", 0.9)),
        intent={"opinion_expressed": True, "confidence": 0.5},
        polarity={"label": "positive", "confidence": 0.30},
        pragmatic={"keep_or_revise": "revise", "final_label": "negative", "confidence": 0.30},
    )
    # use_primary_fallback False → must not adopt the primary label
    label = _run_controller(st, use_primary_fallback=False)
    assert label == "positive"
    assert st.extras[SEQ_KEY]["decided_by"] == "fallback_polarity"


def test_controller_persists_thresholds_and_decided_by():
    st = _seed(
        _state(),
        intent={"opinion_expressed": True, "confidence": 0.9},
        polarity={"label": "positive", "confidence": 0.8},
        pragmatic={"keep_or_revise": "keep", "final_label": "positive", "confidence": 0.7},
    )
    SequentialController(tau_intent=0.6, tau_revise=0.6, tau_low=0.45).run(st)
    store = st.extras[SEQ_KEY]
    assert store["thresholds"]["tau_intent"] == 0.6
    assert store["final_label"] == "positive"
    assert st.consensus_output.label == "positive"   # written for explainability/eval


# ------------------------------------------------------------------- integration
class _AlwaysEscalatePrimary:
    """Minimal primary stub: low-confidence prediction so the router escalates."""

    def run(self, state):
        state.primary_model_output = ModelOutput(
            label="neutral", confidence=0.10,
            probabilities={"positive": 0.3, "negative": 0.3, "neutral": 0.4},
            raw_text=state.input_text,
        )
        return state


def test_end_to_end_escalated_run_produces_valid_label_and_trace():
    o = build_orchestrator(
        _tc(), threshold=0.7, enable_deliberation=False,
        primary_classifier=_AlwaysEscalatePrimary(),
        sentiment_agent_variant="sequential_sentiment_v1",
    )
    st = _state()
    out = o.run(st)
    # escalated → sequential path ran → valid final label, no crash
    assert out.routing_info.decision == "escalate"
    assert out.final_output.label in LABELS
    store = out.extras[SEQ_KEY]
    assert {"intent", "polarity", "pragmatic", "decided_by", "final_label"} <= store.keys()
    assert out.final_output.payload["source"] == "sequential_sentiment_v1"
