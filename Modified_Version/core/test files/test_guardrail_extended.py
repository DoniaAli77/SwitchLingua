"""
test_guardrail_extended.py
Extended OFFLINE coverage for the Rewrite Guardrail + task-aware meet_criteria.
Complements test_refiner_guardrail.py (which covers: quality accepted, quality task-broke
rollback, task fixed accepted, quality-regression rollback). Adds the missing branches:

  5. task_fail  -> candidate STILL fails task            -> ROLLBACK (budget spent)
  6. task_fail  -> task fixed but quality got WORSE      -> ACCEPT (asymmetry: regression tolerated)
  7. validator malfunction (missing 'passed' key)        -> never reject on malfunction -> ACCEPT
  8. refiner returns empty/blank candidate               -> original kept, budget spent
  9. quality_fail -> rescore EQUAL to before             -> ACCEPT (only strict regression rolls back)
 10-12. meet_criteria routing: task-failing high-quality sentence -> Refiner;
        all-pass -> Acceptance; task-failing but budget spent -> Acceptance.
All offline (mocked LLM/prompts); no API.
"""
import os
import sys
from unittest.mock import patch

CORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

import node_engine
import run_french


# ---------------------------------------------------------------------------
# Mock plumbing (same pattern as test_refiner_guardrail.py)
# ---------------------------------------------------------------------------

class _FakePrompt:
    def __init__(self, mode: str):
        self.mode = mode

    def __or__(self, llm):
        if hasattr(llm, "set_mode"):
            llm.set_mode(self.mode)
        return llm


def _quality_payload(mode, score):
    from types import SimpleNamespace
    import json
    key = {"fluency": "fluency_score", "naturalness": "naturalness_score",
           "cs_ratio": "ratio_score", "social": "socio_cultural_score"}[mode]
    body = {key: score, "errors": {}, "observations": {}, "issues": "", "summary": "ok",
            "computed_ratio": "50%:50%", "notes": "ok"}
    return SimpleNamespace(content=json.dumps(body))


class _ConfigurableLLM:
    """One mock for all modes. Class attrs configure behaviour per test."""
    refiner_instances = ["refined sentence"]
    task_refiner_instances = ["task-fixed sentence"]
    validator_response = {"passed": True, "confidence": 0.9, "notes": "ok",
                          "predicted_label": "positive", "errors": []}
    rescore_value = 9.0  # all four quality agents return this -> weighted == value

    def __init__(self, *args, **kwargs):
        self.mode = "unset"

    def set_mode(self, mode):
        self.mode = mode
        return self

    def with_structured_output(self, _):
        return self

    def invoke(self, payload):
        if self.mode == "refiner":
            return {"instances": list(type(self).refiner_instances)}
        if self.mode == "refiner_task":
            return {"instances": list(type(self).task_refiner_instances)}
        if self.mode in {"val_topic", "val_sentiment", "val_ner"}:
            resp = type(self).validator_response
            return dict(resp) if isinstance(resp, dict) else resp
        if self.mode in {"fluency", "naturalness", "cs_ratio", "social"}:
            return _quality_payload(self.mode, type(self).rescore_value)
        return {}


def _patches():
    return [
        patch.object(node_engine, "ChatOpenAI", _ConfigurableLLM),
        patch.object(node_engine, "REFINER_PROMPT", _FakePrompt("refiner")),
        patch.object(node_engine, "REFINER_TASK_TOPIC_PROMPT", _FakePrompt("refiner_task")),
        patch.object(node_engine, "REFINER_TASK_SENTIMENT_PROMPT", _FakePrompt("refiner_task")),
        patch.object(node_engine, "REFINER_TASK_NER_PROMPT", _FakePrompt("refiner_task")),
        patch.object(node_engine, "TASK_VALIDATION_TOPIC_PROMPT", _FakePrompt("val_topic")),
        patch.object(node_engine, "TASK_VALIDATION_SENTIMENT_PROMPT", _FakePrompt("val_sentiment")),
        patch.object(node_engine, "TASK_VALIDATION_NER_PROMPT", _FakePrompt("val_ner")),
        patch.object(node_engine, "FLUENCY_PROMPT", _FakePrompt("fluency")),
        patch.object(node_engine, "NATURALNESS_PROMPT", _FakePrompt("naturalness")),
        patch.object(node_engine, "CS_RATIO_PROMPT", _FakePrompt("cs_ratio")),
        patch.object(node_engine, "SOCIAL_CULTURAL_PROMPT", _FakePrompt("social")),
    ]


def _run_refiner(state):
    ps = _patches()
    for p in ps:
        p.start()
    try:
        return node_engine.RunRefinerAgent(state)
    finally:
        for p in ps:
            p.stop()


def _record(index, text, task_passed=True, weighted_score=6.0, refine_count=0):
    return {
        "index": index, "text": text, "weighted_score": weighted_score,
        "refine_count": refine_count,
        "status": "fail" if weighted_score < 8.0 or not task_passed else "pass",
        "task_passed": task_passed,
        "task_validation": {"passed": task_passed, "feedback": "test feedback"},
        "fluency": {}, "naturalness": {}, "cs_ratio": {}, "socio_cultural": {},
    }


def _state(task, sentences, records, refine_counts):
    return {
        "task": task, "topic": "technology", "label": "positive",
        "first_language": "Arabic", "second_language": "English", "cs_ratio": "30%",
        "task_constraints": {"entity_types": ["PER", "ORG", "LOC"]},
        "task_validation_results_per_instances": [],
        "mcp_result": "", "data_generation_result": list(sentences),
        "sentence_records": records, "instance_refine_counts": list(refine_counts),
    }


# ---------------------------------------------------------------------------
# 5. task_fail -> candidate STILL fails -> ROLLBACK, budget spent
# ---------------------------------------------------------------------------

def test_task_fail_still_failing_rollback():
    _ConfigurableLLM.task_refiner_instances = ["still wrong sentence"]
    _ConfigurableLLM.validator_response = {"passed": False, "confidence": 0.2,
                                           "notes": "still wrong", "predicted_label": "negative", "errors": []}
    original = "task-wrong original"
    st = _state("sentiment", [original], [_record(0, original, task_passed=False, weighted_score=9.0)], [0])
    out = _run_refiner(st)
    assert out["data_generation_result"][0] == original, "candidate must be rolled back"
    assert out["instance_refine_counts"][0] == 1, "budget must be spent on the failed attempt"
    print("PASS test_task_fail_still_failing_rollback")


# ---------------------------------------------------------------------------
# 6. task_fail -> task FIXED but quality WORSE -> ACCEPT (asymmetry)
# ---------------------------------------------------------------------------

def test_task_fix_tolerates_quality_regression():
    _ConfigurableLLM.task_refiner_instances = ["task fixed, clunkier wording"]
    _ConfigurableLLM.validator_response = {"passed": True, "confidence": 0.95,
                                           "notes": "fixed", "predicted_label": "positive", "errors": []}
    _ConfigurableLLM.rescore_value = 3.0  # would be a big regression IF rescore were consulted
    st = _state("sentiment", ["task-wrong original"],
                [_record(0, "task-wrong original", task_passed=False, weighted_score=9.0)], [0])
    out = _run_refiner(st)
    assert out["data_generation_result"][0] == "task fixed, clunkier wording", \
        "task-fix must be kept even when quality would regress (asymmetric guardrail)"
    assert out["instance_refine_counts"][0] == 1
    _ConfigurableLLM.rescore_value = 9.0
    print("PASS test_task_fix_tolerates_quality_regression")


# ---------------------------------------------------------------------------
# 7. validator malfunction (missing 'passed' key) -> treated as pass -> ACCEPT
# ---------------------------------------------------------------------------

def test_validator_malfunction_never_rejects():
    _ConfigurableLLM.refiner_instances = ["improved sentence"]
    _ConfigurableLLM.validator_response = {"confidence": 0.5}  # no 'passed' key at all
    _ConfigurableLLM.rescore_value = 9.0  # quality improves
    st = _state("sentiment", ["mediocre original"],
                [_record(0, "mediocre original", task_passed=True, weighted_score=6.0)], [0])
    out = _run_refiner(st)
    assert out["data_generation_result"][0] == "improved sentence", \
        "malfunctioning validator (no 'passed') must not cause rollback"
    print("PASS test_validator_malfunction_never_rejects")


# ---------------------------------------------------------------------------
# 8. refiner returns blank candidate -> original kept, budget spent
# ---------------------------------------------------------------------------

def test_blank_candidate_keeps_original():
    _ConfigurableLLM.refiner_instances = ["   "]  # blank
    _ConfigurableLLM.validator_response = {"passed": True, "confidence": 0.9,
                                           "notes": "ok", "predicted_label": "positive", "errors": []}
    original = "original text"
    st = _state("sentiment", [original], [_record(0, original, task_passed=True, weighted_score=6.0)], [0])
    out = _run_refiner(st)
    assert out["data_generation_result"][0] == original, "blank candidate must not replace original"
    assert out["instance_refine_counts"][0] == 1, "budget still spent on the attempt"
    _ConfigurableLLM.refiner_instances = ["refined sentence"]
    print("PASS test_blank_candidate_keeps_original")


# ---------------------------------------------------------------------------
# 9. quality_fail -> rescore EQUAL to before -> ACCEPT (strict < rolls back)
# ---------------------------------------------------------------------------

def test_equal_quality_is_accepted():
    _ConfigurableLLM.refiner_instances = ["same-quality rewrite"]
    _ConfigurableLLM.validator_response = {"passed": True, "confidence": 0.9,
                                           "notes": "ok", "predicted_label": "positive", "errors": []}
    _ConfigurableLLM.rescore_value = 6.0  # all four 6.0 -> weighted exactly 6.0 == before
    st = _state("sentiment", ["original 6.0 sentence"],
                [_record(0, "original 6.0 sentence", task_passed=True, weighted_score=6.0)], [0])
    out = _run_refiner(st)
    assert out["data_generation_result"][0] == "same-quality rewrite", \
        "equal (non-regressing) quality must be accepted"
    _ConfigurableLLM.rescore_value = 9.0
    print("PASS test_equal_quality_is_accepted")


# ---------------------------------------------------------------------------
# 10-12. meet_criteria routing (task-aware gate in run_french)
# ---------------------------------------------------------------------------

def test_meet_criteria_routes_task_failing_high_quality():
    st = {"sentence_records": [_record(0, "fluent but wrong task", task_passed=False, weighted_score=9.0)]}
    assert run_french.meet_criteria(st) == "RefinerAgent", \
        "task-failing sentence must be refine-eligible even at quality >= 8"
    print("PASS test_meet_criteria_routes_task_failing_high_quality")


def test_meet_criteria_all_pass_goes_to_acceptance():
    st = {"sentence_records": [_record(0, "good", task_passed=True, weighted_score=9.0),
                               _record(1, "also good", task_passed=True, weighted_score=8.2)]}
    assert run_french.meet_criteria(st) == "AcceptanceAgent"
    print("PASS test_meet_criteria_all_pass_goes_to_acceptance")


def test_meet_criteria_budget_spent_goes_to_acceptance():
    st = {"sentence_records": [_record(0, "still wrong, budget gone", task_passed=False,
                                       weighted_score=9.0, refine_count=1)]}
    assert run_french.meet_criteria(st) == "AcceptanceAgent", \
        "budget-exhausted sentence must not loop forever"
    print("PASS test_meet_criteria_budget_spent_goes_to_acceptance")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running EXTENDED guardrail + gate tests...")
    test_task_fail_still_failing_rollback()
    test_task_fix_tolerates_quality_regression()
    test_validator_malfunction_never_rejects()
    test_blank_candidate_keeps_original()
    test_equal_quality_is_accepted()
    test_meet_criteria_routes_task_failing_high_quality()
    test_meet_criteria_all_pass_goes_to_acceptance()
    test_meet_criteria_budget_spent_goes_to_acceptance()
    print("ALL EXTENDED GUARDRAIL TESTS PASSED")
